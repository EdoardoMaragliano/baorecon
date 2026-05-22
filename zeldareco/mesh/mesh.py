# mesh.py
import numpy as np
import warnings
import logging
from typing import Optional, Union
from .field_ops import project_vector_field_jit, divergence
from zeldareco.utils.loggers import setup_logger
logger = setup_logger(__name__)


class Mesh:
    """
    Class representing a 3D mesh in configuration or Fourier space.
    """

    def __init__(self, nmesh: int, boxsize: float, boxcentre: np.ndarray, los: Optional[str] = None, dtype=np.float32):
        self.nmesh = int(nmesh)
        self.boxsize = float(boxsize)
        self.dtype = np.dtype(dtype)
        # line-of-sight specification: None, 'x', 'y', 'z' or radial
        self.los = None if los is None else str(los).strip().lower()

        self._validate_parameters()

        self.cell_size = self.boxsize / self.nmesh

        # center of the box in the observed frame
        self.boxcentre = np.asarray(boxcentre, dtype=self.dtype)
        if self.boxcentre.shape != (3,):
            raise ValueError("boxcentre must be a length-3 array-like of coordinates")

        # coordinates of the minimum corner of the box in the observed frame
        self.min_corner = self.boxcentre - self.boxsize / 2.0
        
        if self.cell_size % 1 != 0:
            warnings.warn("cell_size is not an integer number.", UserWarning)

        # Mesh attributes initialized as None (lazy)
        self._xmesh = None                  # lives in [0, boxsize]^3
        self._kmesh = None      
        self._radial_versor = None          
        self._radial_versor_k = None
        self._los_versor = None

    def _validate_parameters(self):
        if self.nmesh <= 0:
            raise ValueError("nmesh must be a positive integer")

        if not np.isfinite(self.boxsize) or self.boxsize <= 0:
            raise ValueError("boxsize must be a positive finite number")

        if self.los not in (None, 'x', 'y', 'z'):
            raise ValueError("los must be None, 'x', 'y', or 'z'")

    @property
    def xmesh(self):
        if self._xmesh is None:
            self._create_meshgrid(space='ConfigSpace')
        return self._xmesh

    @property
    def kmesh(self):
        if self._kmesh is None:
            self._create_meshgrid(space='FourierSpace')
        return self._kmesh

    @property
    def los_versor(self):
        if self._los_versor is None:
            self._set_los_versor()
        return self._los_versor

    @property
    def radial_versor(self):

        if self._radial_versor is None:
            # Compute the separation vector from the box center to each mesh point
            separation_vector = self.xmesh + self.min_corner
            norm = np.linalg.norm(separation_vector, axis=-1, keepdims=False)
            mask = norm > 0
            radial_versor = np.zeros_like(separation_vector, dtype=self.dtype)
            radial_versor[mask] = separation_vector[mask] / norm[mask, np.newaxis]
            self._radial_versor = radial_versor
        return self._radial_versor
    
    @property
    def radial_versor_k(self):
        if self._radial_versor_k is None:
            separation_vector = self.kmesh
            norm = np.linalg.norm(separation_vector, axis=-1, keepdims=False)
            mask = norm > 0
            radial_versor_k = np.zeros_like(separation_vector, dtype=self.dtype)
            radial_versor_k[mask] = separation_vector[mask] / norm[mask, np.newaxis]
            self._radial_versor_k = radial_versor_k
        return self._radial_versor_k

    def _create_meshgrid(self, space='ConfigSpace'):
        if space == 'ConfigSpace':
            x = np.linspace(0, self.boxsize, self.nmesh, endpoint=False)
            y = np.linspace(0, self.boxsize, self.nmesh, endpoint=False)
            z = np.linspace(0, self.boxsize, self.nmesh, endpoint=False)
            xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
            self._xmesh = np.stack([xx, yy, zz], axis=-1)
        elif space == 'FourierSpace':
            xx = np.fft.fftfreq(self.nmesh, d=self.cell_size) * 2 * np.pi
            yy = np.fft.fftfreq(self.nmesh, d=self.cell_size) * 2 * np.pi
            zz = np.fft.rfftfreq(self.nmesh, d=self.cell_size) * 2 * np.pi
            xx, yy, zz = np.meshgrid(xx, yy, zz, indexing='ij')
            self._kmesh = np.stack([xx, yy, zz], axis=-1)

    def _set_los_versor(self):
        if self.los is None:
            self._los_versor = self.radial_versor
        elif self.los == 'x':
            arr = np.zeros((self.nmesh, self.nmesh, self.nmesh, 3), dtype=self.dtype)
            arr[..., 0] = 1
            self._los_versor = arr
        elif self.los == 'y':
            arr = np.zeros((self.nmesh, self.nmesh, self.nmesh, 3), dtype=self.dtype)
            arr[..., 1] = 1
            self._los_versor = arr
        elif self.los == 'z':
            arr = np.zeros((self.nmesh, self.nmesh, self.nmesh, 3), dtype=self.dtype)
            arr[..., 2] = 1
            self._los_versor = arr
        else:
            raise ValueError(f"Invalid value for los: {self.los}. Must be 'x', 'y', 'z', or None.")

    def print_info(self):
        logger.info('Mesh Information:')
        logger.info(f'Boxsize: {self.boxsize} Mpc/h')
        logger.info(f'Number of grid points: {self.nmesh}')
        logger.info(f'Cell size: {self.cell_size} Mpc/h')
        logger.info(f'Boxcentre: {self.boxcentre}')
        logger.info(f'Line-of-sight: {self.los}')

    def get_parallel_component(self, vector_field: np.ndarray, out=None, dtype=np.float32) -> np.ndarray:
        if self.los_versor is None:
            self._set_los_versor()
        return project_vector_field_jit(vector_field, self.los_versor, out=out, dtype=dtype)
    
