"""Lightweight geometry-only mesh.

``Mesh`` is a small dataclass that describes the box geometry and nothing
else. It allocates no large arrays: wavevectors, real-space grids and
line-of-sight versors live in the solvers (``solvers/fft/_common.py``) and in
the line-of-sight strategies (``mesh/los.py``).

Both ``boxsize`` and ``nmesh`` may be given either as a scalar (cubic box) or
as a length-3 array (per-axis / rectangular box). Scalars are broadcast to a
length-3 representation so a scalar box and the equivalent cubic array behave
identically everywhere.
"""

from dataclasses import dataclass, field
from typing import Union

import numpy as np

from baorecon.utils.formatters import format_nmesh
from baorecon.utils.loggers import setup_logger

logger = setup_logger(__name__)


@dataclass
class Mesh:
    """Geometry of a 3D mesh.

    Parameters
    ----------
    nmesh : int or array-like of shape (3,)
        Number of grid points per axis. A scalar describes a cubic grid.
    boxsize : float or array-like of shape (3,)
        Physical box size per axis in Mpc/h. A scalar describes a cubic box.
    boxcentre : array-like of shape (3,)
        Centre of the box in the observed frame.
    dtype : data-type, optional
        Floating dtype used for fields painted onto this mesh. Default float32.

    Attributes
    ----------
    cell_size : ndarray, shape (3,)
        Per-axis cell size, ``boxsize / nmesh`` (derived).
    min_corner : ndarray, shape (3,)
        Lower corner of the box in the observed frame, ``boxcentre - boxsize/2``
        (derived).
    """

    nmesh: Union[int, np.ndarray]
    boxsize: Union[float, np.ndarray]
    boxcentre: np.ndarray
    dtype: np.dtype = np.float32

    # Derived quantities (populated in __post_init__).
    cell_size: np.ndarray = field(init=False)
    min_corner: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.dtype = np.dtype(self.dtype)

        self.nmesh = self._normalize_nmesh(self.nmesh)
        self.boxsize = self._normalize_boxsize(self.boxsize)

        self.boxcentre = np.asarray(self.boxcentre, dtype=np.float64)
        if self.boxcentre.shape != (3,):
            raise ValueError("boxcentre must be a length-3 array-like of coordinates")

        self._validate()

        self.cell_size = self.boxsize / self.nmesh
        self.min_corner = self.boxcentre - self.boxsize / 2.0

    @staticmethod
    def _normalize_nmesh(nmesh) -> np.ndarray:
        # format_nmesh broadcasts a scalar to (3,), rejects non-integer and
        # non-positive values, and validates the shape.
        return format_nmesh(nmesh)

    @staticmethod
    def _normalize_boxsize(boxsize) -> np.ndarray:
        arr = np.asarray(boxsize, dtype=np.float64)
        if arr.ndim == 0:
            arr = np.full(3, float(arr), dtype=np.float64)
        elif arr.shape != (3,):
            raise ValueError("boxsize must be a scalar or a length-3 array-like")
        return arr

    def _validate(self) -> None:
        if (self.nmesh <= 0).any():
            raise ValueError("nmesh must be a positive integer (per axis)")
        if not np.all(np.isfinite(self.boxsize)) or (self.boxsize <= 0).any():
            raise ValueError("boxsize must be a positive finite number (per axis)")

    @property
    def shape(self) -> tuple:
        """Grid shape ``(Nx, Ny, Nz)``."""
        return tuple(int(n) for n in self.nmesh)

    def print_info(self) -> None:
        logger.info("Mesh Information:")
        logger.info(f"Boxsize: {self.boxsize} Mpc/h")
        logger.info(f"Number of grid points: {self.nmesh}")
        logger.info(f"Cell size: {self.cell_size} Mpc/h")
        logger.info(f"Boxcentre: {self.boxcentre}")
