"""Overdensity field management.

:class:`DensityManager` paints the data/random catalogues onto a mesh and
builds the overdensity field ``delta``. It is solver-agnostic and centralizes
box/position/weight formatting.
"""

from typing import Optional

import numpy as np

from baorecon.field_ops import smoothed_field
from baorecon.mas import assign
from baorecon.mesh.mesh import Mesh
from baorecon.utils.formatters import (
    format_boxcentre,
    format_boxsize,
    format_mas,
    format_nmesh,
    format_padding,
    format_weights,
    nmesh_boxsize_from_cellsize,
    set_boxcentre_from_positions,
    set_boxsize_from_positions,
    survey_to_box_frame,
)
from baorecon.utils.loggers import setup_logger

logger = setup_logger(__name__)

from baorecon.utils.backend import CUPY_AVAILABLE

if CUPY_AVAILABLE:
    import cupy as cp


class DensityManager:
    """Create and manage the overdensity field (delta) on data/random catalogues."""

    def __init__(
        self,
        data_pos: np.ndarray,
        random_pos: np.ndarray,
        nmesh=None,
        boxsize=None,
        boxcentre: Optional[np.ndarray] = None,
        padding: float = 0.01,
        MAS: str = "CIC",
        dtype=np.float32,
        data_weights: Optional[np.ndarray] = None,
        random_weights: Optional[np.ndarray] = None,
        pbc: bool = False,
        smoothing_radius: float = 0.0,
        device: str = "cpu",
        threshold_randoms: float = 0.01,
        sm_mode: str = "wrap",
        cellsize: Optional[float] = None,
        mas_parallel: bool = False
    ) -> None:
        self._raw_data_pos = data_pos
        self._raw_random_pos = random_pos
        self._raw_data_weights = data_weights
        self._raw_random_weights = random_weights

        # nmesh/boxsize are resolved (and possibly derived from cellsize) in
        # _prepare_inputs; keep the raw values for now.
        self.nmesh = nmesh
        self.cellsize = cellsize
        self.boxsize = boxsize
        self.boxcentre = boxcentre
        self.padding = padding
        self.MAS = MAS
        self.dtype = dtype
        self.pbc = pbc
        self.smoothing_radius = smoothing_radius
        self.device = device
        self.threshold_randoms = threshold_randoms
        self.sm_mode = sm_mode
        self.mas_parallel = mas_parallel

        self.data_pos_box: Optional[np.ndarray] = None
        self.random_pos_box: Optional[np.ndarray] = None
        self.data_weights: Optional[np.ndarray] = None
        self.random_weights: Optional[np.ndarray] = None

        self._mesh: Optional[Mesh] = None
        self._delta_on_mesh: Optional[np.ndarray] = None

        self._prepare_inputs()

    @property
    def mesh(self) -> Mesh:
        if self._mesh is None:
            logger.debug("Initializing Mesh inside DensityManager")
            self._mesh = Mesh(self.nmesh, self.boxsize, self.boxcentre, dtype=self.dtype)
        return self._mesh

    def _prepare_inputs(self) -> None:
        self.padding = format_padding(self.padding, self.pbc)

        if self.cellsize is not None:
            # cellsize fixes both the grid resolution and the box: it is mutually
            # exclusive with an explicit nmesh or boxsize.
            if self.nmesh is not None or self.boxsize is not None:
                raise ValueError(
                    "cellsize is mutually exclusive with nmesh and boxsize; "
                    "provide either cellsize, or nmesh (+optional boxsize)."
                )
            extent = (self._raw_random_pos.max(axis=0) - self._raw_random_pos.min(axis=0)) + self.padding
            self.nmesh, self.boxsize = nmesh_boxsize_from_cellsize(extent, self.cellsize, dtype=self.dtype)
            logger.info(f"cellsize={self.cellsize} -> nmesh={self.nmesh}, boxsize={self.boxsize}.")
        else:
            if self.nmesh is None:
                raise ValueError("Either nmesh or cellsize must be provided.")
            self.nmesh = format_nmesh(self.nmesh)

            if self.boxsize is None:
                self.boxsize = set_boxsize_from_positions(self._raw_random_pos, padding=self.padding, dtype=self.dtype)
                logger.info(f"Box size not provided. Set to {self.boxsize} with padding {self.padding}.")

            self.boxsize = format_boxsize(self.boxsize, positions=self._raw_random_pos, pbc=self.pbc, dtype=self.dtype)

        if self.boxcentre is None:
            self.boxcentre = set_boxcentre_from_positions(self._raw_random_pos, dtype=self.dtype)
            logger.info(f"Box centre not provided. Set to {self.boxcentre} based on positions.")

        self.boxcentre = format_boxcentre(self.boxcentre, dtype=self.dtype)

        self.data_pos_box = survey_to_box_frame(self._raw_data_pos, self.min_corner, self.boxsize,
                                                pbc=self.pbc, dtype=self.dtype)
        self.random_pos_box = survey_to_box_frame(self._raw_random_pos, self.min_corner, self.boxsize,
                                                  pbc=self.pbc, dtype=self.dtype)
        
        del self._raw_data_pos
        del self._raw_random_pos

        self.data_weights = format_weights(self._raw_data_weights, size=len(self.data_pos_box), dtype=self.dtype)
        self.random_weights = format_weights(self._raw_random_weights, size=len(self.random_pos_box), dtype=self.dtype)

        self.MAS = format_mas(self.MAS)

    @property
    def min_corner(self) -> np.ndarray:
        """Lower corner of the survey box in the original survey frame."""
        return self.boxcentre - self.boxsize / 2.0

    def compute_delta(self, sm_mode: str = "wrap") -> np.ndarray:
        if self.device == "gpu":
            if not CUPY_AVAILABLE:
                raise RuntimeError("GPU backend requested, but it is not available (CuPy/CUDA missing).")
            xp = cp
        else:
            xp = np

        logger.debug("Assigning data to mesh...")
        data_rho = assign(self.data_pos_box, self.data_weights, self.mesh,
                          scheme=self.MAS, device=self.device, pbc=self.pbc, parallel=self.mas_parallel)
        data_rho = smoothed_field(data_rho, self.mesh, self.smoothing_radius)

        logger.debug("Assigning randoms to mesh...")
        random_rho = assign(self.random_pos_box, self.random_weights, self.mesh,
                            scheme=self.MAS, device=self.device, pbc=self.pbc, parallel=self.mas_parallel)
        random_rho = smoothed_field(random_rho, self.mesh, self.smoothing_radius)

        logger.debug("Computing overdensity field...")
        alpha = xp.sum(data_rho) / xp.sum(random_rho)
        threshold = self.threshold_randoms * random_rho.sum() / len(self.random_pos_box)
        th_mask = random_rho <= threshold

        xp.multiply(random_rho, alpha, out=random_rho)
        xp.subtract(data_rho, random_rho, out=data_rho)
        xp.putmask(random_rho, th_mask, 1.0)
        xp.divide(data_rho, random_rho, out=data_rho)
        xp.putmask(data_rho, th_mask, 0.0)

        del random_rho
        del th_mask

        target_dtype = np.dtype(self.dtype)
        if data_rho.dtype != target_dtype or not data_rho.flags.c_contiguous:
            logger.info('casting data_rho to %s contiguous array.', target_dtype.name)
            data_rho = xp.ascontiguousarray(data_rho.astype(target_dtype, copy=False))

        self._delta_on_mesh = data_rho
        return self._delta_on_mesh

    @property
    def delta_on_mesh(self) -> np.ndarray:
        if self._delta_on_mesh is None:
            self.compute_delta(sm_mode=self.sm_mode)
        return self._delta_on_mesh
