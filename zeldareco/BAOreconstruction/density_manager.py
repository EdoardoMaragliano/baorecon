import numpy as np
from typing import Optional
from zeldareco.mesh.mesh import Mesh
from zeldareco.mass_assignment import mass_assignment
from zeldareco.mesh.field_ops import smoothed_field
from zeldareco.utils.loggers import setup_logger
from zeldareco.utils.formatters import (
    format_boxsize,
    format_boxcentre,
    format_padding,
    set_boxcentre_from_positions,
    set_boxsize_from_positions,
    survey_to_box_frame,
    format_weights,
    format_mas,
)

logger = setup_logger(__name__)


class DensityManager:
    """
    Create and manage the overdensity field (delta) on data/random catalogues.

    This class is independent and reusable by any solver. It centralizes
    box/position/weights formatting and supports periodic wrapping (pbc).
    """

    def __init__(
        self,
        data_pos: np.ndarray,
        random_pos: np.ndarray,
        nmesh: int,
        boxsize,
        boxcentre: Optional[np.ndarray] = None,
        padding: float = 0.01,
        MAS: str = "CIC",
        dtype=np.float32,
        data_weights: Optional[np.ndarray] = None,
        random_weights: Optional[np.ndarray] = None,
        pbc: bool = False,
        los: Optional[str] = None,
        smoothing_radius: float = 0.0,
    ) -> None:
        # store raw inputs
        self._raw_data_pos = np.array(data_pos, copy=True)
        self._raw_random_pos = np.array(random_pos, copy=True)
        self._raw_data_weights = data_weights
        self._raw_random_weights = random_weights

        self.nmesh = int(nmesh)
        self.boxsize = boxsize 
        self.boxcentre = boxcentre
        self.padding = padding
        self.MAS = MAS
        self.dtype = dtype
        self.pbc = pbc
        self.los = los
        self.smoothing_radius = smoothing_radius

        # formatted/validated attributes (filled by _prepare_inputs)
        self.data_pos_box: Optional[np.ndarray] = None
        self.random_pos_box: Optional[np.ndarray] = None
        self.data_weights: Optional[np.ndarray] = None
        self.random_weights: Optional[np.ndarray] = None

        self._mesh: Optional[Mesh] = None
        self._delta_on_mesh: Optional[np.ndarray] = None

        # perform formatting and validation
        self._prepare_inputs()

    @property
    def mesh(self) -> Mesh:
        if self._mesh is None:
            logger.debug("Initializing Mesh inside DensityManager")
            self._mesh = Mesh(self.nmesh, self.boxsize, self.boxcentre, los=self.los, dtype=self.dtype)
        return self._mesh

    def _prepare_inputs(self) -> None:
        """Infer/validate box parameters, format positions and weights, validate MAS.

        This centralizes the behaviour so callers can pass raw inputs.
        """
        #format padding
        self.padding = format_padding(self.padding, self.pbc)
        
        if self.boxsize is None:
            self.boxsize = set_boxsize_from_positions(self._raw_random_pos, padding=self.padding)
            logger.info(f"Box size not provided. Set to {self.boxsize} based on positions with padding {self.padding}.")

        self.boxsize = format_boxsize(self.boxsize, positions=self._raw_random_pos, pbc=self.pbc)

        # infer or validate boxcentre
        if self.boxcentre is None:
            self.boxcentre = set_boxcentre_from_positions(self._raw_random_pos, dtype=self.dtype)
            logger.info(f"Box centre not provided. Set to {self.boxcentre} based on positions.")
        
        self.boxcentre = format_boxcentre(self.boxcentre, dtype=self.dtype)

        # prepare positions: shift so that min corner is at 0 and optionally wrap
        #min_corner = self.boxcentre - self.boxsize / 2.0

        self.data_pos_box = survey_to_box_frame(self._raw_data_pos, self.min_corner, self.boxsize, 
                                                pbc=self.pbc, dtype=self.dtype)
        self.random_pos_box = survey_to_box_frame(self._raw_random_pos, self.min_corner, self.boxsize, 
                                                  pbc=self.pbc, dtype=self.dtype)

        # format weights
        self.data_weights = format_weights(self._raw_data_weights, size=len(self.data_pos_box), dtype=self.dtype)
        self.random_weights = format_weights(self._raw_random_weights, size=len(self.random_pos_box), dtype=self.dtype)

        # validate MAS string
        self.MAS = format_mas(self.MAS)

    @property
    def min_corner(self) -> np.ndarray:
        """Lower corner of the survey box in the original survey frame."""
        return self.boxcentre - self.boxsize / 2.0

    def compute_delta(self, threshold_randoms: float = 0.7, sm_mode: str = "wrap") -> np.ndarray:
        """Compute the overdensity field on the mesh and cache it."""
        logger.debug("Assigning data to mesh...")
        data_rho = mass_assignment(
            pos=self.data_pos_box,
            boxsize=self.mesh.boxsize,
            nmesh=self.mesh.nmesh,
            weights=self.data_weights,
            pbc=self.pbc,
            method=self.MAS,
            dtype=self.mesh.dtype,
            verbose=False,
            parallel=False,
        )

        data_rho = smoothed_field(data_rho, self.mesh, self.smoothing_radius, pbc=self.pbc, mode=sm_mode)

        logger.debug("Assigning randoms to mesh...")
        random_rho = mass_assignment(
            pos=self.random_pos_box,
            boxsize=self.mesh.boxsize,
            nmesh=self.mesh.nmesh,
            weights=self.random_weights,
            pbc=self.pbc,
            method=self.MAS,
            dtype=self.mesh.dtype,
            verbose=False,
            parallel=False,
        )

        random_rho = smoothed_field(random_rho, self.mesh, self.smoothing_radius, pbc=self.pbc, mode=sm_mode)

        logger.debug("Computing overdensity field...")
        alpha = np.sum(data_rho) / np.sum(random_rho)
        delta_field = np.zeros_like(random_rho, dtype=self.mesh.dtype)
        mask = random_rho > 0.0
        delta_field[mask] = (data_rho[mask] - alpha * random_rho[mask])

        threshold = threshold_randoms * random_rho.sum() / len(self.random_pos_box)  # random_rho.size
        th_mask = random_rho > threshold

        delta_field[th_mask] /= (alpha * random_rho[th_mask])
        delta_field[~th_mask] = 0.0

        self._delta_on_mesh = delta_field
        return self._delta_on_mesh

    @property
    def delta_on_mesh(self) -> np.ndarray:
        if self._delta_on_mesh is None:
            self.compute_delta()
        return self._delta_on_mesh
