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
        mas_parallel:bool = False
    ) -> None:
        
        self.nmesh = int(nmesh)
        self.boxsize = boxsize 
        self.boxcentre = boxcentre
        self.padding = padding
        self.MAS = MAS
        self.dtype = dtype
        self.pbc = pbc
        self.los = los
        self.smoothing_radius = smoothing_radius
        self.mas_parallel = mas_parallel

        # Salviamo la lunghezza dei random qui, perché non conserveremo l'array grezzo
        self.n_randoms = len(random_pos)

        self._mesh: Optional[Mesh] = None
        self._delta_on_mesh: Optional[np.ndarray] = None

        # Eseguiamo la formattazione passando direttamente le variabili locali
        self._prepare_inputs(data_pos, random_pos, data_weights, random_weights)

    def _prepare_inputs(self, data_pos, random_pos, data_weights, random_weights) -> None:
        """Elabora le posizioni senza salvare copie inutili nello stato."""
        
        self.padding = format_padding(self.padding, self.pbc)
        
        if self.boxsize is None:
            self.boxsize = set_boxsize_from_positions(random_pos, padding=self.padding)
        self.boxsize = format_boxsize(self.boxsize, positions=random_pos, pbc=self.pbc)

        if self.boxcentre is None:
            self.boxcentre = set_boxcentre_from_positions(random_pos, dtype=self.dtype)
        self.boxcentre = format_boxcentre(self.boxcentre, dtype=self.dtype)

        # Creiamo gli array formattati.
        self.data_pos_box = survey_to_box_frame(data_pos, self.min_corner, self.boxsize, 
                                                pbc=self.pbc, dtype=self.dtype)
        self.random_pos_box = survey_to_box_frame(random_pos, self.min_corner, self.boxsize, 
                                                  pbc=self.pbc, dtype=self.dtype)

        self.data_weights = format_weights(data_weights, size=len(data_pos), dtype=self.dtype)
        self.random_weights = format_weights(random_weights, size=self.n_randoms, dtype=self.dtype)
        self.MAS = format_mas(self.MAS)

    @property
    def min_corner(self) -> np.ndarray:
        return self.boxcentre - self.boxsize / 2.0

    @property
    def mesh(self) -> Mesh:
        if self._mesh is None:
            self._mesh = Mesh(self.nmesh, self.boxsize, self.boxcentre, los=self.los, dtype=self.dtype)
        return self._mesh

    def compute_delta(self, threshold_randoms: float = 0.01, sm_mode: str = "wrap", mas_parallel:bool=False) -> np.ndarray:
        # --- 1. PROCESSIAMO I DATI ---
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
            parallel=mas_parallel,
        )
        
        # Eliminiamo subito i dati dei cataloghi che non servono più
        self.data_pos_box = None

        data_rho = smoothed_field(data_rho, self.mesh, self.smoothing_radius, pbc=self.pbc, mode=sm_mode)

        # --- 2. PROCESSIAMO I RANDOM ---
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
            parallel=mas_parallel,
        )

        # Eliminiamo subito i cataloghi random
        self.random_pos_box = None

        random_rho = smoothed_field(random_rho, self.mesh, self.smoothing_radius, pbc=self.pbc, mode=sm_mode)

        # --- 3. CALCOLO OVERDENSITY ---
        logger.debug("Computing overdensity field...")
        alpha = np.sum(data_rho) / np.sum(random_rho)
        threshold = threshold_randoms * random_rho.sum() / self.n_randoms  # <--- Usiamo il valore salvato
        th_mask = random_rho > threshold

        data_rho /= alpha 
        np.divide(data_rho, random_rho, out=data_rho, where=th_mask)
        np.subtract(data_rho, 1.0, out=data_rho, where=th_mask)
        data_rho[~th_mask] = 0.0

        del random_rho
        del th_mask

        self._delta_on_mesh = data_rho
        return self._delta_on_mesh

    @property
    def delta_on_mesh(self) -> np.ndarray:
        if self._delta_on_mesh is None:
            self.compute_delta(mas_parallel=self.mas_parallel)
        return self._delta_on_mesh