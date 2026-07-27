"""BAO reconstruction orchestrator.

Builds the overdensity field, solves for the displacement, and shifts the
data/random catalogues. The compute device, displacement solver class and
line-of-sight strategy are all resolved once in ``__init__``; there is no
``if device == ...`` branching scattered through the methods.
"""

from typing import Optional, Tuple

import numpy as np

from baorecon.field_ops import project_vector_field
from baorecon.mesh.los import FixedAxisLOS, LocalLOS
from baorecon.mesh.mesh import Mesh
from baorecon.reconstruction.density import DensityManager
from baorecon.solvers.fft import FFTSolverCPU, FFTSolverGPU
from baorecon.solvers.multigrid import MultigridSolver
from baorecon.utils.formatters import format_positions, format_rectype, format_rsd_space, survey_to_box_frame
from baorecon.utils.loggers import setup_logger

logger = setup_logger(__name__)

try:
    import cupy as cp
    from numba import cuda
    CUPY_AVAILABLE = cuda.is_available()
except ImportError:
    CUPY_AVAILABLE = False

_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


class BAOReconstructor:
    """Unified BAO reconstruction orchestrator (FFT or multigrid solver)."""

    def __init__(
        self,
        data_pos: np.ndarray,
        random_pos: np.ndarray,
        data_weights: Optional[np.ndarray] = None,
        random_weights: Optional[np.ndarray] = None,
        data_ids: Optional[np.ndarray] = None,
        RSDspace: str = "RedshiftSpace",
        nmesh=None,
        boxsize=None,
        boxcentre: Optional[np.ndarray] = None,
        padding: float = 0.01,
        los: Optional[str] = None,
        R_sm: float = 15,
        pbc: bool = False,
        rectype: str = "rec-sym",
        f: float = None,
        bias: float = None,
        MAS: str = "CIC",
        dtype=np.float32,
        threshold_randoms: float = 0.01,
        solver_type: str = "multigrid",
        solver_args: Optional[dict] = None,
        device: str = "cpu",
        cellsize: Optional[float] = None,
        mas_parallel: bool = False,
        **kwargs,
    ) -> None:
        self._padding = padding
        self._dtype = dtype

        # nmesh may be a scalar (cubic) or a per-axis (3,) array. When neither
        # nmesh nor cellsize is given, fall back to the historical 256^3 grid.
        if nmesh is None and cellsize is None:
            raise ValueError("Either nmesh or cellsize MUST be provided!")
        if bias is None:
            raise ValueError("bias must be provided!")
        if f is None:
            raise ValueError("growth rate f must be provided!")

        # Cast the (often float64) input catalogues to the working precision
        # (float32 by default). This is the single biggest memory saving: the
        # random catalogue is typically 10-50x larger than the data catalogue.
        # Double precision is opt-in via ``dtype`` and propagates from here.
        self.data_pos = format_positions(data_pos, dtype=self._dtype)
        self.random_pos = format_positions(random_pos, dtype=self._dtype)

        # Normalize the LOS string once.
        self._los = None if los is None else str(los).strip().lower()
        if self._los is not None and self._los not in _AXIS_INDEX:
            raise ValueError("los must be None, 'x', 'y' or 'z'")

        self._density_manager = DensityManager(
            data_pos=self.data_pos,
            random_pos=self.random_pos,
            nmesh=nmesh,
            boxsize=boxsize,
            boxcentre=boxcentre,
            padding=self._padding,
            MAS=MAS,
            dtype=dtype,
            data_weights=data_weights,
            random_weights=random_weights,
            pbc=pbc,
            smoothing_radius=R_sm,
            device=device,
            threshold_randoms=threshold_randoms,
            cellsize=cellsize,
            mas_parallel=mas_parallel
        )

        # Read back the resolved geometry (nmesh/boxsize may be derived from cellsize).
        self._nmesh = self._density_manager.nmesh
        self._boxsize = self._density_manager.boxsize
        self._boxcentre = self._density_manager.boxcentre
        self.data_weights = self._density_manager.data_weights
        self.random_weights = self._density_manager.random_weights
        self.MAS = self._density_manager.MAS
        self.mas_parallel = mas_parallel

        if data_ids is not None and len(data_ids) != len(self.data_pos):
            raise ValueError("data_ids length must match data_pos length")
        self.data_ids = data_ids

        self._rectype = format_rectype(rectype)
        self._RSDspace = format_rsd_space(RSDspace)
        self._R_sm = R_sm
        self._pbc = pbc
        self._f = f
        self._bias = bias
        self._threshold_randoms = threshold_randoms
        self._solver_type = solver_type
        self._solver_args = solver_args or {}
        self._device = device

        # --- Resolve device-dependent strategy ONCE ---
        self._los_strategy = self._build_los_strategy()
        self._solver_class = self._select_solver_class()

        self._solver = None

    # ------------------------------------------------------------------
    # One-time strategy resolution
    # ------------------------------------------------------------------
    def _build_los_strategy(self):
        if self._los is None:
            return LocalLOS(
                boxcentre=self._density_manager.boxcentre,
                min_corner=self._density_manager.min_corner,
                boxsize=self._density_manager.boxsize,
                nmesh=self.mesh.nmesh,
                device=self._device,
            )
        return FixedAxisLOS(_AXIS_INDEX[self._los])

    def _select_solver_class(self):
        if self._solver_type == "ifft":
            return FFTSolverGPU if self._device == "gpu" else FFTSolverCPU
        elif self._solver_type == "multigrid":
            return MultigridSolver
        raise ValueError(f"Unknown solver_type: {self._solver_type}")

    # ------------------------------------------------------------------
    # Properties (lazy)
    # ------------------------------------------------------------------
    @property
    def mesh(self) -> Mesh:
        return self._density_manager.mesh

    @property
    def delta_on_mesh(self) -> np.ndarray:
        return self._density_manager.delta_on_mesh

    @property
    def solver(self):
        if self._solver is None:
            logger.debug(f"Initializing {self._solver_type.upper()} solver ({self._device})...")
            if self._solver_type == "ifft":
                n_iter = self._solver_args.get("n_iterations", 3)
                self._solver = self._solver_class(
                    delta_on_mesh=self.delta_on_mesh,
                    mesh=self.mesh,
                    los=self._los_strategy,
                    f=self._f,
                    bias=self._bias,
                    RSDspace=self._RSDspace,
                    n_iterations=n_iter,
                    pbc=self._pbc,
                )
            else:  # multigrid 
                delta_on_mesh = self.delta_on_mesh
                if CUPY_AVAILABLE and isinstance(delta_on_mesh, cp.ndarray):
                    delta_on_mesh = cp.asnumpy(delta_on_mesh)
                smoother = self._solver_args.get("smoother", "jacobi")
                self._solver = self._solver_class(
                    delta_on_mesh=delta_on_mesh,
                    mesh=self.mesh,
                    los=self._los_strategy,
                    f=self._f if self._RSDspace == "RedshiftSpace" else 0.0,
                    bias=self._bias,
                    RSDspace=self._RSDspace,
                    use_plane_parallel=self._los is not None,
                    smoother = smoother,
                )
        return self._solver

    @property
    def boxsize(self):
        return self._boxsize

    @property
    def nmesh(self):
        return self._nmesh

    @property
    def cell_size(self):
        return self._boxsize / self._nmesh

    @property
    def boxcentre(self):
        return self._boxcentre

    @property
    def rectype(self):
        return self._rectype

    @property
    def los(self):
        return self._los

    @property
    def R_sm(self):
        return self._R_sm

    @property
    def pbc(self):
        return self._pbc

    @property
    def f(self):
        return self._f

    @property
    def bias(self):
        return self._bias

    @property
    def dtype(self):
        return self._dtype

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------
    def print_info(self):
        logger.info("=" * 60)
        logger.info(f"BAO Reconstruction ({self._solver_type.upper()}) Information:")
        logger.info("=" * 60)
        logger.info(f"Mesh: {self._nmesh}^3 points")
        logger.info(f"Box size: {self._boxsize} Mpc/h")
        logger.info(f"Box centre: {self._boxcentre}")
        logger.info(f"Padding: {self._padding} Mpc/h")
        logger.info(f"Cell size: {np.array2string(np.asarray(self.cell_size), precision=4, separator=', ')} Mpc/h")
        logger.info(f"Line-of-sight: {self._los}")
        logger.info(f"random-density threshold: {self._threshold_randoms}")
        logger.info(f"Smoothing radius: {self._R_sm} Mpc/h")
        logger.info(f"Periodic BC: {self._pbc}")
        logger.info(f"Reconstruction type: {self._rectype}")
        logger.info(f"Mass Assignment Scheme: {self.MAS}")
        logger.info(f"RSD Space: {self._RSDspace}")
        logger.info(f"Growth rate f: {self._f}")
        logger.info(f"Bias: {self._bias}")
        logger.info(f"Data points: {len(self.data_pos)}")
        logger.info(f"Random points: {len(self.random_pos)}")
        logger.info(f"mas parallel is: {self.mas_parallel}")
        logger.info(f"solver args {self._solver_args}")
        logger.info("=" * 60)

    def _interpolate_displacement(self, positions: np.ndarray, survey_frame: bool = True) -> np.ndarray:
        if survey_frame:
            pos_for_interp = survey_to_box_frame(positions, self._density_manager.min_corner,
                                                 self._boxsize, pbc=self._pbc, dtype=self._dtype)
        else:
            pos_for_interp = positions

        # Every solver reads the displacement out through the same interface and
        # returns a host (N, 3) array; the FFT/multigrid difference (interpolate a
        # spectral Psi grid vs. differentiate the potential on the fly) is hidden
        # behind read_displacement_at.
        return self.solver.read_displacement_at(pos_for_interp, mas=self.MAS)

    def _get_rsd_displacement(self, tracer_psi: np.ndarray, tracer_pos: np.ndarray) -> np.ndarray:
        if self._RSDspace == "RealSpace":
            logger.warning("RSD displacement requested but RSDspace is 'RealSpace'. Returning zeros.")
            return np.zeros_like(tracer_psi)

        if self._los is None:
            # Local line-of-sight in the original survey frame.
            norm = np.linalg.norm(tracer_pos, axis=1, keepdims=True)
            mask = norm[:, 0] > 0
            tracers_los = np.zeros_like(tracer_pos)
            np.divide(tracer_pos, norm, out=tracers_los, where=norm > 0)
            logger.debug("Using local line-of-sight for RSD displacement")
            psi_parallel = project_vector_field(tracer_psi, tracers_los)
            psi_parallel *= self._f
        else:
            # Fixed plane-parallel line-of-sight (same for all tracers).
            direction = self._los_strategy.direction
            psi_parallel = (tracer_psi @ direction).reshape(-1, 1) * direction
            psi_parallel *= self._f
        return psi_parallel

    def _shift_gals(self) -> np.ndarray:
        if self._RSDspace == "RealSpace":
            logger.info("Computing displacement for galaxies (real space)...")
            return self.data_pos - self._interpolate_displacement(self.data_pos)
        elif self._RSDspace == "RedshiftSpace":
            logger.info("Computing RSD displacement for galaxies...")
            tracer_psi = self._interpolate_displacement(self.data_pos)
            rsd = self._get_rsd_displacement(tracer_psi, self.data_pos)
            tracer_psi += rsd
            return self.data_pos - tracer_psi
        raise ValueError(f"Unknown RSDspace: {self._RSDspace}")

    def _shift_randoms(self) -> np.ndarray:
        if self._rectype == "rec-sym":
            logger.info("Computing displacement for randoms (symmetric reconstruction)...")
            if self._RSDspace == "RealSpace":
                return self.random_pos - self._interpolate_displacement(self.random_pos, survey_frame=True)
            elif self._RSDspace == "RedshiftSpace":
                tracer_psi = self._interpolate_displacement(self.random_pos, survey_frame=True)
                rsd = self._get_rsd_displacement(tracer_psi, self.random_pos)
                tracer_psi += rsd
                return self.random_pos - tracer_psi
            raise ValueError(f"Unknown RSDspace: {self._RSDspace}")
        elif self._rectype == "rec-iso":
            tracer_psi = self._interpolate_displacement(self.random_pos, survey_frame=True)
            return self.random_pos - tracer_psi
        raise ValueError(f"Unknown rectype: {self._rectype}")

    def run_reconstruction(self) -> Tuple[np.ndarray, np.ndarray]:
        logger.info(f"Starting BAO reconstruction ({self._solver_type.upper()})...")
        logger.info(f"Reconstruction type: {self._rectype}, RSD space: {self._RSDspace}")
        self.print_info()

        #logger.info("Solving for displacement field...")
        if self.solver == 'multigrid':
            _ = self.solver.potential
        else:
            _ = self.solver.displacement

        logger.info("Shifting galaxies...")
        data_pos_rec = self._shift_gals()
        logger.info("Shifting randoms...")
        random_pos_rec = self._shift_randoms()

        if self._pbc:
            min_corner = self._density_manager.min_corner
            data_pos_rec = (np.mod(data_pos_rec - min_corner, self._boxsize) + min_corner)
            random_pos_rec = (np.mod(random_pos_rec - min_corner, self._boxsize) + min_corner)

        logger.info("BAO reconstruction completed successfully")
        return data_pos_rec, random_pos_rec
