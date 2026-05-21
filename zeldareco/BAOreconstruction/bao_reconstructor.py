import numpy as np
from typing import Optional, Tuple
from zeldareco.BAOreconstruction.density_manager import DensityManager
from zeldareco.displacement_solver.fft_solver import FFTSolver
from zeldareco.displacement_solver.multigrid_solver import MultigridSolver
from zeldareco.mesh.field_ops import interpolate_vector_field, project_vector_field
from zeldareco.mesh.mesh import Mesh
from zeldareco.utils.loggers import setup_logger
from zeldareco.utils.formatters import (
    format_rectype, format_rsd_space, survey_to_box_frame,
)

logger = setup_logger(__name__)


class BAOReconstructor:
    """
    Unified BAO reconstruction orchestrator supporting both FFT and Multigrid solvers.

    Provides high-level interface for BAO reconstruction with lazy initialization
    of mesh, density field, and solver. Handles both real-space and redshift-space
    reconstructions with flexible reconstruction types (rec-sym, rec-iso).

    Parameters
    ----------
    data_pos : ndarray
        Positions of data objects, shape (N_data, 3)
    random_pos : ndarray
        Positions of random objects, shape (N_random, 3)
    data_weights : ndarray, optional
        Weights for data objects. Default is uniform weights.
    random_weights : ndarray, optional
        Weights for random objects. Default is uniform weights.
    data_ids : ndarray, optional
        IDs for data objects.
    RSDspace : str, optional
        Reconstruction space ('RealSpace' or 'RedshiftSpace'). Default is 'RedshiftSpace'.
    nmesh : int, optional
        Number of mesh points per dimension. Default is 256.
    boxsize : float, optional
        Size of the simulation box in Mpc/h. Default is computed from data/random positions.
    boxcentre : ndarray, optional
        Center of the box. Default is computed from data/random positions.
    padding : float, optional
        Padding for box size computation. Default is 0.0.
    los : str, optional
        Line-of-sight direction ('x', 'y', 'z', or None for local). Default is 'z'.
    R_sm : float, optional
        Smoothing radius in Mpc/h. Default is 15.
    pbc : bool, optional
        Use periodic boundary conditions. Default is True.
    rectype : str, optional
        Reconstruction type ('rec-sym' or 'rec-iso'). Default is 'rec-sym'.
    f : float, optional
        Growth rate. Default is 0.88.
    bias : float, optional
        Linear bias. Default is 1.0.
    MAS : str, optional
        Mass assignment scheme ('NGP', 'CIC', or 'TSC'). Default is 'CIC'.
    dtype : type, optional
        NumPy data type. Default is np.float32.
    threshold_randoms : float, optional
        Default random-density threshold used by compute_delta_mesh(). Default is 0.7.
    solver_type : str, optional
        Type of solver ('ifft' or 'multigrid'). Default is 'ifft'.
    n_iterations : int, optional
        Number of iterations (for compatibility). Default is 3.
    **solver_kwargs : dict
        Additional keyword arguments passed to the solver.

    Methods
    -------
    print_info()
        Print reconstruction parameters and settings.
    compute_delta_mesh(threshold_randoms=0.7, sm_mode='wrap')
        Compute the overdensity field on the mesh.
    run_reconstruction()
        Perform the full BAO reconstruction and return reconstructed positions.
    
    """

    def __init__(
        self,
        data_pos: np.ndarray,
        random_pos: np.ndarray,
        data_weights: Optional[np.ndarray] = None,
        random_weights: Optional[np.ndarray] = None,
        data_ids: Optional[np.ndarray] = None,
        RSDspace: str = "RealSpace",
        nmesh: int = 256,
        boxsize: Optional[float] = None,
        boxcentre: Optional[np.ndarray] = None,
        padding: float = 0.01,
        los: str = "z",
        R_sm: float = 15,
        pbc: bool = False,
        rectype: str = "rec-sym",
        f: float = 0.88,
        bias: float = 1.0,
        MAS: str = "CIC",
        dtype=np.float32,
        threshold_randoms: float = 0.7,
        solver_type: str = "ifft",
        n_iterations: int = 3,
        **solver_kwargs
    ) -> None:

        # store basic params
        self._padding = padding
        self._nmesh = nmesh

        # Keep raw input coordinates on the public API.
        self.data_pos = np.asarray(data_pos)
        self.random_pos = np.asarray(random_pos)

        logger.info('mean data pos: %s', self.data_pos.mean(axis=0))
        logger.info('mean random pos: %s', self.random_pos.mean(axis=0))

        # Delay formatting/validation to DensityManager: pass raw inputs
        self._density_manager = DensityManager(
            data_pos=self.data_pos,
            random_pos=self.random_pos,
            nmesh=self._nmesh,
            boxsize=boxsize,
            boxcentre=boxcentre,
            padding=self._padding,
            MAS=MAS,
            dtype=dtype,
            data_weights=data_weights,
            random_weights=random_weights,
            pbc=pbc,
            los=los,
            smoothing_radius=R_sm,
        )

        
        logger.info('mean data pos: %s', self.data_pos.mean(axis=0))
        logger.info('mean random pos: %s', self.random_pos.mean(axis=0))
        # adopt formatted values from DensityManager for internal numerical use
        self._boxsize = self._density_manager.boxsize
        self._boxcentre = self._density_manager.boxcentre

        self.data_weights = self._density_manager.data_weights
        self.random_weights = self._density_manager.random_weights
        self.MAS = self._density_manager.MAS

        # Validate IDs
        if data_ids is not None and len(data_ids) != len(self.data_pos):
            raise ValueError("data_ids length must match data_pos length")
        self.data_ids = data_ids

        # Validate parameters
        self._rectype = format_rectype(rectype)

        # Store scalar parameters
        self._RSDspace = format_rsd_space(RSDspace)
        self._nmesh = nmesh
        self._los = los
        self._R_sm = R_sm
        self._pbc = pbc
        self._f = f
        self._bias = bias
        self._dtype = dtype
        self._threshold_randoms = threshold_randoms
        self._solver_type = solver_type
        self._n_iterations = n_iterations
        self._solver_kwargs = solver_kwargs

        # Lazy initialization
        self._solver = None

    # ============================================================================
    # Properties (lazy initialization)
    # ============================================================================

    @property
    def mesh(self) -> Mesh:
        """Lazily initialize and return mesh."""
        return self._density_manager.mesh

    @property
    def delta_on_mesh(self) -> np.ndarray:
        """Lazily compute and return overdensity field."""
        return self._density_manager.delta_on_mesh

    @property
    def solver(self):
        """Lazily initialize and return solver (FFT or Multigrid)."""
        if self._solver is None:
            logger.debug(f"Initializing {self._solver_type.upper()} solver...")
            if self._solver_type == "ifft":
                self._solver = FFTSolver(
                    delta_on_mesh=self.delta_on_mesh,
                    mesh=self.mesh,
                    f=self._f,
                    bias=self._bias,
                    RSDspace=self._RSDspace,
                    **self._solver_kwargs,
                )
            elif self._solver_type == "multigrid":
                self._solver = MultigridSolver(
                    delta_on_mesh=self.delta_on_mesh,
                    mesh=self.mesh,
                    f=self._f if self._RSDspace == 'RedshiftSpace' else 0.0,  # Multigrid solver needs f=0 for real-space reconstruction
                    bias=self._bias,
                    RSDspace=self._RSDspace,
                    use_plane_parallel= True if self.los is not None else False,
                    **self._solver_kwargs,
                )
            else:
                raise ValueError(f"Unknown solver_type: {self._solver_type}")
        return self._solver

    # Simple property getters for backward compatibility
    @property
    def n_iterations(self):
        return self._n_iterations

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

    # ============================================================================
    # Core methods
    # ============================================================================

    def print_info(self):
        """Print reconstruction parameters."""
        logger.info("=" * 60)
        logger.info(f"BAO Reconstruction ({self._solver_type.upper()}) Information:")
        logger.info("=" * 60)
        logger.info(f"Mesh: {self._nmesh}^3 points")
        logger.info(f"Box size: {self._boxsize} Mpc/h")
        logger.info(f"Box centre: {self._boxcentre}")
        logger.info(f"Padding: {self._padding} Mpc/h")
        logger.info(f"Cell size: {np.array2string(self.cell_size, precision=4, separator=', ')} Mpc/h")
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
        logger.info("=" * 60)

    def compute_delta_mesh(self, threshold_randoms: Optional[float] = None, sm_mode: str = 'wrap') -> np.ndarray:
        """
        Compute overdensity field delta on the mesh.

        Parameters
        ----------
        threshold_randoms : float
            Minimum fraction of average random density to consider.
        sm_mode : str
            Smoothing mode ('wrap' for periodic, 'reflect' for non-periodic).

        Returns
        -------
        delta_field : ndarray
            Overdensity field on mesh.
        """
        if threshold_randoms is None:
            threshold_randoms = self._threshold_randoms
        return self._density_manager.compute_delta(
            threshold_randoms=threshold_randoms,
            sm_mode=sm_mode,
        )

    def _interpolate_displacement(self, positions: np.ndarray, survey_frame: bool = True) -> np.ndarray:
        """
        Interpolate displacement field at given positions.
        
        Parameters
        ----------
        positions : ndarray
            Positions where to interpolate, shape (N, 3)
        survey_frame : bool
            If True, positions are in survey frame and will be transformed to box frame on-the-fly.
            If False, positions are assumed to be in box frame already.
        
        Returns
        -------
        displacement_field : ndarray
            Interpolated real-space displacement field at input positions, shape (N, 3). 
            This does NOT include RSD component, which is computed separately in _get_rsd_displacement() and added to the final shift for redshift-space reconstruction, when applicable.
        """
        if survey_frame:
            # Transform from survey to box frame on-the-fly
            pos_for_interp = survey_to_box_frame(positions, self._density_manager.min_corner, 
                                                self._boxsize, pbc=self._pbc, dtype=self._dtype)
        else:
            pos_for_interp = positions
        
        displacement_field = interpolate_vector_field(
            pos=pos_for_interp,
            field=self.solver.displacement,
            boxsize=self.mesh.boxsize,
            MAS=self.MAS,
            pbc=self._pbc,
            dtype=self.mesh.dtype
        )
        return displacement_field

    def _get_rsd_displacement(self, tracer_psi: np.ndarray, tracer_pos: np.ndarray) -> np.ndarray:
        """
        Compute RSD displacement component.
        
        Parameters
        ----------
        tracer_psi : ndarray
            Real-space displacement field interpolated at tracer positions, shape (N, 3).
        tracer_pos : ndarray
            Tracer positions corresponding to tracer_psi, shape (N, 3).

        Returns
        -------
        rsd_displacement : ndarray
            RSD displacement component to be added to real-space displacement for redshift-space reconstruction, shape (N, 3).
        """

        if self._RSDspace == 'RealSpace':
            logger.warning("RSD displacement requested but RSDspace is set to 'RealSpace'. Returning zero RSD displacement.")
            return np.zeros_like(tracer_psi)

        if self.los is None:
            # Local line-of-sight in the original survey frame.
            norm = np.linalg.norm(tracer_pos, axis=1, keepdims=True)
            mask = norm[:, 0] > 0
            tracers_los = np.zeros_like(tracer_pos)
            tracers_los[mask] = tracer_pos[mask] / norm[mask]
            logger.debug("Using local line-of-sight for RSD displacement")
            logger.debug(f"Mean LOS direction (for masked tracers): {tracers_los[mask].mean(axis=0)}")
            logger.debug(f"head of LOS directions (for masked tracers): {tracers_los[mask][:5]}")
        else:
            # Fixed line-of-sight extracted from mesh (same for all tracers).
            tracers_los = np.tile(self.mesh.los_versor[0, 0, 0], reps=(len(tracer_psi), 1))

        psi_parallel = project_vector_field(tracer_psi, tracers_los)
        logger.debug(f'psi_parallel shape: {psi_parallel.shape}, mean: {psi_parallel.mean()}, std: {psi_parallel.std()}')
        rsd_displacement = self._f * psi_parallel
        return rsd_displacement

    def _shift_gals(self) -> np.ndarray:
        """
        Shift galaxies according to displacement field.
            - For real-space reconstruction, shift by real-space displacement.
            - For redshift-space reconstruction, shift by real-space displacement + RSD component.
        
        Returns
        -------
        shft_gals : ndarray
            Shifted galaxy positions, shape (N_data, 3).
        """
        if self._RSDspace == 'RealSpace':
            logger.info("Computing displacement for galaxies (real space)...")
            shft_gals = self.data_pos - self._interpolate_displacement(self.data_pos)

        elif self._RSDspace == 'RedshiftSpace':
            logger.info("Computing RSD displacement for galaxies...")
            tracer_psi = self._interpolate_displacement(self.data_pos)
            rsd_displacement = self._get_rsd_displacement(tracer_psi, self.data_pos)
            shft_gals = self.data_pos - (tracer_psi + rsd_displacement)
        else:
            raise ValueError(f"Unknown RSDspace: {self._RSDspace}")

        return shft_gals

    def _shift_randoms(self) -> np.ndarray:
        """
        Shift randoms according to reconstruction type and displacement field.
         - For 'rec-sym' (symmetric reconstruction), shift randoms by the same total displacement as galaxies (real-space + RSD if applicable).
         - For 'rec-iso' (isotropic reconstruction), shift randoms only by the real-space displacement, ignoring RSD component, to preserve isotropy of the random catalog.
         
         Returns
         -------
         shft_randoms : ndarray
             Shifted random positions, shape (N_random, 3)
         """
        if self._rectype == 'rec-sym':
            # Randoms shifted like galaxies
            logger.info("Computing displacement for randoms (symmetric reconstruction)...") 
            if self._RSDspace == 'RealSpace':
                shft_randoms = self.random_pos - self._interpolate_displacement(self.random_pos, survey_frame=True)
            elif self._RSDspace == 'RedshiftSpace':
                tracer_psi = self._interpolate_displacement(self.random_pos, survey_frame=True)
                rsd_displacement = self._get_rsd_displacement(tracer_psi, self.random_pos)
                shft_randoms = self.random_pos - (tracer_psi + rsd_displacement)
            else:
                raise ValueError(f"Unknown RSDspace: {self._RSDspace}")

        elif self._rectype == 'rec-iso':
            # Randoms shifted only by displacement (no RSD)
            tracer_psi = self._interpolate_displacement(self.random_pos, survey_frame=True)
            shft_randoms = self.random_pos - tracer_psi
        else:
            raise ValueError(f"Unknown rectype: {self._rectype}")

        return shft_randoms

    def run_reconstruction(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform full BAO reconstruction.

        Returns
        -------
        data_pos_rec : ndarray
            Reconstructed data positions, shape (N_data, 3)
        random_pos_rec : ndarray
            Reconstructed random positions, shape (N_random, 3)
        """
        logger.info(f"Starting BAO reconstruction ({self._solver_type.upper()})...")
        logger.info(f"Reconstruction type: {self._rectype}, RSD space: {self._RSDspace}")
        self.print_info()

        # Solve for displacement field
        logger.info("Solving for displacement field...")
        _ = self.solver.displacement

        # Shift galaxies and randoms
        logger.info("Shifting galaxies...")
        data_pos_rec = self._shift_gals()

        logger.info("Shifting randoms...")
        random_pos_rec = self._shift_randoms()

        # Apply periodic boundary conditions if needed
        if self._pbc:
            min_corner = self._density_manager.min_corner
            data_pos_rec = (np.mod(data_pos_rec - min_corner, self._boxsize) + min_corner)
            random_pos_rec = (np.mod(random_pos_rec - min_corner, self._boxsize) + min_corner)

        logger.info("BAO reconstruction completed successfully")
        return data_pos_rec, random_pos_rec

