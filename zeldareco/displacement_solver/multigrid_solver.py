import numpy as np
from typing import Optional, Union
from zeldareco.displacement_solver.poisson_solver import PoissonSolver
from zeldareco.mesh.mesh import Mesh
from zeldareco.utils.loggers import setup_logger
from zeldareco.displacement_solver.multigrid_lib import (
    jacobi_jit,
    residual_jit,
    reduce_jit,
    prolong_jit,
    gradient_periodic_jit
)

logger = setup_logger(__name__)


class _RawMultigrid:
    """
    Internal copy of the multigrid solver adapted from the original `multigrid.py`.
    This class operates on flattened vectors and exposes `solve_fmg` and `solve`.
    """

    class GridLevel:
        def __init__(self, N: int, boxsize: np.ndarray, boxcenter: np.ndarray, dtype=np.float32):
            self.N = N
            self._dtype = dtype
            self.boxsize = boxsize
            self.boxcenter = boxcenter
            self.dims = np.array([N, N, N], dtype=np.int32)
            self.local_x = N
            self.offset_x = 0
            self.size = N**3
            self.v = np.zeros(self.size, dtype=dtype)
            self.f = np.zeros(self.size, dtype=dtype)
            self.temp = np.zeros(self.size, dtype=dtype)

    def __init__(self, N_fine: int, boxsize: Union[float, int, np.ndarray], boxcenter: Optional[np.ndarray] = None, use_plane_parallel: bool = True, dtype=np.float32):
        self.levels = []
        self._dtype = dtype
        self.boxsize = self._format_boxsize(boxsize)
        self.boxcenter = boxcenter 
        self.use_plane_parallel = use_plane_parallel
        curr_N = int(N_fine)
        while curr_N >= 4:
            self.levels.append(_RawMultigrid.GridLevel(N=curr_N, boxsize=self.boxsize, boxcenter=self.boxcenter, dtype=dtype))
            curr_N //= 2
        self.num_levels = len(self.levels)
        logger.debug(f"LOS plane-parallel: {self.use_plane_parallel}.")
        logger.info(f"Solver Init: {N_fine} -> {self.levels[-1].N}")

    def _format_boxsize(self, boxsize):
        if isinstance(boxsize, (float, int)):
            boxsize = float(boxsize)
            boxsize = np.array([boxsize, boxsize, boxsize], dtype=self._dtype)
            return boxsize
        elif isinstance(boxsize, np.ndarray):
            if boxsize.shape == (3,):
                boxsize = boxsize.astype(self._dtype)
                return boxsize
            else:
                raise ValueError("boxsize as ndarray must have shape (3,)")
        else:
            raise TypeError("boxsize must be float, int or np.ndarray of shape (3,)")

    def solve_fmg(self, input_density: np.ndarray,
                  beta: float = 0.0,
                  los: np.ndarray = np.array([0.,0.,1.]),
                  v_cycles: int = 6,
                  n_smooth: int = 5,
                  damping: float = 0.4) -> np.ndarray:
        fine = self.levels[0]
        fine.f[:] = input_density.ravel()
        logger.info(f"Starting FMG (Cycles={v_cycles}, Smooth={n_smooth}, Damp={damping})...")
        self._fmg_recursive(0, beta, los, v_cycles, n_smooth, damping)
        return fine.v.reshape((fine.N, fine.N, fine.N))

    def solve(self, input_density: np.ndarray, beta: float = 0.0, los: np.ndarray = np.array([0.,0.,1.]),
              n_cycles: int = 15, n_smooth: int = 5, damping: float = 0.4) -> np.ndarray:
        fine = self.levels[0]
        fine.f[:] = input_density.ravel()
        fine.v.fill(0.0)
        logger.info(f"Starting V-Cycles (N={n_cycles}, Smooth={n_smooth}, Damp={damping})...")
        for i in range(n_cycles):
            self._v_cycle_recursive(0, beta, los, n_smooth, damping)
            if i % 5 == 0:
                residual_jit(fine.v, fine.f, fine.temp, fine.dims, fine.local_x, fine.offset_x,
                             self.boxsize, self.boxcenter, beta, los, self.use_plane_parallel)
                logger.info(f"  Cycle {i}: Res {np.std(fine.temp):.4e}")
        return fine.v.reshape((fine.N, fine.N, fine.N))

    def _v_cycle_recursive(self, lvl_idx: int, beta: float, los: np.ndarray, n_smooth: int, damping: float):
        curr = self.levels[lvl_idx]
        is_bottom = (lvl_idx == self.num_levels - 1)

        for _ in range(n_smooth):
            jacobi_jit(curr.v, curr.f, curr.temp, curr.dims, curr.local_x, curr.offset_x,
                       self.boxsize, self.boxcenter, beta, damping, los, self.use_plane_parallel)

        if not is_bottom:
            next_lvl = self.levels[lvl_idx + 1]
            residual_jit(curr.v, curr.f, curr.temp, curr.dims, curr.local_x, curr.offset_x,
                         self.boxsize, self.boxcenter, beta, los, self.use_plane_parallel)
            reduce_jit(curr.temp, next_lvl.f, curr.dims, next_lvl.local_x, next_lvl.offset_x)
            next_lvl.v.fill(0.0)
            self._v_cycle_recursive(lvl_idx + 1, beta, los, n_smooth, damping)
            prolong_jit(next_lvl.v, curr.temp, curr.dims, curr.local_x, curr.offset_x)
            curr.v += curr.temp

        for _ in range(n_smooth):
            jacobi_jit(curr.v, curr.f, curr.temp, curr.dims, curr.local_x, curr.offset_x,
                       self.boxsize, self.boxcenter, beta, damping, los, self.use_plane_parallel)

    def _fmg_recursive(self, lvl_idx: int, beta: float, los: np.ndarray,
                       n_cycles: int, n_smooth: int, damping: float):
        curr = self.levels[lvl_idx]
        is_bottom = (lvl_idx == self.num_levels - 1)

        if not is_bottom:
            next_lvl = self.levels[lvl_idx + 1]
            reduce_jit(curr.f, next_lvl.f, curr.dims, next_lvl.local_x, next_lvl.offset_x)
            self._fmg_recursive(lvl_idx + 1, beta, los, n_cycles, n_smooth, damping)
            prolong_jit(next_lvl.v, curr.v, curr.dims, curr.local_x, curr.offset_x)
        else:
            curr.v.fill(0.0)

        for _ in range(n_cycles):
            self._v_cycle_recursive(lvl_idx, beta, los, n_smooth, damping)


class MultigridSolver(PoissonSolver):
    """
    Wrapper exposing the multigrid solver through the PoissonSolver interface.
    
    Parameters
    ----------
    delta_on_mesh : ndarray
        Overdensity field on mesh
    mesh : Mesh
        Mesh object defining geometry
    f : float, optional
        Growth rate
    bias : float, optional
        Linear bias. Default is 1.0.
    RSDspace : str, optional
        Reconstruction space ('RealSpace' or 'RedshiftSpace'). Default is 'RealSpace'.
    smoothing_radius : float, optional
        Smoothing radius in Mpc/h. Default is 15.0.
    use_plane_parallel : bool, optional
        If True, use plane-parallel approximation (RSD along z-axis only).
        If False, use full RSD treatment. Default is True.
    **kwargs : dict
        Additional keyword arguments (v_cycles, n_smooth, damping, etc.)
    """

    def __init__(
        self,
        delta_on_mesh: np.ndarray,
        mesh: Mesh,
        f: float,
        bias: float,
        RSDspace: str = "RedshiftSpace",
        use_plane_parallel: bool = False,
        dtype = np.float32,
        **kwargs,
    ) -> None:
        
        if f is None:
            raise ValueError("growth rate f must be provided!")
        if bias is None:
            raise ValueError("bias must be provided!")
        
        super().__init__(delta_on_mesh, mesh, f=f, bias=bias, RSDspace=RSDspace)
        self._use_plane_parallel = use_plane_parallel
        self._raw: Optional[_RawMultigrid] = None
        self._kwargs = kwargs
        self._dtype = dtype

    def _compute_potential(self) -> None:
        N = int(self.mesh.nmesh)
        box = self.mesh.boxsize
        boxcenter = self.mesh.boxcentre
        if np.isscalar(box):
            box = np.array([box, box, box], dtype=self._dtype)
        delta = np.ascontiguousarray(self.delta_on_mesh.astype(self._dtype, copy=False))

        if self._raw is None:
            logger.debug("Initializing internal RawMultigrid")
            self._raw = _RawMultigrid(
                N_fine=N, boxsize=box, 
                boxcenter=boxcenter, 
                use_plane_parallel=self._use_plane_parallel, 
                dtype=self._dtype
            )

        # los from mesh if plane-parallel, otherwise dummy (for jit functions that expect it, but it won't be used in that case)
        if self._use_plane_parallel:
            # prendere il versore globale (prima cella)
            current_los = self.mesh.los_versor[0, 0, 0].astype(self._dtype)
        else:
            current_los = np.array([0.0, 0.0, 0.0], dtype=self._dtype)

        logger.debug("Calling RawMultigrid.solve_fmg()")
        potential = self._raw.solve_fmg(delta.reshape((N, N, N)), beta=self.f/self.bias, los=current_los,
                                        v_cycles=self._kwargs.get("v_cycles", 6),
                                        n_smooth=self._kwargs.get("n_smooth", 5),
                                        damping=self._kwargs.get("damping", 0.4))

        if self.bias is not None and self.bias != 0:
            potential = potential / float(self.bias)
        self._potential = np.asarray(potential)


    def _compute_displacement(self) -> None:
        N = int(self.mesh.nmesh)
        boxsize = self.mesh.boxsize
        if np.isscalar(boxsize):
            dx = dy = dz = float(boxsize) / N
        else:
            boxsize = np.asarray(boxsize, dtype=self._dtype)
            dx = float(boxsize[0]) / N
            dy = float(boxsize[1]) / N
            dz = float(boxsize[2]) / N

        phi = self.potential.reshape((N, N, N))
        displacement = np.empty((N, N, N, 3), dtype=phi.dtype)
        gradient_periodic_jit(phi, displacement, dx, dy, dz)
        displacement *= -1
        self._displacement = displacement
