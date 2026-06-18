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
)

logger = setup_logger(__name__)


class _RawMultigrid:
    """
    Internal copy of the multigrid solver adapted from the original `multigrid.py`.
    This class operates on flattened vectors and exposes `solve_fmg` and `solve`.
    """

    class GridLevel:
        def __init__(self, N: int, boxsize: np.ndarray, boxcenter: np.ndarray):
            self.N = N
            self.boxsize = boxsize
            self.boxcenter = boxcenter
            self.dims = np.array([N, N, N], dtype=np.int64)
            self.local_x = N
            self.offset_x = 0
            self.size = N**3
            self.v = np.zeros(self.size, dtype=np.float64)
            self.f = np.zeros(self.size, dtype=np.float64)
            self.temp = np.zeros(self.size, dtype=np.float64)

    def __init__(self, N_fine: int, boxsize: Union[float, int, np.ndarray], boxcenter: Optional[np.ndarray] = None, use_plane_parallel: bool = True):
        self.levels = []
        self.boxsize = self._format_boxsize(boxsize)
        self.boxcenter = boxcenter 
        self.use_plane_parallel = use_plane_parallel
        curr_N = int(N_fine)
        while curr_N >= 4:
            self.levels.append(_RawMultigrid.GridLevel(N=curr_N, boxsize=self.boxsize, boxcenter=self.boxcenter))
            curr_N //= 2
        self.num_levels = len(self.levels)
        logger.debug(f"LOS plane-parallel: {self.use_plane_parallel}.")
        logger.info(f"Solver Init: {N_fine} -> {self.levels[-1].N}")

    def _format_boxsize(self, boxsize):
        if isinstance(boxsize, (float, int)):
            boxsize = float(boxsize)
            boxsize = np.array([boxsize, boxsize, boxsize], dtype=float)
            return boxsize
        elif isinstance(boxsize, np.ndarray):
            if boxsize.shape == (3,):
                boxsize = boxsize.astype(float)
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
        f: float = None,
        bias: float = 1.0,
        RSDspace: str = "RealSpace",
        use_plane_parallel: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(delta_on_mesh, mesh, f=f, bias=bias, RSDspace=RSDspace)
        self._use_plane_parallel = use_plane_parallel
        self._raw: Optional[_RawMultigrid] = None
        self._kwargs = kwargs

    def _compute_potential(self) -> None:
        N = int(self.mesh.nmesh)
        box = self.mesh.boxsize
        boxcenter = self.mesh.boxcentre
        if np.isscalar(box):
            box = np.array([box, box, box], dtype=float)
        delta = np.ascontiguousarray(self.delta_on_mesh.astype(np.float64, copy=False))

        if self._raw is None:
            logger.debug("Initializing internal RawMultigrid")
            self._raw = _RawMultigrid(N_fine=N, boxsize=box, boxcenter=boxcenter, use_plane_parallel=self._use_plane_parallel)

        # los from mesh if plane-parallel, otherwise dummy (for jit functions that expect it, but it won't be used in that case)
        if self._use_plane_parallel:
            # prendere il versore globale (prima cella)
            current_los = self.mesh.los_versor[0, 0, 0].astype(np.float64)
        else:
            current_los = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        logger.debug("Calling RawMultigrid.solve_fmg()")
        potential = self._raw.solve_fmg(delta.reshape((N, N, N)), beta=self.f/self.bias, los=current_los,
                                        v_cycles=self._kwargs.get("v_cycles", 6),
                                        n_smooth=self._kwargs.get("n_smooth", 5),
                                        damping=self._kwargs.get("damping", 0.4))

        if self.bias is not None and self.bias != 0:
            potential = potential / float(self.bias)
        self._potential = np.asarray(potential)

        # Compute displacement on the mesh via centered finite differences (periodic)
        # This is currently done in Python for simplicity, but could be optimized with a JIT function if needed.
        # Moreover, if one only needs the displacement at particle positions, 
        # it would be more efficient to directly interpolate the potential gradient at those positions 
        # without computing the full displacement field on the mesh. 
        # See the `interpolate_potential_jit` function in `multigrid_lib.py` for a potential implementation of this approach.



    def _compute_displacement(self):
        """
        Computes the displacement field (gradient) from the scalar potential on the grid.
        Uses centered finite differences with periodic boundary conditions via np.roll.
            
        Returns
        -------
        displacement : ndarray (N, N, N, 3)
            The displacement field vectors on the grid nodes.
        """

        N = int(self.mesh.nmesh)
        box = self.mesh.boxsize
        if np.isscalar(box):
            box = np.array([box, box, box], dtype=float)

        phi = self.potential.reshape((N, N, N))

        dx = box[0] / float(N)
        dy = box[1] / float(N)
        dz = box[2] / float(N)

        grad_x = (np.roll(phi, -1, axis=0) - np.roll(phi, 1, axis=0)) / (2.0 * dx)
        grad_y = (np.roll(phi, -1, axis=1) - np.roll(phi, 1, axis=1)) / (2.0 * dy)
        grad_z = (np.roll(phi, -1, axis=2) - np.roll(phi, 1, axis=2)) / (2.0 * dz)

        disp = -np.stack((grad_x, grad_y, grad_z), axis=-1)
        self._displacement = np.asarray(disp)