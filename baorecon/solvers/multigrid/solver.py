"""Multigrid displacement solver.

Wraps the low-level multigrid kernels (:mod:`baorecon.solvers.multigrid._kernels`)
behind the :class:`~baorecon.solvers._interface.PoissonSolver` API. The
hierarchy supports anisotropic grids: it coarsens by full (factor-of-two)
coarsening of every axis together, so each axis must be *multigrid-friendly*
(``k * 2**p`` with ``k in {1,3,5,7}``, ``p >= 2``). Geometry comes from the
:class:`~baorecon.mesh.mesh.Mesh`; the LOS strategy is injected by the
reconstructor.
"""

import warnings
from typing import Optional, Union

import numpy as np

from baorecon.solvers._interface import PoissonSolver
from baorecon.field_ops.cpu import (
    gradient_periodic_jit,
    differentiate_potential_jit_cic,
    differentiate_potential_tsc_jit,
)
from baorecon.solvers.multigrid._kernels import (
    jacobi_jit,
    prolong_jit,
    reduce_jit,
    residual_jit,
    multicolor_gs_jit
)
from baorecon.utils.formatters import format_mas, round_to_multigrid_friendly
from baorecon.utils.loggers import setup_logger
from baorecon.mesh.los import FixedAxisLOS

logger = setup_logger(__name__)

# Smallest grid size (per axis) kept at the bottom of the multigrid hierarchy.
# Coarsening halts before any axis would drop below this; chosen as 4 to
# reproduce the historical cubic chain (the old ``while curr_N >= 4`` loop).
MIN_COARSE = 4

# Odd multipliers ``k`` allowed in a multigrid-friendly size ``k * 2**p``.
_MULTIGRID_FACTORS = (1, 3, 5, 7)


def is_multigrid_friendly(n: int) -> bool:
    """Return True if ``n`` has the form ``k * 2**p`` with ``k in {1,3,5,7}`` and ``p >= 2``."""
    n = int(n)
    if n < MIN_COARSE:
        return False
    p = 0
    while n % 2 == 0:
        n //= 2
        p += 1
    return p >= 2 and n in _MULTIGRID_FACTORS


def _coarsening_levels(n: int) -> int:
    """Number of times ``n`` can be halved before becoming odd (the 2-adic valuation)."""
    n, k = int(n), 0
    while n > 0 and n % 2 == 0:
        n //= 2
        k += 1
    return k


class _RawMultigrid:
    """Low-level FMG/V-cycle multigrid operating on flattened vectors (anisotropic)."""

    class GridLevel:
        def __init__(self, dims: np.ndarray, boxsize: np.ndarray, boxcenter: np.ndarray, temp_pool:np.ndarray, 
                     dtype=np.float32, is_top_level=False):
            # dims: per-axis number of cells [Nx, Ny, Nz] (anisotropic grids)
            self._dtype = dtype
            self.dims = np.asarray(dims, dtype=np.int32)
            self.shape = tuple(int(d) for d in self.dims)
            self.boxsize = boxsize
            self.boxcenter = boxcenter
            self.local_x = int(self.dims[0])
            self.offset_x = 0
            self.size = int(np.prod(self.dims))
            self.v = np.zeros(self.size, dtype=dtype)
            
            self.temp = temp_pool[:self.size]

            if not is_top_level:
                self.f = np.zeros(self.size, dtype=dtype)
            

    def __init__(self, N_fine, boxsize: Union[float, int, np.ndarray],
                 boxcenter: Optional[np.ndarray] = None, use_plane_parallel: bool = True, dtype=np.float32, smoother="jacobi"):
        self._SMOOTHERS = ("jacobi", "mcgs")
        if smoother not in self._SMOOTHERS:
            raise ValueError(f"smoother must be one of {self._SMOOTHERS}, got {smoother!r}.")
        self.smoother = smoother
        self.levels = []
        self._dtype = dtype
        self.boxsize = self._format_boxsize(boxsize)
        self.boxcenter = boxcenter
        self.use_plane_parallel = use_plane_parallel

        # Per-axis fine dimensions: a scalar broadcasts to a cubic grid.
        dims = np.asarray(N_fine, dtype=np.int32)
        if dims.ndim == 0:
            dims = np.full(3, dims, dtype=np.int32)
        elif dims.shape != (3,):
            raise ValueError(f"N_fine must be a scalar or array-like of shape (3,). Got shape {dims.shape}.")

        self._validate_multigrid_friendly(dims)
        self._warn_if_unbalanced(dims)

        fine_size = int(np.prod(dims))
        self._global_temp_pool = np.empty(fine_size, dtype=self._dtype)

        # Build the hierarchy by FULL coarsening: halve every axis together
        # (required by reduce_jit/prolong_jit, which assume coarse == fine // 2
        # on every axis), while all axes stay even and >= MIN_COARSE.
        self.levels.append(_RawMultigrid.GridLevel(
            dims, boxsize=self.boxsize, boxcenter=self.boxcenter, dtype=self._dtype, 
            temp_pool=self._global_temp_pool, is_top_level=True
            ))
        while np.all(dims % 2 == 0) and np.all(dims // 2 >= MIN_COARSE):
            dims = dims // 2
            self.levels.append(_RawMultigrid.GridLevel(dims, boxsize=self.boxsize, boxcenter=self.boxcenter, 
                                                       temp_pool=self._global_temp_pool))

        self.num_levels = len(self.levels)
        # mcgs requires even levels (8-color on a periodic grid). If the bottom
        # level is odd, it falls back to Jacobi there, which is parity-agnostic.
        bottom_even = bool(np.all(self.levels[-1].dims % 2 == 0))
        self._bottom_smoother = "jacobi" if (smoother == "mcgs" and not bottom_even) else smoother
        if self._bottom_smoother != smoother:
            logger.info(f"Bottom level {self.levels[-1].shape} is odd: smoothing there with Jacobi.")
        logger.debug(f"LOS plane-parallel: {self.use_plane_parallel}.")
        logger.info(f"Solver Init: {self.levels[0].shape} -> {self.levels[-1].shape}")

    @staticmethod
    def _validate_multigrid_friendly(dims: np.ndarray) -> None:
        """Require every axis to be a multigrid-friendly ``k * 2**p``."""
        dims = [int(d) for d in dims]
        bad = [d for d in dims if not is_multigrid_friendly(d)]
        if bad:
            suggestion = [round_to_multigrid_friendly(d) for d in dims]
            raise ValueError(
                f"Multigrid requires every nmesh axis to be of the form k*2**n with "
                f"k in {{1, 3, 5, 7}} and n >= 2 (so the grid coarsens well). "
                f"Got nmesh={tuple(dims)}; offending axes: {bad}. "
                f"Nearest valid sizes (rounded up): {tuple(suggestion)}. "
                f"Note the FFT solver accepts any nmesh; this constraint only "
                f"applies to the multigrid solver."
            )

    @staticmethod
    def _warn_if_unbalanced(dims: np.ndarray) -> None:
        """Warn (non-blocking) when the grid is poorly balanced for full coarsening."""
        levels = [_coarsening_levels(d) for d in dims]
        spread = max(levels) - min(levels)
        if spread >= 3:
            warnings.warn(
                f"Multigrid grid {tuple(int(d) for d in dims)} is poorly balanced for "
                f"full coarsening: factors of 2 per axis = {levels} (spread {spread} >= 3). "
                f"Coarsening halts at the axis with the fewest factors of 2, leaving the "
                f"other axes under-coarsened, so multigrid convergence will be very slow. "
                f"Prefer dimensions with a similar number of factors of 2 "
                f"(all 2^n, or m*2^n with matching n; m in {{1,3,5,7}}).",
                UserWarning,
                stacklevel=2,
            )

    def _format_boxsize(self, boxsize):
        if isinstance(boxsize, (float, int)):
            boxsize = np.float32(boxsize)
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
        
    def _smooth(self, curr, beta, los, damping, smoother):
        if smoother == "jacobi":
            jacobi_jit(curr.v, curr.f, curr.temp, curr.dims, curr.local_x, curr.offset_x,
                    self.boxsize, self.boxcenter, beta, damping, los, self.use_plane_parallel)
        else:  # "mcgs"
            for color in range(8):
                multicolor_gs_jit(curr.v, curr.f, curr.dims, curr.local_x, curr.offset_x,
                                self.boxsize, self.boxcenter, beta, los,
                                self.use_plane_parallel, color)

    def solve_fmg(self, input_density, beta=0.0, los=np.array([0., 0., 1.]),
                  v_cycles=6, n_smooth=5, damping=0.4):
        fine = self.levels[0]
        fine.f = input_density.ravel()
        logger.info(f"Starting FMG (Cycles={v_cycles}, Smooth={n_smooth}, Damp={damping})...")
        self._fmg_recursive(0, beta, los, v_cycles, n_smooth, damping)
        return fine.v.reshape(fine.shape)

    def solve(self, input_density, beta=0.0, los=np.array([0., 0., 1.]),
              n_cycles=15, n_smooth=5, damping=0.4):
        fine = self.levels[0]
        fine.f = input_density.ravel()
        fine.v.fill(0.0)
        logger.info(f"Starting V-Cycles (N={n_cycles}, Smooth={n_smooth}, Damp={damping})...")
        for i in range(n_cycles):
            self._v_cycle_recursive(0, beta, los, n_smooth, damping)
            if i % 5 == 0:
                residual_jit(fine.v, fine.f, fine.temp, fine.dims, fine.local_x, fine.offset_x,
                             self.boxsize, self.boxcenter, beta, los, self.use_plane_parallel)
                
        return fine.v.reshape(fine.shape)

    def _v_cycle_recursive(self, lvl_idx, beta, los, n_smooth, damping):
        curr = self.levels[lvl_idx]
        is_bottom = (lvl_idx == self.num_levels - 1)
        sm = self._bottom_smoother if is_bottom else self.smoother

        #for _ in range(n_smooth):
        #    jacobi_jit(curr.v, curr.f, curr.temp, curr.dims, curr.local_x, curr.offset_x,
        #               self.boxsize, self.boxcenter, beta, damping, los, self.use_plane_parallel)
        for _ in range(n_smooth):
            self._smooth(curr, beta, los, damping, sm)

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
            self._smooth(curr, beta, los, damping, sm)

    def _fmg_recursive(self, lvl_idx, beta, los, n_cycles, n_smooth, damping):
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
    """Multigrid Poisson/Zel'dovich solver (CPU-only); supports anisotropic grids."""

    def __init__(self, delta_on_mesh, mesh, f: float, bias: float, los=None, 
                 RSDspace: str ="RedshiftSpace", use_plane_parallel=False, dtype = np.float32, smoother="jacobi", **kwargs) -> None:
        if f is None:
            raise ValueError("growth rate f must be provided!")
        if bias is None:
            raise ValueError("bias must be provided!")
        super().__init__(delta_on_mesh, mesh, f=f, bias=bias, RSDspace=RSDspace)
        self.los = los
        self._use_plane_parallel = use_plane_parallel
        self._raw: Optional[_RawMultigrid] = None
        self._kwargs = kwargs
        self._dtype = dtype
        self._smoother = smoother

    def _set_los(self) -> np.ndarray:
        if self._use_plane_parallel:
            if self.los is not None:
                if isinstance(self.los, FixedAxisLOS):
                    return np.asarray(self.los.direction, dtype=self._dtype)
                
                elif isinstance(self.los, (np.ndarray, list, tuple)):
                    current_los = np.asarray(self.los, dtype=self._dtype)
                    # Extra safety: verify it is a 3D vector
                    if current_los.shape != (3,):
                        raise ValueError(f"The 'los' array must have shape (3,), but has shape {current_los.shape}.")
                    return current_los
                    
                else:
                    raise TypeError("With use_plane_parallel=True, 'los' must be FixedAxisLOS or an array (3,).")
            else:
                # Nessuna LOS fornita nonostante l'approccio plane-parallel
                raise ValueError("With use_plane_parallel=True, 'los' cannot be None.")
                
        else:
            # LOCAL LOS (Radial): JIT kernels compute it internally on the fly.
            # We pass a dummy array.
            return np.array([0.0, 0.0, 0.0], dtype=self._dtype)

    def _compute_potential_mesh(self) -> None:
        nmesh = np.asarray(self.mesh.nmesh, dtype=np.int32)
        box = np.asarray(self.mesh.boxsize, dtype=self._dtype)
        boxcenter = np.asarray(self.mesh.boxcentre, dtype=self._dtype)
        delta = self.delta_on_mesh
    
        if self._raw is None:
            logger.debug("Initializing internal RawMultigrid")
            self._raw = _RawMultigrid(N_fine=nmesh, boxsize=box, boxcenter=boxcenter,
                                      use_plane_parallel=self._use_plane_parallel, dtype=self._dtype, smoother=self._smoother)

        current_los = self._set_los()

        logger.debug("Calling RawMultigrid.solve_fmg()")
        potential = self._raw.solve_fmg(
            delta.reshape(tuple(int(n) for n in nmesh)), beta=self.f / self.bias, los=current_los,
            v_cycles=self._kwargs.get("v_cycles", 6),
            n_smooth=self._kwargs.get("n_smooth", 5),
            damping=self._kwargs.get("damping", 0.4),
        )
        self._raw = None
        self._potential = np.divide(potential, self.bias)

    def _compute_displacement_mesh(self) -> None:
        """Displacement on mesh from the potential on mesh via centered periodic finite differences."""
        nmesh = np.asarray(self.mesh.nmesh, dtype=np.int32)
        box = np.asarray(self.mesh.boxsize, dtype=self._dtype)

        phi = self.potential.reshape(tuple(int(n) for n in nmesh))
        dx = box[0] / np.float32(nmesh[0])
        dy = box[1] / np.float32(nmesh[1])
        dz = box[2] / np.float32(nmesh[2])

        displacement = np.empty(tuple(int(n) for n in nmesh) + (3,), dtype=phi.dtype)
        gradient_periodic_jit(phi, displacement, dx, dy, dz)
        displacement *= -1  # Psi = -grad phi
        self._displacement = displacement

    
    def read_displacement_at(self, positions: np.ndarray, mas: str = "CIC") -> np.ndarray:
        if self._potential is None:
            self._compute_potential_mesh()

        mas = format_mas(mas)
        disp = np.empty_like(positions)
        nmesh = np.asarray(self.mesh.nmesh, dtype=np.int32)

        if mas == "CIC":
            differentiate_potential_jit_cic(
                self.potential.reshape(tuple(nmesh)), positions, disp,
                nmesh, offset_x=0, boxsize=self.mesh.boxsize
            )
        elif mas == "TSC":
            differentiate_potential_tsc_jit(
                self.potential.reshape(tuple(nmesh)), positions, disp,
                nmesh, self.mesh.boxsize
            )
        else:
            raise ValueError(f"MAS {mas!r} not supported by the multigrid read-out (use 'CIC' or 'TSC').")

        return disp
