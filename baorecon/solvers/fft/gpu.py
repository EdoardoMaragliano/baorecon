"""GPU FFT displacement solver (iFFT / Burden algorithm), optimized for GPU.

Everything stays resident on the device: the iterative loop, the final
displacement field, and (if requested) the potential are all computed and kept
on the GPU. Nothing is copied back to host here -- only the small per-particle
read-out at the end of reconstruction leaves the device. The potential, when
asked for, is recomputed from the device-resident displacement rather than from
a stored psi_k grid.
"""

from baorecon.solvers._interface import PoissonSolver
from baorecon.solvers.fft._common import (
    divergence_from_components,
    prepare_k_components,
)
from baorecon.utils.backend import get_fft_backend
from baorecon.utils.loggers import setup_logger

try:
    import cupy as _cupy
except ImportError:  # keep this module importable on CPU-only hosts (cupy is runtime-only)
    _cupy = None

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# On-the-fly radial (LocalLOS) projection kernels, on the device.
#
# Device analogues of the numba kernels in ``_radial_stream``: they evaluate the
# radial unit versor cell by cell -- n_hat_a = coord_a / |coord|, with
# coord_a = min_corner[a] + idx_a * cell_size[a] -- so no (Nx,Ny,Nz,3) versor
# grid is ever stored (on host or device). The cell index (ix,iy,iz) is recovered
# from the flat ElementwiseKernel index ``i`` of a C-contiguous (Nx,Ny,Nz) array.
# The versor is single precision (matching ``LocalLOS.radial_versor``); the
# gradient/output dtype ``T`` follows the working precision. The cell at the
# observer (|coord| = 0) yields a zero versor, as ``radial_versor`` does.
# ---------------------------------------------------------------------------
_VERSOR_C = '''
    long iz = i % nz;
    long iy = (i / nz) % ny;
    long ix = i / ((long)ny * nz);
    float cx = min_x + (float)ix * cell_x;
    float cy = min_y + (float)iy * cell_y;
    float cz = min_z + (float)iz * cell_z;
    float r2 = cx * cx + cy * cy + cz * cz;
    float versor = 0.0f;
    if (r2 > 0.0f) {
        float ca = (axis == 0) ? cx : ((axis == 1) ? cy : cz);
        versor = ca / sqrtf(r2);
    }
'''

if _cupy is not None:
    # s += grad * n_hat_axis, streamed one gradient component at a time
    # (``init`` writes the axis-0 pass, so s needs no separate zeroing).
    _project_grad_onto_los = _cupy.ElementwiseKernel(
        'T grad, int32 axis, float32 min_x, float32 min_y, float32 min_z, '
        'float32 cell_x, float32 cell_y, float32 cell_z, int32 ny, int32 nz, int32 init',
        'raw T s',
        _VERSOR_C + '''
        T contribution = grad * versor;
        s[i] = init ? contribution : (s[i] + contribution);
        ''',
        'project_grad_onto_los')

    # parallel = s * n_hat_axis (the axis component of the parallel vector field)
    _reconstruct_parallel_vector = _cupy.ElementwiseKernel(
        'T s, int32 axis, float32 min_x, float32 min_y, float32 min_z, '
        'float32 cell_x, float32 cell_y, float32 cell_z, int32 ny, int32 nz',
        'T parallel',
        _VERSOR_C + '''
        parallel = s * versor;
        ''',
        'reconstruct_parallel_vector')
    # out = (sign * i) * k_axis [* 1/(bias k^2)] * v, with |k|^2 evaluated on the
    # fly from the three 1-D wavevector arrays. This replaces the materialized
    # ``inv_k2_bias`` real half-grid (~0.5 grid) and the ``scaled_k`` complex
    # scratch (~1 grid) of the previous implementation: nothing k-dependent is
    # stored beyond the 1-D arrays. The DC handling is positional (k^2 == 0
    # emits 0, exactly what ``build_inv_k2`` encoded by index), which also makes
    # the kernel correct on a ky-slab of a distributed k-grid with no rank-aware
    # special case. ``i`` indexes a C-contiguous (nxk, nyk, nzk) complex grid.
    _scale_component_k = _cupy.ElementwiseKernel(
        'C v, raw float32 kx, raw float32 ky, raw float32 kz, '
        'float32 inv_bias, float32 sign, int32 axis, int32 use_inv_k2, '
        'int32 nyk, int32 nzk',
        'C out',
        '''
        const long iz = i % nzk;
        const long iy = (i / nzk) % nyk;
        const long ix = i / ((long)nyk * nzk);
        const float ka = (axis == 0) ? kx[ix] : ((axis == 1) ? ky[iy] : kz[iz]);
        float fac = sign * ka;
        if (use_inv_k2) {
            const float kxv = kx[ix];
            const float kyv = ky[iy];
            const float kzv = kz[iz];
            const float k2 = kxv * kxv + kyv * kyv + kzv * kzv;
            fac = (k2 > 0.0f) ? fac * inv_bias / k2 : 0.0f;
        }
        out = C(-fac * v.imag(), fac * v.real());
        ''',
        'scale_component_k')
else:
    _project_grad_onto_los = None
    _reconstruct_parallel_vector = None
    _scale_component_k = None


class FFTSolverGPU(PoissonSolver):
    """FFT-based Poisson/Zel'dovich solver on the GPU (CuPy).

    With a distributed environment (``dist`` with ``world_size > 1``),
    ``delta_on_mesh`` is this rank's x-slab, every transform routes through the
    slab :class:`~baorecon.solvers.fft._distributed_fft.DistributedFFT`, the
    1-D ky array is sliced to the rank's ky block, and the displacement is kept
    in a halo-extended slab for cross-rank read-back. The Burden loop itself is
    identical to the single-GPU path.
    """

    # Read-back ghost width for the distributed displacement grid: sized for
    # the widest supported read stencil (TSC), so any MAS can read from it.
    _DISP_HALO = 2

    def __init__(self, delta_on_mesh, mesh, los=None, f=None, bias=1.0,
                 RSDspace="RealSpace", n_iterations=3, dist=None) -> None:
        super().__init__(delta_on_mesh, mesh, f=f, bias=bias, RSDspace=RSDspace)
        from baorecon.utils.distributed import DistEnv

        self.los = los
        self.n_iterations = n_iterations
        self.dist = dist if dist is not None else DistEnv.serial()
        self.backend = get_fft_backend("gpu")
        self.xp = self.backend.xp
        self._init_fft()
        self._complex_j = self.xp.complex64(1j)
        self._k_host = None
        self._displacement_ext = None
        self._disp_halo_filled = False

    def _init_fft(self):
        if self.dist.is_distributed:
            from baorecon.solvers.fft._distributed_fft import DistributedFFT

            self.fft = DistributedFFT(self.dist, self.mesh.shape)
        else:
            self.fft = self.backend.fft

    def __getstate__(self):
        state = self.__dict__.copy()
        # "dist" holds live NCCL/MPI handles and cannot be pickled; a restored
        # solver comes back serial (its grids are this rank's slabs).
        for key in ("backend", "xp", "fft", "_complex_j", "_k_host", "dist"):
            state.pop(key, None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if getattr(self, "dist", None) is None:
            from baorecon.utils.distributed import DistEnv

            self.dist = DistEnv.serial()
        self.backend = get_fft_backend("gpu")
        self.xp = self.backend.xp
        self._init_fft()
        self._complex_j = self.xp.complex64(1j)
        self._k_host = None

    def _k_components(self):
        """Cached host 1-D wavevector arrays ``(kx, ky, kz)``.

        Cached on the host and uploaded per method; recomputing the 1-D arrays
        on every displacement/potential pass is pure waste. In distributed mode
        ``ky`` is this rank's ky block (kx/kz stay global on every rank).
        """
        if self._k_host is None:
            kx, ky, kz = prepare_k_components(self.mesh.cell_size, self.mesh.nmesh)
            if self.dist.is_distributed:
                ky = self.fft.decomp.ky_slice(ky)
            self._k_host = (kx, ky, kz)
        return self._k_host

    def _compute_displacement_mesh(self) -> None:
        n_iter = 0 if self.RSDspace == "RealSpace" else self.n_iterations
        self._compute_displacement_iterative_potential(n_iterations=n_iter)

    def _compute_displacement_iterative_potential(self, n_iterations=3) -> None:
        """Iterative Zel'dovich (Burden) reconstruction of the displacement, on the GPU.

        Each iteration solves the potential ``phi = delta / (bias k^2)``, takes
        its gradient, projects the gradient onto the line of sight, and uses the
        divergence of that parallel field to correct the density. On convergence
        the displacement is ``psi = grad(phi)``. Everything stays device-resident.
        """
        logger.info(f"Computing displacement iteratively with {n_iterations} iterations (GPU)")
        xp = self.xp

        # delta_dev is read-only here (re-added each Burden iteration, never
        # mutated), so no private copy is needed even when the input already lives
        # on the device. Keep it read-only if you touch the loop below.
        delta_dev = self.backend.to_device(self.delta_on_mesh)

        # 1-D wavevectors only: |k|^2 and 1/(bias k^2) are evaluated on the fly
        # inside _scale_component_k, so no k-dependent grid is ever materialized.
        kx_h, ky_h, kz_h = self._k_components()
        kx = xp.asarray(kx_h, dtype=xp.float32)
        ky = xp.asarray(ky_h, dtype=xp.float32)
        kz = xp.asarray(kz_h, dtype=xp.float32)
        k_comps = (kx, ky, kz)
        inv_bias = float(1.0 / self.bias)
        nyk, nzk = int(ky.size), int(kz.size)

        # The line of sight sets the geometry of the projection:
        #  * FixedAxisLOS (plane-parallel): n_hat is a fixed Cartesian axis, so the
        #    parallel field keeps only that component -- one gradient component and
        #    a single-axis divergence.
        #  * LocalLOS (radial): n_hat = x/|x| points from the observer to each cell,
        #    evaluated on the fly by the device kernels (no versor grid). Any other
        #    LOS is unsupported.
        # (The irfft->rfft round trip below is deliberate, not collapsible to
        #  k_a^2/k^2: it cleans the Nyquist plane.)
        axis = getattr(self.los, "axis", None)
        radial = axis is None and getattr(self.los, "min_corner", None) is not None

        _rfftn = lambda a: self.fft.rfftn(xp.ascontiguousarray(a))
        _irfftn = lambda a, s: self.fft.irfftn(a, s=s)

        # Density contrast in Fourier space.
        delta_k = self.fft.rfftn(delta_dev)
        temp_k_comp = xp.empty_like(delta_k)   # reused complex scratch

        if radial and n_iterations > 0:
            # Radial LOS geometry as scalars for the on-the-fly versor, plus the
            # parallel magnitude s and a reused scatter scratch (no versor grid).
            mc, cs = self.los.min_corner, self.los.cell_size
            min_x, min_y, min_z = float(mc[0]), float(mc[1]), float(mc[2])
            cell_x, cell_y, cell_z = float(cs[0]), float(cs[1]), float(cs[2])
            if self.dist.is_distributed:
                # The versor kernels recover the cell x-index from the local
                # flat index; shifting min_x by the slab origin makes the
                # coordinate global with no kernel change (audit E6).
                min_x += self.fft.decomp.x_offset * cell_x
            ny, nz = int(delta_dev.shape[1]), int(delta_dev.shape[2])
            s = xp.empty(delta_dev.shape, dtype=delta_dev.dtype)            # s = grad . n_hat
            proj_scratch = xp.empty(delta_dev.shape, dtype=delta_dev.dtype)  # s * n_hat
        elif axis is None and n_iterations > 0:
            raise TypeError(
                f"FFTSolverGPU does not support line-of-sight strategy "
                f"{type(self.los).__name__!r}: expected FixedAxisLOS (exposes "
                f"'.axis') or LocalLOS (exposes '.min_corner')."
            )

        for iteration in range(n_iterations):
            logger.info(f"Iteration {iteration + 1}")

            if axis is not None:
                # Gradient of the potential along the LOS axis, in one fused pass:
                # grad_a_k = -i k_a delta_k / (bias k^2).
                _scale_component_k(delta_k, kx, ky, kz, inv_bias, -1.0,
                                   axis, 1, nyk, nzk, temp_k_comp)
                del delta_k
                grad_a = self.fft.irfftn(temp_k_comp, s=delta_dev.shape)
                # Divergence of the parallel field (single axis): d_a grad_a.
                corr_k = self.fft.rfftn(grad_a)
                _scale_component_k(corr_k, kx, ky, kz, 1.0, 1.0,
                                   axis, 0, nyk, nzk, temp_k_comp)
                del corr_k
                correction = self.fft.irfftn(temp_k_comp, s=delta_dev.shape)

            elif radial:
                # Project the gradient onto the radial LOS: accumulate the parallel
                # magnitude s = grad.n_hat one gradient component at a time, with the
                # versor evaluated on the fly by the device kernel (no versor grid)
                # and the k-space scaling -i k_i delta_k / (bias k^2) fused into a
                # single pass per component (no scaled_k / inv_k2 grids).
                for i in range(3):
                    _scale_component_k(delta_k, kx, ky, kz, inv_bias, -1.0,
                                       i, 1, nyk, nzk, temp_k_comp)
                    grad_i = self.fft.irfftn(temp_k_comp, s=delta_dev.shape)
                    _project_grad_onto_los(grad_i, i, min_x, min_y, min_z,
                                           cell_x, cell_y, cell_z, ny, nz, int(i == 0), s)
                del delta_k
                self.xp.get_default_memory_pool().free_all_blocks()

                # Divergence of the parallel field s*n_hat, each component scattered
                # on the fly into the reused scratch grid.
                def _parallel_component(i):
                    _reconstruct_parallel_vector(s, i, min_x, min_y, min_z,
                                                 cell_x, cell_y, cell_z, ny, nz, proj_scratch)
                    return proj_scratch

                correction = divergence_from_components(
                    _parallel_component, k_comps, _rfftn, _irfftn, xp)

            # Burden update: reconstructed density = delta - f * div(parallel)
            # (with the RSD factor 1/(1+beta) on the first iteration).
            correction *= - self.f
            if iteration == 0:
                correction /= (1 + self.beta)
            xp.add(delta_dev, correction, out=correction)
            delta_k = self.fft.rfftn(correction)

        if n_iterations > 0 and radial:
            del s, proj_scratch

        # Converged density -> displacement psi = grad(phi), phi = delta/(bias k^2),
        # built and kept on the device (the potential is recomputed from psi on demand).
        # In distributed mode the grid is allocated with read-back ghost planes
        # up front and the components are written into the interior view, so no
        # post-hoc extended copy is ever needed for cross-rank interpolation.
        if self.dist.is_distributed:
            w = self._DISP_HALO
            ext_shape = (delta_dev.shape[0] + 2 * w,) + delta_dev.shape[1:] + (3,)
            self._displacement_ext = xp.empty(ext_shape, dtype=delta_dev.dtype)
            self._disp_halo_filled = False
            displacement_dev = self._displacement_ext[w:-w]
        else:
            displacement_dev = xp.empty(delta_dev.shape + (3,), dtype=delta_dev.dtype)
        for i in range(3):
            _scale_component_k(delta_k, kx, ky, kz, inv_bias, 1.0,
                               i, 1, nyk, nzk, temp_k_comp)
            displacement_dev[..., i] = self.fft.irfftn(temp_k_comp, s=delta_dev.shape)

        del temp_k_comp
        self._displacement = displacement_dev

    def _compute_potential_mesh(self) -> None:
        if self._potential is not None:
            return
        if self._displacement is None:
            self._compute_displacement_mesh()
        xp = self.xp

        kx_h, ky_h, kz_h = self._k_components()
        kx = xp.asarray(kx_h, dtype=xp.float32)
        ky = xp.asarray(ky_h, dtype=xp.float32)
        kz = xp.asarray(kz_h, dtype=xp.float32)
        nyk, nzk = int(ky.size), int(kz.size)

        # Potential from the displacement: phi_k = i k.psi_k / k^2, with the
        # i k_a / k^2 scaling fused on the fly (no inv_k2 half-grid).
        disp = self._displacement
        phi_k = None
        for i in range(3):
            psi_k_comp = self.fft.rfftn(xp.ascontiguousarray(disp[..., i]))
            _scale_component_k(psi_k_comp, kx, ky, kz, 1.0, 1.0,
                               i, 1, nyk, nzk, psi_k_comp)
            if phi_k is None:
                phi_k = psi_k_comp
            else:
                phi_k += psi_k_comp

        potential_dev = self.fft.irfftn(phi_k, s=self.delta_on_mesh.shape, axes=(0, 1, 2))
        self._potential = self.backend.to_host(potential_dev)

    def read_displacement_at(self, position, mas='CIC', pbc=True):
        """Interpolate the device-resident displacement at particle positions.

        ``position`` is an ``(N, 3)`` array in the box frame (``[0, boxsize)``).
        Returns an ``(N, 3)`` CuPy array; the caller decides if/when to bring it
        to host. The displacement grid itself never leaves the device.

        In distributed mode ``position`` must contain only particles owned by
        this rank (see ``SlabDecomp.owned_mask``); the ghost planes of the
        extended displacement slab are filled from the neighbours once (copy
        halos) and the offset-aware read kernels do the rest.
        """
        if self.dist.is_distributed:
            from baorecon.mas import read_field_at
            from baorecon.utils.distributed import halo_exchange_copy

            _ = self.displacement  # ensure the field (and its halo slab) exists
            if not self._disp_halo_filled:
                halo_exchange_copy(self._displacement_ext, self.dist,
                                   self._DISP_HALO, pbc=pbc)
                # the exchange runs on CuPy's stream; the numba read kernels run
                # on numba's -- order them explicitly (audit E7).
                self.xp.cuda.get_current_stream().synchronize()
                self._disp_halo_filled = True
            return read_field_at(self._displacement_ext, position, self.mesh,
                                 self.dist, scheme=mas, pbc=pbc)

        from baorecon.field_ops import interpolate_vector_field

        return interpolate_vector_field(
            pos=self.xp.asarray(position, dtype=self.xp.float32),
            field=self.displacement,
            boxsize=self.mesh.boxsize,
            MAS=mas,
            pbc=pbc,
            dtype=self.mesh.dtype,
        )