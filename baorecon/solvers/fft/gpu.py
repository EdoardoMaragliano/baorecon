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
    build_inv_k2,
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
else:
    _project_grad_onto_los = None
    _reconstruct_parallel_vector = None


class FFTSolverGPU(PoissonSolver):
    """FFT-based Poisson/Zel'dovich solver on the GPU (CuPy)."""

    def __init__(self, delta_on_mesh, mesh, los=None, f=None, bias=1.0,
                 RSDspace="RealSpace", n_iterations=3) -> None:
        super().__init__(delta_on_mesh, mesh, f=f, bias=bias, RSDspace=RSDspace)
        self.los = los
        self.n_iterations = n_iterations
        self.backend = get_fft_backend("gpu")
        self.xp = self.backend.xp
        self.fft = self.backend.fft
        self._complex_j = self.xp.complex64(1j)
        self._k_host = None

    def __getstate__(self):
        state = self.__dict__.copy()
        for key in ("backend", "xp", "fft", "_complex_j", "_k_host"):
            state.pop(key, None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.backend = get_fft_backend("gpu")
        self.xp = self.backend.xp
        self.fft = self.backend.fft
        self._complex_j = self.xp.complex64(1j)
        self._k_host = None

    def _k_components(self):
        """Cached host 1-D wavevector arrays ``(kx, ky, kz)``.

        Cached on the host and uploaded per method; recomputing the 1-D arrays
        on every displacement/potential pass is pure waste.
        """
        if self._k_host is None:
            self._k_host = prepare_k_components(self.mesh.cell_size, self.mesh.nmesh)
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

        kx_h, ky_h, kz_h = self._k_components()
        kx, ky, kz = xp.asarray(kx_h), xp.asarray(ky_h), xp.asarray(kz_h)
        k_comps = (kx, ky, kz)
        k_bcast = (kx[:, None, None], ky[None, :, None], kz[None, None, :])

        inv_k2_bias = build_inv_k2(k_comps, bias=self.bias)

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

        if axis is not None:
            ka = k_bcast[axis]                      # k along the LOS axis
        elif radial and n_iterations > 0:
            # Radial LOS geometry as scalars for the on-the-fly versor, plus the
            # parallel magnitude s and a reused scatter scratch (no versor grid).
            mc, cs = self.los.min_corner, self.los.cell_size
            min_x, min_y, min_z = float(mc[0]), float(mc[1]), float(mc[2])
            cell_x, cell_y, cell_z = float(cs[0]), float(cs[1]), float(cs[2])
            ny, nz = int(delta_dev.shape[1]), int(delta_dev.shape[2])
            s = xp.empty(delta_dev.shape, dtype=delta_dev.dtype)            # s = grad . n_hat
            proj_scratch = xp.empty(delta_dev.shape, dtype=delta_dev.dtype)  # s * n_hat
            scaled_k = xp.empty_like(delta_k)   # reused complex scratch: delta_k * inv_k2_bias
        elif n_iterations > 0:
            raise TypeError(
                f"FFTSolverGPU does not support line-of-sight strategy "
                f"{type(self.los).__name__!r}: expected FixedAxisLOS (exposes "
                f"'.axis') or LocalLOS (exposes '.min_corner')."
            )

        for iteration in range(n_iterations):
            logger.info(f"Iteration {iteration + 1}")

            if axis is not None:
                # Gradient of the potential along the LOS axis: grad_a = d_a phi.
                xp.multiply(delta_k, inv_k2_bias, out=temp_k_comp)
                del delta_k
                temp_k_comp *= ka
                temp_k_comp *= -self._complex_j
                grad_a = self.fft.irfftn(temp_k_comp, s=delta_dev.shape)
                # Divergence of the parallel field (single axis): d_a grad_a.
                corr_k = self.fft.rfftn(grad_a)
                corr_k *= ka
                corr_k *= self._complex_j
                correction = self.fft.irfftn(corr_k, s=delta_dev.shape)
                
            elif radial:
                # Project the gradient onto the radial LOS: accumulate the parallel
                # magnitude s = grad.n_hat one gradient component at a time, with the
                # versor evaluated on the fly by the device kernel (no versor grid).
                xp.multiply(delta_k, inv_k2_bias, out=scaled_k)
                del delta_k
                scaled_k *= -self._complex_j
                for i in range(3):
                    xp.multiply(scaled_k, k_bcast[i], out=temp_k_comp)
                    grad_i = self.fft.irfftn(temp_k_comp, s=delta_dev.shape)
                    _project_grad_onto_los(grad_i, i, min_x, min_y, min_z,
                                           cell_x, cell_y, cell_z, ny, nz, int(i == 0), s)
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
            del s, proj_scratch, scaled_k

        # Converged density -> displacement psi = grad(phi), phi = delta/(bias k^2),
        # built and kept on the device (the potential is recomputed from psi on demand).
        displacement_dev = xp.empty(delta_dev.shape + (3,), dtype=delta_dev.dtype)
        for i in range(3):
            xp.multiply(k_bcast[i], delta_k, out=temp_k_comp)
            temp_k_comp *= inv_k2_bias
            temp_k_comp *= self._complex_j
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
        kx, ky, kz = xp.asarray(kx_h), xp.asarray(ky_h), xp.asarray(kz_h)
        k_bcast = (kx[:, None, None], ky[None, :, None], kz[None, None, :])

        inv_k2 = build_inv_k2((kx, ky, kz))

        # Potential from the displacement: phi_k = i k.psi_k / k^2.
        disp = self._displacement
        phi_k = None
        for i in range(3):
            psi_k_comp = self.fft.rfftn(xp.ascontiguousarray(disp[..., i]))
            psi_k_comp *= k_bcast[i]
            psi_k_comp *= inv_k2
            psi_k_comp *= self._complex_j
            if phi_k is None:
                phi_k = psi_k_comp
            else:
                phi_k += psi_k_comp

        potential_dev = self.fft.irfftn(phi_k, s=self.delta_on_mesh.shape, axes=(0, 1, 2))
        self._potential = self.backend.to_host(potential_dev)

    def read_displacement_at(self, position, mas = 'CIC'):
        # TODO
        pass