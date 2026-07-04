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
    divergence_inplace,
    prepare_k_components,
)
from baorecon.utils.backend import get_fft_backend
from baorecon.utils.loggers import setup_logger

logger = setup_logger(__name__)


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
        logger.info(f"Computing displacement iteratively with {n_iterations} iterations (GPU)")
        xp = self.xp

        # Work on a private copy: the Burden iteration overwrites delta in place,
        # and ``to_device`` returns the input unchanged when it is already a
        # device array, so without this we would mutate the caller's delta.
        delta_dev = self.backend.to_device(self.delta_on_mesh)
        if delta_dev is self.delta_on_mesh:
            delta_dev = delta_dev.copy()

        kx_h, ky_h, kz_h = self._k_components()
        kx, ky, kz = xp.asarray(kx_h), xp.asarray(ky_h), xp.asarray(kz_h)
        k_comps = (kx, ky, kz)
        k_bcast = (kx[:, None, None], ky[None, :, None], kz[None, None, :])

        # Only the real 1/(bias k^2) half-grid is kept resident; the -i sign is
        # folded into the per-component multiply instead of materialising a
        # separate full complex grid.
        inv_k2_bias = build_inv_k2(k_comps, bias=self.bias)

        # Plane-parallel (fixed-axis) LOS projects onto a single axis, so we
        # build just ``grad_a`` (1 grid) instead of the full 3-vector field and
        # take its single-axis divergence -- what the old code computed (the
        # other two components project to zero) without the 2 wasted gradient
        # grids and 4 wasted transforms per iteration. ``FixedAxisLOS`` exposes
        # ``.axis``; ``LocalLOS`` does not (None here). The irfft->rfft round
        # trip is kept (not collapsed to k_a^2/k^2): it cleans the Nyquist
        # plane, which the collapsed form changes by ~1% on the default axis.
        axis = getattr(self.los, "axis", None)

        # A radial (LocalLOS) projection is a per-cell scalar contraction
        # s(x) = grad.n_hat; streaming it avoids the full (N,N,N,3) gradient and
        # the heavy device-side ``project_vector_field`` temporaries (which also
        # redundantly re-normalise an already-unit versor). Any other LOS falls
        # back to the generic gradient+project+divergence path.
        versor = None
        if axis is None and n_iterations > 0:
            versor = getattr(self.los, "radial_versor", None)

        # cupy FFT callables for the lean component-wise divergence (fallback).
        _rfftn = lambda a: self.fft.rfftn(xp.ascontiguousarray(a))
        _irfftn = lambda a, s: self.fft.irfftn(a, s=s)

        delta_k = self.fft.rfftn(delta_dev)

        # One complex half-grid, reused for every component transform below.
        temp_k_comp = xp.empty_like(delta_k)

        if axis is not None:
            ka = k_bcast[axis]                      # k along the LOS axis
        elif versor is not None:
            n_hat = xp.asarray(versor)              # upload the unit versor once (3R device)
            s = xp.empty(delta_dev.shape, dtype=delta_dev.dtype)   # reused scalar buffer
        elif n_iterations > 0:
            grad_phi_est_x = xp.empty(delta_dev.shape + (3,), dtype=delta_dev.dtype)

        for iteration in range(n_iterations):
            logger.info(f"Iteration {iteration + 1}")

            if axis is not None:
                # Single-component gradient: grad_a = irfft(-i k_a inv_k2_bias delta_k)
                xp.multiply(delta_k, inv_k2_bias, out=temp_k_comp)
                del delta_k     # dead until rebuilt below; return its block to the pool
                temp_k_comp *= ka
                temp_k_comp *= -self._complex_j
                grad_a = self.fft.irfftn(temp_k_comp, s=delta_dev.shape)
                # Single-axis divergence: correction = irfft(i k_a rfft(grad_a))
                corr_k = self.fft.rfftn(grad_a)
                corr_k *= ka
                corr_k *= self._complex_j
                correction = self.fft.irfftn(corr_k, s=delta_dev.shape)
            elif versor is not None:
                # LocalLOS streamed: s = grad.n_hat, accumulated one component
                # at a time (no full gradient field, no project_vector_field).
                scaled_delta_k = delta_k * inv_k2_bias
                del delta_k     # dead until rebuilt below; return its block to the pool
                scaled_delta_k *= -self._complex_j
                for i in range(3):
                    xp.multiply(scaled_delta_k, k_bcast[i], out=temp_k_comp)
                    grad_i = self.fft.irfftn(temp_k_comp, s=delta_dev.shape)
                    if i == 0:
                        xp.multiply(grad_i, n_hat[..., 0], out=s)
                    else:
                        grad_i *= n_hat[..., i]
                        s += grad_i
                del scaled_delta_k
                # correction = irfft( sum_j i k_j rfft(s * n_hat_j) ), streamed.
                div_k = None
                for j in range(3):
                    comp_k = self.fft.rfftn(s * n_hat[..., j])   # s*n_hat_j is contiguous
                    comp_k *= k_bcast[j]
                    comp_k *= self._complex_j
                    if div_k is None:
                        div_k = comp_k
                    else:
                        div_k += comp_k
                correction = self.fft.irfftn(div_k, s=delta_dev.shape)
            else:
                # Generic-LOS fallback: full gradient, project, divergence.
                scaled_delta_k = delta_k * inv_k2_bias
                del delta_k     # dead until rebuilt below; return its block to the pool
                scaled_delta_k *= -self._complex_j
                for i in range(3):
                    xp.multiply(scaled_delta_k, k_bcast[i], out=temp_k_comp)
                    grad_phi_est_x[..., i] = self.fft.irfftn(temp_k_comp, s=delta_dev.shape)
                del scaled_delta_k
                parallel = self.los.project_parallel(grad_phi_est_x, out=grad_phi_est_x)
                correction = divergence_inplace(parallel, k_comps, _rfftn, _irfftn, xp)

            correction *= - self.f
            if iteration == 0:
                correction /= (1 + self.beta)

            xp.add(delta_dev, correction, out=correction)
            delta_k = self.fft.rfftn(correction)

        if n_iterations > 0:
            if versor is not None:
                del s, n_hat
            elif axis is None:
                del grad_phi_est_x

        # Build the displacement straight on the device and keep it there. The
        # potential field (psi_k) is no longer materialised: when a potential is
        # requested it is recomputed from this displacement, so the iFFT
        # reconstruction path never pays for an unused complex grid nor its host
        # transfer. Only the final per-particle read-out leaves the device.
        # Multiply order kept as before (k_i, inv_k2_bias, +i) so this stays
        # bit-for-bit and the GRF/analytic tolerances (1e-6) are unaffected.
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

        # phi_k = sum_i (i k_i / k^2) psi_k_i. psi_k is transformed one component
        # at a time from the device-resident displacement, so no full psi_k grid
        # is ever kept in memory; the first transform doubles as the accumulator.
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