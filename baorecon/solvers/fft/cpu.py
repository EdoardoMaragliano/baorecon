"""CPU FFT displacement solver (iFFT / Burden algorithm), optimized for CPU.

Uses ``scipy.fft`` with multithreading (``workers=-1``) and in-place transforms
(``overwrite_x=True``) on throwaway buffers. The original input ``delta`` is
never modified. The line-of-sight projection is delegated to an injected
:class:`~baorecon.mesh.los.LOSStrategy`.
"""

import numpy as np
import scipy.fft as sfft

from baorecon.solvers._interface import PoissonSolver
from baorecon.solvers.fft._common import (
    build_inv_k2,
    divergence_inplace,
    prepare_k_components,
)
from baorecon.utils.backend import use_pyfftw
from baorecon.utils.loggers import setup_logger

logger = setup_logger(__name__)

class FFTSolverCPU(PoissonSolver):
    """FFT-based Poisson/Zel'dovich solver on the CPU."""

    def __init__(self, delta_on_mesh, mesh, los=None, f=None, bias=1.0,
                 RSDspace="RealSpace", n_iterations=3) -> None:
        super().__init__(delta_on_mesh, mesh, f=f, bias=bias, RSDspace=RSDspace)
        self.los = los
        self.n_iterations = n_iterations
        self._complex_j = np.complex64(1j)
        self._k_host = None

    def __getstate__(self):
        state = self.__dict__.copy()
        for key in ("backend", "xp", "fft", "_complex_j", "_k_host"):
            state.pop(key, None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._complex_j = np.complex64(1j)
        self._k_host = None

    def _k_components(self):
        """Return the cached 1-D wavevector arrays ``(kx, ky, kz)``.

        Built once from the mesh geometry and reused by both the displacement
        and potential passes (they are tiny 1-D arrays, but recomputing them is
        pure waste).
        """
        if self._k_host is None:
            self._k_host = prepare_k_components(self.mesh.cell_size, self.mesh.nmesh)
        return self._k_host

    def _compute_displacement_mesh(self) -> None:
        n_iter = 0 if self.RSDspace == "RealSpace" else self.n_iterations
        self._compute_displacement_iterative_potential(n_iterations=n_iter)

    def _compute_displacement_iterative_potential(self, n_iterations=3) -> None:
        logger.info(f"Computing displacement iteratively with {n_iterations} iterations")

        # Opt-in low-memory path: in-place pyfftw transforms (BAORECON_FFT=pyfftw).
        # Only the LOS types the reconstructor builds are handled in place; any
        # other case transparently falls through to the scipy path below.
        if use_pyfftw():
            from baorecon.solvers.fft import _pyfftw_cpu as _ip
            if _ip.supported(self.los, n_iterations):
                logger.info("Using in-place pyfftw CPU FFT backend")
                self._displacement = _ip.displacement_inplace(
                    self.delta_on_mesh, self.mesh, self.los,
                    self.f, self.bias, self.beta, n_iterations,
                )
                return

        delta = self.delta_on_mesh
        logger.info(f"Dtype delta: {self.delta_on_mesh.dtype}")
        kx, ky, kz = self._k_components()
        k_comps = (kx, ky, kz)
        k_bcast = (kx[:, None, None], ky[None, :, None], kz[None, None, :])

        inv_k2_bias = build_inv_k2(k_comps, bias=self.bias)

        # A plane-parallel (fixed-axis) LOS projects onto a single axis, so the
        # gradient/divergence only ever touch that one component: we build just
        # ``grad_a`` (1 grid) instead of the full 3-vector field and take its
        # single-axis divergence -- bit-for-bit what the old code computed (the
        # other two components project to zero) but without the 2 wasted
        # gradient grids and 4 wasted transforms per iteration.
        # ``FixedAxisLOS`` exposes ``.axis``; ``LocalLOS`` does not (None here).
        # NB: the irfft->rfft round trip is kept (not collapsed to k_a^2/k^2)
        # because that round trip is what cleans the Nyquist plane; collapsing
        # it changes the Nyquist modes by ~1% on the default axis.
        axis = getattr(self.los, "axis", None)

        # NB (CPU): a radial (LocalLOS) projection is NOT streamed here. Streaming
        # the scalar s = grad.n_hat through numpy would replace the parallel
        # numba ``project_vector_field_jit`` with several single-threaded passes
        # over the full grid, which benchmarks ~30% slower on the CPU for no RSS
        # gain (the freed grid is not returned to the OS). The GPU solver does
        # stream, because there the elementwise ops are parallel on-device and it
        # avoids a very heavy ``project_vector_field`` temporary allocation.

        # scipy FFT callables for the lean component-wise divergence (LocalLOS).
        _rfftn = lambda a: sfft.rfftn(a, workers=-1)
        _irfftn = lambda a, s: sfft.irfftn(a, s=s, workers=-1, overwrite_x=True)

        # First transform of the original delta -- never overwrite it (Burden).
        delta_k = sfft.rfftn(delta, workers=-1)
        # One complex half-grid, reused for every component transform below
        # (both the loop and the final displacement build) instead of a fresh
        # ``.copy()`` per component.
        tmp_k = np.empty_like(delta_k)

        if axis is not None:
            ka = k_bcast[axis]                      # k along the LOS axis
        elif n_iterations > 0:
            # LocalLOS (and any generic LOS): full real-space gradient buffer.
            grad = np.empty(delta.shape + (3,), dtype=delta.dtype)

        for iteration in range(n_iterations):
            logger.info(f"Iteration {iteration + 1}")

            if axis is not None:
                # Single-component gradient: grad_a = irfft(-i k_a inv_k2_bias delta_k)
                np.multiply(delta_k, inv_k2_bias, out=tmp_k)
                del delta_k     # dead until rebuilt below; free it before the transforms
                tmp_k *= ka
                tmp_k *= -self._complex_j
                grad_a = sfft.irfftn(tmp_k, s=delta.shape, workers=-1, overwrite_x=True)
                # Single-axis divergence: correction = irfft(i k_a rfft(grad_a))
                corr_k = sfft.rfftn(grad_a, workers=-1, overwrite_x=True)
                corr_k *= ka
                corr_k *= self._complex_j
                correction = sfft.irfftn(corr_k, s=delta.shape, workers=-1, overwrite_x=True)
            else:
                # LocalLOS (radial) / generic LOS: build the full gradient,
                # project onto the LOS with the parallel numba JIT kernel, then
                # take the lean component-wise divergence.
                scaled_k = delta_k * inv_k2_bias
                del delta_k     # dead until rebuilt below; free it before the gradient/divergence
                scaled_k *= -self._complex_j
                for i in range(3):
                    np.multiply(scaled_k, k_bcast[i], out=tmp_k)
                    grad[..., i] = sfft.irfftn(tmp_k, s=delta.shape, workers=-1, overwrite_x=True)
                del scaled_k
                parallel = self.los.project_parallel(grad, out=grad)
                correction = divergence_inplace(parallel, k_comps, _rfftn, _irfftn, np)

            np.multiply(correction, -self.f, out=correction)
            if iteration == 0:
                np.divide(correction, (1 + self.beta), out=correction)
            np.add(correction, delta, out=correction)

            delta_k = sfft.rfftn(correction, workers=-1, overwrite_x=True)

        if axis is None and n_iterations > 0:
            del grad

        # Final displacement build. Multiply order kept identical to the
        # original (delta_k * inv_k2_bias, then k_i, then +i) so this stays
        # bit-for-bit and the analytic/GRF tolerances (1e-6) are unaffected.
        host_displacement = np.empty(delta.shape + (3,), dtype=delta.dtype)
        for i in range(3):
            np.multiply(delta_k, inv_k2_bias, out=tmp_k)
            tmp_k *= k_bcast[i]
            tmp_k *= self._complex_j
            host_displacement[..., i] = sfft.irfftn(
                tmp_k, s=delta.shape, workers=-1, overwrite_x=True
            )
        logger.info(f"dtype psi is {host_displacement.dtype}")
        self._displacement = host_displacement

    def _compute_potential_mesh(self) -> None:
        if self._potential is None:
            if self._displacement is None:
                self._compute_displacement_mesh()

            kx, ky, kz = self._k_components()
            k_bcast = (kx[:, None, None], ky[None, :, None], kz[None, None, :])

            inv_k2 = build_inv_k2((kx, ky, kz))

            # phi_k = sum_i (i k_i / k^2) psi_k_i, accumulated one component at a
            # time; the first transform doubles as the accumulator so no extra
            # zeroed grid is allocated.
            phi_k = None
            for i in range(3):
                psi_k_comp = sfft.rfftn(self.displacement[..., i], workers=-1)
                psi_k_comp *= k_bcast[i]
                psi_k_comp *= inv_k2
                psi_k_comp *= self._complex_j
                if phi_k is None:
                    phi_k = psi_k_comp
                else:
                    phi_k += psi_k_comp

            self._potential = sfft.irfftn(phi_k, s=self.delta_on_mesh.shape, workers=-1)
            
    def read_displacement_at(self, pos):
            raise NotImplementedError("Still working on this! Requires to move the logic of the interpolation from bao_reconstructor to here.")