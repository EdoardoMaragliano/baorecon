"""CPU FFT displacement solver (iFFT / Burden algorithm), optimized for CPU.

Uses ``scipy.fft`` with multithreading (``workers=-1``) and in-place transforms
(``overwrite_x=True``) on throwaway buffers. The original input ``delta`` is
never modified. The line-of-sight projection is delegated to an injected
:class:`~baorecon.mesh.los.LOSStrategy`.
"""

import numpy as np
import scipy.fft as sfft

from baorecon.field_ops import interpolate_vector_field
from baorecon.solvers._interface import PoissonSolver
from baorecon.solvers.fft._common import (
    build_inv_k2,
    divergence_from_components,
    prepare_k_components,
)
from baorecon.solvers.fft._radial_stream import (
    project_grad_onto_los,
    reconstruct_parallel_vector,
)
from baorecon.utils.backend import use_pyfftw
from baorecon.utils.formatters import format_mas
from baorecon.utils.loggers import setup_logger

logger = setup_logger(__name__)

class FFTSolverCPU(PoissonSolver):
    """FFT-based Poisson/Zel'dovich solver on the CPU."""

    def __init__(self, delta_on_mesh, mesh, los=None, f=None, bias=1.0,
                 RSDspace="RealSpace", n_iterations=3, pbc=False) -> None:
        super().__init__(delta_on_mesh, mesh, f=f, bias=bias, RSDspace=RSDspace)
        self.los = los
        self.n_iterations = n_iterations
        self._pbc = pbc
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
            # Build the wavevectors at the mesh's working precision so a float64
            # mesh yields float64 k (and hence a float64 potential/displacement).
            self._k_host = prepare_k_components(
                self.mesh.cell_size, self.mesh.nmesh, dtype=self.mesh.dtype
            )
        return self._k_host

    def _compute_displacement_mesh(self) -> None:
        n_iter = 0 if self.RSDspace == "RealSpace" else self.n_iterations
        self._compute_displacement_iterative_potential(n_iterations=n_iter)

    def _compute_displacement_iterative_potential(self, n_iterations=3) -> None:
        """Iterative Zel'dovich (Burden) reconstruction of the displacement.

        Each iteration solves the potential ``phi = delta / (bias k^2)``, takes
        its gradient, projects the gradient onto the line of sight, and uses the
        divergence of that parallel field to correct the density. On convergence
        the displacement is ``psi = grad(phi)``.
        """
        logger.info(f"Computing displacement iteratively with {n_iterations} iterations")

        # Optional low-memory backend (BAORECON_FFT=pyfftw); identical physics.
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

        # The line of sight sets the geometry of the projection:
        #  * FixedAxisLOS (plane-parallel): n_hat is a fixed Cartesian axis, so the
        #    parallel field keeps only that component -- one gradient component and
        #    a single-axis divergence.
        #  * LocalLOS (radial): n_hat = x/|x| points from the observer to each
        #    cell, so the full gradient is projected onto n_hat cell by cell.
        # (The irfft->rfft round trip below is deliberate, not collapsible to
        #  k_a^2/k^2: it cleans the Nyquist plane.)
        axis = getattr(self.los, "axis", None)
        radial = axis is None and getattr(self.los, "min_corner", None) is not None

        _rfftn_scratch = lambda a: sfft.rfftn(a, workers=-1, overwrite_x=True)
        _irfftn = lambda a, s: sfft.irfftn(a, s=s, workers=-1, overwrite_x=True)

        # Density contrast in Fourier space (kept intact: each iteration adds delta).
        delta_k = sfft.rfftn(delta, workers=-1)
        tmp_k = np.empty_like(delta_k)   # reused complex scratch

        if axis is not None:
            ka = k_bcast[axis]                      # k along the LOS axis
        elif radial and n_iterations > 0:
            # Radial LOS geometry, plus the parallel magnitude s and field s*n_hat.
            min_corner = self.los.min_corner
            cell_size = self.los.cell_size
            los_magnitude = np.empty(delta.shape, dtype=delta.dtype)   # s = grad . n_hat
            proj_scratch = np.empty(delta.shape, dtype=delta.dtype)    # s * n_hat
        elif n_iterations > 0:
            raise TypeError(
                f"FFTSolverCPU does not support line-of-sight strategy "
                f"{type(self.los).__name__!r}: expected FixedAxisLOS (exposes "
                f"'.axis') or LocalLOS (exposes '.min_corner')."
            )

        for iteration in range(n_iterations):
            logger.info(f"Iteration {iteration + 1}")

            if axis is not None:
                # Gradient of the potential along the LOS axis: grad_a = d_a phi.
                np.multiply(delta_k, inv_k2_bias, out=tmp_k)
                del delta_k
                tmp_k *= ka
                tmp_k *= -self._complex_j
                grad_a = sfft.irfftn(tmp_k, s=delta.shape, workers=-1, overwrite_x=True)
                # Divergence of the parallel field (single axis): d_a grad_a.
                corr_k = sfft.rfftn(grad_a, workers=-1, overwrite_x=True)
                corr_k *= ka
                corr_k *= self._complex_j
                correction = sfft.irfftn(corr_k, s=delta.shape, workers=-1, overwrite_x=True)
                
            elif radial:
                # Project the gradient onto the radial LOS: accumulate the parallel
                # magnitude s = grad.n_hat one gradient component at a time.
                scaled_k = delta_k * inv_k2_bias
                del delta_k
                scaled_k *= -self._complex_j
                for i in range(3):
                    np.multiply(scaled_k, k_bcast[i], out=tmp_k)
                    grad_i = sfft.irfftn(tmp_k, s=delta.shape, workers=-1, overwrite_x=True)
                    project_grad_onto_los(los_magnitude, grad_i, i,
                                          min_corner[0], min_corner[1], min_corner[2],
                                          cell_size[0], cell_size[1], cell_size[2], i == 0)
                del scaled_k, grad_i

                # Divergence of the parallel field s*n_hat, each component
                # reconstructed on the fly.
                def _parallel_component(i):
                    reconstruct_parallel_vector(proj_scratch, los_magnitude, i,
                                                min_corner[0], min_corner[1], min_corner[2],
                                                cell_size[0], cell_size[1], cell_size[2])
                    return proj_scratch

                correction = divergence_from_components(
                    _parallel_component, k_comps, _rfftn_scratch, _irfftn, np
                )

            # Burden update: reconstructed density = delta - f * div(parallel)
            # (with the RSD factor 1/(1+beta) on the first iteration).
            np.multiply(correction, -self.f, out=correction)
            if iteration == 0:
                np.divide(correction, (1 + self.beta), out=correction)
            np.add(correction, delta, out=correction)

            delta_k = sfft.rfftn(correction, workers=-1, overwrite_x=True)

        if radial and n_iterations > 0:
            del los_magnitude, proj_scratch

        # Converged density -> displacement psi = grad(phi), phi = delta/(bias k^2).
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

            # Potential from the displacement: phi_k = i k.psi_k / k^2.
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
            
    def read_displacement_at(self, positions, mas="CIC"):
        """Interpolate the spectral displacement grid at ``positions`` (host ``(N, 3)``)."""
        return interpolate_vector_field(
            pos=positions,
            field=self.displacement,
            boxsize=self.mesh.boxsize,
            MAS=format_mas(mas),
            pbc=self._pbc,
            dtype=self.mesh.dtype,
        )
