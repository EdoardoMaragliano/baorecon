"""CPU FFT displacement solver (iFFT / Burden algorithm), optimized for CPU.

Uses ``scipy.fft`` with multithreading (``workers=-1``) and in-place transforms
(``overwrite_x=True``) on throwaway buffers. The original input ``delta`` is
never modified. The line-of-sight projection is delegated to an injected
:class:`~baorecon.mesh.los.LOSStrategy`.
"""

import numpy as np
import scipy.fft as sfft

from baorecon.field_ops import divergence_FFT, interpolate_vector_field
from baorecon.solvers._interface import PoissonSolver
from baorecon.solvers.fft._common import compute_k2, prepare_k_components
from baorecon.utils.loggers import setup_logger

logger = setup_logger(__name__)

class FFTSolverCPU(PoissonSolver):
    """FFT-based Poisson/Zel'dovich solver on the CPU."""

    def __init__(self, delta_on_mesh, mesh, los=None, f=None, bias=1.0,
                 RSDspace="RealSpace", n_iterations=3) -> None:
        super().__init__(delta_on_mesh, mesh, f=f, bias=bias, RSDspace=RSDspace)
        self.los = los
        self.n_iterations = n_iterations
        self._psi_k = None
        self._complex_j = np.complex64(1j)

    def __getstate__(self):
        state = self.__dict__.copy()
        for key in ("backend", "xp", "fft", "_complex_j"):
            state.pop(key, None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._complex_j = np.complex64(1j)

    def _compute_displacement_mesh(self) -> None:
        n_iter = 0 if self.RSDspace == "RealSpace" else self.n_iterations
        self._compute_displacement_iterative_potential(n_iterations=n_iter)

    def _compute_displacement_iterative_potential(self, n_iterations=3) -> None:
        logger.info(f"Computing displacement iteratively with {n_iterations} iterations")

        delta = self.delta_on_mesh
        logger.info(f"Dtype delta: {self.delta_on_mesh.dtype}")
        kx, ky, kz = prepare_k_components(self.mesh.cell_size, self.mesh.nmesh)
        k_comps = (kx[:, None, None], ky[None, :, None], kz[None, None, :])

        k2 = compute_k2((kx, ky, kz))
        k2[0, 0, 0] = 1.0
        k2 *= self.bias
        np.divide(1.0, k2, out=k2)
        k2[0, 0, 0] = 0.0
        inv_k2_bias = k2

        # First transform of the original delta -- never overwrite it (Burden).
        delta_k_it = sfft.rfftn(delta, workers=-1)

        # Gradient buffer is only needed when iterating.
        if n_iterations > 0:
            grad_phi_est_x = np.empty(delta.shape + (3,), dtype=delta.dtype)

        for iteration in range(n_iterations):
            logger.info(f"Iteration {iteration + 1}")

            for i in range(3):
                
                step_k = delta_k_it.copy() # Alloca 134 MB temporanei
                step_k *= inv_k2_bias
                step_k *= k_comps[i]
                step_k *= -self._complex_j
                
                # irfftn(overwrite_x=True) distrugge e ricicla step_k internamente
                grad_phi_est_x[..., i] = sfft.irfftn(
                    step_k, s=delta.shape, workers=-1, overwrite_x=True
                )
            del delta_k_it

            # Reuse the gradient buffer for the LOS projection.
            parallel = self.los.project_parallel(grad_phi_est_x, out=grad_phi_est_x)
            correction = divergence_FFT(vector_field=parallel, k_components=(kx, ky, kz))
            np.multiply(correction, -self.f, out=correction)

            if iteration == 0:
                np.divide(correction, (1 + self.beta), out=correction)
            
            np.add(correction, delta, out=correction)

            delta_k_it = sfft.rfftn(correction, workers=-1, overwrite_x=True)

        if n_iterations > 0:
            del grad_phi_est_x

        host_displacement = np.empty(delta.shape + (3,), dtype=delta.dtype)
        for i in range(3):
            # Stesso trucco del buffer temporaneo locale
            step_k = delta_k_it.copy()
            step_k *= inv_k2_bias
            step_k *= k_comps[i]
            step_k *= self._complex_j
            
            host_displacement[..., i] = sfft.irfftn(
                step_k, s=delta.shape, workers=-1, overwrite_x=True
            )
        logger.info(f"dtype psi is {host_displacement.dtype}")
        self._displacement = host_displacement

    def _compute_potential_mesh(self) -> None:
        if self._potential is None:
            if self._displacement is None:
                self._compute_displacement_mesh()

            kx, ky, kz = prepare_k_components(self.mesh.cell_size, self.mesh.nmesh)
            k_comps = (kx[:, None, None], ky[None, :, None], kz[None, None, :])

            k2 = compute_k2((kx, ky, kz))
            k2[0, 0, 0] = 1.0
            np.divide(1.0, k2, out=k2)
            k2[0, 0, 0] = 0.0
            inv_k2 = k2

            phi_k = np.zeros(k2.shape, dtype=np.complex64)

            # Loop over the three physical dimensions (0: x, 1: y, 2: z)
            for i in range(3):
                # 1. FFT one component at a time (no overwrite_x=True!)
                psi_k_comp = sfft.rfftn(self.displacement[..., i], workers=-1)
                
                # 2. Accumulate the divergence into the potential in-place
                psi_k_comp *= k_comps[i]
                psi_k_comp *= inv_k2
                psi_k_comp *= self._complex_j
                
                phi_k += psi_k_comp

            # 3. Final inverse transform. Implicit axes are fine since phi_k is now 3D.
            self._potential = sfft.irfftn(phi_k, s=self.delta_on_mesh.shape, workers=-1)
            
    def read_displacement_at(self, pos):
            raise NotImplementedError("Still working on this! Requires to move the logic of the interpolation from bao_reconstructor to here.")