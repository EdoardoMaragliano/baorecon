import numpy as np
from zeldareco.displacement_solver.poisson_solver import PoissonSolver
from zeldareco.mesh.mesh import Mesh
from zeldareco.mesh.field_ops import divergence_FFT as divergence
from zeldareco.utils.loggers import setup_logger

logger = setup_logger(__name__)


class FFTSolver(PoissonSolver):
    """
    FFT-based Poisson solver (iFFT algorithm) integrated here.

    Implements both direct computation in RealSpace and iterative RSD-corrected
    computation following the previous `iFFT_solver` implementation.
    """

    def __init__(
        self,
        delta_on_mesh: np.ndarray,
        mesh: Mesh,
        f: float = None,
        bias: float = 1.0,
        RSDspace: str = "RealSpace",
    ) -> None:
        super().__init__(delta_on_mesh, mesh, f=f, bias=bias, RSDspace=RSDspace)
        # internal cached Fourier representation
        self._psi_k = None

    def _compute_displacement(self) -> None:
        """Compute displacement field and store in `self._displacement`."""
        # follow the logic of the old iFFT_solver
        if self.RSDspace == 'RealSpace':
            self._compute_displacement_direct()
        else:
            # default to 3 iterations as before
            self._compute_displacement_iterative_potential(n_iterations=3)

    def _compute_displacement_direct(self):
            logger.info('Using direct estimation of displacement field (FFT).')

            # define k^2 and mask for k=0
            k2 = np.sum(self.mesh.kmesh**2, axis=-1)
            mask = k2 > 0  # Esclude k=0

            # Fourier transform delta
            delta_k = np.fft.rfftn(self.delta_on_mesh)

            # Initialize psi_k with the correct shape for vector field (N, N, N/2+1, 3)
            self._psi_k = np.zeros(delta_k.shape + (3,), dtype=np.complex128)

            # Psi_k = i * k * delta_k / (k^2 * bias)
            inv_k2_bias = 1.0 / (k2[mask] * self.bias)
            
            for i in range(3):
                self._psi_k[mask, i] = 1j * self.mesh.kmesh[mask, i] * delta_k[mask] * inv_k2_bias

            # Inverse FFT to get displacement in real space
            self._displacement = np.fft.irfftn(self._psi_k, axes=(0, 1, 2))


            # Apply RSD correction if needed
            if self.RSDspace == 'RedshiftSpace':
                # Nota: beta = f/b. Assicurati che self.beta sia definito
                parallel_comp = self.mesh.get_parallel_component(self._displacement)
                self._displacement -= (self.beta / (1.0 + self.beta)) * parallel_comp


    def _compute_displacement_iterative_potential(self, n_iterations=3):
        logger.info(f'Computing displacement iteratively (RSD), iterations={n_iterations}')

        # Precompute k^2 and mask for k=0
        k2 = np.sum(self.mesh.kmesh**2, axis=-1)
        mask = k2 > 0 

        #
        sm_delta_on_mesh = np.copy(self.delta_on_mesh)
        delta_k_it = np.fft.rfftn(sm_delta_on_mesh)

        for iteration in range(n_iterations):
            logger.info(f'Iteration {iteration+1}')
            
            # fourier space potential estimation
            phi_est_k = np.zeros_like(delta_k_it)
            phi_est_k[mask] = -delta_k_it[mask] / (k2[mask] * self.bias)

            # gradient in Fourier space
            grad_phi_est_k = np.zeros(delta_k_it.shape + (3,), dtype=np.complex128)
            for i in range(3):
                grad_phi_est_k[mask, i] = 1j * self.mesh.kmesh[mask, i] * phi_est_k[mask]

            # Inverse FFT to get gradient in real space
            grad_phi_est_x = np.fft.irfftn(grad_phi_est_k, axes=(0, 1, 2))

            # RSD Correction
            # Note: grad_phi_est_x is the estimated displacement vector (Psi)
            correction = self.f * divergence(
                self.mesh.get_parallel_component(-grad_phi_est_x),
                self.mesh.kmesh
            )

            if iteration == 0:
                correction /= (1 + self.beta)

            # update delta in real space and transform back to Fourier space for next iteration
            delta_real_it = sm_delta_on_mesh + correction
            delta_k_it = np.fft.rfftn(delta_real_it)

        # --- FINE LOOP ---

        # Final displacement
        self._psi_k = np.zeros(delta_k_it.shape + (3,), dtype=np.complex128)
        inv_k2_bias = 1.0 / (k2[mask] * self.bias)
        
        for i in range(3):
            self._psi_k[mask, i] = 1j * self.mesh.kmesh[mask, i] * delta_k_it[mask] * inv_k2_bias

        self._displacement = np.fft.irfftn(self._psi_k, axes=(0, 1, 2))



    def _compute_potential(self) -> None:
        """Compute potential from the current delta_on_mesh and store in `self._potential`."""

        if self._potential is None:
            if self._displacement is None:
                self._compute_displacement()
            
            inv_k2 = np.zeros(self.mesh.kmesh.shape[:-1], dtype=np.float64)
            k2 = np.sum(self.mesh.kmesh**2, axis=-1)
            mask = k2 > 0
            inv_k2[mask] = 1.0 / k2[mask]

            # compute potential from displacement 
            phi_k = np.zeros(self.mesh.kmesh.shape[:-1], dtype=np.complex128)
            for i in range(3):
                phi_k[mask] += 1j * self.mesh.kmesh[mask, i] * self._psi_k[mask, i] * inv_k2[mask]
            self._potential = np.fft.irfftn(phi_k, axes=(0, 1, 2))
