"""GPU FFT displacement solver (iFFT / Burden algorithm), optimized for GPU.

Everything stays resident on the device: the iterative loop, the final
displacement field, and (if requested) the potential are all computed and kept
on the GPU. Nothing is copied back to host here -- only the small per-particle
read-out at the end of reconstruction leaves the device. The potential, when
asked for, is recomputed from the device-resident displacement rather than from
a stored psi_k grid.
"""

from baorecon.field_ops import divergence_FFT
from baorecon.solvers._interface import PoissonSolver
from baorecon.solvers.fft._common import compute_k2, prepare_k_components
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

    def __getstate__(self):
        state = self.__dict__.copy()
        for key in ("backend", "xp", "fft", "_complex_j"):
            state.pop(key, None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.backend = get_fft_backend("gpu")
        self.xp = self.backend.xp
        self.fft = self.backend.fft
        self._complex_j = self.xp.complex64(1j)

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

        kx_h, ky_h, kz_h = prepare_k_components(self.mesh.cell_size, self.mesh.nmesh)
        kx, ky, kz = xp.asarray(kx_h), xp.asarray(ky_h), xp.asarray(kz_h)
        k_comps = (kx[:, None, None], ky[None, :, None], kz[None, None, :])

        k2 = compute_k2((kx, ky, kz))
        k2[0, 0, 0] = 1.0
        k2 *= self.bias
        xp.divide(1.0, k2, out=k2)
        k2[0, 0, 0] = 0.0
        inv_k2_bias = k2

        inv_k2_bias_j_neg = -self._complex_j * inv_k2_bias

        delta_k_it = self.fft.rfftn(delta_dev)

        # ``temp_k_comp`` is reused for every component transform, both in the
        # iterations and in the final displacement build, so it is allocated
        # once and freed at the very end (not per-iteration).
        temp_k_comp = xp.empty_like(delta_k_it)
        if n_iterations > 0:
            grad_phi_est_x = xp.empty(delta_dev.shape + (3,), dtype=delta_dev.dtype)

        for iteration in range(n_iterations):
            logger.info(f"Iteration {iteration + 1}")

            scaled_delta_k = inv_k2_bias_j_neg * delta_k_it

            for i in range(3):
                xp.multiply(k_comps[i], scaled_delta_k, out=temp_k_comp)
                grad_phi_est_x[..., i] = self.fft.irfftn(temp_k_comp, s=delta_dev.shape)

            del scaled_delta_k

            parallel = self.los.project_parallel(grad_phi_est_x, out=grad_phi_est_x)
            correction = divergence_FFT(parallel, (kx, ky, kz))
            correction *= - self.f

            if iteration == 0:
                correction /= (1 + self.beta)

            xp.add(delta_dev, correction, out=correction)
            delta_k_it = self.fft.rfftn(correction)

        if n_iterations > 0:
            del grad_phi_est_x

        # Build the displacement straight on the device and keep it there. The
        # potential field (psi_k) is no longer materialised: when a potential is
        # requested it is recomputed from this displacement, so the iFFT
        # reconstruction path never pays for an unused complex grid nor its host
        # transfer. Only the final per-particle read-out leaves the device.
        displacement_dev = xp.empty(delta_dev.shape + (3,), dtype=delta_dev.dtype)
        for i in range(3):
            xp.multiply(k_comps[i], delta_k_it, out=temp_k_comp)
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

        kx_h, ky_h, kz_h = prepare_k_components(self.mesh.cell_size, self.mesh.nmesh)
        kx, ky, kz = xp.asarray(kx_h), xp.asarray(ky_h), xp.asarray(kz_h)
        k_comps = (kx[:, None, None], ky[None, :, None], kz[None, None, :])

        k2 = compute_k2((kx, ky, kz))
        k2[0, 0, 0] = 1.0
        xp.divide(1.0, k2, out=k2)
        k2[0, 0, 0] = 0.0
        inv_k2 = k2

        # phi_k = sum_i (i k_i / k^2) psi_k_i. psi_k is transformed one component
        # at a time from the device-resident displacement, so no full psi_k grid
        # is ever kept in memory.
        disp = self._displacement
        phi_k = xp.zeros(inv_k2.shape, dtype=xp.complex64)
        for i in range(3):
            psi_k_comp = self.fft.rfftn(xp.ascontiguousarray(disp[..., i]))
            psi_k_comp *= k_comps[i]
            psi_k_comp *= inv_k2
            psi_k_comp *= self._complex_j
            phi_k += psi_k_comp

        potential_dev = self.fft.irfftn(phi_k, s=self.delta_on_mesh.shape, axes=(0, 1, 2))
        self._potential = self.backend.to_host(potential_dev)

    def read_displacement_at(self, position, mas = 'CIC'):
        # TODO
        pass