import pytest
import numpy as np
from zeldareco.displacement_solver.poisson_solver import PoissonSolver

# Ensure this import points to the correct file in your package
from zeldareco.displacement_solver.fft_solver import FFTSolver 
from zeldareco.mesh.mesh import Mesh

# ==========================================
# 1. DUMMY CLASSES TO ISOLATE THE TEST
# ==========================================
class DummyMesh:
    """
    Simulates the zeldareco Mesh object to test the FFTSolver in isolation.
    Creates a spatial frequency grid (kmesh) compatible with rfftn.
    """
    def __init__(self, N=16, boxsize=1.0):
        self.N = N
        self.boxsize = boxsize
        
        k_1d = np.fft.fftfreq(N) * N * (2 * np.pi / boxsize)
        k_1d_r = np.fft.rfftfreq(N) * N * (2 * np.pi / boxsize)
        
        kx, ky, kz = np.meshgrid(k_1d, k_1d, k_1d_r, indexing='ij')
        self.kmesh = np.stack([kx, ky, kz], axis=-1)

    def get_parallel_component(self, vector_field):
        """Simulates the line of sight (LOS) strictly along the Z-axis."""
        par = np.zeros_like(vector_field)
        par[..., 2] = vector_field[..., 2]
        return par

def dummy_divergence_FFT(vector_field, kmesh):
    """Computes the divergence in Fourier space for testing purposes."""
    Vk = np.fft.rfftn(vector_field, axes=(0, 1, 2))
    div_k = 1j * np.sum(kmesh * Vk, axis=-1)
    return np.fft.irfftn(div_k, axes=(0, 1, 2))

@pytest.fixture(autouse=True)
def mock_divergence(monkeypatch):
    monkeypatch.setattr("zeldareco.displacement_solver.fft_solver.divergence", dummy_divergence_FFT)


# ==========================================
# 2. FIXTURES FOR TEST DATA
# ==========================================
@pytest.fixture
def basic_setup():
    N = 16
    mesh = DummyMesh(N=N, boxsize=2*np.pi) 
    delta = np.random.normal(0, 0.1, size=(N, N, N))
    delta -= np.mean(delta) # Force zero mean
    return mesh, delta


# ==========================================
# 3. PARAMETRIZED UNIT TESTS
# ==========================================

# We define the spaces to test once so we can easily inject them into all tests
RSD_SPACES = ["RealSpace", "RedshiftSpace"]


@pytest.mark.parametrize("rsd_space", RSD_SPACES)
def test_fftsolver_smoke(basic_setup, rsd_space):
    """
    Smoke test: verifies that both direct (Real) and iterative (Redshift) computations 
    do not crash and produce arrays of the correct dimension with finite values.
    """
    mesh, delta = basic_setup
    
    solver = FFTSolver(delta_on_mesh=delta, mesh=mesh, f=0.8, bias=2.0, RSDspace=rsd_space)
    
    # In RedshiftSpace, 'beta' is required (f/bias). The base PoissonSolver likely 
    # handles this, but we explicitly set it here to be safe if the mock bypasses init logic.
    solver.beta = 0.8 / 2.0 
    
    solver._compute_displacement()
    
    psi = solver._displacement
    assert psi is not None
    assert psi.shape == (mesh.N, mesh.N, mesh.N, 3)
    assert np.isfinite(psi).all()


@pytest.mark.parametrize("rsd_space", RSD_SPACES)
def test_fftsolver_compute_potential(basic_setup, rsd_space):
    """
    Verifies the computation of the gravitational potential phi in both spaces.
    The potential must be a scalar field derived from the displacement.
    """
    mesh, delta = basic_setup
    
    solver = FFTSolver(delta_on_mesh=delta, mesh=mesh, f=0.8, bias=1.0, RSDspace=rsd_space)
    solver.beta = 0.8 / 1.0
    
    solver._compute_potential()
    
    phi = solver._potential
    assert phi is not None
    assert phi.shape == (mesh.N, mesh.N, mesh.N)
    assert np.isfinite(phi).all()


@pytest.mark.parametrize("rsd_space", RSD_SPACES)
def test_fftsolver_analytic_sine_wave(rsd_space):
    """
    Mathematical accuracy test against a known analytical solution.
    
    We provide a pure sinusoidal density wave along the transverse X-axis: 
        delta(x, y, z) = cos(x)
        
    Physics check:
    Because the wave propagates strictly along X, its displacement is purely in X.
    Our DummyMesh sets the Line of Sight (LOS) along Z. 
    Therefore, the parallel (LOS) component of the displacement is 0.
    This means the Redshift Space distortions (RSD) should mathematically vanish, 
    and BOTH spaces must converge to the exact same analytical solution:
        Psi_x = -sin(x)
        Phi = -cos(x)
    """
    N = 32
    L = 2 * np.pi
    mesh = DummyMesh(N=N, boxsize=L)
    
    x = np.linspace(0, L, N, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    delta = np.cos(X)
    
    solver = FFTSolver(delta_on_mesh=delta, mesh=mesh, f=0.8, bias=1.0, RSDspace=rsd_space)
    solver.beta = 0.8 / 1.0
    
    solver._compute_displacement()
    solver._compute_potential()
    
    psi = solver._displacement
    phi = solver._potential
    
    psi_x_expected = -np.sin(X)
    phi_expected = -np.cos(X)
    
    # Rigorous numerical comparison
    assert np.allclose(psi[..., 0], psi_x_expected, atol=1e-10), f"X component of Psi is incorrect in {rsd_space}!"
    assert np.allclose(psi[..., 1], 0.0, atol=1e-10), f"Y component is not zero in {rsd_space}!"
    assert np.allclose(psi[..., 2], 0.0, atol=1e-10), f"Z component is not zero in {rsd_space}!"
    
    assert np.allclose(phi, phi_expected, atol=1e-10), f"Scalar potential Phi is incorrect in {rsd_space}!"


@pytest.mark.parametrize("rsd_space", RSD_SPACES)
def test_fftsolver_conservation_of_mean(basic_setup, rsd_space):
    """
    Conservation of the mean bulk flow.
    Because the k=0 mode is masked out in Fourier space, the reconstructed 
    displacement field must have a strictly zero global mean in all directions,
    regardless of whether RSD iterative corrections were applied.
    """
    mesh, delta = basic_setup
    
    solver = FFTSolver(delta_on_mesh=delta, mesh=mesh, f=0.5, bias=1.5, RSDspace=rsd_space)
    solver.beta = 0.5 / 1.5
    
    solver._compute_displacement()
    
    mean_displacement = np.mean(solver._displacement, axis=(0, 1, 2))
    
    assert np.allclose(mean_displacement, [0.0, 0.0, 0.0], atol=1e-12), \
        f"Mean displacement is not zero in {rsd_space}: {mean_displacement}"


def test_fftsolver_realistic_grf_closure():
    """
    Realistic Forward-Backward Closure Test using a Gaussian Random Field.
    
    Instead of a single sine wave, this generates a stochastic, multi-frequency 
    density field similar to the actual Universe (a Gaussian Random Field). 
    
    Methodology:
    1. Generate a random scalar potential Phi in Fourier space.
    2. Compute the exact true Displacement (Psi = grad Phi).
    3. Compute the exact true Density (Delta = -nabla^2 Phi).
    4. Feed Delta to the FFTSolver and verify it recovers the true Psi and Phi.
    """
    # 1. Realistic Cosmological Setup
    N = 64  # Use a larger grid for realistic frequency mixing
    boxsize = 1000.0  # e.g., 1000 Mpc/h
    
    # Initialize your actual Mesh class
    mesh = Mesh(nmesh=N, boxsize=boxsize, boxcentre=[boxsize/2]*3)
    
    # 2. Generate a random scalar potential in Fourier space (phi_k)
    np.random.seed(42)
    noise_real = np.random.normal(0, 1, size=(N, N, N))
    phi_k = np.fft.rfftn(noise_real)
    
    k2 = np.sum(mesh.kmesh**2, axis=-1)
    mask = k2 > 0
    
    # Clean division (no 1e-5 buffer needed since k=0 is masked)
    phi_k[~mask] = 0.0
    phi_k[mask] = phi_k[mask] / k2[mask]
    
    # 3. Derive the exact True Fields mathematically
    # A. True Density: delta_k = -k^2 * phi_k 
    delta_k = np.zeros_like(phi_k)
    delta_k[mask] = -k2[mask] * phi_k[mask]
    delta_true_real = np.fft.irfftn(delta_k, axes=(0, 1, 2))
    
    # B. True Displacement: Psi_k = -i * k * phi_k  (FIXED MINUS SIGN HERE)
    psi_k_true = np.zeros(phi_k.shape + (3,), dtype=np.complex128)
    for i in range(3):
        psi_k_true[mask, i] = -1j * mesh.kmesh[mask, i] * phi_k[mask]
    psi_true_real = np.fft.irfftn(psi_k_true, axes=(0, 1, 2))
    
    # C. True Potential in real space
    phi_true_real = np.fft.irfftn(phi_k, axes=(0, 1, 2))

    # 4. Run the FFTSolver (Backward Step)
    # We feed it ONLY the density field. It knows nothing about how we generated it.
    solver = FFTSolver(
        delta_on_mesh=delta_true_real, 
        mesh=mesh, 
        bias=1.0, 
        RSDspace="RealSpace"
    )
    
    solver._compute_displacement()
    solver._compute_potential()
    
    # 5. Rigorous Validation
    # Because FFT operations are global and spectral, the solver should recover
    # the true fields up to machine precision, despite the stochastic noise.
    
    # Check Displacement
    assert np.allclose(solver._displacement, psi_true_real, atol=1e-8), \
        "Solver failed to recover the multi-frequency GRF displacement field."
        
    # Check Potential
    # Note: Potentials are only defined up to an additive constant. 
    # We subtract the mean of both fields before comparing them to align their zero-points.
    phi_solver_centered = solver._potential - np.mean(solver._potential)
    phi_true_centered = phi_true_real - np.mean(phi_true_real)
    
    assert np.allclose(phi_solver_centered, phi_true_centered, atol=1e-8), \
        "Solver failed to recover the multi-frequency GRF scalar potential."