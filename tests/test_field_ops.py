import pytest
import numpy as np

# Adjust this import to match the actual path to your module
# For example: from zeldareco.mesh.field_ops import ...
from zeldareco.mesh.field_ops import (
    project_vector_field,
    project_vector_field_jit,
    interpolate_vector_field,
    divergence,
    smoothed_field,
    _scale_factor
)


# ==========================================
# 1. DUMMY CLASSES FOR ISOLATION
# ==========================================
class DummyMesh:
    """
    Simulates the Mesh object to provide the kmesh required by FFT divergence
    and Gaussian smoothing without relying on the actual Mesh class implementation.
    """
    def __init__(self, N=16, boxsize=1.0):
        self.N = N
        self.boxsize = boxsize
        self.cell_size = boxsize / N
        
        k_1d = np.fft.fftfreq(N) * N * (2 * np.pi / boxsize)
        k_1d_r = np.fft.rfftfreq(N) * N * (2 * np.pi / boxsize)
        
        kx, ky, kz = np.meshgrid(k_1d, k_1d, k_1d_r, indexing='ij')
        self.kmesh = np.stack([kx, ky, kz], axis=-1)


# ==========================================
# 2. UNIT TESTS: SCALE FACTOR
# ==========================================
def test_scale_factor():
    """Test the cosmological scale factor mathematical definition a = 1/(1+z)."""
    assert np.isclose(_scale_factor(0.0), 1.0)
    assert np.isclose(_scale_factor(1.0), 0.5)
    assert np.isclose(_scale_factor(9.0), 0.1)


# ==========================================
# 3. UNIT TESTS: VECTOR PROJECTION
# ==========================================
@pytest.mark.parametrize(
    "axis_idx, expected_mag", 
    [
        (0, 2.0),  # Project onto X-axis, expect magnitude 2.0
        (1, 3.0),  # Project onto Y-axis, expect magnitude 3.0
        (2, 4.0),  # Project onto Z-axis, expect magnitude 4.0
    ]
)
def test_project_vector_field_tilted(axis_idx, expected_mag):
    """
    Test vector projection of a tilted field onto distinct Cartesian axes.
    We use a field V = (2, 3, 4) to ensure the dot product scaling is exactly 
    correct and avoids false positives that uniform fields (1,1,1) might hide.
    """
    N = 4
    # Create a tilted vector field: V = (2.0, 3.0, 4.0)
    field = np.zeros((N, N, N, 3), dtype=np.float32)
    field[..., 0] = 2.0
    field[..., 1] = 3.0
    field[..., 2] = 4.0
    
    # Create the LOS versor strictly along the parameterized axis
    los_versor = np.zeros((N, N, N, 3), dtype=np.float32)
    los_versor[..., axis_idx] = 1.0 
    
    # --- Test Standard Numpy implementation ---
    proj_np = project_vector_field(field, los_versor)
    
    # The active axis should have the expected magnitude
    assert np.allclose(proj_np[..., axis_idx], expected_mag), \
        f"Numpy: Expected magnitude {expected_mag} on axis {axis_idx}"
        
    # The orthogonal axes must be strictly 0
    for i in range(3):
        if i != axis_idx:
            assert np.allclose(proj_np[..., i], 0.0), \
                f"Numpy: Orthogonal axis {i} should be 0.0"

    # --- Test Numba JIT implementation ---
    proj_jit = project_vector_field_jit(field, los_versor)
    
    assert np.allclose(proj_jit[..., axis_idx], expected_mag), \
        f"JIT: Expected magnitude {expected_mag} on axis {axis_idx}"
        
    for i in range(3):
        if i != axis_idx:
            assert np.allclose(proj_jit[..., i], 0.0), \
                f"JIT: Orthogonal axis {i} should be 0.0"


# ==========================================
# 4. UNIT TESTS: INTERPOLATION (CIC & TSC)
# ==========================================
@pytest.mark.parametrize("mas_scheme", ["CIC", "TSC"])
def test_interpolate_constant_vector_field(mas_scheme):
    """
    Test both Cloud-in-Cell (CIC) and Triangular Shaped Cloud (TSC) JIT interpolators.
    If the vector field is completely uniform, interpolating at ANY random 
    coordinate must return the exact same constant value.
    """
    nmesh = 8
    boxsize = 10.0
    
    # Uniform vector field where every cell holds the vector (5.0, -2.0, 3.0)
    field = np.zeros((nmesh, nmesh, nmesh, 3), dtype=np.float64)
    field[..., 0] = 5.0
    field[..., 1] = -2.0
    field[..., 2] = 3.0
    
    # Random particle positions inside the box
    np.random.seed(42)
    pos = np.random.uniform(0, boxsize, size=(100, 3))
    
    # Interpolate
    interp_field = interpolate_vector_field(pos, field, boxsize, MAS=mas_scheme, pbc=True)
    
    assert interp_field.shape == (100, 3)
    assert np.allclose(interp_field[:, 0], 5.0)
    assert np.allclose(interp_field[:, 1], -2.0)
    assert np.allclose(interp_field[:, 2], 3.0)


# ==========================================
# 5. UNIT TESTS: DIVERGENCE (FINITE DIFF)
# ==========================================
def test_divergence_finite_diff():
    """
    Test the finite difference divergence using a linear vector field.
    Given V = (1x, 2y, 3z), the mathematical divergence is exactly 1 + 2 + 3 = 6.
    """
    N = 10
    cell_size = 1.0
    
    x = np.arange(N)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    
    field = np.stack([1.0 * X, 2.0 * Y, 3.0 * Z], axis=-1).astype(np.float64)
    
    div = divergence(field, div_algo='finite_diff', cell_size=cell_size)
    
    # Check only the inner volume (indices 1 to -1) because finite difference 
    # boundary conditions at the edges of a linear field will differ slightly.
    inner_div = div[1:-1, 1:-1, 1:-1]
    
    assert np.allclose(inner_div, 6.0), "Divergence of (x, 2y, 3z) must be 6"


# ==========================================
# 6. UNIT TESTS: DIVERGENCE (FFT)
# ==========================================
def test_divergence_fft():
    """
    Test the Fourier Transform divergence using a periodic sine wave.
    Given V = (sin(x), 0, 0), the analytical divergence is cos(x).
    Because it uses spectral methods, it should match the analytical 
    derivative almost to machine precision.
    """
    N = 32
    boxsize = 2 * np.pi
    mesh = DummyMesh(N=N, boxsize=boxsize)
    
    x = np.linspace(0, boxsize, N, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    
    field = np.zeros((N, N, N, 3), dtype=np.float64)
    field[..., 0] = np.sin(X)  # V_x = sin(x)
    
    expected_div = np.cos(X)
    
    div = divergence(field, div_algo='FFT', kmesh=mesh.kmesh)
    
    assert np.allclose(div, expected_div, atol=1e-7), "FFT Divergence failed on pure sine wave"


# ==========================================
# 7. UNIT TESTS: SMOOTHING KERNEL
# ==========================================
def test_smoothed_field_constant():
    """
    Test FFT-based Gaussian smoothing.
    Smoothing a completely uniform density field should leave it perfectly unchanged,
    since the k=0 mode (the mean) is preserved and higher modes are zero.
    """
    N = 16
    mesh = DummyMesh(N=N, boxsize=10.0)
    
    # Uniform density field of value 3.0
    field = np.ones((N, N, N), dtype=np.float64) * 3.0
    
    smoothed = smoothed_field(field, mesh, smoothing_radius=2.0)
    
    assert np.allclose(smoothed, 3.0, atol=1e-7), "Smoothing altered a constant field"

def test_smoothed_field_reduces_variance():
    """
    Test that Gaussian smoothing effectively dampens high-frequency noise.
    The variance of a white noise field must strictly decrease after smoothing.
    """
    N = 32
    mesh = DummyMesh(N=N, boxsize=10.0)
    
    np.random.seed(42)
    noisy_field = np.random.normal(0, 1.0, size=(N, N, N))
    
    initial_variance = np.var(noisy_field)
    
    smoothed = smoothed_field(noisy_field, mesh, smoothing_radius=2.0)
    smoothed_variance = np.var(smoothed)
    
    assert smoothed_variance < initial_variance, "Smoothing did not reduce the variance of the noise"