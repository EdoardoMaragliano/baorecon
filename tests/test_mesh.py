import pytest
import numpy as np

# Adjust the import path to match your project structure
from zeldareco.mesh.mesh import Mesh

# Mocking the field_ops import just in case the real one isn't available in the test environment
from unittest.mock import patch

@pytest.fixture
def basic_mesh():
    """Fixture providing a standard Mesh instance for reuse in tests."""
    return Mesh(nmesh=16, boxsize=100.0, boxcentre=[50.0, 50.0, 50.0], los=None)


# ==========================================
# 1. INITIALIZATION AND VALIDATION TESTS
# ==========================================

def test_mesh_initialization(basic_mesh):
    """Test that valid initialization computes the correct basic attributes."""
    assert basic_mesh.nmesh == 16
    assert basic_mesh.boxsize == 100.0
    assert basic_mesh.cell_size == 100.0 / 16
    
    # Check minimum corner logic
    np.testing.assert_array_equal(basic_mesh.boxcentre, [50.0, 50.0, 50.0])
    np.testing.assert_array_equal(basic_mesh.min_corner, [0.0, 0.0, 0.0])

    # Ensure lazy loading is intact
    assert basic_mesh._xmesh is None
    assert basic_mesh._kmesh is None
    assert basic_mesh._los_versor is None

@pytest.mark.parametrize("bad_params", [
    {"nmesh": 0, "boxsize": 100, "boxcentre": [50,50,50], "los": 'z'},
    {"nmesh": 16, "boxsize": -50, "boxcentre": [50,50,50], "los": 'z'},
    {"nmesh": 16, "boxsize": 100, "boxcentre": [50,50], "los": 'z'}, # Wrong center shape
    {"nmesh": 16, "boxsize": 100, "boxcentre": [50,50,50], "los": 'invalid_axis'},
])
def test_mesh_validation_errors(bad_params):
    """Test that invalid parameters raise ValueError."""
    with pytest.raises(ValueError):
        Mesh(**bad_params)

def test_mesh_warning_non_integer_cell():
    """Test that a non-integer cell size triggers a UserWarning."""
    with pytest.warns(UserWarning, match="cell_size is not an integer number"):
        # 100 / 3 = 33.333...
        Mesh(nmesh=3, boxsize=100.0, boxcentre=[50,50,50])


# ==========================================
# 2. CONFIGURATION SPACE GRID TESTS
# ==========================================

def test_mesh_xmesh_generation(basic_mesh):
    """Test the generation of the real-space grid (ConfigSpace)."""
    xmesh = basic_mesh.xmesh
    
    # 1. Shape should be (N, N, N, 3)
    assert xmesh.shape == (16, 16, 16, 3)
    
    # 2. The grid should start at 0 and strictly end before boxsize
    assert np.min(xmesh) == 0.0
    assert np.max(xmesh) == basic_mesh.boxsize - basic_mesh.cell_size
    
    # 3. Check lazy loading persistence (it shouldn't compute it twice)
    assert basic_mesh._xmesh is not None
    assert basic_mesh.xmesh is basic_mesh._xmesh


# ==========================================
# 3. FOURIER SPACE GRID TESTS
# ==========================================

def test_mesh_kmesh_generation(basic_mesh):
    """
    Test the generation of the Fourier-space grid (FourierSpace).
    Critically, because rfftn is expected downstream, the Z-axis must be truncated.
    """
    kmesh = basic_mesh.kmesh
    N = basic_mesh.nmesh
    
    # 1. Shape must account for rfftfreq on the last spatial axis: N//2 + 1
    expected_z_dim = (N // 2) + 1
    assert kmesh.shape == (N, N, expected_z_dim, 3)
    
    # 2. k=0 mode should be at the origin (0,0,0)
    assert np.allclose(kmesh[0, 0, 0], [0.0, 0.0, 0.0])


# ==========================================
# 4. VERSOR (UNIT VECTOR) TESTS
# ==========================================

@pytest.mark.parametrize("los_axis, expected_vector", [
    ('x', [1.0, 0.0, 0.0]),
    ('y', [0.0, 1.0, 0.0]),
    ('z', [0.0, 0.0, 1.0]),
])
def test_mesh_cartesian_los_versors(los_axis, expected_vector):
    """Test that specifying 'x', 'y', or 'z' generates uniform Cartesian versors."""
    mesh = Mesh(nmesh=4, boxsize=100.0, boxcentre=[50,50,50], los=los_axis)
    versor = mesh.los_versor
    
    assert versor.shape == (4, 4, 4, 3)
    assert np.allclose(versor[0,0,0], expected_vector)
    assert np.allclose(versor[-1,-1,-1], expected_vector)

def test_mesh_radial_versor():
    """
    Test the radial versor (Observer line-of-sight).
    Verifies that vectors point radially outward and their magnitudes are exactly 1.
    """
    # Place observer at (0,0,0) by setting min_corner to (0,0,0)
    mesh = Mesh(nmesh=4, boxsize=100.0, boxcentre=[50,50,50], los=None)
    
    rad_versor = mesh.radial_versor
    assert rad_versor.shape == (4, 4, 4, 3)
    
    # Calculate magnitudes across the grid
    magnitudes = np.linalg.norm(rad_versor, axis=-1)
    
    # The origin (if present in the grid) will have magnitude 0 due to the mask.
    # All other points MUST have magnitude 1.
    mask = magnitudes > 0
    assert np.allclose(magnitudes[mask], 1.0), "Radial versors must be unit vectors"

def test_mesh_radial_versor_k():
    """Test the k-space radial versor."""
    mesh = Mesh(nmesh=4, boxsize=100.0, boxcentre=[50,50,50])
    
    rad_versor_k = mesh.radial_versor_k
    assert rad_versor_k.shape == (4, 4, 3, 3) # Note the Z dimension is 3 (N//2 + 1)
    
    magnitudes = np.linalg.norm(rad_versor_k, axis=-1)
    mask = magnitudes > 0
    assert np.allclose(magnitudes[mask], 1.0)


# ==========================================
# 5. METHOD INTEGRATION TESTS
# ==========================================

@patch('zeldareco.mesh.mesh.project_vector_field_jit')
def test_get_parallel_component(mock_project_jit, basic_mesh):
    """
    Test that get_parallel_component correctly wraps the JIT projection function.
    """
    # Create a dummy vector field
    dummy_field = np.ones((16, 16, 16, 3), dtype=np.float32)
    
    # Force the mesh to use a Z-axis LOS
    basic_mesh.los = 'z'
    
    # Call the method
    basic_mesh.get_parallel_component(dummy_field)
    
    # Verify the JIT function was called
    mock_project_jit.assert_called_once()
    
    # Extract the arguments passed to the mocked JIT function
    args, kwargs = mock_project_jit.call_args
    passed_field = args[0]
    passed_versor = args[1]
    
    np.testing.assert_array_equal(passed_field, dummy_field)
    
    # The versor passed down should be the Z-axis versor
    expected_versor = np.zeros((16, 16, 16, 3), dtype=np.float32)
    expected_versor[..., 2] = 1.0
    np.testing.assert_array_equal(passed_versor, expected_versor)