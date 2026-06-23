import pytest
import numpy as np

from baorecon.mesh.mesh import Mesh


@pytest.fixture
def basic_mesh():
    """A standard cubic Mesh instance for reuse in tests."""
    return Mesh(nmesh=16, boxsize=100.0, boxcentre=[50.0, 50.0, 50.0])


# ==========================================
# 1. INITIALIZATION AND VALIDATION
# ==========================================
def test_mesh_initialization(basic_mesh):
    """Valid initialization computes the correct geometry attributes."""
    np.testing.assert_array_equal(basic_mesh.nmesh, [16, 16, 16])
    np.testing.assert_array_equal(basic_mesh.boxsize, [100.0, 100.0, 100.0])
    np.testing.assert_allclose(basic_mesh.cell_size, [100.0 / 16] * 3)
    np.testing.assert_array_equal(basic_mesh.boxcentre, [50.0, 50.0, 50.0])
    np.testing.assert_array_equal(basic_mesh.min_corner, [0.0, 0.0, 0.0])
    assert basic_mesh.shape == (16, 16, 16)


@pytest.mark.parametrize("bad_params", [
    {"nmesh": 0, "boxsize": 100, "boxcentre": [50, 50, 50]},
    {"nmesh": 16, "boxsize": -50, "boxcentre": [50, 50, 50]},
    {"nmesh": 16, "boxsize": 100, "boxcentre": [50, 50]},          # wrong centre shape
    {"nmesh": [16, 16], "boxsize": 100, "boxcentre": [50, 50, 50]},  # wrong nmesh shape
    {"nmesh": 16, "boxsize": [100, 100], "boxcentre": [50, 50, 50]},  # wrong boxsize shape
])
def test_mesh_validation_errors(bad_params):
    with pytest.raises(ValueError):
        Mesh(**bad_params)


# ==========================================
# 2. SCALAR vs PER-AXIS EQUIVALENCE
# ==========================================
def test_mesh_scalar_equals_cubic_array():
    """A scalar box and the equivalent cubic array describe identical geometry."""
    mesh_scalar = Mesh(nmesh=16, boxsize=1000.0, boxcentre=[500.0, 500.0, 500.0])
    mesh_array = Mesh(nmesh=16, boxsize=[1000.0, 1000.0, 1000.0], boxcentre=[500.0, 500.0, 500.0])

    np.testing.assert_array_equal(mesh_scalar.boxsize, mesh_array.boxsize)
    np.testing.assert_array_equal(mesh_scalar.cell_size, mesh_array.cell_size)
    np.testing.assert_array_equal(mesh_scalar.min_corner, mesh_array.min_corner)
    assert mesh_scalar.shape == mesh_array.shape


def test_mesh_rectangular_geometry():
    """A genuinely rectangular box yields per-axis cell sizes."""
    mesh = Mesh(nmesh=16, boxsize=[800.0, 1000.0, 1200.0], boxcentre=[400.0, 500.0, 600.0])
    np.testing.assert_allclose(mesh.cell_size, [800.0 / 16, 1000.0 / 16, 1200.0 / 16])
    np.testing.assert_array_equal(mesh.min_corner, [0.0, 0.0, 0.0])


def test_mesh_is_lightweight(basic_mesh):
    """Mesh holds only geometry, no large arrays."""
    for attr in ("xmesh", "kmesh", "k_components", "radial_versor", "los_versor"):
        assert not hasattr(basic_mesh, attr)
