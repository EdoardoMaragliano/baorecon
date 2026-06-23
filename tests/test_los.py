import pytest
import numpy as np

from baorecon.mesh.los import FixedAxisLOS, LocalLOS, project_vector_field_jit


# ==========================================
# FIXED-AXIS LOS
# ==========================================
@pytest.mark.parametrize("axis, expected", [
    (0, [1.0, 0.0, 0.0]),
    (1, [0.0, 1.0, 0.0]),
    (2, [0.0, 0.0, 1.0]),
])
def test_fixed_axis_direction(axis, expected):
    los = FixedAxisLOS(axis)
    np.testing.assert_array_equal(los.direction, expected)


@pytest.mark.parametrize("axis, mag", [(0, 2.0), (1, 3.0), (2, 4.0)])
def test_fixed_axis_project_parallel(axis, mag):
    """Projecting a tilted field keeps only the LOS-axis component."""
    N = 4
    field = np.zeros((N, N, N, 3), dtype=np.float32)
    field[..., 0] = 2.0
    field[..., 1] = 3.0
    field[..., 2] = 4.0

    out = FixedAxisLOS(axis).project_parallel(field)
    assert np.allclose(out[..., axis], mag)
    for i in range(3):
        if i != axis:
            assert np.allclose(out[..., i], 0.0)


def test_fixed_axis_project_in_place():
    """project_parallel must work when out aliases the input."""
    N = 4
    field = np.ones((N, N, N, 3), dtype=np.float32)
    out = FixedAxisLOS(2).project_parallel(field, out=field)
    assert out is field
    assert np.allclose(out[..., 2], 1.0)
    assert np.allclose(out[..., 0], 0.0)
    assert np.allclose(out[..., 1], 0.0)


def test_fixed_axis_rejects_bad_axis():
    with pytest.raises(ValueError):
        FixedAxisLOS(3)


# ==========================================
# LOCAL (RADIAL) LOS
# ==========================================
def test_local_los_radial_versor_is_unit():
    """The cached radial versor field is made of unit vectors (except the origin)."""
    los = LocalLOS(boxcentre=[50, 50, 50], min_corner=[0, 0, 0],
                   boxsize=100.0, nmesh=4)
    versor = los.radial_versor
    assert versor.shape == (4, 4, 4, 3)
    mag = np.linalg.norm(versor, axis=-1)
    mask = mag > 0
    assert np.allclose(mag[mask], 1.0)


def test_local_los_project_parallel():
    """The parallel component lies along the radial direction at every point."""
    los = LocalLOS(boxcentre=[50, 50, 50], min_corner=[0, 0, 0],
                   boxsize=100.0, nmesh=4)
    field = np.random.RandomState(0).normal(size=(4, 4, 4, 3)).astype(np.float32)
    out = los.project_parallel(field)
    versor = los.radial_versor
    # out should be parallel to the versor: cross product ~ 0 where versor != 0.
    cross = np.cross(out, versor)
    assert np.allclose(cross, 0.0, atol=1e-5)


# ==========================================
# project_vector_field_jit kernel
# ==========================================
def test_project_vector_field_jit():
    """The JIT projection keeps only the component along the (z) versor."""
    N = 4
    vf = np.ones((N, N, N, 3), dtype=np.float32)
    los = np.zeros((N, N, N, 3), dtype=np.float32)
    los[..., 2] = 1.0

    out = project_vector_field_jit(vf, los, np.empty_like(vf))
    np.testing.assert_allclose(out[..., 2], 1.0)
    np.testing.assert_allclose(out[..., 0], 0.0)
    np.testing.assert_allclose(out[..., 1], 0.0)
