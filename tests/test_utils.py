"""Tests for the misc numerical helpers in baorecon.utils.utils.

Nothing in baorecon itself calls these (they're standalone public helpers), so
they had no coverage at all; the tests below just check each function against
its documented formula / inverse relationship.
"""

import numpy as np
import pytest

from baorecon.utils.utils import (
    delta_to_rho,
    effective_smoothing_radius,
    euclidean_distance_periodic_mesh,
    gaussian_smoothing_radius_from_effective,
    periodic_distance,
    periodic_distance_mesh,
    rho_to_delta,
    split_box_into_eight,
)


# ==========================================
# DENSITY <-> OVERDENSITY
# ==========================================
def test_rho_to_delta_known_values():
    rho = np.array([0.5, 1.0, 1.5, 3.0])
    delta = rho_to_delta(rho, mean_density=1.0)
    assert np.allclose(delta, [-0.5, 0.0, 0.5, 2.0])


def test_rho_delta_roundtrip():
    rng = np.random.default_rng(0)
    rho = rng.uniform(0.1, 10.0, size=50)
    mean_density = 2.3
    delta = rho_to_delta(rho, mean_density)
    assert np.allclose(delta_to_rho(delta, mean_density), rho)


# ==========================================
# SMOOTHING RADIUS CONVERSION
# ==========================================
def test_smoothing_radius_roundtrip():
    cell_size = 5.0
    r_eff = 15.0
    r_config = gaussian_smoothing_radius_from_effective(cell_size, r_eff)
    assert r_config > r_eff  # configuration-space radius includes the pixel-window term
    assert np.isclose(effective_smoothing_radius(cell_size, r_config), r_eff)


def test_effective_smoothing_radius_zero_cell_is_identity():
    # No pixel-window correction when cell_size=0: effective == configuration radius.
    assert np.isclose(effective_smoothing_radius(0.0, 15.0), 15.0)
    assert np.isclose(gaussian_smoothing_radius_from_effective(0.0, 15.0), 15.0)


# ==========================================
# BOX SPLITTING
# ==========================================
def test_split_box_into_eight_reassembles():
    N = 8
    grid = np.arange(N**3).reshape(N, N, N)
    octants = split_box_into_eight(grid)
    assert len(octants) == 8

    edge = N // 2
    rebuilt = np.empty_like(grid)
    idx = 0
    for i in (0, edge):
        for j in (0, edge):
            for k in (0, edge):
                octant = octants[idx]
                assert octant.shape == (edge, edge, edge)
                rebuilt[i:i + edge, j:j + edge, k:k + edge] = octant
                idx += 1
    assert np.array_equal(rebuilt, grid)


# ==========================================
# PERIODIC DISTANCE (continuous positions)
# ==========================================
def test_periodic_distance_no_wrap():
    pos1 = np.array([[1.0, 1.0, 1.0]])
    pos2 = np.array([[2.0, 1.0, 1.0]])
    d = periodic_distance(pos1, pos2, boxsize=10.0)
    assert np.allclose(d, 1.0)


def test_periodic_distance_wraps_around():
    # Points near opposite box edges are 1 apart through the wrap, not 9.
    boxsize = 10.0
    pos1 = np.array([[0.5, 0.0, 0.0]])
    pos2 = np.array([[9.5, 0.0, 0.0]])
    d = periodic_distance(pos1, pos2, boxsize)
    assert np.isclose(d[0], 1.0)


# ==========================================
# PERIODIC DISTANCE (grid indices)
# ==========================================
def test_periodic_distance_mesh():
    size = 10
    assert periodic_distance_mesh(1, 3, size) == 2
    assert periodic_distance_mesh(0, 9, size) == 1  # wraps: 0 and 9 are neighbours
    assert periodic_distance_mesh(2, 2, size) == 0


def test_euclidean_distance_periodic_mesh_matches_periodic_distance():
    """Cross-check against periodic_distance on the equivalent physical positions."""
    size, cell_size = 10, 2.0
    boxsize = size * cell_size
    i1, j1, k1 = 1, 2, 3
    i2, j2, k2 = 9, 1, 7  # wraps in x and z, not y

    d_mesh = euclidean_distance_periodic_mesh(i1, j1, k1, i2, j2, k2, size, cell_size)

    pos1 = np.array([[i1, j1, k1]], dtype=np.float64) * cell_size
    pos2 = np.array([[i2, j2, k2]], dtype=np.float64) * cell_size
    d_ref = periodic_distance(pos1, pos2, boxsize)[0]
    assert np.isclose(d_mesh, d_ref)


# ---------------------------------------------------------------------------
# position_bounds: numpy's axis-0 min/max, in one parallel pass
# ---------------------------------------------------------------------------

def test_position_bounds_matches_numpy_in_every_dtype():
    import numba
    from baorecon.utils.reductions import position_bounds

    rng = np.random.default_rng(9)
    for dtype in (np.float32, np.float64):
        pos = rng.uniform(-50.0, 2500.0, (100_003, 3)).astype(dtype)
        pos[777] = [-1234.5, 9.0, 7.0]                  # extremes off the diagonal
        pos[-1] = [1.0, 9999.25, -3.0]
        low, high = position_bounds(pos)
        assert low.dtype == dtype and high.dtype == dtype
        np.testing.assert_array_equal(low, pos.min(axis=0))
        np.testing.assert_array_equal(high, pos.max(axis=0))
        before = numba.get_num_threads()
        try:
            numba.set_num_threads(1)
            again = position_bounds(pos)
        finally:
            numba.set_num_threads(before)
        np.testing.assert_array_equal(again[0], low)
        np.testing.assert_array_equal(again[1], high)
    # Few rows, fewer than the chunks: still right.
    tiny = np.array([[3.0, 2.0, 1.0], [0.0, 5.0, -1.0]])
    np.testing.assert_array_equal(position_bounds(tiny)[0], [0.0, 2.0, -1.0])
    np.testing.assert_array_equal(position_bounds(tiny)[1], [3.0, 5.0, 1.0])
    with pytest.raises(ValueError, match="empty"):
        position_bounds(np.empty((0, 3)))
    with pytest.raises(ValueError, match="shape"):
        position_bounds(np.ones((4, 2)))
