"""Tests for the mock-catalogue helpers in :mod:`baorecon.utils.mock_generator`.

These guard the field generators against a regression where ``n_cell`` was left
as a float (``side // spacing``) and passed straight to ``np.fft.fftfreq``, which
requires an integer window length and raised ``ValueError: n should be an
integer`` before any field was produced.
"""

import numpy as np
import pytest

from baorecon.utils.mock_generator import (
    generate_gaussian_map,
    generate_lognormal_map,
    poisson_sample_from_map,
)

SIDE = 200.0
SPACING = 12.5           # 200 / 12.5 == 16, an even, exact grid
N_CELL = int(SIDE // SPACING)


def _pk(k):
    """A smooth, finite toy power spectrum."""
    k = np.asarray(k, dtype=float)
    kk = np.where(k > 0, k, 1e-6)
    return np.where(k > 0, 2.5e3 * (kk / 0.05) / (1.0 + (kk / 0.15) ** 3), 0.0)


def test_generate_lognormal_map_runs_and_has_cubic_shape():
    """Regression: float ``n_cell`` used to crash ``fftfreq`` immediately."""
    np.random.seed(0)
    delta_x = generate_lognormal_map(_pk, SIDE, SPACING)

    assert delta_x.shape == (N_CELL, N_CELL, N_CELL)
    # A lognormal density contrast is bounded below by -1 by construction.
    assert np.all(delta_x > -1.0)
    assert np.all(np.isfinite(delta_x))


def test_generate_gaussian_map_shapes():
    np.random.seed(0)
    delta_k, delta_x = generate_gaussian_map(_pk, SIDE, SPACING)

    assert delta_x.shape == (N_CELL, N_CELL, N_CELL)
    assert delta_k.shape == (N_CELL, N_CELL, N_CELL // 2 + 1)
    assert np.all(np.isfinite(delta_x))


def test_poisson_sample_from_lognormal_map():
    """The lognormal field is directly Poisson-sampleable (1 + delta >= 0)."""
    np.random.seed(0)
    delta_x = generate_lognormal_map(_pk, SIDE, SPACING)

    n_objects = 20_000
    points = poisson_sample_from_map(delta_x, SIDE, SPACING, n_objects, seed=1)

    assert points.ndim == 2 and points.shape[1] == 3
    assert 0.5 * n_objects < len(points) < 2.0 * n_objects
    assert points.min() >= 0.0
    assert points.max() <= SIDE


# --------------------------------------------------------------------------- #
# Hermitian symmetry of the generated field (regression for the half-power bug
# on the k_z = 0 and k_z = Nyquist planes)
# --------------------------------------------------------------------------- #

FLAT_SIDE, FLAT_SPACING, FLAT_P0 = 320.0, 5.0, 1000.0      # 64^3 grid, flat spectrum
FLAT_N = int(FLAT_SIDE // FLAT_SPACING)


def _flat_pk(k):
    return np.full_like(np.asarray(k, dtype=float), FLAT_P0)


def _measured_power(delta_x):
    """Power of the real field actually produced, in the <|d_k|^2> = V P convention."""
    return np.abs(np.fft.rfftn(delta_x) * FLAT_SPACING**3) ** 2 / FLAT_SIDE**3


def test_gaussian_map_fourier_coefficients_are_hermitian():
    """delta(-k) == conj(delta(k)) on the self-conjugate planes k_z = 0 and k_z = Nyquist."""
    np.random.seed(3)
    delta_k, _ = generate_gaussian_map(_flat_pk, FLAT_SIDE, FLAT_SPACING)
    ix, iy = np.meshgrid(np.arange(FLAT_N), np.arange(FLAT_N), indexing="ij")
    for iz in (0, FLAT_N // 2):
        plane = delta_k[:, :, iz]
        mirrored = np.conj(plane[(-ix) % FLAT_N, (-iy) % FLAT_N])
        assert np.allclose(plane, mirrored, rtol=0, atol=1e-6 * np.abs(plane).max())
    assert delta_k[0, 0, 0] == 0


def test_gaussian_map_has_full_power_on_every_plane():
    """With a flat spectrum every region of k-space must carry the same power.
    The amplitude-and-phase drawing gave 0.5 on the two self-conjugate planes."""
    np.random.seed(4)
    _, delta_x = generate_gaussian_map(_flat_pk, FLAT_SIDE, FLAT_SPACING)
    P = _measured_power(delta_x) / FLAT_P0
    generic = P[:, :, 1:-1].mean()
    assert abs(generic - 1.0) < 0.03
    assert abs(P[:, :, 0].mean() - 1.0) < 0.08          # k_z = 0 plane (64^2 modes, ~2% noise)
    assert abs(P[:, :, -1].mean() - 1.0) < 0.08         # k_z = Nyquist plane


def test_gaussian_map_moduli_are_rayleigh_distributed():
    """A complex Gaussian mode has <|d|>^2 / <|d|^2> = pi/4; a half-normal amplitude gives 2/pi."""
    np.random.seed(5)
    delta_k, _ = generate_gaussian_map(_flat_pk, FLAT_SIDE, FLAT_SPACING)
    generic = np.abs(delta_k[:, :, 1:-1]).ravel()
    ratio = generic.mean() ** 2 / (generic**2).mean()
    assert abs(ratio - np.pi / 4) < 0.01


def test_lognormal_map_is_statistically_isotropic():
    """The correlation of the lognormal field along the three axes must agree:
    the half-power bug suppressed xi along z (the rfftn axis) only."""
    np.random.seed(6)
    delta_x = generate_lognormal_map(_pk, 400.0, 5.0)
    n = delta_x.shape[0]
    xi3 = np.fft.irfftn(np.abs(np.fft.rfftn(delta_x)) ** 2, s=delta_x.shape) / n**3
    xi0 = xi3[0, 0, 0]
    for lag in (1, 2, 3):
        axes = np.array([xi3[lag, 0, 0], xi3[0, lag, 0], xi3[0, 0, lag]])
        # the three axes must agree to within a small fraction of the variance
        # (single-realisation scatter); the old generator suppressed z by a constant
        assert axes.max() - axes.min() < 0.1 * xi0, (lag, axes / xi0)


@pytest.mark.parametrize("generate", [generate_gaussian_map, generate_lognormal_map])
def test_seed_makes_the_field_reproducible_and_independent_of_the_global_state(generate):
    def field(**kwargs):
        out = generate(_pk, SIDE, SPACING, **kwargs)
        return out[-1] if isinstance(out, tuple) else out

    np.random.seed(0)
    a = field(seed=123)
    np.random.seed(1)  # a different global state must not matter when a seed is given
    b = field(seed=123)
    c = field(seed=124)
    np.testing.assert_array_equal(a, b)
    assert not np.allclose(a, c)

    # seed=None keeps using the global generator, so np.random.seed still controls it.
    np.random.seed(7)
    d = field()
    np.random.seed(7)
    e = field()
    np.testing.assert_array_equal(d, e)
