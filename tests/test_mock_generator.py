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
