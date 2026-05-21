import numpy as np
import pytest

from zeldareco.BAOreconstruction.density_manager import DensityManager
from zeldareco.mass_assignment import mass_assignment, interpolate_grid_to_particles


def test_density_manager_basic():
    # small mock catalogue
    np.random.seed(0)
    data = np.random.rand(10, 3) * 100.0
    rand = np.random.rand(20, 3) * 100.0
    dm = DensityManager(data, rand, nmesh=8, boxsize=100.0)
    delta = dm.compute_delta()
    assert delta.shape == (8, 8, 8)
    assert np.isfinite(delta).all()


def test_density_manager_infers_boxcentre_from_positions():
    data = np.array([[10.0, 20.0, 30.0], [30.0, 40.0, 50.0]])
    rand = np.array([[12.0, 22.0, 32.0], [28.0, 38.0, 48.0]])

    dm = DensityManager(data, rand, nmesh=8, boxsize=None, boxcentre=None, padding=1.0, pbc=False)

    combined = np.concatenate((data, rand), axis=0)
    expected_center = 0.5 * (combined.min(axis=0) + combined.max(axis=0))

    assert np.allclose(dm.boxcentre, expected_center)
    assert dm.boxsize >= np.max(combined.max(axis=0) - combined.min(axis=0))



def _random_catalog(npart=100, boxsize=100.0, seed=0):
    rng = np.random.RandomState(seed)
    pos = rng.rand(npart, 3) * boxsize
    weights = rng.rand(npart)
    return pos, weights, boxsize


def test_mass_conservation_all_methods():
    pos, weights, boxsize = _random_catalog(npart=300, boxsize=100.0)
    nmesh = 16
    for method in ("NGP", "CIC", "TSC"):
        field = mass_assignment(pos, boxsize, nmesh, weights=weights, method=method, pbc=True, parallel=False)
        assert field.shape == (nmesh, nmesh, nmesh)
        assert np.isfinite(field).all()
        # total assigned mass must equal sum of input weights
        assert np.isclose(field.sum(), weights.sum(), rtol=1e-12, atol=1e-12)


def test_interpolate_grid_to_particles_identity():
    pos, _, boxsize = _random_catalog(npart=50, boxsize=50.0)
    grid = np.full((8, 8, 8), 2.5, dtype=np.float64)
    for method in ("NGP", "CIC", "TSC"):
        vals = interpolate_grid_to_particles(pos, grid, boxsize, method=method)
        assert vals.shape[0] == pos.shape[0]
        assert np.isfinite(vals).all()


def test_pbc_wrapping_equivalence():
    # verify that assigning a particle outside the box with pbc=True
    # is equivalent to its wrapped position
    pos, weights, boxsize = _random_catalog(npart=200, boxsize=100.0)
    nmesh = 16
    # pick a particle and move it by +boxsize along x
    pos_wrapped = pos.copy()
    pos_out = pos.copy()
    pos_out[0, 0] = pos_out[0, 0] + boxsize

    for method in ("CIC", "TSC"):
        f_orig = mass_assignment(pos_wrapped, boxsize, nmesh, weights=weights, method=method, pbc=True)
        f_out = mass_assignment(pos_out, boxsize, nmesh, weights=weights, method=method, pbc=True)
        assert np.allclose(f_orig, f_out)


def test_density_manager_vs_pyrecon():
    import numpy as np; np.product = np.prod; np.asfarray = np.asarray
    pyrecon = pytest.importorskip("pyrecon")
    from pyrecon.multigrid import MultiGridReconstruction

    rng = np.random.RandomState(1)
    data = rng.rand(300, 3) * 100.0
    rand = rng.rand(900, 3) * 100.0
    nmesh = 32
    box = 100.0
    rand_th = 0.01
    BIAS = 2.0

    dm = DensityManager(data, rand, nmesh=nmesh, boxsize=box, boxcentre=[box/2]*3, smoothing_radius=15.0, pbc=True, padding=0.0)
    delta_z = dm.compute_delta(threshold_randoms=rand_th, sm_mode="wrap")

    py = MultiGridReconstruction(f=0.0, bias=BIAS, los=None, boxsize=box, boxcenter=[box/2]*3, nmesh=nmesh, wrap=True, resampler='cic', ran_min=rand_th)
    py.assign_data(data)
    py.assign_randoms(rand)
    py.set_density_contrast(smoothing_radius=15.0, ran_min=rand_th)
    delta_py = BIAS*np.asarray(py.mesh_delta)

    assert delta_z.shape == delta_py.shape
    mask = (delta_z != 0)
    assert mask.sum() > 0
    corr = np.corrcoef(delta_z[mask].ravel(), delta_py[mask].ravel())[0, 1]
    assert corr > 0.99
    # Verifichiamo che la differenza relativa media sia piccola
    diff = np.abs(delta_z[mask] - delta_py[mask])
    rel_error = np.median(diff / (np.abs(delta_py[mask]) + 1e-6))
    assert rel_error < 1e-3  # Errore relativo mediano inferiore all'1%