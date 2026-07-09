import numpy as np
import pytest

from baorecon.reconstruction.density import DensityManager
from baorecon.mesh.mesh import Mesh
from baorecon.mas import assign, readout, CUPY_AVAILABLE

try:
    import cupy as cp
except ImportError:
    cp = None


def _to_host(arr):
    """Bring a possibly-cupy array back to numpy for assertions."""
    if CUPY_AVAILABLE and cp is not None and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return arr


def _mesh(boxsize, nmesh):
    """Build a Mesh for the given box (boxcentre is irrelevant to assignment)."""
    box = np.broadcast_to(np.asarray(boxsize, dtype=np.float64), (3,))
    return Mesh(nmesh=nmesh, boxsize=boxsize, boxcentre=box / 2.0)


gpu_test = pytest.mark.skipif(not CUPY_AVAILABLE, reason="GPU not available or CuPy not installed")


@pytest.mark.parametrize("device", ["cpu", pytest.param("gpu", marks=gpu_test)])
def test_density_manager_basic(device):
    np.random.seed(0)
    data = np.random.rand(10, 3) * 100.0
    rand = np.random.rand(20, 3) * 100.0
    dm = DensityManager(data, rand, nmesh=8, boxsize=100.0, device=device)
    delta = _to_host(dm.compute_delta())
    assert delta.shape == (8, 8, 8)
    assert np.isfinite(delta).all()


def test_density_manager_infers_boxcentre_from_positions():
    # 1. Genera i random
    rand = np.random.uniform(0.0, 50.0, size=(100, 3)).astype(np.float32)
    
    # FORZATURA: Assicuriamoci che il bounding box empirico sia esattamente [0, 50]
    rand[0] = [0.0, 0.0, 0.0]
    rand[1] = [50.0, 50.0, 50.0]

    # 2. Ora i dati da 0.1 a 49.9 saranno SEMPRE dentro il bounding box
    data = np.random.uniform(0.1, 49.9, size=(100, 3)).astype(np.float32)

    dm = DensityManager(data, rand, nmesh=8, boxsize=None, boxcentre=None, padding=1.0, pbc=False)

    expected_center = 0.5 * (rand.min(axis=0) + rand.max(axis=0))

    assert np.allclose(dm.boxcentre, expected_center)
    assert (dm.boxsize >= np.max(rand.max(axis=0) - rand.min(axis=0))).all()


def _random_catalog(npart=100, boxsize=100.0, seed=0):
    rng = np.random.RandomState(seed)
    pos = rng.rand(npart, 3) * boxsize
    weights = rng.rand(npart)
    return pos, weights, boxsize


@pytest.mark.parametrize("device", ["cpu", pytest.param("gpu", marks=gpu_test)])
@pytest.mark.parametrize("pbc", [True, False])
def test_mass_conservation_all_methods(device, pbc):
    """Mass assignment conserves mass for all methods on both CPU and GPU.

    Holds for pbc=True (wrap) and pbc=False (boundary clamp): both fold every
    weight onto a valid cell, so nothing is dropped.
    """
    pos, weights, boxsize = _random_catalog(npart=300, boxsize=100.0)
    nmesh = 16
    mesh = _mesh(boxsize, nmesh)

    methods_to_test = ("NGP", "CIC", "TSC") if device == "cpu" else ("CIC", "TSC")
    for method in methods_to_test:
        field = _to_host(assign(pos, weights, mesh, scheme=method, pbc=pbc, parallel=False, device=device))
        assert field.shape == (nmesh, nmesh, nmesh)
        assert np.isfinite(field).all()
        assert np.isclose(field.sum(), weights.sum(), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("device", ["cpu", pytest.param("gpu", marks=gpu_test)])
@pytest.mark.parametrize("pbc", [True, False])
def test_readout_identity(device, pbc):
    """Grid->particle read-out of a constant field returns the constant.

    For pbc=False this also guards the boundary clamp: dropping out-of-range
    stencil cells would make the interpolation weights near the edge sum to < 1.
    """
    pos, _, boxsize = _random_catalog(npart=50, boxsize=50.0)
    grid = np.full((8, 8, 8), 2.5, dtype=np.float32)
    mesh = _mesh(boxsize, 8)

    methods_to_test = ("NGP", "CIC", "TSC") if device == "cpu" else ("CIC", "TSC")
    for method in methods_to_test:
        vals = _to_host(readout(grid, pos, mesh, scheme=method, device=device, pbc=pbc))
        assert vals.shape[0] == pos.shape[0]
        assert np.isfinite(vals).all()
        assert np.allclose(vals, 2.5)


@pytest.mark.parametrize("device", ["cpu", pytest.param("gpu", marks=gpu_test)])
def test_pbc_wrapping_equivalence(device):
    """pbc=True: assigning a particle moved by +boxsize == its wrapped position."""
    pos, weights, boxsize = _random_catalog(npart=200, boxsize=100.0)
    nmesh = 16
    mesh = _mesh(boxsize, nmesh)
    pos_wrapped = pos.copy()
    pos_out = pos.copy()
    pos_out[0, 0] = pos_out[0, 0] + boxsize

    for method in ("CIC", "TSC"):
        f_orig = _to_host(assign(pos_wrapped, weights, mesh, scheme=method, pbc=True, device=device))
        f_out = _to_host(assign(pos_out, weights, mesh, scheme=method, pbc=True, device=device))
        assert np.allclose(f_orig, f_out)


@pytest.mark.parametrize("device", ["cpu", pytest.param("gpu", marks=gpu_test)])
@pytest.mark.parametrize("method", ["CIC", "TSC"])
def test_pbc_false_clamps_to_boundary(device, method):
    """pbc=False: an out-of-range stencil is clamped onto the near boundary cell
    (mass-conserving), not wrapped to the far side.

    Complements test_pbc_wrapping_equivalence and exercises the clamp path on
    both backends (this is where the CPU serial/parallel kernels and the GPU
    kernels used to diverge).
    """
    boxsize, nmesh = 100.0, 10
    mesh = _mesh(boxsize, nmesh)
    # One particle in the last x-cell (its stencil spills past x=nmesh), interior in y, z.
    pos = np.array([[99.0, 50.0, 50.0]], dtype=np.float64)
    weights = np.array([1.0])

    field = _to_host(assign(pos, weights, mesh, scheme=method, pbc=False, device=device))
    assert np.isclose(field.sum(), 1.0)          # mass conserved (clamped, not dropped)
    assert abs(field[0].sum()) < 1e-6            # nothing wrapped to the far (x=0) face
    assert field[nmesh - 1].sum() > 0.0          # mass sits on the near boundary


def test_density_manager_vs_pyrecon():
    import numpy as np
    np.product = np.prod
    np.asfarray = np.asarray
    pytest.importorskip("pyrecon")
    from pyrecon.multigrid import MultiGridReconstruction

    rng = np.random.RandomState(1)
    data = rng.rand(300, 3) * 100.0
    rand = rng.rand(900, 3) * 100.0
    nmesh = 32
    box = 100.0
    rand_th = 0.01
    BIAS = 2.0

    dm = DensityManager(data, rand, nmesh=nmesh, boxsize=box, boxcentre=[box / 2] * 3,
                        smoothing_radius=15.0, pbc=True, padding=0.0)
    delta_z = dm.compute_delta(sm_mode="wrap")

    py = MultiGridReconstruction(f=0.0, bias=BIAS, los=None, boxsize=box, boxcenter=[box / 2] * 3,
                                 nmesh=nmesh, wrap=True, resampler="cic", ran_min=rand_th)
    py.assign_data(data)
    py.assign_randoms(rand)
    py.set_density_contrast(smoothing_radius=15.0)
    delta_py = BIAS * np.asarray(py.mesh_delta)

    assert delta_z.shape == delta_py.shape
    mask = (delta_z != 0)
    assert mask.sum() > 0
    corr = np.corrcoef(delta_z[mask].ravel(), delta_py[mask].ravel())[0, 1]
    assert corr > 0.99
    diff = np.abs(delta_z[mask] - delta_py[mask])
    rel_error = np.median(diff / (np.abs(delta_py[mask]) + 1e-6))
    assert rel_error < 1e-3
