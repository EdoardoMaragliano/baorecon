"""
Tests for rectangular-box support.

``boxsize`` can be a per-axis array of shape (3,) instead of a single scalar.
These tests check (1) that a scalar box and the equivalent cubic array produce
identical results everywhere, and (2) that genuinely rectangular boxes behave
correctly (mass conservation, finite displacements).
"""

import numpy as np
import pytest

from baorecon.mesh.mesh import Mesh
from baorecon.mesh.los import FixedAxisLOS, project_vector_field_jit
from baorecon.mas import assign
from baorecon.field_ops import interpolate_vector_field
from baorecon.solvers.fft import FFTSolverCPU
from baorecon.utils.formatters import format_boxsize, set_boxsize_from_positions


def _mesh(boxsize, nmesh, boxcentre=None):
    box = np.broadcast_to(np.asarray(boxsize, dtype=np.float64), (3,))
    if boxcentre is None:
        boxcentre = box / 2.0
    return Mesh(nmesh=nmesh, boxsize=boxsize, boxcentre=boxcentre)


def _random_catalog(npart=300, boxsize=(800.0, 1000.0, 1200.0), seed=0):
    rng = np.random.RandomState(seed)
    box = np.broadcast_to(np.asarray(boxsize, dtype=np.float64), (3,))
    pos = rng.rand(npart, 3) * box
    weights = rng.rand(npart).astype(np.float32)
    return pos.astype(np.float32), weights


# ==========================================
# 1. SCALAR vs CUBIC-ARRAY EQUIVALENCE
# ==========================================
def test_format_boxsize_scalar_equals_array():
    pos = np.zeros((1, 3), dtype=np.float64)
    from_scalar = format_boxsize(1000.0, positions=pos, pbc=True)
    from_array = format_boxsize([1000.0, 1000.0, 1000.0], positions=pos, pbc=True)
    assert from_scalar.shape == (3,)
    assert from_array.shape == (3,)
    np.testing.assert_array_equal(from_scalar, from_array)
    np.testing.assert_array_equal(from_scalar, [1000.0, 1000.0, 1000.0])


def test_format_boxsize_rejects_bad_shape():
    pos = np.zeros((1, 3), dtype=np.float64)
    with pytest.raises(ValueError):
        format_boxsize([1000.0, 1000.0], positions=pos, pbc=True)


def test_set_boxsize_from_positions_is_per_axis():
    pos = np.array([[0.0, 0.0, 0.0], [10.0, 20.0, 30.0]], dtype=np.float64)
    box = set_boxsize_from_positions(pos, padding=1.0)
    assert box.shape == (3,)
    np.testing.assert_allclose(box, [11.0, 21.0, 31.0])


def test_mesh_scalar_equals_cubic_array():
    mesh_scalar = _mesh(1000.0, 16, [500.0, 500.0, 500.0])
    mesh_array = _mesh([1000.0, 1000.0, 1000.0], 16, [500.0, 500.0, 500.0])
    np.testing.assert_array_equal(mesh_scalar.boxsize, mesh_array.boxsize)
    np.testing.assert_array_equal(mesh_scalar.cell_size, mesh_array.cell_size)
    np.testing.assert_array_equal(mesh_scalar.min_corner, mesh_array.min_corner)
    assert mesh_scalar.shape == mesh_array.shape


@pytest.mark.parametrize("method", ["NGP", "CIC", "TSC"])
def test_mass_assignment_scalar_equals_cubic_array(method):
    pos, weights = _random_catalog(npart=400, boxsize=(1000.0, 1000.0, 1000.0), seed=1)
    L = 1000.0
    grid_scalar = assign(pos, weights, _mesh(L, 16), scheme=method, pbc=True)
    grid_array = assign(pos, weights, _mesh([L, L, L], 16), scheme=method, pbc=True)
    np.testing.assert_array_equal(grid_scalar, grid_array)


@pytest.mark.parametrize("mas", ["CIC", "TSC"])
def test_interpolation_scalar_equals_cubic_array(mas):
    nmesh = 8
    L = 100.0
    rng = np.random.RandomState(7)
    field = rng.normal(size=(nmesh, nmesh, nmesh, 3)).astype(np.float64)
    pos = rng.uniform(0, L, size=(120, 3))
    interp_scalar = interpolate_vector_field(pos, field, L, MAS=mas, pbc=True, dtype=np.float64)
    interp_array = interpolate_vector_field(pos, field, [L, L, L], MAS=mas, pbc=True, dtype=np.float64)
    np.testing.assert_array_equal(interp_scalar, interp_array)


# ==========================================
# 2. MASS CONSERVATION ON A RECTANGULAR BOX
# ==========================================
@pytest.mark.parametrize("method", ["NGP", "CIC", "TSC"])
def test_mass_conservation_rectangular_box(method):
    boxsize = [800.0, 1000.0, 1200.0]
    pos, weights = _random_catalog(npart=500, boxsize=boxsize, seed=2)
    grid = assign(pos, weights, _mesh(boxsize, 16), scheme=method, pbc=True)
    assert grid.shape == (16, 16, 16)
    assert np.isfinite(grid).all()
    assert np.isclose(grid.sum(), weights.sum(), rtol=1e-5, atol=1e-4)


@pytest.mark.parametrize("method", ["CIC", "TSC"])
def test_mass_conservation_rectangular_parallel(method):
    boxsize = [800.0, 1000.0, 1200.0]
    pos, weights = _random_catalog(npart=500, boxsize=boxsize, seed=3)
    grid = assign(pos, weights, _mesh(boxsize, 16), scheme=method, pbc=True, parallel=True)
    assert np.isclose(grid.sum(), weights.sum(), rtol=1e-5, atol=1e-4)


def test_rectangular_box_differs_from_cubic():
    boxsize_rect = [800.0, 1000.0, 1200.0]
    pos, weights = _random_catalog(npart=400, boxsize=boxsize_rect, seed=4)
    grid_rect = assign(pos, weights, _mesh(boxsize_rect, 16), scheme="CIC", pbc=True)
    grid_cube = assign(pos, weights, _mesh(1200.0, 16), scheme="CIC", pbc=True)
    assert np.isclose(grid_rect.sum(), weights.sum(), rtol=1e-5, atol=1e-4)
    assert np.isclose(grid_cube.sum(), weights.sum(), rtol=1e-5, atol=1e-4)
    assert not np.allclose(grid_rect, grid_cube)


# ==========================================
# 3. DISPLACEMENT FIELD EQUIVALENCE
# ==========================================
def test_displacement_scalar_equals_cubic_array():
    N = 32
    L = 1000.0
    rng = np.random.RandomState(42)
    delta = rng.normal(0, 0.1, size=(N, N, N))
    delta -= delta.mean()

    los = FixedAxisLOS(2)
    psi_scalar = FFTSolverCPU(delta, _mesh(L, N), los=los, f=0.0, bias=1.0, RSDspace="RealSpace").displacement
    psi_array = FFTSolverCPU(delta, _mesh([L, L, L], N), los=los, f=0.0, bias=1.0, RSDspace="RealSpace").displacement
    np.testing.assert_array_equal(psi_scalar, psi_array)


def test_displacement_rectangular_box_runs():
    N = 32
    boxsize = [800.0, 1000.0, 1200.0]
    rng = np.random.RandomState(11)
    delta = rng.normal(0, 0.1, size=(N, N, N))
    delta -= delta.mean()

    mesh = _mesh(boxsize, N)
    psi = FFTSolverCPU(delta, mesh, los=FixedAxisLOS(2), f=0.0, bias=1.0, RSDspace="RealSpace").displacement
    assert psi.shape == (N, N, N, 3)
    assert np.isfinite(psi).all()
    np.testing.assert_allclose(psi.mean(axis=(0, 1, 2)), [0.0, 0.0, 0.0], atol=1e-10)


# ==========================================
# 4. PROJECTION KERNEL + REDSHIFT-SPACE ON A RECTANGULAR MESH
# ==========================================
def test_project_vector_field_jit():
    vf = np.ones((4, 4, 4, 3), dtype=np.float32)
    los = np.zeros((4, 4, 4, 3), dtype=np.float32)
    los[..., 2] = 1.0
    out = project_vector_field_jit(vf, los, np.zeros_like(vf))
    assert out.shape == (4, 4, 4, 3)
    np.testing.assert_allclose(out[..., 2], 1.0)
    np.testing.assert_allclose(out[..., 0], 0.0)


@pytest.mark.parametrize("boxsize", [1000.0, [800.0, 1000.0, 1200.0]])
def test_redshift_space_displacement_on_real_mesh(boxsize):
    """The RSD iterative solver runs on a real (possibly rectangular) Mesh."""
    N = 16
    box = np.broadcast_to(np.asarray(boxsize, dtype=np.float64), (3,))
    rng = np.random.RandomState(0)
    delta = rng.normal(0, 0.1, size=(N, N, N))
    delta -= delta.mean()

    mesh = Mesh(nmesh=N, boxsize=boxsize, boxcentre=box / 2)
    psi = FFTSolverCPU(delta, mesh, los=FixedAxisLOS(2), f=0.8, bias=2.0,
                       RSDspace="RedshiftSpace").displacement
    assert psi.shape == (N, N, N, 3)
    assert np.isfinite(psi).all()


# ==================================================
# 4. PER-AXIS NMESH (NOT just per-axis boxsize)
# ==================================================
# The tests above vary `boxsize` while keeping `nmesh` cubic, which is the one
# rectangular case that cannot catch a read-out assuming a single mesh size: with
# a cubic `nmesh`, `field.shape[0]` happens to be right for all three axes. A
# per-axis `nmesh` — what `cellsize` produces — is the case that bites.

def _sine_potential(nmesh, boxsize):
    """phi = sin(2pi x/Lx) sin(2pi y/Ly) sin(2pi z/Lz) sampled on the grid nodes."""
    nmesh = np.asarray(nmesh, dtype=np.int64)
    boxsize = np.asarray(boxsize, dtype=np.float64)
    k = 2.0 * np.pi / boxsize
    axes = [np.arange(n) * (L / n) for n, L in zip(nmesh, boxsize)]
    xx, yy, zz = np.meshgrid(*axes, indexing="ij")
    return np.sin(k[0] * xx) * np.sin(k[1] * yy) * np.sin(k[2] * zz), (xx, yy, zz), k


def _analytic_minus_grad(pos, boxsize):
    """-grad phi at arbitrary positions, for the same phi."""
    boxsize = np.asarray(boxsize, dtype=np.float64)
    k = 2.0 * np.pi / boxsize
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    return -np.stack([
        k[0] * np.cos(k[0] * x) * np.sin(k[1] * y) * np.sin(k[2] * z),
        k[1] * np.sin(k[0] * x) * np.cos(k[1] * y) * np.sin(k[2] * z),
        k[2] * np.sin(k[0] * x) * np.sin(k[1] * y) * np.cos(k[2] * z),
    ], axis=1)


def _cosine_similarity(a, b):
    return float(np.sum(a * b) / np.sqrt(np.sum(a * a) * np.sum(b * b)))


# Cubic first as the control, then grids whose axes genuinely differ. The last
# two are the shapes a survey run produces through `cellsize`.
PER_AXIS_GRIDS = [
    ((32, 32, 32), (64.0, 64.0, 64.0)),
    ((32, 64, 128), (64.0, 128.0, 256.0)),
    ((56, 80, 80), (448.0, 640.0, 640.0)),
    ((80, 80, 24), (640.0, 640.0, 192.0)),
]


@pytest.mark.parametrize("nmesh, boxsize", PER_AXIS_GRIDS)
def test_vector_interpolation_honours_per_axis_nmesh(nmesh, boxsize):
    """Interpolating an exact vector field must reproduce it, on any grid shape.

    Feeds the analytic field itself onto the nodes, so nothing but the
    interpolation is under test. A read-out that takes one mesh size from axis 0
    mis-sizes the cells on the other two and indexes the wrong nodes; on
    ``(80, 80, 24)`` it also clamps the third index to 79 on an axis of length
    24, reading out of bounds.
    """
    nmesh = np.asarray(nmesh, dtype=np.int64)
    boxsize = np.asarray(boxsize, dtype=np.float64)
    _, (xx, yy, zz), k = _sine_potential(nmesh, boxsize)

    field = -np.stack([
        k[0] * np.cos(k[0] * xx) * np.sin(k[1] * yy) * np.sin(k[2] * zz),
        k[1] * np.sin(k[0] * xx) * np.cos(k[1] * yy) * np.sin(k[2] * zz),
        k[2] * np.sin(k[0] * xx) * np.sin(k[1] * yy) * np.cos(k[2] * zz),
    ], axis=-1).astype(np.float64)

    rng = np.random.RandomState(11)
    pos = rng.uniform(0, 1, size=(5000, 3)) * boxsize
    expected = _analytic_minus_grad(pos, boxsize)

    got = interpolate_vector_field(pos, field, boxsize, MAS="CIC", pbc=True, dtype=np.float64)

    assert np.all(np.isfinite(got))
    assert _cosine_similarity(got, expected) > 0.99, (
        f"interpolated field decorrelates from the exact one on nmesh={tuple(nmesh)}: "
        "the read-out is assuming a single mesh size for all three axes"
    )
    # Trilinear interpolation of a sine is slightly low; a few percent is expected,
    # a wrong cell size is not.
    ratio = np.median(np.linalg.norm(got, axis=1)) / np.median(np.linalg.norm(expected, axis=1))
    assert 0.9 < ratio < 1.1, f"amplitude off by {ratio:.3f} on nmesh={tuple(nmesh)}"


@pytest.mark.parametrize("nmesh, boxsize", PER_AXIS_GRIDS)
def test_fft_and_multigrid_readout_agree_on_per_axis_nmesh(nmesh, boxsize):
    """The two solvers must read the same displacement at the same tracers.

    ``test_solver_equivalence`` compares the displacement *grids* on cubic
    meshes; this compares what the pipeline actually consumes — the per-tracer
    read-out — on grids whose axes differ. The two paths are independent
    implementations (the FFT solver interpolates its displacement grid, the
    multigrid differentiates the potential on the fly), so agreement here is a
    real check rather than a tautology.
    """
    from baorecon.solvers.fft import FFTSolverCPU
    from baorecon.solvers.multigrid import MultigridSolver
    from baorecon.mesh.los import FixedAxisLOS

    nmesh = np.asarray(nmesh, dtype=np.int64)
    boxsize = np.asarray(boxsize, dtype=np.float64)

    # A resolved overdensity: a bump several cells across, mean-subtracted.
    axes = [np.arange(n) * (L / n) for n, L in zip(nmesh, boxsize)]
    xx, yy, zz = np.meshgrid(*axes, indexing="ij")
    centre = boxsize / 2.0
    sigma = boxsize.min() / 12.0
    delta = np.exp(-(((xx - centre[0]) ** 2 + (yy - centre[1]) ** 2 + (zz - centre[2]) ** 2)
                     / (2.0 * sigma ** 2))).astype(np.float32)
    delta -= delta.mean()

    mesh = _mesh(boxsize, nmesh, boxcentre=centre)
    los = FixedAxisLOS(2)
    fft = FFTSolverCPU(delta, mesh, los=los, f=0.0, bias=1.0, RSDspace="RealSpace")
    mg = MultigridSolver(delta, mesh, los=los, f=0.0, bias=1.0, RSDspace="RealSpace",
                         use_plane_parallel=True)

    rng = np.random.RandomState(3)
    pos = (rng.uniform(0.2, 0.8, size=(4000, 3)) * boxsize).astype(np.float32)

    d_fft = fft.read_displacement_at(pos, mas="CIC")
    d_mg = mg.read_displacement_at(pos, mas="CIC")

    assert np.all(np.isfinite(d_fft)) and np.all(np.isfinite(d_mg))
    assert np.abs(d_fft).max() > 1e-6, "FFT read-out is ~zero; the comparison would be vacuous"
    assert np.abs(d_mg).max() > 1e-6, "multigrid read-out is ~zero; the comparison would be vacuous"
    assert _cosine_similarity(d_fft, d_mg) > 0.99, (
        f"FFT and multigrid read-outs disagree at the tracers on nmesh={tuple(nmesh)}"
    )
