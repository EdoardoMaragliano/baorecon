"""Mass assignment and read-out: every scheme, and the survey regime.

``assign`` and ``readout`` sat at 61% coverage, with the untested lines in input
validation and scheme/device dispatch. More to the point, the suite leaned on
``CIC`` with ``pbc=True`` — while a survey run is ``pbc=False``, and ``TSC`` is a
supported alternative that almost nothing exercised. Both are covered here as
first-class cases rather than as parameters tacked onto a CIC test.

The non-periodic path is the one worth being sure about: without wrapping, a
stencil that hangs off the edge is clamped to the boundary cell, and the module
documents that as mass-conserving. That claim is asserted below.
"""

import numpy as np
import pytest

from baorecon.mas import assign, readout
from baorecon.mesh.mesh import Mesh

SCHEMES = ["NGP", "CIC", "TSC"]
SMOOTH_SCHEMES = ["CIC", "TSC"]          # NGP is piecewise constant
BOXSIZE = 100.0
NMESH = 16


def _mesh(nmesh=NMESH, boxsize=BOXSIZE, dtype=np.float64):
    return Mesh(nmesh=nmesh, boxsize=boxsize,
                boxcentre=np.full(3, boxsize / 2.0), dtype=dtype)


def _particles(n=500, seed=0, margin=0.0, boxsize=BOXSIZE):
    """Positions inside the box, optionally kept ``margin`` away from the faces."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(margin, boxsize - margin, size=(n, 3))
    return pos, np.ones(n)


# ---------------------------------------------------------------------------
# mass conservation, periodic and not
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.parametrize("pbc", [True, False])
@pytest.mark.parametrize("parallel", [False, True])
def test_assign_conserves_mass(scheme, pbc, parallel):
    """Painting must move all the weight onto the grid and none off it.

    With ``pbc=False`` the stencil is clamped at the faces rather than wrapped;
    the clamp is documented as mass-conserving, which is what makes the total
    still match.
    """
    if scheme == "NGP" and parallel:
        pytest.skip("NGP has no chunked variant")
    pos, w = _particles()
    grid = assign(pos, w, _mesh(), scheme=scheme, pbc=pbc, parallel=parallel)

    assert grid.shape == (NMESH, NMESH, NMESH)
    assert np.all(np.isfinite(grid))
    assert np.isclose(grid.sum(), w.sum(), rtol=1e-6), (
        f"{scheme} lost or invented mass with pbc={pbc}")


@pytest.mark.parametrize("scheme", SMOOTH_SCHEMES)
def test_assign_conserves_mass_for_particles_on_the_faces(scheme):
    """The clamp only matters for stencils hanging off the edge; put them there."""
    pos = np.array([
        [0.0, 0.0, 0.0], [BOXSIZE, BOXSIZE, BOXSIZE],
        [0.0, BOXSIZE / 2, BOXSIZE / 2], [BOXSIZE, BOXSIZE / 2, BOXSIZE / 2],
        [BOXSIZE / 2, 0.0, BOXSIZE], [1e-6, BOXSIZE - 1e-6, 1e-6],
    ])
    w = np.ones(len(pos))
    grid = assign(pos, w, _mesh(), scheme=scheme, pbc=False)
    assert np.all(np.isfinite(grid))
    assert np.isclose(grid.sum(), w.sum(), rtol=1e-6), (
        f"{scheme} does not conserve mass at the box faces with pbc=False")


@pytest.mark.parametrize("scheme", SMOOTH_SCHEMES)
def test_parallel_and_serial_assignment_agree(scheme):
    """The chunked variant is an optimisation, not a different scheme."""
    pos, w = _particles(n=2000, seed=4)
    serial = assign(pos, w, _mesh(), scheme=scheme, pbc=False, parallel=False)
    chunked = assign(pos, w, _mesh(), scheme=scheme, pbc=False, parallel=True)
    np.testing.assert_allclose(chunked, serial, rtol=1e-10, atol=1e-10)


def test_schemes_differ_from_each_other():
    """Guards the parametrisation: three names must not be one implementation."""
    pos, w = _particles(n=300, seed=9)
    grids = {s: assign(pos, w, _mesh(), scheme=s, pbc=False) for s in SCHEMES}
    assert not np.allclose(grids["NGP"], grids["CIC"])
    assert not np.allclose(grids["CIC"], grids["TSC"])
    # TSC spreads over more cells than CIC, which spreads over more than NGP.
    occupied = {s: int((g != 0).sum()) for s, g in grids.items()}
    assert occupied["NGP"] < occupied["CIC"] < occupied["TSC"], occupied


# ---------------------------------------------------------------------------
# read-out
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scheme", SMOOTH_SCHEMES)
@pytest.mark.parametrize("pbc", [True, False])
def test_readout_of_a_constant_field_is_that_constant(scheme, pbc):
    """Interpolation weights must sum to one, everywhere including at the faces."""
    mesh = _mesh()
    grid = np.full(mesh.shape, 3.25, dtype=np.float64)
    pos, _ = _particles(n=400, seed=1)
    pos = np.vstack([pos, [[0.0, 0.0, 0.0], [BOXSIZE, BOXSIZE, BOXSIZE]]])

    values = readout(grid, pos, mesh, scheme=scheme, pbc=pbc)
    assert values.shape == (pos.shape[0],)
    np.testing.assert_allclose(values, 3.25, rtol=1e-6)


@pytest.mark.parametrize("scheme", SMOOTH_SCHEMES)
def test_readout_recovers_a_linear_field(scheme):
    """A field the stencil can represent exactly must come back exactly.

    Sampled away from the faces, where a non-periodic clamp would truncate the
    ramp rather than interpolate it.
    """
    mesh = _mesh(nmesh=32)
    cell = BOXSIZE / 32
    nodes = np.arange(32) * cell
    grid = np.broadcast_to(nodes[:, None, None], (32, 32, 32)).astype(np.float64).copy()

    pos, _ = _particles(n=500, seed=2, margin=2 * cell)
    values = readout(grid, pos, mesh, scheme=scheme, pbc=False)
    np.testing.assert_allclose(values, pos[:, 0], rtol=0, atol=0.02 * cell)


@pytest.mark.parametrize("scheme", SMOOTH_SCHEMES)
def test_assign_then_readout_is_self_consistent(scheme):
    """Painting one particle and reading at its own position must see its weight."""
    mesh = _mesh(nmesh=8)
    pos = np.array([[51.3, 24.9, 77.1]])
    grid = assign(pos, np.array([1.0]), mesh, scheme=scheme, pbc=False)
    value = readout(grid, pos, mesh, scheme=scheme, pbc=False)
    assert value[0] > 0.0
    # The stencil is normalised, so the read-back cannot exceed the mass painted.
    assert value[0] <= 1.0 + 1e-9


def test_readout_follows_the_field_precision():
    """Documented behaviour: a float64 field is not silently downcast."""
    mesh = _mesh(dtype=np.float32)
    pos, _ = _particles(n=50, seed=3)
    assert readout(np.full(mesh.shape, 1.0, dtype=np.float64), pos, mesh,
                   scheme="CIC", pbc=False).dtype == np.float64
    assert readout(np.full(mesh.shape, 1.0, dtype=np.float32), pos, mesh,
                   scheme="CIC", pbc=False).dtype == np.float32


# ---------------------------------------------------------------------------
# dtype, defaults and validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_assign_follows_the_mesh_precision(dtype):
    pos, w = _particles(n=100, seed=6)
    assert assign(pos, w, _mesh(dtype=dtype), scheme="CIC", pbc=False).dtype == dtype


def test_assign_defaults_weights_to_ones():
    pos, w = _particles(n=120, seed=7)
    np.testing.assert_allclose(
        assign(pos, None, _mesh(), scheme="TSC", pbc=False),
        assign(pos, w, _mesh(), scheme="TSC", pbc=False), rtol=0, atol=0)


@pytest.mark.parametrize("scheme", SCHEMES)
def test_out_of_range_positions_are_rejected_without_pbc(scheme):
    """A survey run has no wrap to fall back on, so this must be an error."""
    pos, w = _particles(n=10, seed=8)
    pos[0, 0] = BOXSIZE * 1.5
    with pytest.raises(ValueError, match=r"range \[0, boxsize\]"):
        assign(pos, w, _mesh(), scheme=scheme, pbc=False)


@pytest.mark.parametrize("scheme", SCHEMES)
def test_out_of_range_positions_only_warn_with_pbc(scheme):
    pos, w = _particles(n=10, seed=8)
    pos[0, 0] = BOXSIZE * 1.5
    with pytest.warns(UserWarning, match="outside the range"):
        grid = assign(pos, w, _mesh(), scheme=scheme, pbc=True)
    assert np.isclose(grid.sum(), w.sum(), rtol=1e-6)


def test_malformed_input_is_rejected():
    mesh = _mesh()
    with pytest.raises(ValueError, match=r"shape \(N, 3\)"):
        assign(np.zeros((10, 2)), np.ones(10), mesh, pbc=False)
    with pytest.raises(ValueError, match="same number of particles"):
        assign(np.zeros((10, 3)), np.ones(9), mesh, pbc=False)


@pytest.mark.parametrize("bad", ["", "cic ", "quadratic", "cloud-in-cell"])
def test_unknown_scheme_is_rejected(bad):
    mesh = _mesh()
    pos, w = _particles(n=5, seed=10)
    if bad.strip().upper() in ("NGP", "CIC", "TSC"):
        pytest.skip("case-insensitive alias, not an invalid scheme")
    with pytest.raises(ValueError, match="Invalid scheme"):
        assign(pos, w, mesh, scheme=bad, pbc=False)
    with pytest.raises(ValueError, match="Invalid scheme"):
        readout(np.zeros(mesh.shape), pos, mesh, scheme=bad, pbc=False)


@pytest.mark.parametrize("alias", ["cic", " CIC ", "Tsc"])
def test_scheme_names_are_case_and_space_insensitive(alias):
    pos, w = _particles(n=50, seed=11)
    grid = assign(pos, w, _mesh(), scheme=alias, pbc=False)
    assert np.isclose(grid.sum(), w.sum(), rtol=1e-6)


def test_gpu_request_without_cupy_is_a_clear_error():
    from baorecon.mas import _interface
    if _interface.CUPY_AVAILABLE:
        pytest.skip("CuPy present: the failure path cannot be exercised")
    pos, w = _particles(n=5, seed=12)
    mesh = _mesh()
    with pytest.raises(RuntimeError, match="GPU backend requested"):
        assign(pos, w, mesh, scheme="CIC", device="gpu", pbc=False)
    with pytest.raises(RuntimeError, match="GPU backend requested"):
        readout(np.zeros(mesh.shape), pos, mesh, scheme="CIC", device="gpu", pbc=False)
