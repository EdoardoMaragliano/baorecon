"""Tests for multigrid-friendly grid sizing, the cellsize entry point,
anisotropic multigrid coarsening, and strict nmesh validation."""

import numpy as np
import pytest

from baorecon.utils.formatters import (
    format_nmesh,
    round_to_multigrid_friendly,
    nmesh_boxsize_from_cellsize,
)
from baorecon.mesh.mesh import Mesh
from baorecon.mesh.los import FixedAxisLOS
from baorecon.solvers.multigrid import MultigridSolver
from baorecon.solvers.multigrid.solver import is_multigrid_friendly
from baorecon.reconstruction.bao_reconstructor import BAOReconstructor


# ==========================================
# 1. MULTIGRID-FRIENDLY SIZING
# ==========================================
@pytest.mark.parametrize("n, expected", [
    (1, 4), (3, 4), (4, 4), (5, 8), (60, 64), (100, 112), (130, 160)
])
def test_round_to_multigrid_friendly(n, expected):
    assert round_to_multigrid_friendly(n) == expected


def test_round_to_multigrid_friendly_rejects_nonpositive():
    with pytest.raises(ValueError):
        round_to_multigrid_friendly(0)


@pytest.mark.parametrize("n, friendly", [
    (4, True), (8, True), (12, True), (16, True), (20, True), (28, True),
    (9, False), (36, False), (100, False), (2, False), (3, False)
])
def test_is_multigrid_friendly(n, friendly):
    assert is_multigrid_friendly(n) is friendly


def test_nmesh_boxsize_from_cellsize():
    extent = np.array([800.0, 1000.0, 500.0])
    nmesh, boxsize = nmesh_boxsize_from_cellsize(extent, 8.0)
    assert nmesh.shape == (3,) and boxsize.shape == (3,)
    # isotropic cells exactly equal to the requested cellsize
    np.testing.assert_allclose(boxsize / nmesh, 8.0)
    # multigrid-friendly per axis and box covers the extent
    assert all(is_multigrid_friendly(int(n)) for n in nmesh)
    assert (boxsize >= extent).all()


def test_nmesh_boxsize_from_cellsize_validation():
    with pytest.raises(ValueError):
        nmesh_boxsize_from_cellsize(np.array([1.0, 2.0]), 8.0)
    with pytest.raises(ValueError):
        nmesh_boxsize_from_cellsize(np.array([1.0, 2.0, 3.0]), -8.0)


# ==========================================
# 2. STRICT NMESH VALIDATION
# ==========================================
@pytest.mark.parametrize("good, expected", [
    (16, (16, 16, 16)),
    ([8, 16, 32], (8, 16, 32)),
])
def test_format_nmesh_valid(good, expected):
    assert tuple(format_nmesh(good)) == expected


@pytest.mark.parametrize("bad", [16.5, [16, 16], 0, -4, [8, 16, 0]])
def test_format_nmesh_rejects(bad):
    with pytest.raises(ValueError):
        format_nmesh(bad)


def test_mesh_rejects_non_integer_nmesh():
    with pytest.raises(ValueError):
        Mesh(nmesh=16.5, boxsize=100.0, boxcentre=[50, 50, 50])


# ==========================================
# 3. ANISOTROPIC MULTIGRID COARSENING
# ==========================================
def test_multigrid_anisotropic_runs():
    N = np.array([16, 16, 32])
    rng = np.random.RandomState(0)
    delta = rng.normal(0, 0.1, size=tuple(N))
    delta -= delta.mean()
    mesh = Mesh(nmesh=N, boxsize=[100.0, 100.0, 200.0], boxcentre=[50, 50, 100])

    mg = MultigridSolver(delta, mesh, los=FixedAxisLOS(2), f=0.0, bias=1.0,
                         RSDspace="RealSpace", use_plane_parallel=True)
    psi = mg.displacement
    assert psi.shape == (16, 16, 32, 3)
    assert np.isfinite(psi).all()


def test_multigrid_rejects_non_friendly_nmesh():
    mesh = Mesh(nmesh=36, boxsize=100.0, boxcentre=[50, 50, 50])
    with pytest.raises(ValueError):
        MultigridSolver(np.zeros((36, 36, 36)), mesh, los=FixedAxisLOS(2),
                        f=0.0, bias=1.0).displacement


def test_multigrid_warns_when_unbalanced():
    # nmesh = [8, 8, 64]: factors of 2 per axis = [3, 3, 6], spread 3 >= 3 -> warns.
    mesh = Mesh(nmesh=[8, 8, 64], boxsize=[100.0, 100.0, 800.0], boxcentre=[50, 50, 400])
    delta = np.zeros((8, 8, 64))
    with pytest.warns(UserWarning):
        MultigridSolver(delta, mesh, los=FixedAxisLOS(2), f=0.0, bias=1.0).displacement


# ==========================================
# 4. CELLSIZE ENTRY POINT
# ==========================================
def test_cellsize_entry_point_derives_grid():
    rng = np.random.RandomState(1)
    data = rng.uniform(0, 100, size=(200, 3))
    rand = rng.uniform(0, 100, size=(600, 3))

    recon = BAOReconstructor(data, rand, cellsize=10.0, los="z", f=0.0, bias=1.7,
                             RSDspace="RealSpace", threshold_randoms=0.01, pbc=False)
    nmesh = np.asarray(recon.nmesh)
    boxsize = np.asarray(recon.boxsize)
    assert nmesh.shape == (3,)
    np.testing.assert_allclose(boxsize / nmesh, 10.0, rtol=1e-5)
    assert all(is_multigrid_friendly(int(n)) for n in nmesh)

    data_rec, random_rec = recon.run_reconstruction()
    assert data_rec.shape == data.shape
    assert np.isfinite(data_rec).all()


def test_cellsize_mutually_exclusive_with_nmesh():
    rng = np.random.RandomState(1)
    data = rng.uniform(0, 100, size=(50, 3))
    rand = rng.uniform(0, 100, size=(150, 3))
    with pytest.raises(ValueError):
        BAOReconstructor(data, rand, cellsize=10.0, nmesh=64)
