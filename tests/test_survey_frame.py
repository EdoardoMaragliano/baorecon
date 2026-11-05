"""Painting and reading out straight from the survey frame.

``DensityManager`` used to keep a box-frame copy of both catalogues
(``survey_to_box_frame``: subtract the lower corner, wrap under pbc) for the
whole solve: two more ``(N, 3)`` arrays the size of the input. The kernels now
take the lower corner and do the same two operations on each coordinate as
they read it. The claim is exactness, not closeness: the subtraction and the
wrap are the same operations in the same dtype, so every grid, every read-out
and every reconstructed position must come out bit for bit as before. That is
what is asserted here, against ``survey_to_box_frame`` itself.
"""

import numpy as np
import pytest

from baorecon.field_ops import interpolate_vector_field
from baorecon.field_ops.cpu import differentiate_potential_jit_cic, differentiate_potential_tsc_jit
from baorecon.mas import assign, readout
from baorecon.mesh.mesh import Mesh
from baorecon.reconstruction.bao_reconstructor import BAOReconstructor
from baorecon.reconstruction.density import DensityManager
from baorecon.utils.formatters import survey_to_box_frame

DTYPES = [np.float32, np.float64]
BOXSIZE = 100.0
NMESH = 16
# An off-origin box, so the subtraction is not a no-op on any axis.
BOXCENTRE = np.array([1234.5, -321.25, 777.0])


def _mesh(dtype):
    return Mesh(nmesh=NMESH, boxsize=BOXSIZE, boxcentre=BOXCENTRE, dtype=dtype)


def _survey_positions(mesh, n=2000, seed=0, outside=False):
    """Survey-frame positions in the box; with ``outside`` some spill past the faces
    (only meaningful under pbc, where they are wrapped back in)."""
    rng = np.random.RandomState(seed)
    lo = np.asarray(mesh.min_corner, dtype=np.float64)
    margin = 0.3 * BOXSIZE if outside else 0.0
    pos = rng.uniform(lo - margin, lo + BOXSIZE + margin, size=(n, 3))
    # A few particles exactly on cell boundaries: the floor is the fragile step.
    cell = BOXSIZE / NMESH
    pos[:20] = lo + cell * rng.randint(0, NMESH, size=(20, 3))
    return pos.astype(mesh.dtype)


def _box_positions(mesh, pos, pbc):
    return survey_to_box_frame(pos, mesh.min_corner, mesh.boxsize, pbc=pbc, dtype=mesh.dtype)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("scheme", ["NGP", "CIC", "TSC"])
@pytest.mark.parametrize("pbc", [True, False])
@pytest.mark.parametrize("parallel", [False, True])
def test_assign_from_the_survey_frame_is_bit_identical(dtype, scheme, pbc, parallel):
    mesh = _mesh(dtype)
    pos = _survey_positions(mesh, outside=pbc)
    w = np.random.RandomState(1).uniform(0.5, 1.5, len(pos)).astype(dtype)
    direct = assign(pos, w, mesh, scheme=scheme, pbc=pbc, parallel=parallel, survey_frame=True)
    via_copy = assign(_box_positions(mesh, pos, pbc), w, mesh, scheme=scheme, pbc=pbc, parallel=parallel)
    assert direct.dtype == via_copy.dtype == dtype
    assert np.array_equal(direct, via_copy)
    assert direct.sum() > 0


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("scheme", ["NGP", "CIC", "TSC"])
@pytest.mark.parametrize("pbc", [True, False])
def test_readout_from_the_survey_frame_is_bit_identical(dtype, scheme, pbc):
    mesh = _mesh(dtype)
    pos = _survey_positions(mesh, outside=pbc)
    grid = np.random.RandomState(2).normal(size=mesh.shape).astype(dtype)
    direct = readout(grid, pos, mesh, scheme=scheme, pbc=pbc, survey_frame=True)
    via_copy = readout(grid, _box_positions(mesh, pos, pbc), mesh, scheme=scheme, pbc=pbc)
    assert np.array_equal(direct, via_copy)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("scheme", ["CIC", "TSC"])
@pytest.mark.parametrize("pbc", [True, False])
def test_vector_interpolation_from_the_survey_frame_is_bit_identical(dtype, scheme, pbc):
    mesh = _mesh(dtype)
    pos = _survey_positions(mesh, outside=pbc)
    field = np.random.RandomState(3).normal(size=mesh.shape + (3,)).astype(dtype)
    direct = interpolate_vector_field(pos, field, mesh.boxsize, MAS=scheme, pbc=pbc, dtype=dtype,
                                      origin=mesh.min_corner)
    via_copy = interpolate_vector_field(_box_positions(mesh, pos, pbc), field, mesh.boxsize,
                                        MAS=scheme, pbc=pbc, dtype=dtype)
    assert np.array_equal(direct, via_copy)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("scheme", ["CIC", "TSC"])
@pytest.mark.parametrize("pbc", [True, False])
def test_potential_differentiation_from_the_survey_frame_is_bit_identical(dtype, scheme, pbc):
    mesh = _mesh(dtype)
    pos = _survey_positions(mesh, outside=pbc)
    phi = np.random.RandomState(4).normal(size=mesh.shape).astype(dtype)
    nmesh = np.asarray(mesh.nmesh, dtype=np.int32)
    direct = np.empty_like(pos)
    via_copy = np.empty_like(pos)
    origin = np.asarray(mesh.min_corner, dtype=dtype)
    if scheme == "CIC":
        differentiate_potential_jit_cic(phi, pos, direct, nmesh, 0, mesh.boxsize, origin=origin, pbc=pbc)
        differentiate_potential_jit_cic(phi, _box_positions(mesh, pos, pbc), via_copy, nmesh, 0, mesh.boxsize, pbc=pbc)
    else:
        differentiate_potential_tsc_jit(phi, pos, direct, nmesh, mesh.boxsize, origin=origin, pbc=pbc)
        differentiate_potential_tsc_jit(phi, _box_positions(mesh, pos, pbc), via_copy, nmesh, mesh.boxsize, pbc=pbc)
    assert np.array_equal(direct, via_copy)


def test_survey_frame_positions_outside_the_box_are_rejected_without_pbc():
    mesh = _mesh(np.float64)
    pos = _survey_positions(mesh, outside=True)
    with pytest.raises(ValueError, match="range"):
        assign(pos, np.ones(len(pos)), mesh, scheme="CIC", pbc=False, survey_frame=True)


@pytest.mark.parametrize("dtype", DTYPES)
def test_density_manager_keeps_no_box_frame_copy(dtype):
    """The catalogues are held in the survey frame, by reference; the box-frame
    view is computed on request and is a fresh array every time."""
    rng = np.random.RandomState(5)
    lo = BOXCENTRE - BOXSIZE / 2
    data = rng.uniform(lo + 5, lo + BOXSIZE - 5, size=(500, 3)).astype(dtype)
    rand = rng.uniform(lo + 5, lo + BOXSIZE - 5, size=(5000, 3)).astype(dtype)
    dm = DensityManager(data_pos=data, random_pos=rand, nmesh=NMESH, boxsize=BOXSIZE,
                        boxcentre=BOXCENTRE, dtype=dtype, pbc=False)

    assert dm.data_pos is data and dm.random_pos is rand
    assert not any(k.endswith("pos_box") for k in vars(dm))
    box = dm.data_pos_box
    assert box.dtype == dtype
    assert np.array_equal(box, survey_to_box_frame(data, dm.min_corner, dm.boxsize, pbc=False, dtype=dtype))
    assert dm.data_pos_box is not box
    assert dm.delta_on_mesh.shape == (NMESH,) * 3


@pytest.mark.parametrize("solver_type", ["ifft", "multigrid"])
@pytest.mark.parametrize("dtype", DTYPES)
def test_reconstruction_reads_the_displacement_straight_from_the_survey_frame(solver_type, dtype):
    """Psi read at the survey positions equals Psi read at their box-frame copy."""
    rng = np.random.RandomState(6)
    lo = BOXCENTRE - BOXSIZE / 2
    data = rng.uniform(lo + 5, lo + BOXSIZE - 5, size=(2000, 3)).astype(dtype)
    rand = rng.uniform(lo + 5, lo + BOXSIZE - 5, size=(20000, 3)).astype(dtype)
    recon = BAOReconstructor(data, rand, nmesh=NMESH, boxsize=BOXSIZE, boxcentre=BOXCENTRE,
                             bias=1.5, f=0.8, R_sm=10.0, dtype=dtype,
                             solver_type=solver_type, pbc=False)
    direct = recon.interpolate_displacement(recon.data_pos, survey_frame=True)
    via_copy = recon.interpolate_displacement(recon._density_manager.data_pos_box, survey_frame=False)
    assert np.array_equal(direct, via_copy)
    assert np.isfinite(direct).all() and np.abs(direct).max() > 0
