"""Precision-compliance tests.

Verify the working precision propagates end-to-end for both supported dtypes:
- the default is ``float32`` (float64 inputs are downcast), and
- ``float64`` is honoured when explicitly requested via ``dtype``.

Covers the formatters, :class:`DensityManager`, and :class:`BAOReconstructor`.
"""

import numpy as np
import pytest

from baorecon.utils import formatters as F
from baorecon.reconstruction.density import DensityManager
from baorecon.reconstruction.bao_reconstructor import BAOReconstructor

DTYPES = [np.float32, np.float64]


def _catalogues(seed=0, n_data=200, n_rand=800, box=100.0):
    """float64 catalogues, to check downcasting to the working precision."""
    rng = np.random.RandomState(seed)
    data = rng.uniform(0.1 * box, 0.9 * box, size=(n_data, 3)).astype(np.float64)
    rand = rng.uniform(0.0, box, size=(n_rand, 3)).astype(np.float64)
    return data, rand


# ---------------------------------------------------------------------------
# formatters
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", DTYPES)
def test_format_positions_dtype(dtype):
    pos = np.random.RandomState(0).rand(20, 3).astype(np.float64)
    out = F.format_positions(pos, dtype=dtype)
    assert out.dtype == dtype
    assert out.shape == (20, 3)


def test_format_positions_default_is_float32():
    pos = np.random.RandomState(0).rand(20, 3).astype(np.float64)
    assert F.format_positions(pos).dtype == np.float32


def test_format_positions_bad_shape_raises():
    with pytest.raises(ValueError, match="shape"):
        F.format_positions(np.random.rand(20, 2))


@pytest.mark.parametrize("dtype", DTYPES)
def test_format_boxsize_dtype(dtype):
    pos = np.random.RandomState(0).rand(10, 3).astype(np.float64) * 10.0
    box = F.format_boxsize(100.0, pos, pbc=True, dtype=dtype)
    assert box.dtype == dtype
    assert box.shape == (3,)


def test_format_boxsize_default_is_float32():
    pos = np.random.RandomState(0).rand(10, 3).astype(np.float64) * 10.0
    assert F.format_boxsize(100.0, pos, pbc=True).dtype == np.float32


@pytest.mark.parametrize("dtype", DTYPES)
def test_set_boxsize_from_positions_dtype(dtype):
    pos = np.random.RandomState(0).rand(10, 3).astype(np.float64) * 10.0
    assert F.set_boxsize_from_positions(pos, padding=1.0, dtype=dtype).dtype == dtype


def test_set_boxsize_from_positions_default_is_float32():
    pos = np.random.RandomState(0).rand(10, 3).astype(np.float64) * 10.0
    assert F.set_boxsize_from_positions(pos, padding=1.0).dtype == np.float32


@pytest.mark.parametrize("dtype", DTYPES)
def test_nmesh_boxsize_from_cellsize_dtype(dtype):
    extent = np.array([100.0, 100.0, 100.0])
    _, boxsize = F.nmesh_boxsize_from_cellsize(extent, 5.0, dtype=dtype)
    assert boxsize.dtype == dtype


# ---------------------------------------------------------------------------
# DensityManager
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", DTYPES)
def test_density_manager_dtype_propagation(dtype):
    data, rand = _catalogues(seed=1)
    dm = DensityManager(
        data_pos=data, random_pos=rand, nmesh=16, boxsize=100.0,
        boxcentre=[50, 50, 50], dtype=dtype,
    )
    assert dm.boxsize.dtype == dtype
    assert dm.boxcentre.dtype == dtype
    assert dm.data_pos_box.dtype == dtype
    assert dm.random_pos_box.dtype == dtype
    assert dm.data_weights.dtype == dtype
    assert dm.random_weights.dtype == dtype
    assert dm.delta_on_mesh.dtype == dtype


@pytest.mark.parametrize("dtype", DTYPES)
def test_density_manager_dtype_from_cellsize(dtype):
    data, rand = _catalogues(seed=2)
    dm = DensityManager(
        data_pos=data, random_pos=rand, cellsize=5.0,
        boxcentre=[50, 50, 50], dtype=dtype,
    )
    assert dm.boxsize.dtype == dtype
    assert dm.delta_on_mesh.dtype == dtype


# ---------------------------------------------------------------------------
# BAOReconstructor
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", DTYPES)
def test_bao_reconstructor_dtype_attributes(dtype):
    data, rand = _catalogues(seed=3)
    recon = BAOReconstructor(
        data_pos=data, random_pos=rand, nmesh=16, boxsize=100.0,
        boxcentre=[50, 50, 50], f=0.8, bias=2.0, dtype=dtype, RSDspace="RealSpace",
    )
    assert recon.data_pos.dtype == dtype
    assert recon.random_pos.dtype == dtype
    assert recon.boxsize.dtype == dtype
    assert recon.boxcentre.dtype == dtype
    assert recon.delta_on_mesh.dtype == dtype


def test_bao_reconstructor_downcasts_float64_by_default():
    """float64 catalogues are downcast to float32 unless double precision is asked."""
    data, rand = _catalogues(seed=4)
    assert data.dtype == np.float64 and rand.dtype == np.float64
    recon = BAOReconstructor(
        data_pos=data, random_pos=rand, nmesh=8, boxsize=100.0,
        boxcentre=[50, 50, 50], f=0.8, bias=2.0,
    )
    assert recon.data_pos.dtype == np.float32
    assert recon.random_pos.dtype == np.float32
    assert recon.delta_on_mesh.dtype == np.float32


@pytest.mark.parametrize("solver_type", ["multigrid", "ifft"])
@pytest.mark.parametrize("dtype", DTYPES)
def test_run_reconstruction_dtype(solver_type, dtype):
    """Full reconstruction keeps the working precision for both solvers."""
    data, rand = _catalogues(seed=5, n_data=300, n_rand=1200)
    recon = BAOReconstructor(
        data_pos=data, random_pos=rand, nmesh=16, boxsize=100.0,
        boxcentre=[50, 50, 50], f=0.8, bias=2.0, dtype=dtype,
        RSDspace="RealSpace", solver_type=solver_type,
    )
    data_rec, random_rec = recon.run_reconstruction()
    assert data_rec.dtype == dtype
    assert random_rec.dtype == dtype
    assert data_rec.shape == data.shape
    assert random_rec.shape == rand.shape
    assert np.isfinite(data_rec).all()
    assert np.isfinite(random_rec).all()
