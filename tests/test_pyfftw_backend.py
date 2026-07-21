"""Tests for the opt-in in-place pyfftw CPU FFT backend (BAORECON_FFT=pyfftw).

baorecon.solvers.fft._pyfftw_cpu mirrors baorecon.solvers.fft.cpu.FFTSolverCPU's
scipy path exactly (same physics, one reused in-place buffer instead of a fresh
allocation per transform), so "correct" here means agreement with the scipy
path to its documented round-off. Skipped entirely if pyfftw isn't installed
(it's an optional extra, not a core dependency).
"""

import numpy as np
import pytest

pytest.importorskip("pyfftw")

from baorecon.mesh.los import FixedAxisLOS, LocalLOS
from baorecon.mesh.mesh import Mesh
from baorecon.solvers.fft import FFTSolverCPU
from baorecon.solvers.fft import _pyfftw_cpu


def _make_mesh(N, boxsize, boxcentre=None):
    if boxcentre is None:
        boxcentre = [boxsize / 2] * 3
    return Mesh(nmesh=N, boxsize=boxsize, boxcentre=boxcentre)


class _NoGeometryLOS:
    """A LOS strategy exposing neither .axis nor .min_corner."""


# ==========================================
# supported(): which LOS/iteration combinations this path can handle
# ==========================================
def test_supported_realspace_always_true():
    # n_iterations=0 (RealSpace): only the final build runs, LOS is irrelevant.
    assert _pyfftw_cpu.supported(_NoGeometryLOS(), n_iterations=0) is True


def test_supported_fixed_axis_los():
    assert _pyfftw_cpu.supported(FixedAxisLOS(2), n_iterations=3) is True


def test_supported_radial_los():
    los = LocalLOS(boxcentre=[50, 50, 50], min_corner=[-50, -50, -50], boxsize=100.0, nmesh=16)
    assert _pyfftw_cpu.supported(los, n_iterations=3) is True


def test_supported_unknown_los_rejected():
    assert _pyfftw_cpu.supported(_NoGeometryLOS(), n_iterations=3) is False


# ==========================================
# displacement_inplace() vs the scipy path (FFTSolverCPU), identical inputs
# ==========================================
@pytest.fixture
def basic_setup():
    N = 16
    mesh = _make_mesh(N, 2 * np.pi)
    rng = np.random.default_rng(0)
    delta = rng.normal(0, 0.1, size=(N, N, N)).astype(np.float32)
    delta -= delta.mean()
    return mesh, delta


@pytest.mark.parametrize("n_iterations", [0, 3])
def test_displacement_inplace_matches_scipy_fixed_axis(basic_setup, n_iterations):
    mesh, delta = basic_setup
    los = FixedAxisLOS(2)
    rsd_space = "RealSpace" if n_iterations == 0 else "RedshiftSpace"

    scipy_solver = FFTSolverCPU(delta, mesh, los=los, f=0.8, bias=2.0, RSDspace=rsd_space)
    psi_scipy = scipy_solver.displacement

    psi_pyfftw = _pyfftw_cpu.displacement_inplace(
        delta, mesh, los, f=0.8, bias=2.0, beta=scipy_solver.beta, n_iterations=n_iterations
    )
    np.testing.assert_allclose(psi_pyfftw, psi_scipy, rtol=1e-4, atol=1e-5)


def test_displacement_inplace_matches_scipy_radial():
    # Box pushed off the origin so the observer-at-origin radial versor is
    # well-defined everywhere on the grid (mirrors tests_multigrid/test_smoother_consistency.py).
    N, boxsize = 16, 2 * np.pi
    mesh = _make_mesh(N, boxsize, boxcentre=[1.5 * boxsize] * 3)
    rng = np.random.default_rng(2)
    delta = rng.normal(0, 0.1, size=(N, N, N)).astype(np.float32)
    delta -= delta.mean()

    los = LocalLOS(boxcentre=mesh.boxcentre, min_corner=mesh.min_corner,
                   boxsize=mesh.boxsize, nmesh=mesh.nmesh)

    scipy_solver = FFTSolverCPU(delta, mesh, los=los, f=0.8, bias=2.0, RSDspace="RedshiftSpace")
    psi_scipy = scipy_solver.displacement

    psi_pyfftw = _pyfftw_cpu.displacement_inplace(
        delta, mesh, los, f=0.8, bias=2.0, beta=scipy_solver.beta, n_iterations=3
    )
    np.testing.assert_allclose(psi_pyfftw, psi_scipy, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_displacement_inplace_is_dtype_neutral(dtype):
    N = 16
    mesh = _make_mesh(N, 2 * np.pi)
    rng = np.random.default_rng(1)
    delta = rng.normal(0, 0.1, size=(N, N, N)).astype(dtype)
    delta -= delta.mean()

    psi = _pyfftw_cpu.displacement_inplace(
        delta, mesh, FixedAxisLOS(0), f=0.8, bias=2.0, beta=0.4, n_iterations=3
    )
    assert psi.dtype == np.dtype(dtype)
    assert np.isfinite(psi).all()


# ==========================================
# Integration: FFTSolverCPU actually dispatches here under BAORECON_FFT=pyfftw
# ==========================================
def test_fftsolver_cpu_dispatches_to_pyfftw(monkeypatch, basic_setup):
    mesh, delta = basic_setup
    los = FixedAxisLOS(2)

    scipy_solver = FFTSolverCPU(delta, mesh, los=los, f=0.8, bias=2.0, RSDspace="RedshiftSpace")
    psi_scipy = scipy_solver.displacement

    monkeypatch.setenv("BAORECON_FFT", "pyfftw")
    pyfftw_solver = FFTSolverCPU(delta, mesh, los=los, f=0.8, bias=2.0, RSDspace="RedshiftSpace")
    psi_pyfftw = pyfftw_solver.displacement

    np.testing.assert_allclose(psi_pyfftw, psi_scipy, rtol=1e-4, atol=1e-5)
