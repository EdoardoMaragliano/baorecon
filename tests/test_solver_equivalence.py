import numpy as np

from baorecon.solvers.fft import FFTSolverCPU
from baorecon.solvers.multigrid import MultigridSolver
from baorecon.mesh.mesh import Mesh
from baorecon.mesh.los import FixedAxisLOS


def _gaussian_bump(N=64, sigma=3.0):
    """An overdensity the grid can actually resolve.

    The previous version used ``sigma = 0.05`` on a unit-spaced grid and centred
    the bump at 0.5, between nodes. Sampled on the grid it peaked at 7e-66 —
    identically zero in double precision — so both solvers returned a zero
    displacement and ``np.allclose(0, 0)`` passed without comparing anything.
    Three cells wide and centred on a node, it is a real field; the solvers then
    agree to ~1%, which is the statement these tests are meant to make.

    Mean-subtracted, so it is an overdensity rather than a density.
    """
    x = np.linspace(0, 64, N, endpoint=False)
    xx, yy, zz = np.meshgrid(x, x, x, indexing="ij")
    centre = 32.0
    field = np.exp(-((xx - centre) ** 2 + (yy - centre) ** 2 + (zz - centre) ** 2)
                   / (2 * sigma ** 2))
    return field - field.mean()


def _assert_not_vacuous(d1, d2):
    """Guard against the comparison silently becoming 0 vs 0 again."""
    assert np.abs(d1).max() > 1e-3, (
        "FFT displacement is ~zero: the equivalence assertion below would pass "
        "vacuously. Check that the input overdensity is resolved by the grid.")
    assert np.abs(d2).max() > 1e-3, (
        "Multigrid displacement is ~zero: same problem.")


def test_solvers_agree_realspace():
    N = 64
    delta = _gaussian_bump(N)
    mesh = Mesh(N, 100.0, np.array([50.0, 50.0, 50.0]))
    los = FixedAxisLOS(2)

    fft = FFTSolverCPU(delta, mesh, los=los, f=0.0, bias=1.0, RSDspace="RealSpace")
    mg = MultigridSolver(delta, mesh, los=los, f=0.0, bias=1.0, RSDspace="RealSpace",
                         use_plane_parallel=True)

    d1 = fft.displacement
    d2 = mg.displacement

    assert isinstance(d1, np.ndarray) and isinstance(d2, np.ndarray)
    assert d1.shape == d2.shape == (N, N, N, 3)
    assert np.all(np.isfinite(d1)) and np.all(np.isfinite(d2))
    _assert_not_vacuous(d1, d2)
    assert np.allclose(d1, d2, atol=1e-1, rtol=1e-2), \
        "FFT and Multigrid displacements disagree in real space"


def test_solvers_agree_redshiftspace():
    N = 64
    delta = _gaussian_bump(N)
    mesh = Mesh(N, 100.0, np.array([50.0, 50.0, 50.0]))
    los = FixedAxisLOS(2)

    fft = FFTSolverCPU(delta, mesh, los=los, f=0.8, bias=1.0, RSDspace="RedshiftSpace")
    mg = MultigridSolver(delta, mesh, los=los, f=0.8, bias=1.0, RSDspace="RedshiftSpace",
                         use_plane_parallel=True)

    d1 = fft.displacement
    d2 = mg.displacement

    assert isinstance(d1, np.ndarray) and isinstance(d2, np.ndarray)
    assert d1.shape == d2.shape == (N, N, N, 3)
    assert np.all(np.isfinite(d1)) and np.all(np.isfinite(d2))
    _assert_not_vacuous(d1, d2)
    assert np.allclose(d1, d2, atol=1e-1, rtol=1e-2), \
        "FFT and Multigrid displacements disagree in redshift space"
