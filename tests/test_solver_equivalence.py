import numpy as np

from baorecon.solvers.fft import FFTSolverCPU
from baorecon.solvers.multigrid import MultigridSolver
from baorecon.mesh.mesh import Mesh
from baorecon.mesh.los import FixedAxisLOS


def _gaussian_bump(N=64):
    x = np.linspace(0, 64, N, endpoint=False)
    xx, yy, zz = np.meshgrid(x, x, x, indexing="ij")
    return np.exp(-((xx - 0.5) ** 2 + (yy - 0.5) ** 2 + (zz - 0.5) ** 2) / (2 * 0.05 ** 2))


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
    assert np.allclose(d1, d2, atol=1e-1, rtol=1e-2), \
        "FFT and Multigrid displacements disagree in redshift space"
