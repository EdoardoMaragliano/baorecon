import numpy as np
from zeldareco.displacement_solver.fft_solver import FFTSolver
from zeldareco.displacement_solver.multigrid_solver import MultigridSolver

def test_solvers_agree_realspace():
    # small synthetic delta (Gaussian bump)
    N = 64
    x = np.linspace(0, 64, N, endpoint=False)
    xx, yy, zz = np.meshgrid(x, x, x, indexing='ij')
    delta = np.exp(-((xx-0.5)**2 + (yy-0.5)**2 + (zz-0.5)**2) / (2*0.05**2))

    from zeldareco.mesh.mesh import Mesh
    mesh = Mesh(N, 100.0, np.array([50.0,50.0,50.0]))

    fft = FFTSolver(delta, mesh, f=0.0, bias=1.0, RSDspace='RealSpace')
    mg = MultigridSolver(delta, mesh, f=0.0, bias=1.0, RSDspace='RealSpace')

    d1 = fft.displacement
    d2 = mg.displacement

    assert isinstance(d1, np.ndarray)
    assert isinstance(d2, np.ndarray)
    assert d1.shape == d2.shape == (N, N, N, 3)
    assert np.all(np.isfinite(d1))
    assert np.all(np.isfinite(d2))
    assert np.allclose(d1, d2, atol=1e-1, rtol=1e-2), "Displacements from FFT and Multigrid solvers do not match in real space"


def test_solvers_agree_redshiftspace():
    # small synthetic delta (Gaussian bump)
    N = 64
    x = np.linspace(0, 64, N, endpoint=False)
    xx, yy, zz = np.meshgrid(x, x, x, indexing='ij')
    delta = np.exp(-((xx-0.5)**2 + (yy-0.5)**2 + (zz-0.5)**2) / (2*0.05**2))

    from zeldareco.mesh.mesh import Mesh
    mesh = Mesh(N, 100.0, np.array([50.0,50.0,50.0]))

    fft = FFTSolver(delta, mesh, f=0.8, bias=1.0, RSDspace='RedshiftSpace')
    mg = MultigridSolver(delta, mesh, f=0.8, bias=1.0, RSDspace='RedshiftSpace')

    d1 = fft.displacement
    d2 = mg.displacement

    assert isinstance(d1, np.ndarray)
    assert isinstance(d2, np.ndarray)
    assert d1.shape == d2.shape == (N, N, N, 3)
    assert np.all(np.isfinite(d1))
    assert np.all(np.isfinite(d2))
    assert np.allclose(d1, d2, atol=1e-1, rtol=1e-2), "Displacements from FFT and Multigrid solvers do not match in redshift space"