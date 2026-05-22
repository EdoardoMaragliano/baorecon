import numpy as np
import pytest


def test_bao_reconstructor_instantiation():
    """Smoke test for BAOReconstructor constructor."""
    from zeldareco.BAOreconstruction.bao_reconstructor import BAOReconstructor
    
    data = np.random.rand(10, 3) * 100.0
    rand = np.random.rand(20, 3) * 100.0
    recon = BAOReconstructor(data, rand, nmesh=8, boxsize=100, boxcentre=[50,50,50], padding=0.2)
    assert recon is not None
    assert np.allclose(recon.data_pos, data)
    assert np.allclose(recon.random_pos, rand)


def test_bao_reconstructor_infers_box_when_missing():
    """Test boxsize and boxcenter are properly set when not provided"""
    from zeldareco.BAOreconstruction.bao_reconstructor import BAOReconstructor

    data = np.random.uniform(0, 100, size=(100, 3))
    rand = np.random.uniform(0, 100, size=(200, 3))

    recon = BAOReconstructor(data, rand, nmesh=8, boxsize=None, boxcentre=None, padding=0.5)

    combined = np.concatenate((data, rand), axis=0)
    expected_center = 0.5 * (combined.min(axis=0) + combined.max(axis=0))

    assert np.allclose(recon.data_pos, data)
    assert np.allclose(recon.random_pos, rand)
    assert np.allclose(recon.boxcentre, expected_center)


def test_bao_ifft_reconstruction_instantiation():
    """Test BAOiFFTreconstruction instantiation with standard parameters."""
    from zeldareco.BAOreconstruction.bao_ifft import BAOiFFTreconstruction
    
    # Create synthetic data
    np.random.seed(42)
    n_data = 50
    n_rand = 150
    boxsize = 100.0
    data_pos = np.random.uniform(0, boxsize, size=(n_data, 3))
    random_pos = np.random.uniform(0, boxsize, size=(n_rand, 3))
    
    # Standard reconstruction parameters
    recon = BAOiFFTreconstruction(
        data_pos=data_pos,
        random_pos=random_pos,
        boxsize=boxsize,
        RSDspace="RealSpace",
        nmesh=16,
        boxcentre=np.array([boxsize/2, boxsize/2, boxsize/2]),
        los='z',
        f=0.88,
        bias=1.7,
        R_sm=15.0,
        threshold_randoms=0.01,
    )
    assert recon is not None
    assert recon.data_pos.shape == (n_data, 3)
    assert recon.random_pos.shape == (n_rand, 3)


def test_bao_ifft_compute_delta_mesh():
    """Test density contrast computation in BAOiFFTreconstruction."""
    from zeldareco.BAOreconstruction.bao_ifft import BAOiFFTreconstruction
    
    np.random.seed(42)
    n_data = 50
    n_rand = 150
    boxsize = 100.0
    data_pos = np.random.uniform(0, boxsize, size=(n_data, 3))
    random_pos = np.random.uniform(0, boxsize, size=(n_rand, 3))
    
    recon = BAOiFFTreconstruction(
        data_pos=data_pos,
        random_pos=random_pos,
        boxsize=boxsize,
        nmesh=16,
        boxcentre=np.array([boxsize/2, boxsize/2, boxsize/2]),
    )
    
    delta_mesh = recon.compute_delta_mesh()
    assert delta_mesh is not None
    assert delta_mesh.shape == (16, 16, 16)
    assert np.isfinite(delta_mesh).all()


def test_bao_ifft_solver_potential_displacement():
    """Test that FFT solver computes both potential and displacement."""
    from zeldareco.BAOreconstruction.bao_ifft import BAOiFFTreconstruction
    
    np.random.seed(42)
    n_data = 50
    n_rand = 150
    boxsize = 100.0
    data_pos = np.random.uniform(0, boxsize, size=(n_data, 3))
    random_pos = np.random.uniform(0, boxsize, size=(n_rand, 3))
    
    recon = BAOiFFTreconstruction(
        data_pos=data_pos,
        random_pos=random_pos,
        boxsize=boxsize,
        nmesh=16,
        boxcentre=np.array([boxsize/2, boxsize/2, boxsize/2]),
        RSDspace="RealSpace",
    )
    
    # Trigger solver computation via delta mesh
    _ = recon.compute_delta_mesh()
    
    # Access solver outputs
    phi = recon.solver.potential
    psi = recon.solver.displacement
    
    assert phi is not None
    assert psi is not None
    assert phi.shape == (16, 16, 16)
    assert psi.shape == (16, 16, 16, 3)
    assert np.isfinite(phi).all()
    assert np.isfinite(psi).all()


def test_bao_ifft_run_reconstruction():
    """Test full BAO reconstruction workflow."""
    from zeldareco.BAOreconstruction.bao_ifft import BAOiFFTreconstruction
    
    np.random.seed(42)
    n_data = 50
    n_rand = 150
    boxsize = 100.0
    data_pos = np.random.uniform(0, boxsize, size=(n_data, 3))
    random_pos = np.random.uniform(0, boxsize, size=(n_rand, 3))
    
    recon = BAOiFFTreconstruction(
        data_pos=data_pos,
        random_pos=random_pos,
        boxsize=boxsize,
        nmesh=16,
        boxcentre=np.array([boxsize/2, boxsize/2, boxsize/2]),
        f=0.88,
        bias=1.7,
    )
    
    # Run full reconstruction
    data_rec, random_rec = recon.run_reconstruction()
    
    assert data_rec is not None
    assert random_rec is not None
    assert data_rec.shape == data_pos.shape
    assert random_rec.shape == random_pos.shape
    assert np.isfinite(data_rec).all()
    assert np.isfinite(random_rec).all()


def test_bao_ifft_interpolate_displacement():
    """Test displacement interpolation at tracer positions."""
    from zeldareco.BAOreconstruction.bao_ifft import BAOiFFTreconstruction
    
    np.random.seed(42)
    n_data = 50
    n_rand = 150
    boxsize = 100.0
    data_pos = np.random.uniform(0, boxsize, size=(n_data, 3))
    random_pos = np.random.uniform(0, boxsize, size=(n_rand, 3))
    
    recon = BAOiFFTreconstruction(
        data_pos=data_pos,
        random_pos=random_pos,
        boxsize=boxsize,
        nmesh=16,
        boxcentre=np.array([boxsize/2, boxsize/2, boxsize/2]),
    )
    
    # Trigger solver computation
    _ = recon.compute_delta_mesh()
    
    # Interpolate displacement
    psi_tracers = recon._interpolate_displacement(data_pos)
    
    assert psi_tracers is not None
    assert psi_tracers.shape == (n_data, 3)
    assert np.isfinite(psi_tracers).all()


def test_bao_ifft_comparison_vs_pyrecon():
    """Test BAOiFFTreconstruction density output against pyrecon (if available)."""
    # Monkey-patch numpy for pmesh/pyrecon compatibility with NumPy 2.x
    np.product = np.prod
    np.asfarray = np.asarray
    
    pytest.importorskip("pyrecon")
    from zeldareco.BAOreconstruction.bao_ifft import BAOiFFTreconstruction
    from pyrecon.multigrid import MultiGridReconstruction
    
    np.random.seed(42)
    n_data = 30
    n_rand = 90
    boxsize = 100.0
    nmesh = 8
    data_pos = np.random.uniform(0, boxsize, size=(n_data, 3))
    random_pos = np.random.uniform(0, boxsize, size=(n_rand, 3))
    
    bias = 1.7
    f = 0.88
    sm_rad = 15.0
    rand_th = 0.01
    
    # Zelda reconstruction
    recon_zelda = BAOiFFTreconstruction(
        data_pos=data_pos,
        random_pos=random_pos,
        boxsize=boxsize,
        nmesh=nmesh,
        boxcentre=np.array([boxsize/2, boxsize/2, boxsize/2]),
        f=f,
        bias=bias,
        R_sm=sm_rad,
        threshold_randoms=rand_th,
        RSDspace="RealSpace",
    )
    
    delta_zelda = recon_zelda.compute_delta_mesh()
    
    # Pyrecon reconstruction
    py_recon = MultiGridReconstruction(
        f=f, bias=bias, los='z',
        boxsize=boxsize, boxcenter=[boxsize/2]*3,
        nmesh=nmesh
    )
    py_recon.assign_data(data_pos)
    py_recon.assign_randoms(random_pos)
    py_recon.set_density_contrast(smoothing_radius=sm_rad)
    delta_pyrecon = np.asarray(py_recon.mesh_delta)
    
    # Compare delta fields: correlation should be high
    # Note: normalize by bias since pyrecon stores matter field
    delta_zelda_flat = delta_zelda.flatten()
    delta_pyrecon_flat = bias * delta_pyrecon.flatten()
    
    correlation = np.corrcoef(delta_zelda_flat, delta_pyrecon_flat)[0, 1]
    assert correlation > 0.95, f"Delta fields correlation too low: {correlation}"
