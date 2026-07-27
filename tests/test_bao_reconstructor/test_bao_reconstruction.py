import numpy as np
import pytest

from baorecon.reconstruction.bao_reconstructor import BAOReconstructor


def test_bao_reconstructor_instantiation():
    """Smoke test for BAOReconstructor constructor."""
    boxsize = 100.0
    data = np.random.uniform(0, boxsize, size=(50, 3))
    rand = np.random.uniform(0, boxsize, size=(150, 3))
    recon = BAOReconstructor(data, rand, nmesh=8, boxsize=boxsize, boxcentre=[50, 50, 50], padding=0.2, bias=1.0, f=0.88)
    assert recon is not None
    assert np.allclose(recon.data_pos, data)
    assert np.allclose(recon.random_pos, rand)


def test_bao_reconstructor_infers_box_when_missing():
    """boxsize and boxcentre are set when not provided."""
    data = np.random.uniform(10.0, 50, size=(100, 3)).astype(np.float32)
    rand = np.random.uniform(0.0, 50, size=(100, 3)).astype(np.float32)

    rand[0] = [0,0,0]
    rand[1] = [50,50,50]


    recon = BAOReconstructor(data, rand, nmesh=8, boxsize=None, boxcentre=None, padding=0.5, bias=1.0, f=0.88)

    combined = np.concatenate((data, rand), axis=0)
    expected_center = 0.5 * (rand.min(axis=0) + rand.max(axis=0))

    assert np.allclose(recon.data_pos, data)
    assert np.allclose(recon.random_pos, rand)
    assert np.allclose(recon.boxcentre, expected_center)


def test_bao_reconstructor_with_standard_params():
    """Instantiation with standard reconstruction parameters."""
    np.random.seed(42)
    boxsize = 100.0
    data_pos = np.random.uniform(0, boxsize, size=(50, 3))
    random_pos = np.random.uniform(0, boxsize, size=(150, 3))

    recon = BAOReconstructor(
        data_pos=data_pos, random_pos=random_pos, boxsize=boxsize, RSDspace="RealSpace",
        nmesh=16, boxcentre=np.array([boxsize / 2] * 3), los="z", f=0.88, bias=1.7,
        R_sm=15.0, threshold_randoms=0.01,
    )
    assert recon.data_pos.shape == (50, 3)
    assert recon.random_pos.shape == (150, 3)


def test_bao_reconstructor_delta_mesh():
    """Density contrast computation."""
    np.random.seed(42)
    boxsize = 100.0
    data_pos = np.random.uniform(0, boxsize, size=(50, 3))
    random_pos = np.random.uniform(0, boxsize, size=(150, 3))

    recon = BAOReconstructor(data_pos=data_pos, random_pos=random_pos, boxsize=boxsize, bias=1.0, f=0.88,
                             nmesh=16, boxcentre=np.array([boxsize / 2] * 3), threshold_randoms=0.01)
    delta_mesh = recon.delta_on_mesh
    assert delta_mesh.shape == (16, 16, 16)
    assert np.isfinite(delta_mesh).all()


def test_bao_reconstructor_solver_potential_displacement():
    """The FFT solver computes both potential and displacement."""
    np.random.seed(42)
    boxsize = 100.0
    data_pos = np.random.uniform(0, boxsize, size=(50, 3))
    random_pos = np.random.uniform(0, boxsize, size=(150, 3))

    recon = BAOReconstructor(data_pos=data_pos, random_pos=random_pos, boxsize=boxsize,
                             bias = 1.0, f=0.88,
                             nmesh=16, boxcentre=np.array([boxsize / 2] * 3),
                             RSDspace="RealSpace", threshold_randoms=0.01)
    phi = recon.solver.potential
    psi = recon.solver.displacement
    assert phi.shape == (16, 16, 16)
    assert psi.shape == (16, 16, 16, 3)
    assert np.isfinite(phi).all()
    assert np.isfinite(psi).all()


def test_bao_reconstructor_run_reconstruction():
    """Full BAO reconstruction workflow."""
    np.random.seed(42)
    boxsize = 100.0
    data_pos = np.random.uniform(0, boxsize, size=(50, 3))
    random_pos = np.random.uniform(0, boxsize, size=(150, 3))

    recon = BAOReconstructor(data_pos=data_pos, random_pos=random_pos, boxsize=boxsize,
                             nmesh=16, boxcentre=np.array([boxsize / 2] * 3), f=0.88, bias=1.7,
                             threshold_randoms=0.01)
    data_rec, random_rec = recon.run_reconstruction()
    assert data_rec.shape == data_pos.shape
    assert random_rec.shape == random_pos.shape
    assert np.isfinite(data_rec).all()
    assert np.isfinite(random_rec).all()


def test_bao_reconstructor_interpolate_displacement():
    """Displacement interpolation at tracer positions."""
    np.random.seed(42)
    boxsize = 100.0
    data_pos = np.random.uniform(0, boxsize, size=(50, 3))
    random_pos = np.random.uniform(0, boxsize, size=(150, 3))

    recon = BAOReconstructor(data_pos=data_pos, random_pos=random_pos, boxsize=boxsize, f=0.88, bias=1.7,
                             nmesh=16, boxcentre=np.array([boxsize / 2] * 3), threshold_randoms=0.01)
    psi_tracers = recon.interpolate_displacement(data_pos)
    assert psi_tracers.shape == (50, 3)
    assert np.isfinite(psi_tracers).all()

@pytest.mark.parametrize("rsd_space", ["RealSpace", "RedshiftSpace"])
@pytest.mark.parametrize("los", ["x", "y", "z", None])
def test_bao_reconstructor_comparison_vs_pyrecon(los, rsd_space):
    """Density output against pyrecon (if available)."""
    np.product = np.prod
    np.asfarray = np.asarray
    pytest.importorskip("pyrecon")
    from pyrecon import MultiGridReconstruction

    np.random.seed(42)
    boxsize = 100.0
    nmesh = 8
    data_pos = np.random.uniform(0, boxsize, size=(30, 3))
    random_pos = np.random.uniform(0, boxsize, size=(90, 3))
    bias, f, sm_rad, rand_th = 1.7, 0.88, 15.0, 0.01

    recon_zelda = BAOReconstructor(
        data_pos=data_pos, random_pos=random_pos, boxsize=boxsize, nmesh=nmesh,
        boxcentre=np.array([boxsize / 2] * 3), f=f, bias=bias, R_sm=sm_rad,
        threshold_randoms=rand_th, RSDspace=rsd_space, los=los
    )
    delta_zelda = recon_zelda.delta_on_mesh

    f_pyrecon = f if rsd_space == "RedshiftSpace" else 0.0

    py_recon = MultiGridReconstruction(f=f_pyrecon, bias=bias, los=los, boxsize=boxsize,
                                       boxcenter=[boxsize / 2] * 3, nmesh=nmesh,
                                       randoms_threshold=rand_th)
    py_recon.assign_data(data_pos)
    py_recon.assign_randoms(random_pos)
    py_recon.set_density_contrast(smoothing_radius=sm_rad)
    delta_pyrecon = np.asarray(py_recon.mesh_delta)

    correlation = np.corrcoef(delta_zelda.flatten(), (bias * delta_pyrecon).flatten())[0, 1]
    assert correlation > 0.95, f"Delta fields correlation too low: {correlation}"
