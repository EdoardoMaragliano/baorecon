import pytest
import numpy as np

from baorecon.field_ops import (
    project_vector_field,
    interpolate_vector_field,
    divergence,
    smoothed_field,
)
from baorecon.mesh.los import project_vector_field_jit


# ==========================================
# DUMMY MESH (geometry only, mimics the real Mesh interface used here)
# ==========================================
class DummyMesh:
    """Minimal mesh exposing the attributes smoothing/divergence need."""

    def __init__(self, N=16, boxsize=1.0):
        self.nmesh = np.array([N, N, N], dtype=np.int64)
        self.cell_size = np.array([boxsize / N] * 3, dtype=np.float64)
        self._kx = np.fft.fftfreq(N, d=boxsize / N) * 2 * np.pi
        self._ky = np.fft.fftfreq(N, d=boxsize / N) * 2 * np.pi
        self._kz = np.fft.rfftfreq(N, d=boxsize / N) * 2 * np.pi

    @property
    def shape(self):
        return tuple(int(n) for n in self.nmesh)

    @property
    def k_components(self):
        return self._kx, self._ky, self._kz


# ==========================================
# VECTOR PROJECTION
# ==========================================
@pytest.mark.parametrize("axis_idx, expected_mag", [(0, 2.0), (1, 3.0), (2, 4.0)])
def test_project_vector_field_tilted(axis_idx, expected_mag):
    N = 4
    field = np.zeros((N, N, N, 3), dtype=np.float32)
    field[..., 0] = 2.0
    field[..., 1] = 3.0
    field[..., 2] = 4.0

    los_versor = np.zeros((N, N, N, 3), dtype=np.float32)
    los_versor[..., axis_idx] = 1.0

    proj_np = project_vector_field(field, los_versor)
    assert np.allclose(proj_np[..., axis_idx], expected_mag)
    for i in range(3):
        if i != axis_idx:
            assert np.allclose(proj_np[..., i], 0.0)

    proj_jit = project_vector_field_jit(field, los_versor, out=np.empty_like(field))
    assert np.allclose(proj_jit[..., axis_idx], expected_mag)
    for i in range(3):
        if i != axis_idx:
            assert np.allclose(proj_jit[..., i], 0.0)


# ==========================================
# INTERPOLATION (CIC & TSC)
# ==========================================
@pytest.mark.parametrize("mas_scheme", ["CIC", "TSC"])
def test_interpolate_constant_vector_field(mas_scheme):
    nmesh = 8
    boxsize = 10.0
    field = np.zeros((nmesh, nmesh, nmesh, 3), dtype=np.float64)
    field[..., 0] = 5.0
    field[..., 1] = -2.0
    field[..., 2] = 3.0

    np.random.seed(42)
    pos = np.random.uniform(0, boxsize, size=(100, 3))

    interp_field = interpolate_vector_field(pos, field, boxsize, MAS=mas_scheme, pbc=True, dtype=np.float64)
    assert interp_field.shape == (100, 3)
    assert np.allclose(interp_field[:, 0], 5.0)
    assert np.allclose(interp_field[:, 1], -2.0)
    assert np.allclose(interp_field[:, 2], 3.0)


# ==========================================
# DIVERGENCE (FINITE DIFF)
# ==========================================
def test_divergence_finite_diff():
    N = 10
    cell_size = 1.0
    x = np.arange(N)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    field = np.stack([1.0 * X, 2.0 * Y, 3.0 * Z], axis=-1).astype(np.float64)

    div = divergence(field, div_algo="finite_diff", cell_size=cell_size)
    inner_div = div[1:-1, 1:-1, 1:-1]
    assert np.allclose(inner_div, 6.0)


# ==========================================
# DIVERGENCE (FFT)
# ==========================================
def test_divergence_fft():
    N = 32
    boxsize = 2 * np.pi
    mesh = DummyMesh(N=N, boxsize=boxsize)

    x = np.linspace(0, boxsize, N, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    field = np.zeros((N, N, N, 3), dtype=np.float64)
    field[..., 0] = np.sin(X)

    div = divergence(field, div_algo="FFT", k_components=mesh.k_components)
    assert np.allclose(div, np.cos(X), atol=1e-7)


# ==========================================
# SMOOTHING KERNEL
# ==========================================
def test_smoothed_field_constant():
    N = 16
    mesh = DummyMesh(N=N, boxsize=10.0)
    field = np.ones((N, N, N), dtype=np.float64) * 3.0
    smoothed = smoothed_field(field, mesh, smoothing_radius=2.0)
    assert np.allclose(smoothed, 3.0, atol=1e-7)


def test_smoothed_field_reduces_variance():
    N = 32
    mesh = DummyMesh(N=N, boxsize=10.0)
    np.random.seed(42)
    noisy_field = np.random.normal(0, 1.0, size=(N, N, N))
    smoothed = smoothed_field(noisy_field, mesh, smoothing_radius=2.0)
    assert np.var(smoothed) < np.var(noisy_field)


# ==========================================
# GPU PARITY (skipped without CUDA)
# ==========================================
from baorecon.utils.backend import CUPY_AVAILABLE  # noqa: E402

gpu_test = pytest.mark.skipif(not CUPY_AVAILABLE, reason="GPU not available or CuPy not installed")


@gpu_test
@pytest.mark.parametrize("mas_scheme", ["CIC", "TSC"])
def test_interpolate_vector_field_gpu_matches_cpu(mas_scheme):
    import cupy as cp

    nmesh, boxsize = 8, 10.0
    rng = np.random.default_rng(42)
    field = rng.normal(0, 1, size=(nmesh, nmesh, nmesh, 3)).astype(np.float32)
    pos = rng.uniform(0, boxsize, size=(200, 3)).astype(np.float32)

    ref = interpolate_vector_field(pos, field, boxsize, MAS=mas_scheme, pbc=True,
                                   dtype=np.float32)
    got = interpolate_vector_field(cp.asarray(pos), cp.asarray(field), boxsize,
                                   MAS=mas_scheme, pbc=True, dtype=np.float32)
    np.testing.assert_allclose(cp.asnumpy(got), ref, rtol=1e-4, atol=1e-5)


@gpu_test
def test_divergence_fft_gpu_matches_cpu():
    import cupy as cp

    N = 32
    mesh = DummyMesh(N=N, boxsize=2 * np.pi)
    x = np.linspace(0, 2 * np.pi, N, endpoint=False)
    X = np.meshgrid(x, x, x, indexing="ij")[0]
    field = np.zeros((N, N, N, 3), dtype=np.float32)
    field[..., 0] = np.sin(X)

    k_dev = tuple(cp.asarray(k, dtype=cp.float32) for k in mesh.k_components)
    div = divergence(cp.asarray(field), div_algo="FFT", k_components=k_dev)
    np.testing.assert_allclose(cp.asnumpy(div), np.cos(X), atol=1e-4)


@gpu_test
def test_smoothed_field_gpu_matches_cpu():
    import cupy as cp

    N = 32
    mesh = DummyMesh(N=N, boxsize=10.0)
    rng = np.random.default_rng(42)
    field = rng.normal(0, 1.0, size=(N, N, N)).astype(np.float32)
    ref = smoothed_field(field.copy(), mesh, smoothing_radius=2.0)
    got = smoothed_field(cp.asarray(field), mesh, smoothing_radius=2.0)
    np.testing.assert_allclose(cp.asnumpy(got), ref, rtol=1e-4, atol=1e-5)
