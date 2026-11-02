"""GPU field-operations tests (CuPy / numba.cuda).

:mod:`baorecon.field_ops.gpu` mirrors the CPU kernels in
:mod:`baorecon.field_ops.cpu`. Unlike the mass-assignment interface, the
field_ops dispatcher (:mod:`baorecon.field_ops._interface`) has no explicit
``device=`` kwarg -- it picks CPU vs GPU from the array module of the inputs
(see ``_get_array_module``) -- so these tests build CuPy inputs directly to
route through the GPU path, and check them against the same analytic
reference used for the CPU kernels in ``test_field_ops.py`` plus direct
agreement with the CPU result on identical inputs.
"""

import numpy as np
import pytest

from baorecon.field_ops import divergence, interpolate_vector_field
from baorecon.utils.backend import CUPY_AVAILABLE

try:
    import cupy as cp
except ImportError:
    cp = None

gpu_test = pytest.mark.skipif(not CUPY_AVAILABLE, reason="GPU not available or CuPy not installed")


def _to_host(arr):
    return cp.asnumpy(arr) if CUPY_AVAILABLE and cp is not None and isinstance(arr, cp.ndarray) else arr


# ==========================================
# INTERPOLATION (CIC & TSC)
# ==========================================
@gpu_test
@pytest.mark.parametrize("mas_scheme", ["CIC", "TSC"])
def test_interpolate_constant_vector_field_gpu(mas_scheme):
    """GPU read-out of a constant field returns the constant (mirrors the CPU
    check in test_field_ops.py) and agrees with the CPU result."""
    nmesh = 8
    boxsize = 10.0
    field = np.zeros((nmesh, nmesh, nmesh, 3), dtype=np.float64)
    field[..., 0] = 5.0
    field[..., 1] = -2.0
    field[..., 2] = 3.0

    np.random.seed(42)
    pos = np.random.uniform(0, boxsize, size=(100, 3))

    interp_cpu = interpolate_vector_field(pos, field, boxsize, MAS=mas_scheme, pbc=True, dtype=np.float64)

    interp_gpu = interpolate_vector_field(cp.asarray(pos), cp.asarray(field), boxsize,
                                          MAS=mas_scheme, pbc=True, dtype=np.float32)
    interp_gpu = _to_host(interp_gpu)

    assert interp_gpu.shape == (100, 3)
    assert np.allclose(interp_gpu[:, 0], 5.0, atol=1e-4)
    assert np.allclose(interp_gpu[:, 1], -2.0, atol=1e-4)
    assert np.allclose(interp_gpu[:, 2], 3.0, atol=1e-4)
    # GPU kernels are float32-only; compare against the float64 CPU result at
    # float32 precision.
    np.testing.assert_allclose(interp_gpu, interp_cpu, atol=1e-4)


@gpu_test
@pytest.mark.parametrize("pbc", [True, False])
@pytest.mark.parametrize("mas_scheme", ["CIC", "TSC"])
def test_interpolate_boundary_policy_matches_cpu_gpu(mas_scheme, pbc):
    """GPU read-out honours ``pbc`` like the CPU kernels: wrap when periodic,
    clamp to the boundary cell otherwise.

    A constant field cannot tell wrap from clamp (both give the constant), so
    use a field that is linear in each coordinate -- its wrapped and clamped
    values at the box edge differ by the full field range -- and sample inside
    the last and first cell of every axis, where the stencil leaves the grid.
    The GPU kernels used to ignore ``pbc`` and always wrap, so a ``pbc=False``
    reconstruction read a Psi from the far face of the box near the edge.
    """
    nmesh = 8
    boxsize = 10.0
    cell = boxsize / nmesh
    idx = np.arange(nmesh, dtype=np.float64)
    field = np.zeros((nmesh, nmesh, nmesh, 3), dtype=np.float64)
    field[..., 0] = idx[:, None, None]      # grows along x
    field[..., 1] = idx[None, :, None]      # grows along y
    field[..., 2] = idx[None, None, :]      # grows along z

    rng = np.random.default_rng(0)
    # Last half-cell and first half-cell of every axis, plus a few interior points.
    edge_hi = rng.uniform(boxsize - 0.5 * cell, boxsize - 1e-6, size=(40, 3))
    edge_lo = rng.uniform(0.0, 0.5 * cell, size=(40, 3))
    inner = rng.uniform(cell, boxsize - cell, size=(40, 3))
    pos = np.concatenate([edge_hi, edge_lo, inner])

    interp_cpu = interpolate_vector_field(pos, field, boxsize, MAS=mas_scheme, pbc=pbc, dtype=np.float64)
    interp_gpu = _to_host(interpolate_vector_field(cp.asarray(pos), cp.asarray(field), boxsize,
                                                   MAS=mas_scheme, pbc=pbc, dtype=np.float32))

    np.testing.assert_allclose(interp_gpu, interp_cpu, atol=1e-4)
    if not pbc:
        # Clamped: the field keeps growing up to the boundary node, never
        # falls back towards the x=0 face (which wrapping would do).
        assert (interp_gpu[:40] > nmesh - 2).all()


# ==========================================
# DIVERGENCE (FFT)
# ==========================================
@gpu_test
def test_divergence_fft_gpu_matches_analytic_and_cpu():
    """delta(x)=sin(x) -> div=cos(x); GPU path agrees with the analytic result
    and with the CPU FFT divergence on the same field."""
    N = 32
    boxsize = 2 * np.pi
    x = np.linspace(0, boxsize, N, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    field = np.zeros((N, N, N, 3), dtype=np.float64)
    field[..., 0] = np.sin(X)

    kx = np.fft.fftfreq(N, d=boxsize / N) * 2 * np.pi
    ky = np.fft.fftfreq(N, d=boxsize / N) * 2 * np.pi
    kz = np.fft.rfftfreq(N, d=boxsize / N) * 2 * np.pi

    div_cpu = divergence(field, div_algo="FFT", k_components=(kx, ky, kz))

    field_gpu = cp.asarray(field)
    k_gpu = (cp.asarray(kx), cp.asarray(ky), cp.asarray(kz))
    div_gpu = _to_host(divergence(field_gpu, div_algo="FFT", k_components=k_gpu))

    assert np.allclose(div_gpu, np.cos(X), atol=1e-6)
    assert np.allclose(div_gpu, div_cpu, atol=1e-6)
