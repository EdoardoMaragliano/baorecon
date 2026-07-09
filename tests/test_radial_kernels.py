"""Parity tests for the streamed radial-LOS projection kernels.

The radial versor ``n_hat_a = coord_a / |coord|`` is evaluated on the fly by two
independent implementations that must stay in lockstep:

* the numba kernels in :mod:`baorecon.solvers.fft._radial_stream`
  (``project_grad_onto_los`` / ``reconstruct_parallel_vector``) used by the CPU
  solvers;
* the cupy ``ElementwiseKernel`` twins in :mod:`baorecon.solvers.fft.gpu`
  (``_project_grad_onto_los`` / ``_reconstruct_parallel_vector``) used by the GPU
  solver.

There is no shared source between numba and CUDA C, so these tests guard against
the two drifting apart: the CPU kernels are checked against a plain-numpy
reference of the same formula, and (when CuPy is present) the GPU kernels are
checked against the CPU kernels on identical inputs.
"""

import numpy as np
import pytest

from baorecon.solvers.fft._radial_stream import (
    project_grad_onto_los,
    reconstruct_parallel_vector,
)
from baorecon.utils.backend import CUPY_AVAILABLE

gpu_test = pytest.mark.skipif(
    not CUPY_AVAILABLE, reason="GPU not available or CuPy not installed"
)

N = 12
BOXSIZE = 2 * np.pi

# ``with-origin`` puts the (0,0,0) cell at |coord| = 0 so the zero-versor guard
# in both implementations is exercised.
GEOMETRIES = [
    pytest.param((100.0, 100.0, 100.0), id="off-origin"),
    pytest.param((0.0, 0.0, 0.0), id="with-origin"),
]


def _cell_size():
    return np.full(3, BOXSIZE / N, dtype=np.float32)


def _versor_reference(min_corner, cell):
    """(N,N,N,3) radial unit versor via the same on-the-fly formula, in numpy."""
    idx = np.arange(N**3)
    iz = idx % N
    iy = (idx // N) % N
    ix = idx // (N * N)
    cx = (np.float32(min_corner[0]) + ix.astype(np.float32) * cell[0]).astype(np.float32)
    cy = (np.float32(min_corner[1]) + iy.astype(np.float32) * cell[1]).astype(np.float32)
    cz = (np.float32(min_corner[2]) + iz.astype(np.float32) * cell[2]).astype(np.float32)
    r2 = (cx * cx + cy * cy + cz * cz).astype(np.float32)
    versor = np.zeros((N**3, 3), np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        sq = np.sqrt(r2, dtype=np.float32)
        for a, ca in enumerate((cx, cy, cz)):
            versor[:, a] = np.where(r2 > 0.0, ca / sq, np.float32(0.0)).astype(np.float32)
    return versor.reshape(N, N, N, 3)


# ---------------------------------------------------------------------------
# CPU numba kernels vs a numpy reference of the same formula (always runs).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("min_corner", GEOMETRIES)
def test_project_grad_onto_los_matches_reference(min_corner):
    cell = _cell_size()
    rng = np.random.default_rng(0)
    grads = rng.standard_normal((3, N, N, N)).astype(np.float32)

    s = np.empty((N, N, N), dtype=np.float32)
    for a in range(3):
        project_grad_onto_los(s, grads[a], a, *min_corner, *cell, a == 0)

    versor = _versor_reference(min_corner, cell)
    s_ref = np.zeros((N, N, N), dtype=np.float32)
    for a in range(3):
        s_ref += (grads[a] * versor[..., a]).astype(np.float32)

    np.testing.assert_allclose(s, s_ref, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("min_corner", GEOMETRIES)
def test_reconstruct_parallel_vector_matches_reference(min_corner):
    cell = _cell_size()
    rng = np.random.default_rng(1)
    s_field = rng.standard_normal((N, N, N)).astype(np.float32)
    versor = _versor_reference(min_corner, cell)

    for a in range(3):
        parallel = np.empty((N, N, N), dtype=np.float32)
        reconstruct_parallel_vector(parallel, s_field, a, *min_corner, *cell)
        parallel_ref = (s_field * versor[..., a]).astype(np.float32)
        np.testing.assert_allclose(parallel, parallel_ref, rtol=1e-4, atol=1e-5)


# ---------------------------------------------------------------------------
# GPU ElementwiseKernels vs the CPU numba kernels on identical inputs.
# ---------------------------------------------------------------------------
@gpu_test
@pytest.mark.parametrize("min_corner", GEOMETRIES)
def test_gpu_kernels_agree_with_cpu(min_corner):
    import cupy as cp

    from baorecon.solvers.fft.gpu import (
        _project_grad_onto_los,
        _reconstruct_parallel_vector,
    )

    cell = _cell_size()
    mc = [float(x) for x in min_corner]
    cs = [float(x) for x in cell]
    rng = np.random.default_rng(2)
    grads = rng.standard_normal((3, N, N, N)).astype(np.float32)

    # Accumulation: s = sum_a grad_a . n_hat_a
    s_cpu = np.empty((N, N, N), dtype=np.float32)
    for a in range(3):
        project_grad_onto_los(s_cpu, grads[a], a, *min_corner, *cell, a == 0)

    s_gpu = cp.empty((N, N, N), dtype=cp.float32)
    for a in range(3):
        _project_grad_onto_los(cp.asarray(grads[a]), a, *mc, *cs, N, N, int(a == 0), s_gpu)

    np.testing.assert_allclose(cp.asnumpy(s_gpu), s_cpu, rtol=1e-4, atol=1e-5)

    # Scatter: parallel_a = s . n_hat_a
    s_field = rng.standard_normal((N, N, N)).astype(np.float32)
    s_field_gpu = cp.asarray(s_field)
    for a in range(3):
        parallel_cpu = np.empty((N, N, N), dtype=np.float32)
        reconstruct_parallel_vector(parallel_cpu, s_field, a, *min_corner, *cell)

        parallel_gpu = cp.empty((N, N, N), dtype=cp.float32)
        _reconstruct_parallel_vector(s_field_gpu, a, *mc, *cs, N, N, parallel_gpu)

        np.testing.assert_allclose(cp.asnumpy(parallel_gpu), parallel_cpu, rtol=1e-4, atol=1e-5)
