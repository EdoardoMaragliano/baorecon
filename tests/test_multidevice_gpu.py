"""Single-process multi-GPU tests (run_multi_gpu / MultiDeviceComm).

The notebook-launcher counterpart of tests/test_distributed_gpu.py: no MPI,
no mpirun -- just::

    python -m pytest tests/test_multidevice_gpu.py -q

on a host with at least 2 CUDA devices. Asserts that the thread-per-GPU NCCL
path reproduces the single-GPU (P=1) results, mirroring the MPI suite.
"""

import numpy as np
import pytest

from baorecon.utils.backend import CUPY_AVAILABLE

if CUPY_AVAILABLE:
    import cupy as cp

    N_DEVICES = cp.cuda.runtime.getDeviceCount()
else:
    N_DEVICES = 0

pytestmark = pytest.mark.skipif(
    N_DEVICES < 2, reason="requires CUDA + CuPy and at least 2 GPUs"
)

NMESH = 64
BOXSIZE = 500.0


def _mesh():
    from baorecon.mesh.mesh import Mesh

    return Mesh(nmesh=NMESH, boxsize=BOXSIZE, boxcentre=np.full(3, BOXSIZE / 2))


def _catalog(n, seed):
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0, BOXSIZE, size=(n, 3)).astype(np.float32)
    weights = rng.uniform(0.5, 1.5, size=n).astype(np.float32)
    return pos, weights


class TestRunMultiGpu:
    def test_fft_closure_and_forward(self):
        from baorecon.solvers.fft._distributed_fft import DistributedFFT
        from baorecon.utils.distributed import run_multi_gpu

        shape = (NMESH, NMESH, NMESH)
        full = np.random.default_rng(0).standard_normal(shape).astype(np.float32)

        def job(env):
            d = env.decomp(shape)
            dfft = DistributedFFT(env, shape)
            slab = cp.asarray(full[d.x_offset:d.x_offset + d.nx_local])
            k = dfft.rfftn(slab)
            back = dfft.irfftn(k, s=d.local_real_shape)
            return cp.asnumpy(k), cp.asnumpy(back), d.x_offset, d.nx_local

        results = run_multi_gpu(job)
        got_k = np.concatenate([r[0] for r in results], axis=1)
        ref_k = np.fft.rfftn(full)
        np.testing.assert_allclose(got_k, ref_k, rtol=1e-3, atol=1e-2)
        for k_loc, back, off, nx in results:
            np.testing.assert_allclose(back, full[off:off + nx], rtol=1e-4, atol=1e-4)

    def test_mas_paint_matches_single_gpu(self):
        from baorecon.mas import assign
        from baorecon.utils.distributed import run_multi_gpu

        mesh = _mesh()
        pos, weights = _catalog(50_000, seed=1)

        def job(env):
            local = assign(pos, weights, mesh, scheme="CIC", device="gpu",
                           pbc=True, dist=env)
            total = env.allreduce_sum(float(cp.sum(local)))
            return cp.asnumpy(local), total

        results = run_multi_gpu(job)
        full = np.concatenate([r[0] for r in results], axis=0)
        assert all(np.isclose(r[1], weights.sum(), rtol=1e-4) for r in results)

        cp.cuda.Device(0).use()
        ref = cp.asnumpy(assign(pos, weights, mesh, scheme="CIC", device="gpu", pbc=True))
        np.testing.assert_allclose(full, ref, rtol=1e-4, atol=1e-4)

    def test_reconstructor_matches_single_gpu(self):
        from baorecon.reconstruction.bao_reconstructor import BAOReconstructor
        from baorecon.utils.distributed import run_multi_gpu

        data_pos, data_w = _catalog(5_000, seed=2)
        rand_pos, rand_w = _catalog(50_000, seed=3)
        kwargs = dict(
            data_weights=data_w, random_weights=rand_w,
            RSDspace="RedshiftSpace", nmesh=NMESH, boxsize=BOXSIZE,
            boxcentre=np.full(3, BOXSIZE / 2), los=None, R_sm=15.0, pbc=True,
            rectype="rec-sym", f=0.8, bias=1.5, MAS="CIC",
            solver_type="ifft", device="gpu",
        )

        def job(env):
            rec = BAOReconstructor(data_pos=data_pos, random_pos=rand_pos,
                                   dist=env, **kwargs)
            return rec.run_reconstruction()

        results = run_multi_gpu(job)
        # every rank returns the full, identical catalogues
        np.testing.assert_allclose(results[0][0], results[-1][0], rtol=1e-6)

        cp.cuda.Device(0).use()
        ref = BAOReconstructor(data_pos=data_pos, random_pos=rand_pos, **kwargs)
        data_ref, rand_ref = ref.run_reconstruction()
        np.testing.assert_allclose(results[0][0], data_ref, rtol=1e-3, atol=5e-3)
        np.testing.assert_allclose(results[0][1], rand_ref, rtol=1e-3, atol=5e-3)

    def test_single_device_falls_back_to_serial(self):
        from baorecon.utils.distributed import run_multi_gpu

        out = run_multi_gpu(lambda env: env.is_distributed, devices=[0])
        assert out == [False]
