"""Multi-GPU integration tests (NCCL + mpi4py transport).

Run with::

    mpirun -np 2 python -m pytest tests/test_distributed_gpu.py -q

Every test is skipped unless CUDA, CuPy and mpi4py are available AND the
process was launched with at least 2 MPI ranks (one GPU per rank; ranks share
a GPU if the node has fewer). The P=2 results are gathered and compared on
rank 0 against the single-GPU (P=1) reference computed there, per the
verification plan in docs/multigpu_migration_plan.md; the communication-free
logic underneath is covered on CPU by tests/test_distributed.py.
"""

import numpy as np
import pytest

from baorecon.utils.backend import CUPY_AVAILABLE

try:
    from mpi4py import MPI

    WORLD_SIZE = MPI.COMM_WORLD.Get_size()
except ImportError:
    WORLD_SIZE = 1

pytestmark = pytest.mark.skipif(
    not CUPY_AVAILABLE or WORLD_SIZE < 2,
    reason="requires CUDA + CuPy + mpi4py and `mpirun -np >= 2`",
)

if CUPY_AVAILABLE:
    import cupy as cp

NMESH = 64
BOXSIZE = 500.0


@pytest.fixture(scope="module")
def env():
    from baorecon.utils.distributed import DistEnv

    return DistEnv.from_mpi()


@pytest.fixture()
def mesh():
    from baorecon.mesh.mesh import Mesh

    return Mesh(nmesh=NMESH, boxsize=BOXSIZE, boxcentre=np.full(3, BOXSIZE / 2))


def _catalog(n, seed):
    rng = np.random.default_rng(seed)   # same seed => identical on every rank
    pos = rng.uniform(0, BOXSIZE, size=(n, 3)).astype(np.float32)
    weights = rng.uniform(0.5, 1.5, size=n).astype(np.float32)
    return pos, weights


class TestDistributedMAS:
    @pytest.mark.parametrize("scheme", ["CIC", "TSC"])
    @pytest.mark.parametrize("pbc", [True, False])
    def test_paint_matches_single_gpu(self, env, mesh, scheme, pbc):
        from baorecon.mas import assign

        pos, weights = _catalog(50_000, seed=10)
        local = assign(pos, weights, mesh, scheme=scheme, device="gpu",
                       pbc=pbc, dist=env)

        # mass conservation across ranks
        total = env.allreduce_sum(float(cp.sum(local)))
        np.testing.assert_allclose(total, float(weights.sum()), rtol=1e-4)

        full = env.gather_x_slabs(cp.asnumpy(local))
        if env.rank == 0:
            ref = cp.asnumpy(assign(pos, weights, mesh, scheme=scheme,
                                    device="gpu", pbc=pbc))
            np.testing.assert_allclose(full, ref, rtol=1e-4, atol=1e-4)
        env.barrier()


class TestDistributedFFTGPU:
    def test_forward_and_closure(self, env):
        from baorecon.solvers.fft._distributed_fft import DistributedFFT

        shape = (NMESH, NMESH, NMESH)
        rng = np.random.default_rng(11)
        full = rng.standard_normal(shape).astype(np.float32)
        d = env.decomp(shape)
        slab = cp.asarray(full[d.x_offset:d.x_offset + d.nx_local])

        dfft = DistributedFFT(env, shape)
        k_local = dfft.rfftn(slab)
        back = dfft.irfftn(k_local, s=d.local_real_shape)
        np.testing.assert_allclose(cp.asnumpy(back),
                                   full[d.x_offset:d.x_offset + d.nx_local],
                                   rtol=1e-4, atol=1e-4)

        # gather ky-blocks (axis 1) and compare the full spectrum on rank 0
        parts = env.comm.mpi.gather(cp.asnumpy(k_local), root=0)
        if env.rank == 0:
            got = np.concatenate(parts, axis=1)
            ref = cp.asnumpy(cp.fft.rfftn(cp.asarray(full)))
            np.testing.assert_allclose(got, ref, rtol=1e-3, atol=1e-2)
        env.barrier()


class TestDistributedSolver:
    @pytest.mark.parametrize("los", [None, "z"])
    def test_displacement_matches_single_gpu(self, env, mesh, los):
        from baorecon.mesh.los import FixedAxisLOS, LocalLOS
        from baorecon.solvers.fft import FFTSolverGPU

        rng = np.random.default_rng(12)
        delta = rng.normal(0, 0.1, size=mesh.shape).astype(np.float32)

        if los is None:
            los_strategy = LocalLOS(boxcentre=mesh.boxcentre,
                                    min_corner=mesh.min_corner,
                                    boxsize=mesh.boxsize, nmesh=mesh.nmesh,
                                    device="gpu")
        else:
            los_strategy = FixedAxisLOS(2)

        d = env.decomp(mesh.shape)
        slab = cp.asarray(delta[d.x_offset:d.x_offset + d.nx_local])
        solver = FFTSolverGPU(delta_on_mesh=slab, mesh=mesh, los=los_strategy,
                              f=0.8, bias=1.5, RSDspace="RedshiftSpace",
                              n_iterations=2, dist=env)
        disp_local = cp.asnumpy(solver.displacement)
        full = env.gather_x_slabs(disp_local)

        if env.rank == 0:
            ref_solver = FFTSolverGPU(delta_on_mesh=cp.asarray(delta), mesh=mesh,
                                      los=los_strategy, f=0.8, bias=1.5,
                                      RSDspace="RedshiftSpace", n_iterations=2)
            ref = cp.asnumpy(ref_solver.displacement)
            np.testing.assert_allclose(full, ref, rtol=1e-3, atol=1e-4)
        env.barrier()


class TestEndToEnd:
    def test_reconstructor_matches_single_gpu(self, env):
        from baorecon.reconstruction.bao_reconstructor import BAOReconstructor

        data_pos, data_w = _catalog(5_000, seed=13)
        rand_pos, rand_w = _catalog(50_000, seed=14)
        kwargs = dict(
            data_weights=data_w, random_weights=rand_w,
            RSDspace="RedshiftSpace", nmesh=NMESH, boxsize=BOXSIZE,
            boxcentre=np.full(3, BOXSIZE / 2), los=None, R_sm=15.0, pbc=True,
            rectype="rec-sym", f=0.8, bias=1.5, MAS="CIC",
            solver_type="ifft", device="gpu",
        )
        rec = BAOReconstructor(data_pos=data_pos, random_pos=rand_pos,
                               dist=env, **kwargs)
        data_rec, rand_rec = rec.run_reconstruction()

        if env.rank == 0:
            ref = BAOReconstructor(data_pos=data_pos, random_pos=rand_pos, **kwargs)
            data_ref, rand_ref = ref.run_reconstruction()
            np.testing.assert_allclose(data_rec, data_ref, rtol=1e-3, atol=5e-3)
            np.testing.assert_allclose(rand_rec, rand_ref, rtol=1e-3, atol=5e-3)
        env.barrier()
