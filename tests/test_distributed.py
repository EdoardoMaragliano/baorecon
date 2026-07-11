"""CPU tests for the multi-GPU distribution logic (no GPU/MPI required).

Everything communication-shaped runs through the thread-based LoopbackComm,
so the slab decomposition, the distributed-FFT transpose layout, the halo
exchanges and the distributed smoothing are verified with numpy against their
serial references. The NCCL/mpi4py transport itself is exercised by the
mpirun-gated suite in ``tests/test_distributed_gpu.py``.
"""

import numpy as np
import pytest

from baorecon.mas._interface import HALO_WIDTH
from baorecon.solvers.fft._distributed_fft import (
    DistributedFFT,
    pack_forward,
    pack_inverse,
    unpack_forward,
    unpack_inverse,
)
from baorecon.utils.distributed import (
    DistEnv,
    LoopbackComm,
    SlabDecomp,
    halo_exchange_add,
    halo_exchange_copy,
    run_loopback,
)


def _loopback_envs(world_size):
    comms = LoopbackComm.world(world_size)
    return [DistEnv(rank=r, world_size=world_size, comm=comms[r])
            for r in range(world_size)]


def _run_ranks(envs, fn):
    """Run fn(env) on each rank's own thread (they share loopback barriers)."""
    return run_loopback(len(envs), lambda r, _comm: fn(envs[r]))


# ---------------------------------------------------------------------------
# SlabDecomp
# ---------------------------------------------------------------------------
class TestSlabDecomp:
    def test_partition_covers_mesh(self):
        for p in (1, 2, 4):
            decomps = [SlabDecomp((8, 12, 10), p, r) for r in range(p)]
            assert sum(d.nx_local for d in decomps) == 8
            assert sum(d.ny_local for d in decomps) == 12
            offsets = [d.x_offset for d in decomps]
            assert offsets == sorted(offsets)
            assert decomps[0].x_offset == 0
            assert decomps[-1].x_offset + decomps[-1].nx_local == 8

    def test_shapes(self):
        d = SlabDecomp((8, 12, 10), 4, 1)
        assert d.local_real_shape == (2, 12, 10)
        assert d.local_k_shape == (8, 3, 6)
        assert d.nz_half == 6

    def test_indivisible_rejected(self):
        with pytest.raises(ValueError, match="divisible"):
            SlabDecomp((10, 8, 8), 4, 0)
        with pytest.raises(ValueError, match="divisible"):
            SlabDecomp((8, 10, 8), 4, 0)

    def test_neighbours_and_dc(self):
        d0, d1, d2 = (SlabDecomp((6, 6, 6), 3, r) for r in range(3))
        assert (d0.left_rank, d0.right_rank) == (2, 1)
        assert (d2.left_rank, d2.right_rank) == (1, 0)
        assert d0.owns_dc and not d1.owns_dc
        assert d0.is_low_edge and d2.is_high_edge

    def test_owner_ranks_pbc_wrap(self):
        d = SlabDecomp((8, 8, 8), 2, 0)
        box = 100.0
        x = np.array([0.0, 49.9, 50.0, 99.9, 100.0, 101.0, -1.0])
        owners = d.owner_ranks(x, box, pbc=True)
        #  cells:      0     3      4     7     0(wrap) 0(wrap) 7(wrap)
        assert owners.tolist() == [0, 0, 1, 1, 0, 0, 1]

    def test_owner_ranks_clamped(self):
        d = SlabDecomp((8, 8, 8), 2, 0)
        owners = d.owner_ranks(np.array([-5.0, 105.0]), 100.0, pbc=False)
        assert owners.tolist() == [0, 1]

    def test_owned_masks_are_a_partition(self):
        rng = np.random.default_rng(1)
        x = rng.uniform(0, 200.0, size=1000)
        masks = [SlabDecomp((16, 16, 16), 4, r).owned_mask(x, 200.0) for r in range(4)]
        assert np.all(np.sum(masks, axis=0) == 1)

    def test_ky_slice(self):
        d = SlabDecomp((8, 8, 8), 2, 1)
        ky = np.fft.fftfreq(8)
        assert np.array_equal(d.ky_slice(ky), ky[4:])


# ---------------------------------------------------------------------------
# LoopbackComm
# ---------------------------------------------------------------------------
class TestLoopbackComm:
    def test_alltoall(self):
        def fn(rank, comm):
            send = np.arange(6, dtype=np.float32) + 100 * rank
            recv = np.empty_like(send)
            comm.alltoall(send, recv)
            return recv

        r = run_loopback(3, fn)
        # rank 0 receives block 0 of every rank
        assert np.array_equal(r[0], [0, 1, 100, 101, 200, 201])
        assert np.array_equal(r[2], [4, 5, 104, 105, 204, 205])

    def test_exchange_halos_ring(self):
        def fn(rank, comm):
            to_left = np.array([10.0 * rank], dtype=np.float32)
            to_right = np.array([10.0 * rank + 1], dtype=np.float32)
            from_left = np.empty(1, dtype=np.float32)
            from_right = np.empty(1, dtype=np.float32)
            comm.exchange_halos(to_left, to_right, from_left, from_right)
            return from_left[0], from_right[0]

        r = run_loopback(3, fn)
        # from_left = left neighbour's to_right; from_right = right neighbour's to_left
        assert r[0] == (21.0, 10.0)
        assert r[1] == (1.0, 20.0)
        assert r[2] == (11.0, 0.0)

    def test_allreduce_sum(self):
        assert run_loopback(4, lambda r, c: c.allreduce_sum(r + 1)) == [10.0] * 4

    def test_worker_exception_propagates(self):
        def fn(rank, comm):
            if rank == 1:
                raise RuntimeError("boom")
            comm.barrier()

        with pytest.raises(RuntimeError, match="boom"):
            run_loopback(2, fn)


# ---------------------------------------------------------------------------
# Distributed FFT (loopback, numpy) vs numpy.fft reference
# ---------------------------------------------------------------------------
class TestDistributedFFT:
    @pytest.mark.parametrize("world_size", [2, 4])
    @pytest.mark.parametrize("shape", [(8, 8, 8), (8, 12, 10), (16, 4, 6)])
    def test_forward_matches_numpy(self, world_size, shape):
        rng = np.random.default_rng(0)
        x = rng.standard_normal(shape).astype(np.float32)
        ref = np.fft.rfftn(x)
        envs = _loopback_envs(world_size)

        def fn(env):
            d = env.decomp(shape)
            dfft = DistributedFFT(env, shape, xp=np, fft=np.fft)
            slab = x[d.x_offset:d.x_offset + d.nx_local]
            return dfft.rfftn(slab)

        parts = _run_ranks(envs, fn)
        got = np.concatenate(parts, axis=1)   # ky-split -> global
        np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("world_size", [2, 4])
    @pytest.mark.parametrize("shape", [(8, 8, 8), (8, 12, 10)])
    def test_roundtrip_and_inverse(self, world_size, shape):
        rng = np.random.default_rng(1)
        x = rng.standard_normal(shape).astype(np.float32)
        ref_k = np.fft.rfftn(x)
        envs = _loopback_envs(world_size)

        def fn(env):
            d = env.decomp(shape)
            dfft = DistributedFFT(env, shape, xp=np, fft=np.fft)
            slab = x[d.x_offset:d.x_offset + d.nx_local]
            k = dfft.rfftn(slab)
            back = dfft.irfftn(k, s=d.local_real_shape)
            # inverse of the *reference* spectrum's local ky block too
            k_ref_local = ref_k[:, d.ky_offset:d.ky_offset + d.ny_local, :]
            back_ref = dfft.irfftn(np.ascontiguousarray(k_ref_local))
            return back, back_ref

        parts = _run_ranks(envs, fn)
        got = np.concatenate([p[0] for p in parts], axis=0)
        got_ref = np.concatenate([p[1] for p in parts], axis=0)
        np.testing.assert_allclose(got, x, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(got_ref, x, rtol=1e-4, atol=1e-5)

    def test_p1_delegates(self):
        env = DistEnv.serial()
        dfft = DistributedFFT(env, (8, 8, 8), xp=np, fft=np.fft)
        x = np.random.default_rng(2).standard_normal((8, 8, 8))
        np.testing.assert_allclose(dfft.rfftn(x), np.fft.rfftn(x))

    def test_rejects_partial_axes(self):
        env = DistEnv.serial()
        dfft = DistributedFFT(env, (8, 8, 8), xp=np, fft=np.fft)
        with pytest.raises(ValueError, match="axes"):
            dfft.rfftn(np.zeros((8, 8, 8)), axes=(1, 2))

    def test_pack_unpack_are_inverse_permutations(self):
        rng = np.random.default_rng(3)
        p, nx_loc, ny_loc, nzh = 4, 2, 3, 5
        a = rng.standard_normal((nx_loc, p * ny_loc, nzh)).astype(np.complex64)
        # forward pack on rank r, blocks land as unpack_inverse's recv blocks
        send = pack_forward(a, p)
        back = unpack_inverse(send, p, nx_loc, ny_loc, nzh)
        np.testing.assert_array_equal(back, a)
        b = rng.standard_normal((p * nx_loc, ny_loc, nzh)).astype(np.complex64)
        send = pack_inverse(b, p)
        back = unpack_forward(send, p, nx_loc, ny_loc, nzh)
        np.testing.assert_array_equal(back, b)


# ---------------------------------------------------------------------------
# Halo exchanges (loopback, numpy)
# ---------------------------------------------------------------------------
def _slab_index(ixg, x_start, nx_global, nx_ext, pbc):
    """Python replica of the CUDA slab-index mapping in baorecon/mas/gpu.py."""
    d = ixg - x_start
    if pbc:
        if d < 0:
            d += nx_global
        elif d >= nx_global:
            d -= nx_global
    if d < 0 or d >= nx_ext:
        return -1
    return d


def _paint_cic(grid, x_index_map, pos, weights, boxsize, nshape, pbc):
    """Reference CIC painter using the kernels' exact index formulas.

    ``x_index_map(ixg)`` maps a (wrapped/clamped) global x cell to a grid row,
    or -1 to skip -- identity for the global reference, the slab mapping for
    the distributed emulation.
    """
    nx, ny, nz = nshape
    for p, wgt in zip(pos, weights):
        gx, gy, gz = p[0] * nx / boxsize[0], p[1] * ny / boxsize[1], p[2] * nz / boxsize[2]
        ix0, iy0, iz0 = int(np.floor(gx)), int(np.floor(gy)), int(np.floor(gz))
        dx, dy, dz = gx - ix0, gy - iy0, gz - iz0
        for l in range(2):
            ixg = (ix0 + l) % nx if pbc else min(max(ix0 + l, 0), nx - 1)
            row = x_index_map(ixg)
            if row < 0:
                continue
            wx = 1.0 - dx if l == 0 else dx
            for m in range(2):
                iy = (iy0 + m) % ny if pbc else min(max(iy0 + m, 0), ny - 1)
                wy = 1.0 - dy if m == 0 else dy
                for n in range(2):
                    iz = (iz0 + n) % nz if pbc else min(max(iz0 + n, 0), nz - 1)
                    wz = 1.0 - dz if n == 0 else dz
                    grid[row, iy, iz] += wgt * wx * wy * wz


class TestHaloExchange:
    @pytest.mark.parametrize("pbc", [True, False])
    @pytest.mark.parametrize("world_size", [2, 4])
    def test_distributed_cic_paint_matches_global(self, world_size, pbc):
        """Emulated slab painting + halo fold == global reference paint."""
        shape = (8, 4, 4)
        boxsize = np.array([100.0, 50.0, 50.0])
        rng = np.random.default_rng(4)
        npart = 200
        pos = rng.uniform(0, 1, size=(npart, 3)) * boxsize
        weights = rng.uniform(0.5, 1.5, size=npart)
        w = HALO_WIDTH["CIC"]

        ref = np.zeros(shape)
        _paint_cic(ref, lambda ixg: ixg, pos, weights, boxsize, shape, pbc)

        envs = _loopback_envs(world_size)

        def fn(env):
            d = env.decomp(shape)
            mask = d.owned_mask(pos[:, 0], boxsize[0], pbc=pbc)
            ext = np.zeros((d.nx_local + 2 * w,) + shape[1:])
            x_start = d.x_offset - w
            _paint_cic(ext,
                       lambda ixg: _slab_index(ixg, x_start, shape[0], ext.shape[0], pbc),
                       pos[mask], weights[mask], boxsize, shape, pbc)
            return halo_exchange_add(ext, env, w, pbc=pbc).copy()

        parts = _run_ranks(envs, fn)
        got = np.concatenate(parts, axis=0)
        np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-12)
        # mass conservation
        expected_mass = weights.sum() if pbc else ref.sum()
        np.testing.assert_allclose(got.sum(), expected_mass if pbc else ref.sum(), rtol=1e-12)

    @pytest.mark.parametrize("pbc", [True, False])
    def test_halo_exchange_copy_fills_neighbour_planes(self, pbc):
        shape = (12, 3, 3)
        world_size, w = 3, 2
        full = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
        envs = _loopback_envs(world_size)

        def fn(env):
            d = env.decomp(shape)
            owned = full[d.x_offset:d.x_offset + d.nx_local]
            ext = np.full((d.nx_local + 2 * w,) + shape[1:], -1.0)
            ext[w:-w] = owned
            halo_exchange_copy(ext, env, w, pbc=pbc)
            return ext

        parts = _run_ranks(envs, fn)
        for r, ext in enumerate(parts):
            d = envs[r].decomp(shape)
            lo = [(d.x_offset - w + i) % shape[0] for i in range(w)]
            hi = [(d.x_offset + d.nx_local + i) % shape[0] for i in range(w)]
            expect_lo = full[lo]
            expect_hi = full[hi]
            if not pbc and r == 0:
                expect_lo = np.zeros_like(expect_lo)
            if not pbc and r == world_size - 1:
                expect_hi = np.zeros_like(expect_hi)
            np.testing.assert_array_equal(ext[:w], expect_lo)
            np.testing.assert_array_equal(ext[-w:], expect_hi)
            np.testing.assert_array_equal(ext[w:-w], full[d.x_offset:d.x_offset + d.nx_local])

    def test_halo_add_serial_pbc_fold(self):
        env = DistEnv.serial()
        ext = np.zeros((6, 2, 2))
        ext[0] = 1.0   # low halo -> high owned edge
        ext[-1] = 2.0  # high halo -> low owned edge
        ext[1:-1] = 10.0
        owned = halo_exchange_add(ext, env, 1, pbc=True)
        assert owned.shape == (4, 2, 2)
        np.testing.assert_array_equal(owned[0], np.full((2, 2), 12.0))
        np.testing.assert_array_equal(owned[-1], np.full((2, 2), 11.0))


# ---------------------------------------------------------------------------
# Distributed Gaussian smoothing (loopback, numpy) vs the serial smoother
# ---------------------------------------------------------------------------
class TestDistributedSmoothing:
    @pytest.mark.parametrize("world_size", [2, 4])
    def test_matches_serial_smoothing(self, world_size):
        from baorecon.field_ops import smoothed_field
        from baorecon.mesh.mesh import Mesh

        shape = (8, 8, 8)
        mesh = Mesh(nmesh=8, boxsize=100.0, boxcentre=np.zeros(3))
        rng = np.random.default_rng(5)
        field = rng.standard_normal(shape).astype(np.float32)
        ref = smoothed_field(field.copy(), mesh, smoothing_radius=10.0)
        envs = _loopback_envs(world_size)

        def fn(env):
            d = env.decomp(shape)
            dfft = DistributedFFT(env, shape, xp=np, fft=np.fft)
            slab = np.ascontiguousarray(field[d.x_offset:d.x_offset + d.nx_local])
            return smoothed_field(slab, mesh, smoothing_radius=10.0, dist_fft=dfft)

        parts = _run_ranks(envs, fn)
        got = np.concatenate(parts, axis=0)
        np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-5)


# ---------------------------------------------------------------------------
# Reduction helpers (loopback)
# ---------------------------------------------------------------------------
class TestReductions:
    def test_allreduce_inplace_reassembles_disjoint_rows(self):
        n, world_size = 10, 2
        full = np.arange(n * 3, dtype=np.float32).reshape(n, 3)

        def fn(rank, comm):
            env = DistEnv(rank=rank, world_size=world_size, comm=comm)
            buf = np.zeros_like(full)
            mine = slice(rank, n, world_size)   # disjoint rows
            buf[mine] = full[mine]
            env.allreduce_inplace(buf)
            return buf

        for out in run_loopback(world_size, fn):
            np.testing.assert_array_equal(out, full)

    def test_gather_x_slabs(self):
        full = np.arange(24, dtype=np.float32).reshape(8, 3)

        def fn(rank, comm):
            env = DistEnv(rank=rank, world_size=2, comm=comm)
            return env.gather_x_slabs(full[rank * 4:(rank + 1) * 4])

        r = run_loopback(2, fn)
        np.testing.assert_array_equal(r[0], full)
        assert r[1] is None

    def test_serial_env_passthrough(self):
        env = DistEnv.serial()
        a = np.ones(3)
        assert env.allreduce_sum(2.5) == 2.5
        assert env.gather_x_slabs(a) is a
        np.testing.assert_array_equal(env.allreduce_inplace(a), np.ones(3))


class TestMultiDeviceComm:
    def test_device_ops_override_loopback(self):
        """NCCL device collectives must win over the loopback host copies."""
        from baorecon.utils.distributed import (
            LoopbackComm,
            MultiDeviceComm,
            _NcclDeviceOps,
        )

        assert MultiDeviceComm.alltoall is _NcclDeviceOps.alltoall
        assert MultiDeviceComm.exchange_halos is _NcclDeviceOps.exchange_halos
        # host-side collectives stay shared-memory (thread hub)
        assert MultiDeviceComm.allreduce_sum is LoopbackComm.allreduce_sum
        assert MultiDeviceComm.allreduce_inplace is LoopbackComm.allreduce_inplace
        assert MultiDeviceComm.gather_slabs is LoopbackComm.gather_slabs
        assert MultiDeviceComm.barrier is LoopbackComm.barrier

    def test_run_multi_gpu_requires_cuda(self):
        from baorecon.utils.backend import CUPY_AVAILABLE
        from baorecon.utils.distributed import run_multi_gpu

        if not CUPY_AVAILABLE:
            with pytest.raises(RuntimeError, match="CuPy"):
                run_multi_gpu(lambda env: None)
