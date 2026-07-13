"""Single-node multi-GPU distribution utilities (slab decomposition).

This module holds the pieces that are shared by every distributed component:

* :class:`SlabDecomp` -- the 1-D (slab) domain-decomposition descriptor. Real
  space is split in contiguous blocks along x (axis 0); after the distributed
  forward rFFT, k-space is split along ky (axis 1). Pure geometry, no arrays.
* :class:`DistEnv` -- the per-process distribution context: rank, world size,
  the communicator, and the bound GPU. ``DistEnv.serial()`` is the ``P = 1``
  no-communication special case used by the unchanged single-device path.
* :class:`NcclComm` -- the real transport: NCCL (via ``cupy.cuda.nccl``) for
  device-buffer collectives, mpi4py for host-side scalars/objects.
* :class:`LoopbackComm` / :func:`run_loopback` -- an in-process, thread-based
  communicator with the same interface, so all decomposition/transpose/halo
  logic is unit-testable with numpy on hosts without GPUs or MPI.

Communicator interface (duck-typed; all buffers must be C-contiguous and
equally sized across ranks):

* ``alltoall(send, recv)`` -- regular all-to-all of ``world_size`` equal
  blocks of the flat ``send`` buffer.
* ``exchange_halos(to_left, to_right, from_left, from_right)`` -- ring
  exchange with the x-neighbours: ``to_left`` is delivered to rank-1's
  ``from_right`` and ``to_right`` to rank+1's ``from_left`` (periodic ring;
  the *caller* decides whether to use the wrapped edges when ``pbc=False``).
* ``allreduce_sum(x)`` -- global sum of a host scalar.
* ``barrier()``.
"""

import threading
from dataclasses import dataclass

import numpy as np

from baorecon.utils.backend import CUPY_AVAILABLE
from baorecon.utils.loggers import setup_logger

logger = setup_logger(__name__)

if CUPY_AVAILABLE:
    import cupy as cp


# ---------------------------------------------------------------------------
# Decomposition descriptor
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SlabDecomp:
    """Slab (1-D) decomposition of an ``(Nx, Ny, Nz)`` mesh over ``world_size`` ranks.

    Real space: rank ``r`` owns the contiguous x-block
    ``[r * nx_local, (r+1) * nx_local)``. k-space (after the distributed rfftn):
    rank ``r`` owns the contiguous ky-block ``[r * ny_local, (r+1) * ny_local)``;
    kx and kz stay global on every rank (kz is the reduced rfft axis).

    Divisibility (``Nx % P == 0`` and ``Ny % P == 0``) is required so the FFT
    transpose is a regular (equal-block) AllToAll; generalizing to an
    AllToAllv is future work (see the migration plan/audit).
    """

    global_shape: tuple
    world_size: int
    rank: int

    def __post_init__(self):
        nx, ny, nz = (int(n) for n in self.global_shape)
        object.__setattr__(self, "global_shape", (nx, ny, nz))
        p = self.world_size
        if not (0 <= self.rank < p):
            raise ValueError(f"rank {self.rank} out of range for world_size {p}")
        if nx % p or ny % p:
            raise ValueError(
                f"Slab decomposition requires Nx and Ny divisible by the number of "
                f"ranks: got Nx={nx}, Ny={ny}, P={p}. Choose nmesh accordingly "
                f"(remainder/AllToAllv support is future work)."
            )
        if nx < p:
            raise ValueError(f"more ranks ({p}) than x-planes ({nx})")

    # --- real space (x-split) ---
    @property
    def nx_local(self) -> int:
        return self.global_shape[0] // self.world_size

    @property
    def x_offset(self) -> int:
        return self.rank * self.nx_local

    @property
    def local_real_shape(self) -> tuple:
        return (self.nx_local, self.global_shape[1], self.global_shape[2])

    # --- k space (ky-split) ---
    @property
    def ny_local(self) -> int:
        return self.global_shape[1] // self.world_size

    @property
    def ky_offset(self) -> int:
        return self.rank * self.ny_local

    @property
    def nz_half(self) -> int:
        return self.global_shape[2] // 2 + 1

    @property
    def local_k_shape(self) -> tuple:
        return (self.global_shape[0], self.ny_local, self.nz_half)

    @property
    def owns_dc(self) -> bool:
        """True on the rank whose ky-block contains ky = 0 (rank 0)."""
        return self.rank == 0

    # --- neighbours (periodic ring along x) ---
    @property
    def left_rank(self) -> int:
        return (self.rank - 1) % self.world_size

    @property
    def right_rank(self) -> int:
        return (self.rank + 1) % self.world_size

    @property
    def is_low_edge(self) -> bool:
        return self.rank == 0

    @property
    def is_high_edge(self) -> bool:
        return self.rank == self.world_size - 1

    # --- particle ownership ---
    def owner_ranks(self, pos_x, boxsize_x, pbc=True) -> np.ndarray:
        """Owner rank of each particle from its box-frame x coordinate.

        Ownership follows the paint/read kernels' cell mapping
        ``gx = pos_x * Nx / boxsize_x``: the owner is the rank whose x-block
        contains ``floor(gx)`` (wrapped when ``pbc``, clamped otherwise), so a
        particle's CIC/TSC stencil always fits in its owner's halo-extended grid.
        """
        nx = self.global_shape[0]
        gx = np.floor(np.asarray(pos_x, dtype=np.float64) * (nx / float(boxsize_x)))
        gx = gx.astype(np.int64)
        if pbc:
            gx %= nx
        else:
            np.clip(gx, 0, nx - 1, out=gx)
        return gx // self.nx_local

    def owned_mask(self, pos_x, boxsize_x, pbc=True) -> np.ndarray:
        """Boolean mask of the particles owned by this rank."""
        return self.owner_ranks(pos_x, boxsize_x, pbc=pbc) == self.rank

    def ky_slice(self, ky_full):
        """This rank's ky block of a full 1-D ky array."""
        return ky_full[self.ky_offset:self.ky_offset + self.ny_local]


# ---------------------------------------------------------------------------
# NCCL communicators (the real transports)
# ---------------------------------------------------------------------------
class _NcclDeviceOps:
    """Device-buffer collectives shared by the NCCL-backed communicators.

    Requires ``self._nccl_mod`` (the ``cupy.cuda.nccl`` module), ``self._nccl``
    (this endpoint's ``NcclCommunicator``), ``self.rank`` and
    ``self.world_size``. All device buffers must be C-contiguous CuPy arrays
    of float32 or complex64 (complex buffers travel as float32 pairs, which
    is exact).
    """

    @staticmethod
    def _as_flat_f32(a):
        """Flat float32 view of a contiguous float32/complex64 device buffer."""
        if not a.flags.c_contiguous:
            raise ValueError("NCCL buffers must be C-contiguous")
        if a.dtype == cp.complex64:
            a = a.view(cp.float32)
        elif a.dtype != cp.float32:
            raise TypeError(f"unsupported NCCL buffer dtype {a.dtype}")
        return a.ravel()

    def _stream_ptr(self):
        return cp.cuda.get_current_stream().ptr

    def alltoall(self, send, recv):
        nccl = self._nccl_mod
        send = self._as_flat_f32(send)
        recv = self._as_flat_f32(recv)
        if send.size != recv.size or send.size % self.world_size:
            raise ValueError("alltoall buffers must be equal-size, divisible by P")
        cnt = send.size // self.world_size
        stream = self._stream_ptr()
        nccl.groupStart()
        for peer in range(self.world_size):
            self._nccl.send(send[peer * cnt:].data.ptr, cnt,
                            nccl.NCCL_FLOAT32, peer, stream)
            self._nccl.recv(recv[peer * cnt:].data.ptr, cnt,
                            nccl.NCCL_FLOAT32, peer, stream)
        nccl.groupEnd()

    def exchange_halos(self, to_left, to_right, from_left, from_right):
        nccl = self._nccl_mod
        to_left, to_right = self._as_flat_f32(to_left), self._as_flat_f32(to_right)
        from_left, from_right = self._as_flat_f32(from_left), self._as_flat_f32(from_right)
        lo = (self.rank - 1) % self.world_size
        hi = (self.rank + 1) % self.world_size
        stream = self._stream_ptr()
        nccl.groupStart()
        self._nccl.send(to_right.data.ptr, to_right.size, nccl.NCCL_FLOAT32, hi, stream)
        self._nccl.send(to_left.data.ptr, to_left.size, nccl.NCCL_FLOAT32, lo, stream)
        self._nccl.recv(from_left.data.ptr, from_left.size, nccl.NCCL_FLOAT32, lo, stream)
        self._nccl.recv(from_right.data.ptr, from_right.size, nccl.NCCL_FLOAT32, hi, stream)
        nccl.groupEnd()


class NcclComm(_NcclDeviceOps):
    """Multi-process transport: NCCL device collectives, mpi4py host scalars.

    One endpoint per MPI rank (``mpirun -np P``). The CUDA device must already
    be bound (``cp.cuda.Device(id).use()``) before construction.
    """

    def __init__(self, mpi_comm):
        from cupy.cuda import nccl  # noqa: PLC0415 -- optional dependency

        self._nccl_mod = nccl
        self.mpi = mpi_comm
        self.rank = mpi_comm.Get_rank()
        self.world_size = mpi_comm.Get_size()
        uid = nccl.get_unique_id() if self.rank == 0 else None
        uid = mpi_comm.bcast(uid, root=0)
        self._nccl = nccl.NcclCommunicator(self.world_size, uid, self.rank)

    def allreduce_sum(self, x):
        return self.mpi.allreduce(float(x))

    def allreduce_inplace(self, host_array):
        """Element-wise global sum of a host numpy array, in place."""
        from mpi4py import MPI  # noqa: PLC0415

        self.mpi.Allreduce(MPI.IN_PLACE, host_array, op=MPI.SUM)
        return host_array

    def gather_slabs(self, host_array):
        """Gather equal-shaped host x-slabs to rank 0 (None on other ranks)."""
        parts = self.mpi.gather(host_array, root=0)
        if self.rank != 0:
            return None
        return np.concatenate(parts, axis=0)

    def barrier(self):
        self.mpi.Barrier()


# ---------------------------------------------------------------------------
# In-process loopback communicator (tests / CPU simulation)
# ---------------------------------------------------------------------------
class _LoopbackHub:
    """Shared state for the P endpoints of a LoopbackComm world."""

    def __init__(self, world_size):
        self.world_size = world_size
        self.barrier = threading.Barrier(world_size)
        self.posted = [None] * world_size
        self.result = None
        self.lock = threading.Lock()


class LoopbackComm:
    """Thread-based fake communicator with the NcclComm interface.

    ``LoopbackComm.world(P)`` returns the P endpoints; each must be driven
    from its own thread (see :func:`run_loopback`). Works with numpy or cupy
    arrays -- collectives are plain array copies under barriers -- so the
    distributed FFT/halo logic can be verified on a CPU-only host.
    """

    def __init__(self, hub, rank):
        self._hub = hub
        self.rank = rank
        self.world_size = hub.world_size

    @classmethod
    def world(cls, world_size):
        hub = _LoopbackHub(world_size)
        return [cls(hub, r) for r in range(world_size)]

    def _post_and_sync(self, payload):
        self._hub.posted[self.rank] = payload
        self._hub.barrier.wait()          # all payloads visible

    def _done(self):
        self._hub.barrier.wait()          # all reads done; safe to repost

    def alltoall(self, send, recv):
        send = send.ravel()
        recv = recv.ravel()
        if send.size != recv.size or send.size % self.world_size:
            raise ValueError("alltoall buffers must be equal-size, divisible by P")
        self._post_and_sync(send)
        cnt = send.size // self.world_size
        for peer in range(self.world_size):
            block = self._hub.posted[peer][self.rank * cnt:(self.rank + 1) * cnt]
            recv[peer * cnt:(peer + 1) * cnt] = block
        self._done()

    def exchange_halos(self, to_left, to_right, from_left, from_right):
        self._post_and_sync((to_left, to_right))
        lo = (self.rank - 1) % self.world_size
        hi = (self.rank + 1) % self.world_size
        from_left[...] = self._hub.posted[lo][1]   # left neighbour's to_right
        from_right[...] = self._hub.posted[hi][0]  # right neighbour's to_left
        self._done()

    def allreduce_sum(self, x):
        self._post_and_sync(float(x))
        total = sum(self._hub.posted)
        self._done()
        return total

    def allreduce_inplace(self, host_array):
        # The sum is computed ONCE (rank 0) instead of once per rank: with
        # full-catalogue (N, 3) shift buffers this is the difference between
        # O(N*P) and O(N*P^2) host traffic -- the dominant host cost of the
        # single-process launcher on large random catalogues.
        self._post_and_sync(host_array)          # post the live buffers
        if self.rank == 0:
            total = self._hub.posted[0].copy()
            for peer in range(1, self.world_size):
                total += self._hub.posted[peer]
            self._hub.result = total
        self._hub.barrier.wait()                 # result ready
        host_array[...] = self._hub.result
        self._done()                             # all read; safe to reuse
        return host_array

    def gather_slabs(self, host_array):
        self._post_and_sync(host_array)
        result = None
        if self.rank == 0:
            xp = _xp_of(host_array)
            result = xp.concatenate(self._hub.posted, axis=0)
        self._done()
        return result

    def barrier(self):
        self._hub.barrier.wait()


def run_loopback(world_size, fn):
    """Run ``fn(rank, comm)`` on ``world_size`` loopback ranks (one thread each).

    Returns the per-rank results in rank order; re-raises the first exception.
    """
    comms = LoopbackComm.world(world_size)
    results = [None] * world_size
    errors = [None] * world_size

    def _target(r):
        try:
            results[r] = fn(r, comms[r])
        except BaseException as exc:  # noqa: BLE001 -- reported to the caller
            errors[r] = exc
            # release peers stuck on the barrier
            comms[r]._hub.barrier.abort()

    threads = [threading.Thread(target=_target, args=(r,)) for r in range(world_size)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for exc in errors:
        if exc is not None and not isinstance(exc, threading.BrokenBarrierError):
            raise exc
    for exc in errors:
        if exc is not None:
            raise exc
    return results


# ---------------------------------------------------------------------------
# Single-process multi-GPU (notebook / interactive) communicator
# ---------------------------------------------------------------------------
class MultiDeviceComm(_NcclDeviceOps, LoopbackComm):
    """One process, P GPUs: NCCL ``initAll`` device collectives, shared-memory
    host reductions.

    The notebook-friendly counterpart of :class:`NcclComm` (no ``mpirun``, no
    mpi4py): every endpoint runs on its own *thread* of the same process --
    the canonical NCCL one-thread-per-device pattern -- so device buffers move
    over NCCL exactly as in the MPI path, while host-side collectives
    (scalar sums, in-place array reductions, slab gathers on *numpy* arrays)
    are plain shared-memory operations inherited from :class:`LoopbackComm`.
    Build the endpoints with :meth:`world` and drive them with
    :func:`run_multi_gpu`.
    """

    def __init__(self, hub, rank, nccl_comm, device_id):
        super().__init__(hub, rank)
        from cupy.cuda import nccl  # noqa: PLC0415 -- optional dependency

        self._nccl_mod = nccl
        self._nccl = nccl_comm
        self.device_id = device_id

    @classmethod
    def world(cls, devices):
        """Endpoints (one per CUDA device id in ``devices``) sharing one clique."""
        from cupy.cuda import nccl  # noqa: PLC0415

        devices = [int(d) for d in devices]
        comms = nccl.NcclCommunicator.initAll(devices)
        hub = _LoopbackHub(len(devices))
        return [cls(hub, r, comms[r], devices[r]) for r in range(len(devices))]


def run_multi_gpu(fn, devices=None):
    """Run ``fn(env)`` once per GPU of this process -- no ``mpirun`` needed.

    The interactive/notebook launcher: one thread per device executes the same
    SPMD code the MPI path runs, with a :class:`DistEnv` whose communicator is
    a :class:`MultiDeviceComm` endpoint. Each thread binds its device for both
    CUDA runtimes (CuPy and numba.cuda) before calling ``fn``. Returns the
    per-rank results in rank order; with one visible device it simply calls
    ``fn(DistEnv.serial())``.

    Example (identical results on every rank)::

        from baorecon.utils.distributed import run_multi_gpu

        def job(env):
            rec = BAOReconstructor(..., device="gpu", solver_type="ifft", dist=env)
            return rec.run_reconstruction()

        data_rec, rand_rec = run_multi_gpu(job)[0]

    Notes: the Python-side orchestration shares the GIL (GPU work is
    asynchronous, so the impact is small at P <= 8, but ``mpirun`` remains the
    recommended launcher for large production runs), and the usual slab
    constraints apply (``Nx % P == 0``, ``Ny % P == 0``).
    """
    if not CUPY_AVAILABLE:
        raise RuntimeError("run_multi_gpu requires CuPy + CUDA.")
    if devices is None:
        devices = list(range(cp.cuda.runtime.getDeviceCount()))
    if len(devices) == 0:
        raise RuntimeError("no CUDA devices visible")
    if len(devices) == 1:
        cp.cuda.Device(int(devices[0])).use()
        return [fn(DistEnv.serial())]

    comms = MultiDeviceComm.world(devices)
    envs = [DistEnv(rank=r, world_size=len(devices), comm=comms[r],
                    device_id=comms[r].device_id) for r in range(len(devices))]
    results = [None] * len(devices)
    errors = [None] * len(devices)

    def _target(r):
        try:
            cp.cuda.Device(envs[r].device_id).use()
            from numba import cuda as _numba_cuda  # noqa: PLC0415

            _numba_cuda.select_device(envs[r].device_id)
            results[r] = fn(envs[r])
        except BaseException as exc:  # noqa: BLE001 -- reported to the caller
            errors[r] = exc
            comms[r]._hub.barrier.abort()

    threads = [threading.Thread(target=_target, args=(r,), name=f"baorecon-gpu-{r}")
               for r in range(len(devices))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for exc in errors:
        if exc is not None and not isinstance(exc, threading.BrokenBarrierError):
            raise exc
    for exc in errors:
        if exc is not None:
            raise exc
    return results


# ---------------------------------------------------------------------------
# Ghost-zone (halo) exchanges along the x-split
# ---------------------------------------------------------------------------
def _xp_of(a):
    if CUPY_AVAILABLE and isinstance(a, cp.ndarray):
        return cp
    return np


def halo_exchange_add(mesh_ext, env, w, pbc=True):
    """Fold the ±w halo planes of an extended slab into their owners (add).

    ``mesh_ext`` has shape ``(nx_local + 2w, Ny, Nz[, ...])``: planes ``[:w]``
    hold contributions to the left neighbour's last ``w`` owned planes, planes
    ``[-w:]`` to the right neighbour's first ``w``. Mass-conserving: every halo
    plane is *added* into the owning rank's edge. With ``pbc`` the ring wraps;
    otherwise the outer edges are skipped (the paint kernels clamp at the
    global boundary, so edge halos are empty). Returns the owned
    ``(nx_local, ...)`` block (a contiguous view of ``mesh_ext``).
    """
    if w == 0:
        return mesh_ext
    xp = _xp_of(mesh_ext)
    owned = mesh_ext[w:-w]
    if not env.is_distributed:
        if pbc:
            owned[:w] += mesh_ext[-w:]
            owned[-w:] += mesh_ext[:w]
        return owned

    to_left = xp.ascontiguousarray(mesh_ext[:w])
    to_right = xp.ascontiguousarray(mesh_ext[-w:])
    from_left = xp.empty_like(to_left)
    from_right = xp.empty_like(to_right)
    env.comm.exchange_halos(to_left, to_right, from_left, from_right)
    if pbc or env.rank != 0:
        owned[:w] += from_left
    if pbc or env.rank != env.world_size - 1:
        owned[-w:] += from_right
    return owned


def halo_exchange_copy(field_ext, env, w, pbc=True):
    """Fill the ±w halo planes of an extended slab from the neighbours (copy).

    The low halo receives the left neighbour's last ``w`` owned planes and the
    high halo the right neighbour's first ``w`` (periodic ring). Used before
    grid->particle read-back so interpolation stencils can cross the slab
    boundary. With ``pbc=False`` the outer edge halos are zeroed instead (the
    read kernels clamp there and never dereference them).
    """
    if w == 0:
        return field_ext
    xp = _xp_of(field_ext)
    owned = field_ext[w:-w]
    if not env.is_distributed:
        if pbc:
            field_ext[:w] = owned[-w:]
            field_ext[-w:] = owned[:w]
        else:
            field_ext[:w] = 0
            field_ext[-w:] = 0
        return field_ext

    to_left = xp.ascontiguousarray(owned[:w])     # becomes the left neighbour's high halo
    to_right = xp.ascontiguousarray(owned[-w:])   # becomes the right neighbour's low halo
    from_left = xp.empty_like(to_left)
    from_right = xp.empty_like(to_right)
    env.comm.exchange_halos(to_left, to_right, from_left, from_right)
    if pbc or env.rank != 0:
        field_ext[:w] = from_left
    else:
        field_ext[:w] = 0
    if pbc or env.rank != env.world_size - 1:
        field_ext[-w:] = from_right
    else:
        field_ext[-w:] = 0
    return field_ext


# ---------------------------------------------------------------------------
# Distribution environment
# ---------------------------------------------------------------------------
@dataclass
class DistEnv:
    """Per-process distribution context.

    ``DistEnv.serial()`` (the default everywhere) is the single-device case:
    ``world_size == 1``, no communicator, and every component takes its
    existing, unchanged code path.
    """

    rank: int = 0
    world_size: int = 1
    comm: object = None
    device_id: int = 0

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    @classmethod
    def serial(cls) -> "DistEnv":
        return cls()

    @classmethod
    def from_mpi(cls, mpi_comm=None) -> "DistEnv":
        """Build a DistEnv from an mpi4py communicator, binding one GPU per rank.

        Binds device ``rank % n_gpus`` for BOTH CUDA runtimes in play (CuPy for
        FFT/field ops, numba.cuda for the MAS kernels) *before* creating the
        NCCL communicator. With a single MPI rank this degrades to
        ``DistEnv.serial()`` (no NCCL, no communication).
        """
        if mpi_comm is None:
            from mpi4py import MPI  # noqa: PLC0415 -- optional dependency

            mpi_comm = MPI.COMM_WORLD
        world_size = mpi_comm.Get_size()
        if world_size == 1:
            return cls.serial()
        if not CUPY_AVAILABLE:
            raise RuntimeError("Distributed (multi-GPU) mode requires CuPy + CUDA.")

        rank = mpi_comm.Get_rank()
        n_dev = cp.cuda.runtime.getDeviceCount()
        device_id = rank % n_dev
        cp.cuda.Device(device_id).use()
        from numba import cuda as _numba_cuda  # noqa: PLC0415

        _numba_cuda.select_device(device_id)
        comm = NcclComm(mpi_comm)
        logger.info(f"DistEnv: rank {rank}/{world_size} bound to GPU {device_id} "
                    f"({n_dev} visible)")
        return cls(rank=rank, world_size=world_size, comm=comm, device_id=device_id)

    def decomp(self, global_shape) -> SlabDecomp:
        return SlabDecomp(tuple(int(n) for n in global_shape), self.world_size, self.rank)

    def allreduce_sum(self, x):
        if not self.is_distributed:
            return x
        return self.comm.allreduce_sum(x)

    def allreduce_inplace(self, host_array):
        if self.is_distributed:
            self.comm.allreduce_inplace(host_array)
        return host_array

    def gather_x_slabs(self, host_array):
        """Reassemble x-slabs on rank 0 (returns None on other ranks; identity at P=1)."""
        if not self.is_distributed:
            return host_array
        return self.comm.gather_slabs(host_array)

    def barrier(self):
        if self.is_distributed:
            self.comm.barrier()


def auto_dist_env() -> DistEnv:
    """Detect the launch context: DistEnv.from_mpi() under ``mpirun -np P>1``,
    DistEnv.serial() otherwise (including when mpi4py is not installed)."""
    try:
        from mpi4py import MPI  # noqa: PLC0415
    except ImportError:
        return DistEnv.serial()
    if MPI.COMM_WORLD.Get_size() == 1:
        return DistEnv.serial()
    return DistEnv.from_mpi(MPI.COMM_WORLD)
