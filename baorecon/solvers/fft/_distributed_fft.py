"""Slab-decomposed 3-D real FFT (local FFTs + one AllToAll transpose).

Real space is split in contiguous blocks along x (axis 0); k-space is split in
contiguous blocks along ky (axis 1); kz (axis 2) is the reduced rfft axis and
stays global, as does kx. One regular AllToAll per transform re-slabs the data:

* forward:  (Nx_loc, Ny, Nz) real --rfftn(y,z)--> (Nx_loc, Ny, Nzh)
            --AllToAll transpose--> (Nx, Ny_loc, Nzh) --fft(x)--> ky-split k-grid
* inverse:  exactly the three steps reversed.

The pack/unpack layout is the *block* distribution on both sides
(``ix_global = src_rank * Nx_loc + ix_local``; likewise for ky). Getting this
right is subtle -- the original migration plan mixed a block pack with a cyclic
unpack (see docs/multigpu_migration_audit.md, E1) -- so the pack/unpack steps
live in pure array-module-agnostic functions that are unit-tested on CPU
against ``numpy.fft`` via the loopback communicator.

At ``world_size == 1`` every call delegates directly to the underlying FFT
module (``cupy.fft`` on the GPU), so the single-device path is bit-identical
to the non-distributed code.
"""

import numpy as np

from baorecon.utils.backend import CUPY_AVAILABLE

if CUPY_AVAILABLE:
    import cupy as cp


def _array_module(a):
    if CUPY_AVAILABLE and isinstance(a, cp.ndarray):
        return cp
    return np


def pack_forward(ayz, world_size):
    """Pack the y/z-transformed slab for the x->ky AllToAll.

    ``ayz``: (Nx_loc, Ny, Nzh). Block ``p`` of the returned flat buffer holds
    this rank's x-slab of destination rank ``p``'s ky-block.
    """
    xp = _array_module(ayz)
    nx_loc, ny, nzh = ayz.shape
    ny_loc = ny // world_size
    blocks = ayz.reshape(nx_loc, world_size, ny_loc, nzh).transpose(1, 0, 2, 3)
    return xp.ascontiguousarray(blocks).ravel()

def unpack_forward(recv, world_size, nx_local, ny_local, nz_half):
    """Unpack the x->ky AllToAll receive buffer to (Nx, Ny_loc, Nzh).

    Received block ``p`` is source rank ``p``'s x-slab (a contiguous x block),
    so global x order is just the blocks concatenated -- no transpose.
    """
    return recv.reshape(world_size * nx_local, ny_local, nz_half)

def pack_inverse(bx, world_size):
    """Pack the x-transformed k-grid for the ky->x AllToAll.

    ``bx``: (Nx, Ny_loc, Nzh). Destination rank ``p`` owns the contiguous x
    block ``p`` -- already contiguous along axis 0, no transpose.
    """
    xp = _array_module(bx)
    nx, ny_loc, nzh = bx.shape
    nx_loc = nx // world_size
    return xp.ascontiguousarray(bx.reshape(world_size, nx_loc, ny_loc, nzh)).ravel()

def unpack_inverse(recv, world_size, nx_local, ny_local, nz_half):
    """Unpack the ky->x AllToAll receive buffer to (Nx_loc, Ny, Nzh).

    Received block ``p`` is source rank ``p``'s ky-block of *my* x-slab;
    reassembling the global ky axis interleaves the blocks (transpose + copy).
    """
    xp = _array_module(recv)
    blocks = recv.reshape(world_size, nx_local, ny_local, nz_half).transpose(1, 0, 2, 3)
    return xp.ascontiguousarray(blocks).reshape(nx_local, world_size * ny_local, nz_half)


class DistributedFFT:
    """Drop-in ``rfftn``/``irfftn`` provider over a slab decomposition.

    Exposes the (subset of the) ``cupy.fft`` call signatures the solvers and
    ``smoothed_field`` use, so it can replace the ``fft`` module attribute of
    the backend without touching the iterative solver loop. ``xp``/``fft`` are
    injectable (numpy + any fft module) for CPU testing with a loopback
    communicator; in production they are CuPy and ``cupy.fft``.
    """

    _FULL_AXES = (0, 1, 2)

    def __init__(self, env, global_shape, xp=None, fft=None):
        self.env = env
        self.decomp = env.decomp(global_shape)
        if xp is None:
            if not CUPY_AVAILABLE:
                raise RuntimeError("DistributedFFT defaults require CuPy; "
                                   "pass xp/fft explicitly for CPU use.")
            xp = cp
        self.xp = xp
        self.fft = fft if fft is not None else xp.fft

    def _check_axes(self, axes):
        if axes is not None and tuple(axes) != self._FULL_AXES:
            raise ValueError(f"DistributedFFT only supports full 3-D transforms, got axes={axes}")

    def rfftn(self, a, axes=None):
        """(Nx_loc, Ny, Nz) real -> (Nx, Ny_loc, Nzh) complex (ky-split)."""
        self._check_axes(axes)
        if not self.env.is_distributed:
            return self.fft.rfftn(a)
        d, p = self.decomp, self.env.world_size
        if tuple(a.shape) != d.local_real_shape:
            raise ValueError(f"expected local real slab {d.local_real_shape}, got {a.shape}")
        ayz = self.fft.rfftn(a, axes=(1, 2))
        send = pack_forward(ayz, p)
        del ayz
        recv = self.xp.empty_like(send)
        self.env.comm.alltoall(send, recv)
        del send
        b = unpack_forward(recv, p, d.nx_local, d.ny_local, d.nz_half)
        return self.fft.fft(b, axis=0)

    def irfftn(self, a, s=None, axes=None):
        """(Nx, Ny_loc, Nzh) complex (ky-split) -> (Nx_loc, Ny, Nz) real."""
        self._check_axes(axes)
        if not self.env.is_distributed:
            return self.fft.irfftn(a, s=s)
        d, p = self.decomp, self.env.world_size
        if tuple(a.shape) != d.local_k_shape:
            raise ValueError(f"expected local k-grid {d.local_k_shape}, got {a.shape}")
        if s is not None and tuple(s) != d.local_real_shape:
            raise ValueError(f"expected local real shape {d.local_real_shape}, got s={tuple(s)}")
        _, ny, nz = d.local_real_shape
        bx = self.fft.ifft(a, axis=0)
        send = pack_inverse(bx, p)
        del bx
        recv = self.xp.empty_like(send)
        self.env.comm.alltoall(send, recv)
        del send
        ayz = unpack_inverse(recv, p, d.nx_local, d.ny_local, d.nz_half)
        del recv
        return self.fft.irfftn(ayz, axes=(1, 2), s=(ny, nz))
