"""Mass-assignment interface.

Validates inputs, allocates the output grid, and dispatches to the CPU or GPU
kernels. The kernels themselves perform no validation and never allocate the
output grid.

Precision is device-dependent. On the CPU the interface is type-neutral: the
grid, positions and weights follow the mesh's working precision
(``mesh.dtype``), so a float64 mesh is honoured end-to-end. The GPU kernels
operate in float32 only, so a ``device='gpu'`` call always works in (and
returns) float32 regardless of ``mesh.dtype``.
"""

import warnings

import numpy as np

from baorecon.mas import cpu as _cpu
from baorecon.utils.loggers import setup_logger

logger = setup_logger(__name__)

from baorecon.utils.backend import CUPY_AVAILABLE

if CUPY_AVAILABLE:
    import cupy as cp

    from baorecon.mas import gpu as _gpu

_VALID = ("NGP", "CIC", "TSC")

# Ghost-zone half-widths per scheme for the slab decomposition. CIC (floor
# based) reaches at most +1/-0 planes outside an owned slab; TSC (round based)
# reaches +2 on the high side (round(gx) can equal nx_local for an owned
# particle) and -1 on the low side -- so 2, not the 1 the migration plan
# assumed (audit finding E2). Widths are symmetric for simplicity.
HALO_WIDTH = {"NGP": 1, "CIC": 1, "TSC": 2}


def _validate_positions(pos, weights, boxsize, pbc):
    if pos.shape[1] != 3:
        raise ValueError("pos must have shape (N, 3)")
    if pos.shape[0] != weights.shape[0]:
        raise ValueError("pos and weights must have the same number of particles")
    if (pos < 0).any() or (pos > boxsize).any():
        if pbc:
            warnings.warn("Particles are outside the range [0, boxsize]. PBC is true and will be applied.")
        else:
            raise ValueError(
                f"pos must be in the range [0, boxsize]. Got min {pos.min()} and max {pos.max()}."
            )


def assign(pos, weights, mesh, scheme="CIC", device="cpu", pbc=True, parallel=False,
           dist=None):
    """Paint particles onto a fresh grid using the requested scheme.

    Returns a ``(Nx, Ny, Nz)`` grid. On the CPU the grid dtype follows the
    mesh's working precision (``mesh.dtype``); on the GPU it is always a float32
    CuPy array.

    In distributed mode (``dist`` is a :class:`~baorecon.utils.distributed.DistEnv`
    with ``world_size > 1``; GPU only) ``pos``/``weights`` are the *full*
    catalogue on every rank: this rank's particles are selected by x-slab
    ownership, painted into a halo-extended local slab, and the halos are
    accumulated into their owners over NCCL. The returned grid is this rank's
    owned ``(nx_local, Ny, Nz)`` block.
    """
    scheme = scheme.strip().upper()
    if scheme not in _VALID:
        raise ValueError(f"Invalid scheme '{scheme}'. Valid options are: {_VALID}")

    distributed = dist is not None and dist.is_distributed
    if distributed and device != "gpu":
        raise ValueError("Distributed mass assignment is only implemented for device='gpu'.")

    # GPU kernels are float32-only; on the CPU we stay type-neutral and follow
    # the mesh's working precision so a float64 mesh is not silently downcast.
    work_dtype = np.dtype(np.float32) if device == "gpu" else np.dtype(mesh.dtype)

    pos = np.asarray(pos, dtype=work_dtype)
    if weights is None:
        weights = np.ones(pos.shape[0], dtype=work_dtype)
    else:
        weights = np.asarray(weights, dtype=work_dtype)

    boxsize = np.asarray(mesh.boxsize, dtype=work_dtype)
    _validate_positions(pos, weights, boxsize, pbc)
    grid_shape = mesh.shape

    if device == "gpu":
        if not CUPY_AVAILABLE:
            raise RuntimeError("GPU backend requested, but it is not available (CuPy/CUDA missing).")
        if distributed:
            return _assign_distributed(pos, weights, mesh, scheme, pbc, dist)
        pos_dev = cp.asarray(pos, dtype=cp.float32)
        weights_dev = cp.asarray(weights, dtype=cp.float32)
        mesh_dev = cp.zeros(grid_shape, dtype=cp.float32)
        boxsize_dev = cp.asarray(boxsize)
        if scheme == "CIC":
            _gpu.assign_cic(mesh_dev, pos_dev, weights_dev, boxsize_dev, pbc)
        elif scheme == "TSC":
            _gpu.assign_tsc(mesh_dev, pos_dev, weights_dev, boxsize_dev, pbc)
        else:
            raise ValueError(f"GPU mass assignment for scheme '{scheme}' is not implemented.")
        return mesh_dev

    grid = np.zeros(grid_shape, dtype=work_dtype)
    if scheme == "NGP":
        return _cpu.ngp_assign(pos, boxsize, weights, grid, pbc=pbc)
    if scheme == "CIC":
        if parallel:
            return _cpu.cic_assign_chunks(pos, boxsize, weights, grid, pbc=pbc)
        return _cpu.cic_assign_serial(pos, boxsize, weights, grid, pbc=pbc)
    if scheme == "TSC":
        if parallel:
            return _cpu.tsc_assign_chunks(pos, boxsize, weights, grid, pbc=pbc)
        return _cpu.tsc_assign_serial(pos, boxsize, weights, grid, pbc=pbc)


def _assign_distributed(pos, weights, mesh, scheme, pbc, dist):
    """Paint this rank's particles into a halo-extended x-slab and fold halos.

    ``pos``/``weights`` are the full (host) catalogue, already cast to float32
    by :func:`assign`. Returns the owned ``(nx_local, Ny, Nz)`` float32 CuPy
    block, halo contributions accumulated into their owners (mass-conserving).
    """
    from numba import cuda as _numba_cuda  # noqa: PLC0415

    from baorecon.utils.distributed import halo_exchange_add

    if scheme not in ("CIC", "TSC"):
        raise ValueError(f"Distributed GPU mass assignment for scheme '{scheme}' is not implemented.")
    w = HALO_WIDTH[scheme]
    decomp = dist.decomp(mesh.shape)
    nx, ny, nz = decomp.global_shape
    if decomp.nx_local < w or decomp.nx_local + 2 * w > nx:
        raise ValueError(
            f"nx_local={decomp.nx_local} too small for halo width {w} "
            f"(P={decomp.world_size}, Nx={nx}); use fewer ranks or a finer mesh."
        )

    boxsize = np.asarray(mesh.boxsize, dtype=np.float32)
    mask = decomp.owned_mask(pos[:, 0], boxsize[0], pbc=pbc)
    pos_dev = cp.asarray(pos[mask], dtype=cp.float32)
    weights_dev = cp.asarray(weights[mask], dtype=cp.float32)
    boxsize_dev = cp.asarray(boxsize)

    mesh_ext = cp.zeros((decomp.nx_local + 2 * w, ny, nz), dtype=cp.float32)
    x_start = decomp.x_offset - w
    if scheme == "CIC":
        _gpu.assign_cic_slab(mesh_ext, pos_dev, weights_dev, boxsize_dev, nx, x_start, pbc)
    else:
        _gpu.assign_tsc_slab(mesh_ext, pos_dev, weights_dev, boxsize_dev, nx, x_start, pbc)

    # The paint runs on numba.cuda's stream, the NCCL exchange on CuPy's:
    # synchronize before shipping the halos (audit finding E7).
    _numba_cuda.synchronize()
    owned = halo_exchange_add(mesh_ext, dist, w, pbc=pbc)
    return cp.ascontiguousarray(owned)


def read_field_at(field_ext, pos, mesh, dist, scheme="CIC", pbc=True):
    """Distributed grid->particle read-back of a halo-extended vector field.

    ``field_ext`` is this rank's ``(nx_local + 2w, Ny, Nz, C)`` device slab
    with its halo planes already filled (see
    :func:`baorecon.utils.distributed.halo_exchange_copy`); ``pos`` are
    box-frame positions owned by this rank. Returns an ``(N, C)`` CuPy array.
    """
    scheme = scheme.strip().upper()
    if scheme not in ("CIC", "TSC"):
        raise ValueError(f"Distributed GPU read-out for scheme '{scheme}' is not implemented.")
    decomp = dist.decomp(mesh.shape)
    w = (field_ext.shape[0] - decomp.nx_local) // 2
    x_start = decomp.x_offset - w

    pos_dev = cp.asarray(pos, dtype=cp.float32)
    out_dev = cp.empty((pos_dev.shape[0], field_ext.shape[-1]), dtype=cp.float32)
    boxsize_dev = cp.asarray(np.asarray(mesh.boxsize, dtype=np.float32))
    nx = decomp.global_shape[0]
    if scheme == "CIC":
        _gpu.read_cic_slab(field_ext, pos_dev, boxsize_dev, out_dev, nx, x_start, pbc)
    else:
        _gpu.read_tsc_slab(field_ext, pos_dev, boxsize_dev, out_dev, nx, x_start, pbc)
    return out_dev


def readout(grid, pos, mesh, scheme="CIC", device="cpu", pbc=True):
    """Read out a scalar field defined on ``grid`` at particle positions.

    Returns a ``(N,)`` array of interpolated values (CuPy if ``device='gpu'``).
    On the CPU the values follow the field's floating precision (float64 fields
    are not downcast); the GPU path works in float32.
    """
    scheme = scheme.strip().upper()
    if scheme not in _VALID:
        raise ValueError(f"Invalid scheme '{scheme}'. Valid options are: {_VALID}")

    if device == "gpu":
        if not CUPY_AVAILABLE:
            raise RuntimeError("GPU backend requested, but it is not available (CuPy/CUDA missing).")
        grid_dev = cp.asarray(grid, dtype=cp.float32)
        if grid_dev.ndim == 3:
            grid_dev = grid_dev[..., cp.newaxis]
        pos_dev = cp.asarray(pos, dtype=cp.float32)
        out_dev = cp.empty((pos_dev.shape[0], grid_dev.shape[-1]), dtype=cp.float32)
        boxsize_dev = cp.asarray(mesh.boxsize, dtype=cp.float32)
        if scheme == "CIC":
            _gpu.read_cic(grid_dev, pos_dev, boxsize_dev, out_dev, pbc)
        elif scheme == "TSC":
            _gpu.read_tsc(grid_dev, pos_dev, boxsize_dev, out_dev, pbc)
        else:
            raise ValueError(f"GPU read-out for scheme '{scheme}' is not implemented.")
        return out_dev.flatten() if grid_dev.shape[-1] == 1 else out_dev

    # CPU path: stay type-neutral. Follow the field's own floating precision so
    # a float64 field is sampled (and returned) in float64; fall back to the
    # mesh dtype for a non-floating grid.
    grid = np.asarray(grid)
    work_dtype = grid.dtype if grid.dtype.kind == "f" else np.dtype(mesh.dtype)
    grid = np.asarray(grid, dtype=work_dtype)
    pos = np.asarray(pos, dtype=work_dtype)
    boxsize = np.asarray(mesh.boxsize, dtype=work_dtype)
    out = np.zeros(pos.shape[0], dtype=work_dtype)
    if scheme == "NGP":
        return _cpu.ngp_read(pos, grid, boxsize, out, pbc=pbc)
    if scheme == "CIC":
        return _cpu.cic_read(pos, grid, boxsize, out, pbc=pbc)
    if scheme == "TSC":
        return _cpu.tsc_read(pos, grid, boxsize, out, pbc=pbc)
