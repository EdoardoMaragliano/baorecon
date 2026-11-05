"""Mass-assignment interface.

Validates inputs, allocates the output grid, and dispatches to the CPU or GPU
kernels. The kernels themselves perform no validation and never allocate the
output grid.

Precision is device-dependent. On the CPU the interface is type-neutral: the
grid, positions and weights follow the mesh's working precision
(``mesh.dtype``), so a float64 mesh is honoured end-to-end. The GPU kernels
operate in float32 only, so a ``device='gpu'`` call always works in (and
returns) float32 regardless of ``mesh.dtype``.

Positions may be given in the box frame (``[0, boxsize]``, the default) or in
the survey frame (``survey_frame=True``): the mesh knows its own lower corner,
and the CPU kernels move each coordinate to the box frame as they read it, so
no box-frame copy of the catalogue is ever built. ``pbc`` applies to the
coordinate and to the stencil alike: a periodic box wraps both. The result is
the same bit for bit as painting ``survey_to_box_frame(pos, mesh.min_corner,
...)``.
"""

import warnings

import numpy as np

from baorecon.mas import cpu as _cpu
from baorecon.utils.loggers import setup_logger

logger = setup_logger(__name__)

try:
    import cupy as cp
    from numba import cuda
    CUPY_AVAILABLE = cuda.is_available()
except ImportError:
    CUPY_AVAILABLE = False

if CUPY_AVAILABLE:
    from baorecon.mas import gpu as _gpu

_VALID = ("NGP", "CIC", "TSC")


def _origin(mesh, pos, survey_frame):
    """The box's lower corner in the frame of ``pos``, for the kernels; ``None``
    when ``pos`` is already in the box frame."""
    if not survey_frame:
        return None
    return np.asarray(mesh.min_corner, dtype=pos.dtype)


def _validate_positions(pos, weights, boxsize, pbc, origin=None):
    if pos.shape[1] != 3:
        raise ValueError("pos must have shape (N, 3)")
    if pos.shape[0] != weights.shape[0]:
        raise ValueError("pos and weights must have the same number of particles")
    # One parallel pass for the six extremes, instead of two boolean (N, 3)
    # masks the size of the catalogue.
    from baorecon.utils.reductions import position_bounds

    low, high = position_bounds(pos)
    if origin is not None:
        # Six numbers: the bounds in the box frame, without the (N, 3) array.
        low = low - origin
        high = high - origin
    if (low < 0).any() or (high > boxsize).any():
        if pbc:
            warnings.warn("Particles are outside the range [0, boxsize]. PBC is true and will be applied.")
        else:
            raise ValueError(
                f"pos must be in the range [0, boxsize]. Got min {low.min()} and max {high.max()}."
            )


def assign(pos, weights, mesh, scheme="CIC", device="cpu", pbc=True, parallel=False,
           survey_frame=False):
    """Paint particles onto a fresh grid using the requested scheme.

    Returns a ``(Nx, Ny, Nz)`` grid. On the CPU the grid dtype follows the
    mesh's working precision (``mesh.dtype``); on the GPU it is always a float32
    CuPy array. With ``survey_frame=True`` the positions are in the survey
    frame and are moved to the box frame inside the kernel (see the module
    docstring); otherwise they must already be in ``[0, boxsize]``.
    """
    scheme = scheme.strip().upper()
    if scheme not in _VALID:
        raise ValueError(f"Invalid scheme '{scheme}'. Valid options are: {_VALID}")

    # GPU kernels are float32-only; on the CPU we stay type-neutral and follow
    # the mesh's working precision so a float64 mesh is not silently downcast.
    # Weights are never promoted: a float32 column stays float32 under a
    # float64 mesh, the kernel promotes each one exactly as it reads it.
    from baorecon.utils.formatters import weights_in

    work_dtype = np.dtype(np.float32) if device == "gpu" else np.dtype(mesh.dtype)

    pos = np.asarray(pos, dtype=work_dtype)
    if weights is None:
        weights = np.ones(pos.shape[0], dtype=work_dtype)
    else:
        weights = weights_in(weights, work_dtype)

    boxsize = np.asarray(mesh.boxsize, dtype=work_dtype)
    origin = _origin(mesh, pos, survey_frame)
    _validate_positions(pos, weights, boxsize, pbc, origin)
    grid_shape = mesh.shape

    if device == "gpu":
        if not CUPY_AVAILABLE:
            raise RuntimeError("GPU backend requested, but it is not available (CuPy/CUDA missing).")
        pos_dev = cp.asarray(pos, dtype=cp.float32)
        # The GPU kernels read box-frame positions: shift and wrap a device copy.
        if origin is not None:
            pos_dev = pos_dev - cp.asarray(origin, dtype=cp.float32)
        if pbc:
            pos_dev = cp.mod(pos_dev, cp.asarray(boxsize, dtype=cp.float32))
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
        return _cpu.ngp_assign(pos, boxsize, weights, grid, pbc=pbc, origin=origin)
    if scheme == "CIC":
        if parallel:
            return _cpu.cic_assign_chunks(pos, boxsize, weights, grid, pbc=pbc, origin=origin)
        return _cpu.cic_assign_serial(pos, boxsize, weights, grid, pbc=pbc, origin=origin)
    if scheme == "TSC":
        if parallel:
            return _cpu.tsc_assign_chunks(pos, boxsize, weights, grid, pbc=pbc, origin=origin)
        return _cpu.tsc_assign_serial(pos, boxsize, weights, grid, pbc=pbc, origin=origin)


def readout(grid, pos, mesh, scheme="CIC", device="cpu", pbc=True, survey_frame=False):
    """Read out a scalar field defined on ``grid`` at particle positions.

    Returns a ``(N,)`` array of interpolated values (CuPy if ``device='gpu'``).
    On the CPU the values follow the field's floating precision (float64 fields
    are not downcast); the GPU path works in float32. ``survey_frame`` is as
    in :func:`assign`.
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
        boxsize_dev = cp.asarray(mesh.boxsize, dtype=cp.float32)
        origin = _origin(mesh, pos_dev, survey_frame)
        if origin is not None:
            pos_dev = pos_dev - cp.asarray(origin, dtype=cp.float32)
        if pbc:
            pos_dev = cp.mod(pos_dev, boxsize_dev)
        out_dev = cp.empty((pos_dev.shape[0], grid_dev.shape[-1]), dtype=cp.float32)
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
    origin = _origin(mesh, pos, survey_frame)
    out = np.zeros(pos.shape[0], dtype=work_dtype)
    if scheme == "NGP":
        return _cpu.ngp_read(pos, grid, boxsize, out, pbc=pbc, origin=origin)
    if scheme == "CIC":
        return _cpu.cic_read(pos, grid, boxsize, out, pbc=pbc, origin=origin)
    if scheme == "TSC":
        return _cpu.tsc_read(pos, grid, boxsize, out, pbc=pbc, origin=origin)
