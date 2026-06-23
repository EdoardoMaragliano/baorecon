"""Mass-assignment interface.

Validates inputs, allocates the (float32) output grid, and dispatches to the
CPU or GPU kernels. The kernels themselves perform no validation and never
allocate the output grid.
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


def assign(pos, weights, mesh, scheme="CIC", device="cpu", pbc=True, parallel=False):
    """Paint particles onto a fresh grid using the requested scheme.

    Returns a ``(Nx, Ny, Nz)`` float64 grid (a CuPy array if ``device='gpu'``).
    """
    scheme = scheme.strip().upper()
    if scheme not in _VALID:
        raise ValueError(f"Invalid scheme '{scheme}'. Valid options are: {_VALID}")

    pos = np.asarray(pos, dtype=np.float64)
    if weights is None:
        weights = np.ones(pos.shape[0], dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64)

    boxsize = np.asarray(mesh.boxsize, dtype=np.float64)
    _validate_positions(pos, weights, boxsize, pbc)
    grid_shape = mesh.shape

    if device == "gpu":
        if not CUPY_AVAILABLE:
            raise RuntimeError("GPU backend requested, but it is not available (CuPy/CUDA missing).")
        pos_dev = cp.asarray(pos, dtype=cp.float32)
        weights_dev = cp.asarray(weights, dtype=cp.float32)
        mesh_dev = cp.zeros(grid_shape, dtype=cp.float32)
        boxsize_dev = cp.asarray(boxsize)
        if scheme == "CIC":
            _gpu.assign_cic(mesh_dev, pos_dev, weights_dev, boxsize_dev)
        elif scheme == "TSC":
            _gpu.assign_tsc(mesh_dev, pos_dev, weights_dev, boxsize_dev)
        else:
            raise ValueError(f"GPU mass assignment for scheme '{scheme}' is not implemented.")
        return mesh_dev

    grid = np.zeros(grid_shape, dtype=np.float64)
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


def readout(grid, pos, mesh, scheme="CIC", device="cpu", pbc=True):
    """Read out a scalar field defined on ``grid`` at particle positions.

    Returns a ``(N,)`` array of interpolated values (CuPy if ``device='gpu'``).
    """
    scheme = scheme.strip().upper()
    if scheme not in _VALID:
        raise ValueError(f"Invalid scheme '{scheme}'. Valid options are: {_VALID}")

    pos = np.asarray(pos, dtype=np.float32)
    boxsize = np.asarray(mesh.boxsize, dtype=np.float64)

    if device == "gpu":
        if not CUPY_AVAILABLE:
            raise RuntimeError("GPU backend requested, but it is not available (CuPy/CUDA missing).")
        grid_dev = cp.asarray(grid, dtype=cp.float32)
        if grid_dev.ndim == 3:
            grid_dev = grid_dev[..., cp.newaxis]
        pos_dev = cp.asarray(pos, dtype=cp.float32)
        out_dev = cp.empty((pos_dev.shape[0], grid_dev.shape[-1]), dtype=cp.float32)
        boxsize_dev = cp.asarray(boxsize)
        if scheme == "CIC":
            _gpu.read_cic(grid_dev, pos_dev, boxsize_dev, out_dev)
        elif scheme == "TSC":
            _gpu.read_tsc(grid_dev, pos_dev, boxsize_dev, out_dev)
        else:
            raise ValueError(f"GPU read-out for scheme '{scheme}' is not implemented.")
        return out_dev.flatten() if grid_dev.shape[-1] == 1 else out_dev

    grid = np.asarray(grid, dtype=np.float32)
    out = np.zeros(pos.shape[0], dtype=np.float32)
    if scheme == "NGP":
        return _cpu.ngp_read(pos, grid, boxsize, out, pbc=pbc)
    if scheme == "CIC":
        return _cpu.cic_read(pos, grid, boxsize, out, pbc=pbc)
    if scheme == "TSC":
        return _cpu.tsc_read(pos, grid, boxsize, out, pbc=pbc)
