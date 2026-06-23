"""Backend-dispatching field operations.

Validates inputs and dispatches to the CPU (:mod:`baorecon.field_ops.cpu`) or
GPU (:mod:`baorecon.field_ops.gpu`) implementations based on the array module
of the data.
"""

import time

import numpy as np

from baorecon.field_ops import cpu as _cpu
from baorecon.utils.loggers import setup_logger

logger = setup_logger(__name__)

try:
    import cupy as cp
    from numba import cuda
    CUPY_AVAILABLE = cuda.is_available()
except ImportError:
    CUPY_AVAILABLE = False

if CUPY_AVAILABLE:
    from baorecon.field_ops import gpu as _gpu


def _get_array_module(*arrays):
    """Return the array module (cupy or numpy) matching the given arrays."""
    if CUPY_AVAILABLE:
        for arr in arrays:
            if isinstance(arr, cp.ndarray):
                return cp
    return np


def _as_boxsize3(boxsize) -> np.ndarray:
    """Normalize a scalar or length-3 boxsize to a float64 array of shape (3,)."""
    return np.broadcast_to(np.asarray(boxsize, dtype=np.float64), (3,)).copy()


def project_vector_field(vector_field1, vector_field2, out=None):
    """Project ``vector_field1`` onto the direction of ``vector_field2``.

    Generic numpy/cupy implementation. When ``out`` is provided the result is
    written into it (used by the GPU LOS path).
    """
    xp = _get_array_module(vector_field1, vector_field2)

    norm = xp.linalg.norm(vector_field2, axis=-1, keepdims=False)
    mask = norm > 0
    unit_vector_field2 = xp.zeros_like(vector_field2)
    unit_vector_field2[mask] = vector_field2[mask] / norm[mask, xp.newaxis]

    v_dot_r = xp.sum(vector_field1 * unit_vector_field2, axis=-1, keepdims=True)
    v_parallel = v_dot_r * unit_vector_field2

    if out is not None:
        out[...] = v_parallel
        return out
    return v_parallel


def divergence(vector_field, div_algo="FFT", cell_size=None, k_components=None):
    """Divergence of a vector field using the requested algorithm."""
    xp = _get_array_module(vector_field)
    if div_algo == "finite_diff":
        if cell_size is None:
            raise ValueError("cell_size must be provided for finite difference divergence.")
        return _cpu.divergence_finite_diff(vector_field, cell_size)
    elif div_algo == "FFT":
        if k_components is None:
            raise ValueError("k_components must be provided for FFT divergence.")
        if xp is np:
            return _cpu.divergence_FFT(vector_field, k_components)
        return _gpu.divergence_FFT(vector_field, k_components)
    else:
        raise ValueError("Invalid algorithm. Options are 'finite_diff' and 'FFT'.")


def divergence_FFT(vector_field, k_components):
    """Divergence via FFT, dispatched to the matching backend."""
    xp = _get_array_module(vector_field, *k_components)
    if xp is np:
        return _cpu.divergence_FFT(vector_field, k_components)
    return _gpu.divergence_FFT(vector_field, k_components)


def interpolate_vector_field(pos, field, boxsize, MAS="CIC", pbc=True, dtype=np.float32):
    """Interpolate a vector field at particle positions using a CIC/TSC scheme."""
    start_time = time.time()
    boxsize = _as_boxsize3(boxsize)
    xp = _get_array_module(pos, field)
    backend = _cpu if xp is np else _gpu

    if MAS == "CIC":
        interp_field = backend.interpolate_cic_vector(pos, field, boxsize, pbc=pbc, dtype=dtype)
    elif MAS == "TSC":
        interp_field = backend.interpolate_tsc_vector(pos, field, boxsize, pbc=pbc, dtype=dtype)
    else:
        raise ValueError("MAS must be one of 'CIC' or 'TSC'")

    logger.debug(f"{MAS} interpolation took {time.time() - start_time:.4f} seconds.")
    return interp_field


def smoothed_field(field_on_mesh, mesh, smoothing_radius):
    xp = _get_array_module(field_on_mesh)
    nx, ny, nz = mesh.shape
    orig_dtype = field_on_mesh.dtype
    cs = np.asarray(mesh.cell_size, dtype=orig_dtype)
    h = 0.5 * smoothing_radius ** 2

    # 1-D Gaussian factors: exp(-0.5 R^2 k^2) is separable across axes.
    sx = xp.exp(-h * (xp.asarray(np.fft.fftfreq(nx,  d=cs[0]) * 2*np.pi, dtype=orig_dtype) ** 2))
    sy = xp.exp(-h * (xp.asarray(np.fft.fftfreq(ny,  d=cs[1]) * 2*np.pi, dtype=orig_dtype) ** 2))
    sz = xp.exp(-h * (xp.asarray(np.fft.rfftfreq(nz, d=cs[2]) * 2*np.pi, dtype=orig_dtype) ** 2))

    if xp is np:
        import scipy.fft as sfft
        delta_k = sfft.rfftn(field_on_mesh, axes=(0, 1, 2), workers=-1)
        delta_k *= sx[:, None, None]          # in-place, no full-grid temp
        delta_k *= sy[None, :, None]
        delta_k *= sz[None, None, :]
        return sfft.irfftn(delta_k, axes=(0, 1, 2), s=field_on_mesh.shape,
                           workers=-1, overwrite_x=True)   # reuse delta_k's buffer

    delta_k = xp.fft.rfftn(field_on_mesh, axes=(0, 1, 2))
    delta_k *= sx[:, None, None]; delta_k *= sy[None, :, None]; delta_k *= sz[None, None, :]
    return xp.fft.irfftn(delta_k, axes=(0, 1, 2), s=field_on_mesh.shape)
