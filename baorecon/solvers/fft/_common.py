"""Backend-independent helpers shared by the FFT solvers.

These are pure numpy functions that build the wavevector grids from the mesh
geometry. They are used by both :class:`FFTSolverCPU` and
:class:`FFTSolverGPU` (the latter uploads the result to the device).
"""

import numpy as np


import numpy as np

def prepare_k_components(cell_size, nmesh, dtype=np.float32):
    """Return the three 1D wavevector arrays (kx, ky, kz)."""
    cell_size = np.broadcast_to(np.asarray(cell_size, dtype=dtype), (3,))
    nmesh = np.asarray(nmesh)
    if nmesh.ndim == 0:
        nmesh = np.full(3, int(nmesh), dtype=np.int64)

    # np.fft.fftfreq returns float64 by default.
    # Multiply by 2*pi and immediately cast to the desired dtype (float32).
    two_pi = dtype(2 * np.pi)
    kx = (np.fft.fftfreq(int(nmesh[0]), d=cell_size[0]) * two_pi).astype(dtype)
    ky = (np.fft.fftfreq(int(nmesh[1]), d=cell_size[1]) * two_pi).astype(dtype)
    kz = (np.fft.rfftfreq(int(nmesh[2]), d=cell_size[2]) * two_pi).astype(dtype)
    
    return kx, ky, kz


def compute_k2(k_comps):
    """Return |k|^2 of shape (Nx, Ny, Nz//2+1) from the 1D components.

    Array-module agnostic: if the components are CuPy arrays the result is built
    on the device, so the GPU solver never round-trips |k|^2 through the host.
    """
    kx, ky, kz = k_comps

    try:
        import cupy
        xp = cupy.get_array_module(kx)
    except ImportError:
        xp = np

    # 1. Square the 1D vectors directly.
    # This is faster and avoids extra operations on the full 3D grid.
    kx2 = kx ** 2
    ky2 = ky ** 2
    kz2 = kz ** 2

    # 2. Allocate the final 3D array once.
    # Use xp.empty() to avoid the cost of zero-filling the array.
    k2 = xp.empty((len(kx2), len(ky2), len(kz2)), dtype=kx.dtype)

    # 3. In-place assignment and sums via broadcasting.
    # This avoids creating large hidden temporary arrays under the hood.
    k2[...] = kx2[:, None, None]
    k2 += ky2[None, :, None]
    k2 += kz2[None, None, :]
    
    return k2