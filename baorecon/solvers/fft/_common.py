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


def build_inv_k2(k_comps, bias=1.0):
    """Return ``1 / (bias * |k|^2)`` as a single real half-grid.

    Wraps :func:`compute_k2` and inverts it in place, handling the ``k=0`` DC
    mode (set to 0 in the output). Array-module agnostic: on CuPy inputs the
    whole thing stays on the device. This folds the ``compute_k2 -> set DC ->
    reciprocal -> restore DC`` boilerplate that both FFT solvers repeated in
    ``_compute_displacement_*`` and ``_compute_potential_mesh`` into one place.
    """
    k2 = compute_k2(k_comps)

    try:
        import cupy
        xp = cupy.get_array_module(k2)
    except ImportError:
        xp = np

    k2[0, 0, 0] = 1.0
    k2 *= bias
    xp.divide(1.0, k2, out=k2)
    k2[0, 0, 0] = 0.0
    return k2


def divergence_inplace(vector_field, k_comps, rfftn, irfftn, xp):
    """Divergence of a vector field via the rFFT, one component at a time.

    Unlike a fused ``rfftn`` over the whole ``(Nx, Ny, Nz, 3)`` field (which
    materialises three complex half-grids plus temporaries at once), this
    transforms a single component at a time and accumulates into one complex
    half-grid, so the peak is ``div_k`` + one in-flight transform.

    ``rfftn(real3d) -> complex half-grid`` and ``irfftn(complex, s) -> real3d``
    are backend-specific callables supplied by the caller (scipy with
    ``workers``/``overwrite_x`` on the CPU, ``cupy.fft`` on the GPU), keeping
    this helper array-module agnostic. It is private to the FFT solvers and
    replaces the shared ``field_ops.divergence_FFT`` on this path.
    """
    if vector_field.shape[-1] != 3:
        raise ValueError("The last dimension of vector_field must be of size 3.")

    kx, ky, kz = k_comps
    k_bcast = (kx[:, None, None], ky[None, :, None], kz[None, None, :])
    grid_shape = vector_field.shape[:-1]
    complex_j = xp.complex64(1j)

    div_k = None
    for i in range(3):
        v_k = rfftn(vector_field[..., i])
        v_k *= complex_j
        v_k *= k_bcast[i]
        if div_k is None:
            div_k = v_k          # reuse the first transform as the accumulator
        else:
            div_k += v_k
    return irfftn(div_k, grid_shape)