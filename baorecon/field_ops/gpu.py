"""GPU field operations (CuPy / numba.cuda).

Mirrors :mod:`baorecon.field_ops.cpu` on the device: FFT divergence via
``cupy.fft`` and CUDA-kernel CIC/TSC vector interpolation.
"""

import numpy as np

from baorecon.utils.backend import CUPY_AVAILABLE

if CUPY_AVAILABLE:
    import cupy as cp
    from numba import cuda


def divergence_FFT(vector_field, k_components):
    """Divergence of a vector field via the (real) FFT, on the GPU."""
    if vector_field.shape[-1] != 3:
        raise ValueError("The last dimension of vector_field must be of size 3.")

    kx, ky, kz = k_components
    v_k = cp.fft.rfftn(vector_field, axes=(0, 1, 2))
    div_k = (v_k[..., 0] * (1j * kx[:, None, None]) +
             v_k[..., 1] * (1j * ky[None, :, None]) +
             v_k[..., 2] * (1j * kz[None, None, :]))
    return cp.fft.irfftn(div_k, axes=(0, 1, 2))


if CUPY_AVAILABLE:

    @cuda.jit
    def _read_cic_kernel(field, positions, boxsize, out_values):
        i = cuda.grid(1)
        if i >= positions.shape[0]:
            return
        nx, ny, nz = field.shape[:3]
        inv_x = nx / boxsize[0]
        inv_y = ny / boxsize[1]
        inv_z = nz / boxsize[2]

        gx = positions[i, 0] * inv_x
        gy = positions[i, 1] * inv_y
        gz = positions[i, 2] * inv_z
        ix0, iy0, iz0 = int(gx), int(gy), int(gz)
        dx, dy, dz = gx - ix0, gy - iy0, gz - iz0

        for c in range(out_values.shape[1]):
            val = 0.0
            for l in range(2):
                ix = (ix0 + l) % nx
                wx = 1.0 - dx if l == 0 else dx
                for m in range(2):
                    iy = (iy0 + m) % ny
                    wy = 1.0 - dy if m == 0 else dy
                    for n in range(2):
                        iz = (iz0 + n) % nz
                        wz = 1.0 - dz if n == 0 else dz
                        val += field[ix, iy, iz, c] * wx * wy * wz
            out_values[i, c] = val

    @cuda.jit
    def _read_tsc_kernel(field, positions, boxsize, out_values):
        i = cuda.grid(1)
        if i >= positions.shape[0]:
            return
        nx, ny, nz = field.shape[:3]
        inv_x = nx / boxsize[0]
        inv_y = ny / boxsize[1]
        inv_z = nz / boxsize[2]

        gx = positions[i, 0] * inv_x
        gy = positions[i, 1] * inv_y
        gz = positions[i, 2] * inv_z
        ix_c, iy_c, iz_c = int(round(gx)), int(round(gy)), int(round(gz))
        dx, dy, dz = gx - ix_c, gy - iy_c, gz - iz_c

        wx = (0.5 * (0.5 - dx) ** 2, 0.75 - dx ** 2, 0.5 * (0.5 + dx) ** 2)
        wy = (0.5 * (0.5 - dy) ** 2, 0.75 - dy ** 2, 0.5 * (0.5 + dy) ** 2)
        wz = (0.5 * (0.5 - dz) ** 2, 0.75 - dz ** 2, 0.5 * (0.5 + dz) ** 2)

        for c in range(out_values.shape[1]):
            val = 0.0
            for l in range(3):
                ix = (ix_c - 1 + l) % nx
                for m in range(3):
                    iy = (iy_c - 1 + m) % ny
                    for n in range(3):
                        iz = (iz_c - 1 + n) % nz
                        val += field[ix, iy, iz, c] * wx[l] * wy[m] * wz[n]
            out_values[i, c] = val


def interpolate_cic_vector(pos, field, boxsize, pbc=True, dtype=np.float32):
    """CIC interpolation of a vector field on the GPU (PBC only)."""
    pos_dev = cp.asarray(pos, dtype=dtype)
    field_dev = cp.asarray(field, dtype=dtype)
    out = cp.empty((pos_dev.shape[0], 3), dtype=dtype)
    boxsize_dev = cp.asarray(np.broadcast_to(np.asarray(boxsize, dtype=np.float64), (3,)))
    tpb = 256
    bpg = (pos_dev.shape[0] + tpb - 1) // tpb
    _read_cic_kernel[bpg, tpb](field_dev, pos_dev, boxsize_dev, out)
    return out


def interpolate_tsc_vector(pos, field, boxsize, pbc=True, dtype=np.float32):
    """TSC interpolation of a vector field on the GPU (PBC only)."""
    pos_dev = cp.asarray(pos, dtype=dtype)
    field_dev = cp.asarray(field, dtype=dtype)
    out = cp.empty((pos_dev.shape[0], 3), dtype=dtype)
    boxsize_dev = cp.asarray(np.broadcast_to(np.asarray(boxsize, dtype=np.float64), (3,)))
    tpb = 128
    bpg = (pos_dev.shape[0] + tpb - 1) // tpb
    _read_tsc_kernel[bpg, tpb](field_dev, pos_dev, boxsize_dev, out)
    return out
