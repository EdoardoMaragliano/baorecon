"""GPU mass-assignment and read-out kernels (numba.cuda).

Atomic-add painting and trilinear/TSC read-out. Grids/positions/weights are
float32; ``boxsize`` is a length-3 device array.
"""

import numpy as np

try:
    import cupy as cp
    from numba import cuda
    CUPY_AVAILABLE = cuda.is_available()
except ImportError:
    CUPY_AVAILABLE = False


if CUPY_AVAILABLE:

    @cuda.jit
    def _assign_cic_kernel(mesh, positions, weights, boxsize):
        i = cuda.grid(1)
        if i >= positions.shape[0]:
            return
        nx, ny, nz = mesh.shape
        inv_x = nx / boxsize[0]
        inv_y = ny / boxsize[1]
        inv_z = nz / boxsize[2]
        gx = positions[i, 0] * inv_x
        gy = positions[i, 1] * inv_y
        gz = positions[i, 2] * inv_z
        ix0, iy0, iz0 = int(gx), int(gy), int(gz)
        dx, dy, dz = gx - ix0, gy - iy0, gz - iz0
        for l in range(2):
            ix = (ix0 + l) % nx
            wx = 1.0 - dx if l == 0 else dx
            for m in range(2):
                iy = (iy0 + m) % ny
                wy = 1.0 - dy if m == 0 else dy
                for n in range(2):
                    iz = (iz0 + n) % nz
                    wz = 1.0 - dz if n == 0 else dz
                    cuda.atomic.add(mesh, (ix, iy, iz), weights[i] * wx * wy * wz)

    @cuda.jit
    def _assign_tsc_kernel(mesh, positions, weights, boxsize):
        i = cuda.grid(1)
        if i >= positions.shape[0]:
            return
        nx, ny, nz = mesh.shape
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
        for l in range(3):
            ix = (ix_c - 1 + l) % nx
            for m in range(3):
                iy = (iy_c - 1 + m) % ny
                for n in range(3):
                    iz = (iz_c - 1 + n) % nz
                    cuda.atomic.add(mesh, (ix, iy, iz), weights[i] * wx[l] * wy[m] * wz[n])

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


def assign_cic(mesh, positions, weights, boxsize):
    tpb = 256
    bpg = (positions.shape[0] + tpb - 1) // tpb
    _assign_cic_kernel[bpg, tpb](mesh, positions, weights, boxsize)


def assign_tsc(mesh, positions, weights, boxsize):
    tpb = 128
    bpg = (positions.shape[0] + tpb - 1) // tpb
    _assign_tsc_kernel[bpg, tpb](mesh, positions, weights, boxsize)


def read_cic(field, positions, boxsize, out_values):
    tpb = 256
    bpg = (positions.shape[0] + tpb - 1) // tpb
    _read_cic_kernel[bpg, tpb](field, positions, boxsize, out_values)


def read_tsc(field, positions, boxsize, out_values):
    tpb = 128
    bpg = (positions.shape[0] + tpb - 1) // tpb
    _read_tsc_kernel[bpg, tpb](field, positions, boxsize, out_values)
