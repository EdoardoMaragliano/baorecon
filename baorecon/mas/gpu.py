"""GPU mass-assignment and read-out kernels (numba.cuda).

Atomic-add painting and trilinear/TSC read-out. Grids/positions/weights are
always float32 (the interface forces this regardless of ``mesh.dtype``);
``boxsize`` is a length-3 device array.

Boundary handling mirrors the CPU kernels (:mod:`baorecon.mas.cpu`): with
``pbc`` the stencil wraps periodically; without it, out-of-range stencil cells
are clamped to the nearest boundary cell (mass-conserving).
"""

import math

import numpy as np

from baorecon.utils.backend import CUPY_AVAILABLE

if CUPY_AVAILABLE:
    import cupy as cp
    from numba import cuda


if CUPY_AVAILABLE:

    @cuda.jit
    def _assign_cic_kernel(mesh, positions, weights, boxsize, pbc):
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
        ix0 = int(math.floor(gx))   # floor, not trunc: matches the CPU kernels
        iy0 = int(math.floor(gy))
        iz0 = int(math.floor(gz))
        dx, dy, dz = gx - ix0, gy - iy0, gz - iz0
        for l in range(2):
            ix = (ix0 + l) % nx if pbc else min(max(ix0 + l, 0), nx - 1)
            wx = 1.0 - dx if l == 0 else dx
            for m in range(2):
                iy = (iy0 + m) % ny if pbc else min(max(iy0 + m, 0), ny - 1)
                wy = 1.0 - dy if m == 0 else dy
                for n in range(2):
                    iz = (iz0 + n) % nz if pbc else min(max(iz0 + n, 0), nz - 1)
                    wz = 1.0 - dz if n == 0 else dz
                    cuda.atomic.add(mesh, (ix, iy, iz), weights[i] * wx * wy * wz)

    @cuda.jit
    def _assign_tsc_kernel(mesh, positions, weights, boxsize, pbc):
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
            ix = (ix_c - 1 + l) % nx if pbc else min(max(ix_c - 1 + l, 0), nx - 1)
            for m in range(3):
                iy = (iy_c - 1 + m) % ny if pbc else min(max(iy_c - 1 + m, 0), ny - 1)
                for n in range(3):
                    iz = (iz_c - 1 + n) % nz if pbc else min(max(iz_c - 1 + n, 0), nz - 1)
                    cuda.atomic.add(mesh, (ix, iy, iz), weights[i] * wx[l] * wy[m] * wz[n])

    @cuda.jit
    def _read_cic_kernel(field, positions, boxsize, out_values, pbc):
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
        ix0 = int(math.floor(gx))   # floor, not trunc: matches the CPU kernels
        iy0 = int(math.floor(gy))
        iz0 = int(math.floor(gz))
        dx, dy, dz = gx - ix0, gy - iy0, gz - iz0
        for c in range(out_values.shape[1]):
            val = 0.0
            for l in range(2):
                ix = (ix0 + l) % nx if pbc else min(max(ix0 + l, 0), nx - 1)
                wx = 1.0 - dx if l == 0 else dx
                for m in range(2):
                    iy = (iy0 + m) % ny if pbc else min(max(iy0 + m, 0), ny - 1)
                    wy = 1.0 - dy if m == 0 else dy
                    for n in range(2):
                        iz = (iz0 + n) % nz if pbc else min(max(iz0 + n, 0), nz - 1)
                        wz = 1.0 - dz if n == 0 else dz
                        val += field[ix, iy, iz, c] * wx * wy * wz
            out_values[i, c] = val

    @cuda.jit
    def _read_tsc_kernel(field, positions, boxsize, out_values, pbc):
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
                ix = (ix_c - 1 + l) % nx if pbc else min(max(ix_c - 1 + l, 0), nx - 1)
                for m in range(3):
                    iy = (iy_c - 1 + m) % ny if pbc else min(max(iy_c - 1 + m, 0), ny - 1)
                    for n in range(3):
                        iz = (iz_c - 1 + n) % nz if pbc else min(max(iz_c - 1 + n, 0), nz - 1)
                        val += field[ix, iy, iz, c] * wx[l] * wy[m] * wz[n]
            out_values[i, c] = val


    # ------------------------------------------------------------------
    # Slab-decomposed variants (multi-GPU): the grid is a halo-extended
    # x-slab of the global mesh. Positions/weights are the particles OWNED
    # by this rank; the stencil may spill into the halo planes, which are
    # later folded into their owners by halo_exchange_add. Global stencil
    # cells are mapped to extended-local indices via
    #   d = ix_global - x_start   (x_start = x_offset - w, one modular wrap
    #                              for PBC images)
    # and out-of-window cells are skipped (cannot happen for owned
    # particles; kept as a guard). y/z handling is identical to the
    # single-GPU kernels (those axes are not decomposed).
    # ------------------------------------------------------------------

    @cuda.jit(device=True)
    def _slab_index(ixg, x_start, nx_global, nx_ext, pbc):
        d = ixg - x_start
        if pbc:
            if d < 0:
                d += nx_global
            elif d >= nx_global:
                d -= nx_global
        if d < 0 or d >= nx_ext:
            return -1
        return d

    @cuda.jit
    def _assign_cic_slab_kernel(mesh_ext, positions, weights, boxsize,
                                nx_global, x_start, pbc):
        i = cuda.grid(1)
        if i >= positions.shape[0]:
            return
        nx_ext, ny, nz = mesh_ext.shape
        inv_x = nx_global / boxsize[0]
        inv_y = ny / boxsize[1]
        inv_z = nz / boxsize[2]
        gx = positions[i, 0] * inv_x
        gy = positions[i, 1] * inv_y
        gz = positions[i, 2] * inv_z
        ix0 = int(math.floor(gx))
        iy0 = int(math.floor(gy))
        iz0 = int(math.floor(gz))
        dx, dy, dz = gx - ix0, gy - iy0, gz - iz0
        for l in range(2):
            ixg = (ix0 + l) % nx_global if pbc else min(max(ix0 + l, 0), nx_global - 1)
            d = _slab_index(ixg, x_start, nx_global, nx_ext, pbc)
            if d < 0:
                continue
            wx = 1.0 - dx if l == 0 else dx
            for m in range(2):
                iy = (iy0 + m) % ny if pbc else min(max(iy0 + m, 0), ny - 1)
                wy = 1.0 - dy if m == 0 else dy
                for n in range(2):
                    iz = (iz0 + n) % nz if pbc else min(max(iz0 + n, 0), nz - 1)
                    wz = 1.0 - dz if n == 0 else dz
                    cuda.atomic.add(mesh_ext, (d, iy, iz), weights[i] * wx * wy * wz)

    @cuda.jit
    def _assign_tsc_slab_kernel(mesh_ext, positions, weights, boxsize,
                                nx_global, x_start, pbc):
        i = cuda.grid(1)
        if i >= positions.shape[0]:
            return
        nx_ext, ny, nz = mesh_ext.shape
        inv_x = nx_global / boxsize[0]
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
            ixg = (ix_c - 1 + l) % nx_global if pbc else min(max(ix_c - 1 + l, 0), nx_global - 1)
            d = _slab_index(ixg, x_start, nx_global, nx_ext, pbc)
            if d < 0:
                continue
            for m in range(3):
                iy = (iy_c - 1 + m) % ny if pbc else min(max(iy_c - 1 + m, 0), ny - 1)
                for n in range(3):
                    iz = (iz_c - 1 + n) % nz if pbc else min(max(iz_c - 1 + n, 0), nz - 1)
                    cuda.atomic.add(mesh_ext, (d, iy, iz),
                                    weights[i] * wx[l] * wy[m] * wz[n])

    @cuda.jit
    def _read_cic_slab_kernel(field_ext, positions, boxsize, out_values,
                              nx_global, x_start, pbc):
        i = cuda.grid(1)
        if i >= positions.shape[0]:
            return
        nx_ext, ny, nz = field_ext.shape[:3]
        inv_x = nx_global / boxsize[0]
        inv_y = ny / boxsize[1]
        inv_z = nz / boxsize[2]
        gx = positions[i, 0] * inv_x
        gy = positions[i, 1] * inv_y
        gz = positions[i, 2] * inv_z
        ix0 = int(math.floor(gx))
        iy0 = int(math.floor(gy))
        iz0 = int(math.floor(gz))
        dx, dy, dz = gx - ix0, gy - iy0, gz - iz0
        for c in range(out_values.shape[1]):
            val = 0.0
            for l in range(2):
                ixg = (ix0 + l) % nx_global if pbc else min(max(ix0 + l, 0), nx_global - 1)
                d = _slab_index(ixg, x_start, nx_global, nx_ext, pbc)
                if d < 0:
                    continue
                wx = 1.0 - dx if l == 0 else dx
                for m in range(2):
                    iy = (iy0 + m) % ny if pbc else min(max(iy0 + m, 0), ny - 1)
                    wy = 1.0 - dy if m == 0 else dy
                    for n in range(2):
                        iz = (iz0 + n) % nz if pbc else min(max(iz0 + n, 0), nz - 1)
                        wz = 1.0 - dz if n == 0 else dz
                        val += field_ext[d, iy, iz, c] * wx * wy * wz
            out_values[i, c] = val

    @cuda.jit
    def _read_tsc_slab_kernel(field_ext, positions, boxsize, out_values,
                              nx_global, x_start, pbc):
        i = cuda.grid(1)
        if i >= positions.shape[0]:
            return
        nx_ext, ny, nz = field_ext.shape[:3]
        inv_x = nx_global / boxsize[0]
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
                ixg = (ix_c - 1 + l) % nx_global if pbc else min(max(ix_c - 1 + l, 0), nx_global - 1)
                d = _slab_index(ixg, x_start, nx_global, nx_ext, pbc)
                if d < 0:
                    continue
                for m in range(3):
                    iy = (iy_c - 1 + m) % ny if pbc else min(max(iy_c - 1 + m, 0), ny - 1)
                    for n in range(3):
                        iz = (iz_c - 1 + n) % nz if pbc else min(max(iz_c - 1 + n, 0), nz - 1)
                        val += field_ext[d, iy, iz, c] * wx[l] * wy[m] * wz[n]
            out_values[i, c] = val


def assign_cic(mesh, positions, weights, boxsize, pbc=True):
    tpb = 256
    bpg = (positions.shape[0] + tpb - 1) // tpb
    _assign_cic_kernel[bpg, tpb](mesh, positions, weights, boxsize, pbc)


def assign_tsc(mesh, positions, weights, boxsize, pbc=True):
    tpb = 128
    bpg = (positions.shape[0] + tpb - 1) // tpb
    _assign_tsc_kernel[bpg, tpb](mesh, positions, weights, boxsize, pbc)


def read_cic(field, positions, boxsize, out_values, pbc=True):
    tpb = 256
    bpg = (positions.shape[0] + tpb - 1) // tpb
    _read_cic_kernel[bpg, tpb](field, positions, boxsize, out_values, pbc)


def read_tsc(field, positions, boxsize, out_values, pbc=True):
    tpb = 128
    bpg = (positions.shape[0] + tpb - 1) // tpb
    _read_tsc_kernel[bpg, tpb](field, positions, boxsize, out_values, pbc)


# --- slab-decomposed launchers (multi-GPU) ---

def assign_cic_slab(mesh_ext, positions, weights, boxsize, nx_global, x_start, pbc=True):
    tpb = 256
    bpg = max((positions.shape[0] + tpb - 1) // tpb, 1)
    _assign_cic_slab_kernel[bpg, tpb](mesh_ext, positions, weights, boxsize,
                                      nx_global, x_start, pbc)


def assign_tsc_slab(mesh_ext, positions, weights, boxsize, nx_global, x_start, pbc=True):
    tpb = 128
    bpg = max((positions.shape[0] + tpb - 1) // tpb, 1)
    _assign_tsc_slab_kernel[bpg, tpb](mesh_ext, positions, weights, boxsize,
                                      nx_global, x_start, pbc)


def read_cic_slab(field_ext, positions, boxsize, out_values, nx_global, x_start, pbc=True):
    tpb = 256
    bpg = max((positions.shape[0] + tpb - 1) // tpb, 1)
    _read_cic_slab_kernel[bpg, tpb](field_ext, positions, boxsize, out_values,
                                    nx_global, x_start, pbc)


def read_tsc_slab(field_ext, positions, boxsize, out_values, nx_global, x_start, pbc=True):
    tpb = 128
    bpg = max((positions.shape[0] + tpb - 1) // tpb, 1)
    _read_tsc_slab_kernel[bpg, tpb](field_ext, positions, boxsize, out_values,
                                    nx_global, x_start, pbc)
