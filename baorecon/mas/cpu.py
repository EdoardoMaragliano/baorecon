"""CPU mass-assignment and read-out kernels (numpy / numba).

Pure kernels: no validation and no output-grid allocation (the grid/output is
passed in pre-zeroed by the interface). Internal temporaries follow the dtype of
the supplied output array, so the kernels preserve the caller's floating
precision (the interface allocates that array at the mesh's working precision,
``mesh.dtype``, and casts positions/weights to match). ``boxsize`` is a length-3
array giving the per-axis box size, so cubic and rectangular boxes share the
same code path.

Boundary handling is uniform across schemes and across the serial/parallel
variants: with ``pbc`` the stencil wraps periodically; without it, out-of-range
stencil cells are clamped to the nearest boundary cell (mass-conserving), which
matches the read-out kernels.
"""

import numpy as np
from numba import njit, prange


# ============================================================================
# Particles -> grid (painting)
# ============================================================================
def ngp_assign(pos, boxsize, weights, density_grid, pbc=True):
    """Nearest-grid-point painting (vectorized numpy)."""
    nx, ny, nz = density_grid.shape
    inv = np.array([nx, ny, nz], dtype=np.float64) / boxsize
    indices = np.floor(pos * inv + 0.5).astype(np.int64)
    if pbc:
        indices = np.mod(indices, np.array([nx, ny, nz]))
    else:
        # Clamp out-of-range cells to the boundary (matches ngp_read).
        np.clip(indices, 0, np.array([nx, ny, nz]) - 1, out=indices)
    flat_idx = np.ravel_multi_index(indices.T, (nx, ny, nz))
    np.add.at(density_grid.ravel(), flat_idx, weights)
    return density_grid


@njit(parallel=False, fastmath=True)
def cic_assign_serial(pos, boxsize, weights, density_grid, pbc=True):
    """Cloud-in-cell painting, single-threaded."""
    nx, ny, nz = density_grid.shape
    inv0 = nx / boxsize[0]
    inv1 = ny / boxsize[1]
    inv2 = nz / boxsize[2]
    n_axis = (nx, ny, nz)

    u = np.ones(3, dtype=density_grid.dtype)
    d = np.ones(3, dtype=density_grid.dtype)
    index_u = np.zeros(3, dtype=np.int64)
    index_d = np.zeros(3, dtype=np.int64)
    invs = (inv0, inv1, inv2)

    for i in range(pos.shape[0]):
        for axis in range(3):
            dist = pos[i, axis] * invs[axis]
            cell = int(np.floor(dist))   # floor, not trunc: correct for dist < 0
            u[axis] = dist - cell
            d[axis] = 1.0 - u[axis]
            if pbc:
                index_d[axis] = cell % n_axis[axis]
                index_u[axis] = (index_d[axis] + 1) % n_axis[axis]
            else:
                index_d[axis] = min(max(cell, 0), n_axis[axis] - 1)
                index_u[axis] = min(index_d[axis] + 1, n_axis[axis] - 1)

        w = weights[i]
        density_grid[index_d[0], index_d[1], index_d[2]] += d[0] * d[1] * d[2] * w
        density_grid[index_d[0], index_d[1], index_u[2]] += d[0] * d[1] * u[2] * w
        density_grid[index_d[0], index_u[1], index_d[2]] += d[0] * u[1] * d[2] * w
        density_grid[index_d[0], index_u[1], index_u[2]] += d[0] * u[1] * u[2] * w
        density_grid[index_u[0], index_d[1], index_d[2]] += u[0] * d[1] * d[2] * w
        density_grid[index_u[0], index_d[1], index_u[2]] += u[0] * d[1] * u[2] * w
        density_grid[index_u[0], index_u[1], index_d[2]] += u[0] * u[1] * d[2] * w
        density_grid[index_u[0], index_u[1], index_u[2]] += u[0] * u[1] * u[2] * w
    return density_grid


@njit(parallel=True, fastmath=True)
def cic_assign_chunks(pos, boxsize, weights, density_grid, num_chunks=16, pbc=True):
    """Cloud-in-cell painting, multi-threaded via per-chunk private grids."""
    N = pos.shape[0]
    nx, ny, nz = density_grid.shape
    inv0 = nx / boxsize[0]
    inv1 = ny / boxsize[1]
    inv2 = nz / boxsize[2]

    local_grids = np.zeros((num_chunks, nx, ny, nz), dtype=density_grid.dtype)
    chunk_size = (N + num_chunks - 1) // num_chunks

    for c in prange(num_chunks):
        start = c * chunk_size
        end = min(start + chunk_size, N)
        grid = local_grids[c]
        for p in range(start, end):
            mass = weights[p]
            gx = pos[p, 0] * inv0
            gy = pos[p, 1] * inv1
            gz = pos[p, 2] * inv2
            i = int(np.floor(gx))
            j = int(np.floor(gy))
            k = int(np.floor(gz))
            dx = gx - i
            dy = gy - j
            dz = gz - k
            tx = 1.0 - dx
            ty = 1.0 - dy
            tz = 1.0 - dz
            w000 = tx * ty * tz * mass
            w001 = tx * ty * dz * mass
            w010 = tx * dy * tz * mass
            w011 = tx * dy * dz * mass
            w100 = dx * ty * tz * mass
            w101 = dx * ty * dz * mass
            w110 = dx * dy * tz * mass
            w111 = dx * dy * dz * mass
            i1 = i + 1
            j1 = j + 1
            k1 = k + 1
            if pbc:
                i = i % nx; i1 = i1 % nx
                j = j % ny; j1 = j1 % ny
                k = k % nz; k1 = k1 % nz
                grid[i, j, k] += w000
                grid[i, j, k1] += w001
                grid[i, j1, k] += w010
                grid[i, j1, k1] += w011
                grid[i1, j, k] += w100
                grid[i1, j, k1] += w101
                grid[i1, j1, k] += w110
                grid[i1, j1, k1] += w111
            else:
                # Clamp out-of-range corners to the boundary (mass-conserving,
                # matches cic_assign_serial) instead of dropping them.
                i = min(max(i, 0), nx - 1); i1 = min(i + 1, nx - 1)
                j = min(max(j, 0), ny - 1); j1 = min(j + 1, ny - 1)
                k = min(max(k, 0), nz - 1); k1 = min(k + 1, nz - 1)
                grid[i, j, k] += w000
                grid[i, j, k1] += w001
                grid[i, j1, k] += w010
                grid[i, j1, k1] += w011
                grid[i1, j, k] += w100
                grid[i1, j, k1] += w101
                grid[i1, j1, k] += w110
                grid[i1, j1, k1] += w111

    density_grid[:, :, :] = np.sum(local_grids, axis=0)
    return density_grid


@njit(parallel=False, fastmath=True)
def tsc_assign_serial(pos, boxsize, weights, density_grid, pbc=True):
    """Triangular-shaped-cloud painting, single-threaded."""
    nx, ny, nz = density_grid.shape
    n_axis = (nx, ny, nz)
    invs = (nx / boxsize[0], ny / boxsize[1], nz / boxsize[2])

    tsc_w = np.ones((3, 3), dtype=density_grid.dtype)
    index = np.zeros((3, 3), dtype=np.int64)

    for i in range(pos.shape[0]):
        for axis in range(3):
            dist = pos[i, axis] * invs[axis]
            minimum = np.int64(np.floor(dist - 1.5))
            for j in range(3):
                raw = minimum + j + 1
                if pbc:
                    index[axis, j] = (raw + n_axis[axis]) % n_axis[axis]
                else:
                    index[axis, j] = min(max(raw, 0), n_axis[axis] - 1)
                diff = np.abs(raw - dist)
                if diff < 0.5:
                    tsc_w[axis, j] = 0.75 - diff ** 2
                elif diff < 1.5:
                    tsc_w[axis, j] = 0.5 * (1.5 - diff) ** 2
                else:
                    tsc_w[axis, j] = 0.0
        for l in range(3):
            for m in range(3):
                for n in range(3):
                    density_grid[index[0, l], index[1, m], index[2, n]] += (
                        tsc_w[0, l] * tsc_w[1, m] * tsc_w[2, n] * weights[i]
                    )
    return density_grid


@njit(parallel=True, fastmath=True)
def tsc_assign_chunks(pos, boxsize, weights, density_grid, num_chunks=16, pbc=True):
    """Triangular-shaped-cloud painting, multi-threaded via per-chunk grids."""
    N = pos.shape[0]
    nx, ny, nz = density_grid.shape
    inv0 = nx / boxsize[0]
    inv1 = ny / boxsize[1]
    inv2 = nz / boxsize[2]

    local_grids = np.zeros((num_chunks, nx, ny, nz), dtype=density_grid.dtype)
    chunk_size = (N + num_chunks - 1) // num_chunks

    for c in prange(num_chunks):
        start = c * chunk_size
        end = min(start + chunk_size, N)
        grid = local_grids[c]
        for p in range(start, end):
            mass = weights[p]
            gx = pos[p, 0] * inv0
            gy = pos[p, 1] * inv1
            gz = pos[p, 2] * inv2
            ix = int(np.floor(gx + 0.5))
            iy = int(np.floor(gy + 0.5))
            iz = int(np.floor(gz + 0.5))
            dx = gx - ix
            dy = gy - iy
            dz = gz - iz
            wx = (0.5 * (0.5 - dx) ** 2, 0.75 - dx ** 2, 0.5 * (0.5 + dx) ** 2)
            wy = (0.5 * (0.5 - dy) ** 2, 0.75 - dy ** 2, 0.5 * (0.5 + dy) ** 2)
            wz = (0.5 * (0.5 - dz) ** 2, 0.75 - dz ** 2, 0.5 * (0.5 + dz) ** 2)
            if pbc:
                for di in range(3):
                    cx = (ix + di - 1) % nx
                    wx_mass = wx[di] * mass
                    for dj in range(3):
                        cy = (iy + dj - 1) % ny
                        wxy_mass = wx_mass * wy[dj]
                        for dk in range(3):
                            cz = (iz + dk - 1) % nz
                            grid[cx, cy, cz] += wxy_mass * wz[dk]
            else:
                # Clamp out-of-range cells to the boundary (mass-conserving,
                # matches tsc_assign_serial) instead of dropping them.
                for di in range(3):
                    cx = min(max(ix + di - 1, 0), nx - 1)
                    wx_mass = wx[di] * mass
                    for dj in range(3):
                        cy = min(max(iy + dj - 1, 0), ny - 1)
                        wxy_mass = wx_mass * wy[dj]
                        for dk in range(3):
                            cz = min(max(iz + dk - 1, 0), nz - 1)
                            grid[cx, cy, cz] += wxy_mass * wz[dk]

    density_grid[:, :, :] = np.sum(local_grids, axis=0)
    return density_grid


# ============================================================================
# Grid -> particles (read-out, scalar field)
# ============================================================================
@njit(parallel=False, fastmath=True)
def ngp_read(pos, grid, boxsize, out, pbc=True):
    nx, ny, nz = grid.shape
    invs = (nx / boxsize[0], ny / boxsize[1], nz / boxsize[2])
    n_axis = (nx, ny, nz)
    index = np.zeros(3, dtype=np.int64)
    for i in range(pos.shape[0]):
        for axis in range(3):
            idx = int(round(pos[i, axis] * invs[axis]))
            if pbc:
                index[axis] = idx % n_axis[axis]
            else:
                index[axis] = min(max(idx, 0), n_axis[axis] - 1)
        out[i] = grid[index[0], index[1], index[2]]
    return out


@njit(parallel=False, fastmath=True)
def cic_read(pos, grid, boxsize, out, pbc=True):
    nx, ny, nz = grid.shape
    invs = (nx / boxsize[0], ny / boxsize[1], nz / boxsize[2])
    n_axis = (nx, ny, nz)
    u = np.ones(3, dtype=out.dtype)
    d = np.ones(3, dtype=out.dtype)
    index_u = np.zeros(3, dtype=np.int64)
    index_d = np.zeros(3, dtype=np.int64)
    for i in range(pos.shape[0]):
        for axis in range(3):
            dist = pos[i, axis] * invs[axis]
            cell = int(np.floor(dist))   # floor, not trunc: correct for dist < 0
            u[axis] = dist - cell
            d[axis] = 1.0 - u[axis]
            if pbc:
                index_d[axis] = cell % n_axis[axis]
                index_u[axis] = (index_d[axis] + 1) % n_axis[axis]
            else:
                index_d[axis] = min(max(cell, 0), n_axis[axis] - 1)
                index_u[axis] = min(index_d[axis] + 1, n_axis[axis] - 1)
        out[i] = (d[0]*d[1]*d[2]*grid[index_d[0], index_d[1], index_d[2]] +
                  d[0]*d[1]*u[2]*grid[index_d[0], index_d[1], index_u[2]] +
                  d[0]*u[1]*d[2]*grid[index_d[0], index_u[1], index_d[2]] +
                  d[0]*u[1]*u[2]*grid[index_d[0], index_u[1], index_u[2]] +
                  u[0]*d[1]*d[2]*grid[index_u[0], index_d[1], index_d[2]] +
                  u[0]*d[1]*u[2]*grid[index_u[0], index_d[1], index_u[2]] +
                  u[0]*u[1]*d[2]*grid[index_u[0], index_u[1], index_d[2]] +
                  u[0]*u[1]*u[2]*grid[index_u[0], index_u[1], index_u[2]])
    return out


@njit(parallel=False, fastmath=True)
def tsc_read(pos, grid, boxsize, out, pbc=True):
    nx, ny, nz = grid.shape
    invs = (nx / boxsize[0], ny / boxsize[1], nz / boxsize[2])
    n_axis = (nx, ny, nz)
    for p in range(pos.shape[0]):
        gx = pos[p, 0] * invs[0]
        gy = pos[p, 1] * invs[1]
        gz = pos[p, 2] * invs[2]
        ix_c = int(round(gx))
        iy_c = int(round(gy))
        iz_c = int(round(gz))
        dx = gx - ix_c
        dy = gy - iy_c
        dz = gz - iz_c
        wx = (0.5 * (0.5 - dx) ** 2, 0.75 - dx ** 2, 0.5 * (0.5 + dx) ** 2)
        wy = (0.5 * (0.5 - dy) ** 2, 0.75 - dy ** 2, 0.5 * (0.5 + dy) ** 2)
        wz = (0.5 * (0.5 - dz) ** 2, 0.75 - dz ** 2, 0.5 * (0.5 + dz) ** 2)
        val = 0.0
        for i in range(3):
            ix = ix_c - 1 + i
            if pbc:
                ix = ix % n_axis[0]
            else:
                # Clamp to the boundary (mass-conserving) so read stays the
                # adjoint of the TSC assign kernels; dropping would lose weight.
                ix = min(max(ix, 0), n_axis[0] - 1)
            for j in range(3):
                iy = iy_c - 1 + j
                if pbc:
                    iy = iy % n_axis[1]
                else:
                    iy = min(max(iy, 0), n_axis[1] - 1)
                for k in range(3):
                    iz = iz_c - 1 + k
                    if pbc:
                        iz = iz % n_axis[2]
                    else:
                        iz = min(max(iz, 0), n_axis[2] - 1)
                    val += grid[ix, iy, iz] * wx[i] * wy[j] * wz[k]
        out[p] = val
    return out
