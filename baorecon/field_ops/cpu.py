"""CPU field operations (numpy / scipy / numba).

Pure kernels and CPU implementations: FFT divergence (scipy with
``workers=-1``), finite-difference divergence and the JIT CIC/TSC vector
interpolators. ``boxsize`` is always passed as a length-3 array so a cubic box
given as a scalar and as ``[L, L, L]`` produce bit-identical results.
"""

import numpy as np
import scipy.fft as sfft
from numba import njit, prange


def divergence_FFT(vector_field: np.ndarray, k_components) -> np.ndarray:
    """Divergence of a vector field via the (real) FFT, on the CPU.

    Parameters
    ----------
    vector_field : ndarray, shape (Nx, Ny, Nz, 3)
        Vector field in configuration space.
    k_components : tuple of ndarray
        The three 1D wavevector arrays ``(kx, ky, kz)`` (``kz`` reduced to
        ``Nz//2 + 1`` samples) combined via broadcasting.
    """
    if vector_field.shape[-1] != 3:
        raise ValueError("The last dimension of vector_field must be of size 3.")

    # 1. Set up geometry and dtypes
    kx, ky, kz = k_components
    k_comps = (kx[:, None, None], ky[None, :, None], kz[None, None, :])
    
    grid_shape = vector_field.shape[:-1]
    
    k_shape = (grid_shape[0], grid_shape[1], grid_shape[2] // 2 + 1)
    
    c_dtype = np.complex64 if vector_field.dtype == np.float32 else np.complex128
    
    # 2. Accumulator for the divergence in Fourier space
    div_k = np.zeros(k_shape, dtype=c_dtype)
    
    for i in range(3):
        v_k_comp = sfft.rfftn(vector_field[..., i], axes=(0, 1, 2), workers=-1)
        
        v_k_comp *= k_comps[i]
        v_k_comp *= np.complex64(1j)
        div_k += v_k_comp
        
    return sfft.irfftn(div_k, s=grid_shape, axes=(0, 1, 2), workers=-1)


def divergence_finite_diff(vector_field: np.ndarray, cell_size: float) -> np.ndarray:
    """Divergence of a vector field via centered finite differences."""
    if vector_field.shape[-1] != 3:
        raise ValueError("The last dimension of vector_field must be of size 3.")
    return (np.gradient(vector_field[..., 0], cell_size, axis=0) +
            np.gradient(vector_field[..., 1], cell_size, axis=1) +
            np.gradient(vector_field[..., 2], cell_size, axis=2))


@njit(parallel=True, fastmath=True)
def interpolate_cic_vector(pos, field, boxsize, pbc=True, dtype=np.float32):
    """Trilinear CIC interpolation of a vector field at particle positions.

    ``boxsize`` is a length-3 array giving the per-axis box size.
    """
    N = pos.shape[0]
    nmesh = field.shape[0]
    out = np.zeros((N, 3), dtype=dtype)
    csx = boxsize[0] / nmesh
    csy = boxsize[1] / nmesh
    csz = boxsize[2] / nmesh

    for idx in prange(N):
        fx = pos[idx, 0] / csx
        fy = pos[idx, 1] / csy
        fz = pos[idx, 2] / csz

        i0 = int(fx)
        j0 = int(fy)
        k0 = int(fz)

        tx = fx - i0
        ty = fy - j0
        tz = fz - k0

        if pbc:
            i0 = i0 % nmesh
            j0 = j0 % nmesh
            k0 = k0 % nmesh
            i1 = (i0 + 1) % nmesh
            j1 = (j0 + 1) % nmesh
            k1 = (k0 + 1) % nmesh
        else:
            i0 = min(i0, nmesh - 1)
            j0 = min(j0, nmesh - 1)
            k0 = min(k0, nmesh - 1)
            i1 = min(i0 + 1, nmesh - 1)
            j1 = min(j0 + 1, nmesh - 1)
            k1 = min(k0 + 1, nmesh - 1)

        for c in range(3):
            c000 = field[i0, j0, k0, c]
            c100 = field[i1, j0, k0, c]
            c010 = field[i0, j1, k0, c]
            c001 = field[i0, j0, k1, c]
            c101 = field[i1, j0, k1, c]
            c011 = field[i0, j1, k1, c]
            c110 = field[i1, j1, k0, c]
            c111 = field[i1, j1, k1, c]

            c00 = c000 * (1 - tx) + c100 * tx
            c01 = c001 * (1 - tx) + c101 * tx
            c10 = c010 * (1 - tx) + c110 * tx
            c11 = c011 * (1 - tx) + c111 * tx

            c0 = c00 * (1 - ty) + c10 * ty
            c1 = c01 * (1 - ty) + c11 * ty

            out[idx, c] = c0 * (1 - tz) + c1 * tz

    return out


@njit(inline="always")
def tsc_weight(dx):
    """Triangular-shaped-cloud weight function."""
    dx = abs(dx)
    if dx < 0.5:
        return 0.75 - dx * dx
    elif dx < 1.5:
        return 0.5 * (1.5 - dx) ** 2
    else:
        return 0.0


@njit(parallel=True, fastmath=True)
def interpolate_tsc_vector(pos, field, boxsize, pbc=True, dtype=np.float32):
    """TSC interpolation of a vector field at particle positions."""
    N = pos.shape[0]
    nmesh = field.shape[0]
    out = np.zeros((N, 3), dtype=dtype)
    csx = boxsize[0] / nmesh
    csy = boxsize[1] / nmesh
    csz = boxsize[2] / nmesh

    for idx in prange(N):
        fx = pos[idx, 0] / csx
        fy = pos[idx, 1] / csy
        fz = pos[idx, 2] / csz

        i0 = int(np.floor(fx + 0.5))
        j0 = int(np.floor(fy + 0.5))
        k0 = int(np.floor(fz + 0.5))

        for di in range(-1, 2):
            wx = tsc_weight(fx - (i0 + di))
            if wx == 0.0:
                continue
            i_idx = i0 + di
            if pbc:
                i_idx = i_idx % nmesh
            else:
                i_idx = min(max(i_idx, 0), nmesh - 1)

            for dj in range(-1, 2):
                wy = tsc_weight(fy - (j0 + dj))
                if wy == 0.0:
                    continue
                j_idx = j0 + dj
                if pbc:
                    j_idx = j_idx % nmesh
                else:
                    j_idx = min(max(j_idx, 0), nmesh - 1)

                for dk in range(-1, 2):
                    wz = tsc_weight(fz - (k0 + dk))
                    if wz == 0.0:
                        continue
                    k_idx = k0 + dk
                    if pbc:
                        k_idx = k_idx % nmesh
                    else:
                        k_idx = min(max(k_idx, 0), nmesh - 1)

                    w_tot = wx * wy * wz
                    for c in range(3):
                        out[idx, c] += field[i_idx, j_idx, k_idx, c] * w_tot

    return out


########################################################
#### interpolate potential at particle positions    ####
########################################################

@njit(fastmath=True, parallel=True, cache=True, nogil=True)
def differentiate_potential_jit_cic(
                mesh: np.ndarray, 
                positions: np.ndarray, 
                shifts: np.ndarray, 
                nmesh_dims: np.ndarray, 
                offset_x: int, 
                boxsize: np.ndarray
                ) -> None:
    """
    Interpolates the gravitational potential 'mesh' at particle 'positions' using the Cloud-In-Cell (CIC) scheme.
    Calculates displacement vectors at particle positions and writes the results directly into the 'shifts' buffer.
    """
    
    nx = int(nmesh_dims[0])
    ny = int(nmesh_dims[1])
    nz = int(nmesh_dims[2])
    n_particles = positions.shape[0]

    bs = boxsize

    inv_dx = nx / bs[0]
    inv_dy = ny / bs[1]
    inv_dz = nz / bs[2]

    norm_x = nx / (2.0 * bs[0])
    norm_y = ny / (2.0 * bs[1])
    norm_z = nz / (2.0 * bs[2])

    for ii in prange(n_particles):
        
        gx = positions[ii, 0] * inv_dx
        gy = positions[ii, 1] * inv_dy
        gz = positions[ii, 2] * inv_dz
        
        ix0 = int(gx)
        iy0 = int(gy)
        iz0 = int(gz)
        
        dx = gx - ix0
        dy = gy - iy0
        dz = gz - iz0
        
        ix0_loc = ix0 - offset_x

        ix_m  = (ix0_loc - 1 + nx) % nx
        ix_0  = ix0_loc % nx
        ix_p  = (ix0_loc + 1) % nx
        ix_pp = (ix0_loc + 2) % nx
        
        iy_m  = (iy0 - 1 + ny) % ny
        iy_0  = iy0 % ny
        iy_p  = (iy0 + 1) % ny
        iy_pp = (iy0 + 2) % ny
        
        iz_m  = (iz0 - 1 + nz) % nz
        iz_0  = iz0 % nz
        iz_p  = (iz0 + 1) % nz
        iz_pp = (iz0 + 2) % nz

        px = 0.0
        py = 0.0
        pz = 0.0
        
        # 1. Vertex 000
        wt = (1.0 - dx) * (1.0 - dy) * (1.0 - dz)
        px += (mesh[ix_p, iy_0, iz_0] - mesh[ix_m, iy_0, iz_0]) * wt
        py += (mesh[ix_0, iy_p, iz_0] - mesh[ix_0, iy_m, iz_0]) * wt
        pz += (mesh[ix_0, iy_0, iz_p] - mesh[ix_0, iy_0, iz_m]) * wt

        # 2. Vertex 010
        wt = (1.0 - dx) * dy * (1.0 - dz)
        px += (mesh[ix_p, iy_p, iz_0] - mesh[ix_m, iy_p, iz_0]) * wt
        py += (mesh[ix_0, iy_pp, iz_0] - mesh[ix_0, iy_0, iz_0]) * wt
        pz += (mesh[ix_0, iy_p, iz_p] - mesh[ix_0, iy_p, iz_m]) * wt

        # 3. Vertex 001
        wt = (1.0 - dx) * (1.0 - dy) * dz
        px += (mesh[ix_p, iy_0, iz_p] - mesh[ix_m, iy_0, iz_p]) * wt
        py += (mesh[ix_0, iy_p, iz_p] - mesh[ix_0, iy_m, iz_p]) * wt
        pz += (mesh[ix_0, iy_0, iz_pp] - mesh[ix_0, iy_0, iz_0]) * wt

        # 4. Vertex 011
        wt = (1.0 - dx) * dy * dz
        px += (mesh[ix_p, iy_p, iz_p] - mesh[ix_m, iy_p, iz_p]) * wt
        py += (mesh[ix_0, iy_pp, iz_p] - mesh[ix_0, iy_0, iz_p]) * wt
        pz += (mesh[ix_0, iy_p, iz_pp] - mesh[ix_0, iy_p, iz_0]) * wt

        # 5. Vertex 100
        wt = dx * (1.0 - dy) * (1.0 - dz)
        px += (mesh[ix_pp, iy_0, iz_0] - mesh[ix_0, iy_0, iz_0]) * wt
        py += (mesh[ix_p, iy_p, iz_0] - mesh[ix_p, iy_m, iz_0]) * wt
        pz += (mesh[ix_p, iy_0, iz_p] - mesh[ix_p, iy_0, iz_m]) * wt

        # 6. Vertex 110
        wt = dx * dy * (1.0 - dz)
        px += (mesh[ix_pp, iy_p, iz_0] - mesh[ix_0, iy_p, iz_0]) * wt
        py += (mesh[ix_p, iy_pp, iz_0] - mesh[ix_p, iy_0, iz_0]) * wt
        pz += (mesh[ix_p, iy_p, iz_p] - mesh[ix_p, iy_p, iz_m]) * wt

        # 7. Vertex 101
        wt = dx * (1.0 - dy) * dz
        px += (mesh[ix_pp, iy_0, iz_p] - mesh[ix_0, iy_0, iz_p]) * wt
        py += (mesh[ix_p, iy_p, iz_p] - mesh[ix_p, iy_m, iz_p]) * wt
        pz += (mesh[ix_p, iy_0, iz_pp] - mesh[ix_p, iy_0, iz_0]) * wt

        # 8. Vertex 111
        wt = dx * dy * dz
        px += (mesh[ix_pp, iy_p, iz_p] - mesh[ix_0, iy_p, iz_p]) * wt
        py += (mesh[ix_p, iy_pp, iz_p] - mesh[ix_p, iy_0, iz_p]) * wt
        pz += (mesh[ix_p, iy_p, iz_pp] - mesh[ix_p, iy_p, iz_0]) * wt

        # Positive output (+ grad phi), consistent with the documentation and with TSC
        shifts[ii, 0] = - px * norm_x
        shifts[ii, 1] = - py * norm_y
        shifts[ii, 2] = - pz * norm_z


@njit(fastmath=True, parallel=True, cache=True, nogil=True)
def differentiate_potential_tsc_jit(
                mesh: np.ndarray, 
                positions: np.ndarray, 
                shifts: np.ndarray, 
                nmesh_dims: np.ndarray, 
                boxsize: np.ndarray,
                ) -> None:
    """
    Interpolates the gravitational potential 'mesh' at particle 'positions' using the Triangular Shaped Cloud (TSC) scheme.
    """
    
    nx = int(nmesh_dims[0])
    ny = int(nmesh_dims[1])
    nz = int(nmesh_dims[2])
    
    n_particles = positions.shape[0]
    inv_cell = nmesh_dims / boxsize
    norm = nmesh_dims / (2.0 * boxsize)

    for ii in prange(n_particles):
        gx = positions[ii, 0] * inv_cell[0]
        gy = positions[ii, 1] * inv_cell[1]
        gz = positions[ii, 2] * inv_cell[2]
        
        # Replaced round() with floor(x + 0.5)
        ix_c = int(np.floor(gx + 0.5))
        iy_c = int(np.floor(gy + 0.5))
        iz_c = int(np.floor(gz + 0.5))
        
        dx = gx - ix_c
        dy = gy - iy_c
        dz = gz - iz_c
        
        wx = (0.5*(0.5-dx)**2, 0.75-dx**2, 0.5*(0.5+dx)**2)
        wy = (0.5*(0.5-dy)**2, 0.75-dy**2, 0.5*(0.5+dy)**2)
        wz = (0.5*(0.5-dz)**2, 0.75-dz**2, 0.5*(0.5+dz)**2)

        px = 0.0; py = 0.0; pz = 0.0
        
        for i in range(3):
            ix = (ix_c - 1 + i) % nx
            ix_m, ix_p = (ix - 1) % nx, (ix + 1) % nx
            
            for j in range(3):
                iy = (iy_c - 1 + j) % ny
                iy_m, iy_p = (iy - 1) % ny, (iy + 1) % ny
                
                for k in range(3):
                    iz = (iz_c - 1 + k) % nz
                    iz_m, iz_p = (iz - 1) % nz, (iz + 1) % nz
                    
                    weight = wx[i] * wy[j] * wz[k]
                    
                    px += (mesh[ix_p, iy, iz] - mesh[ix_m, iy, iz]) * weight
                    py += (mesh[ix, iy_p, iz] - mesh[ix, iy_m, iz]) * weight
                    pz += (mesh[ix, iy, iz_p] - mesh[ix, iy, iz_m]) * weight

        # Negative output (- grad phi), consistent with CIC
        shifts[ii, 0] = - px * norm[0]
        shifts[ii, 1] = - py * norm[1]
        shifts[ii, 2] = - pz * norm[2]


###########################
########## GRADIENT
###########################
@njit(parallel=True, fastmath=True)
def gradient_periodic_jit(phi, displacement, dx, dy, dz):
    """Centered finite-difference gradient with periodic indices.

    Writes ``grad phi`` into the pre-allocated ``displacement`` buffer of shape
    ``(Nx, Ny, Nz, 3)``. The minus sign for ``Psi = -grad phi`` is applied by
    the caller, not here.
    """
    nx, ny, nz = phi.shape
    for i in prange(nx):
        ip = (i + 1) % nx
        im = (i - 1 + nx) % nx
        for j in range(ny):
            jp = (j + 1) % ny
            jm = (j - 1 + ny) % ny
            for k in range(nz):
                kp = (k + 1) % nz
                km = (k - 1 + nz) % nz
                displacement[i,j,k,0] = (phi[ip,j,k] - phi[im,j,k]) / (2*dx)
                displacement[i,j,k,1] = (phi[i,jp,k] - phi[i,jm,k]) / (2*dy)
                displacement[i,j,k,2] = (phi[i,j,kp] - phi[i,j,km]) / (2*dz)
