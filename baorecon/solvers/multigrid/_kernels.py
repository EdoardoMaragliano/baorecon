# multigrid_lib.py

import numpy as np
from numba import njit, prange
from baorecon.utils.loggers import setup_logger

logger = setup_logger(__name__)

##############################################
#### Prolong the coarse grid to fine grid ####
##############################################

@njit(fastmath=True, parallel=True, nogil=True, cache=True)
def prolong_jit(v2h: np.ndarray, v1h: np.ndarray, nmesh_dims: np.ndarray, localnmeshx: int, offsetx: int):

    """
    Prolongation (interpolation).
    v2h: Coarse (Input)
    v1h: Fine (Output)
    nmesh_dims: Global dimensions of the FINE grid [Nx, Ny, Nz]
    """
    # --- Derive the coarse dimensions from the fine ones ---
    nx_f = nmesh_dims[0]
    ny_f = nmesh_dims[1]
    nz_f = nmesh_dims[2]
    
    # Coarse dimensions are half of Fine dimensions
    nx_c = nx_f // 2
    ny_c = ny_f // 2
    nz_c = nz_f // 2

    
    # Compute strides
    stride_c_z = 1
    stride_c_y = nz_c
    stride_c_yz = ny_c * nz_c
    
    stride_f_z = 1
    stride_f_y = 2 * nz_c
    stride_f_yz = (2 * ny_c) * (2 * nz_c)
    
    # Parallel loop
    for i1x in prange(localnmeshx):
        i1x0 = stride_f_yz * i1x
        i1xo_global = i1x + offsetx

        # --- Periodicity logic ---
        # Use nx_c (global dimension) for correct wrapping
        ix_c_base = (i1xo_global // 2) % nx_c
        ix_c_next = ((i1xo_global // 2) + 1) % nx_c 
        # -------------------------------------
        
        ix0 = stride_c_yz * ix_c_base
        ixp = stride_c_yz * ix_c_next
        
        for iy in range(ny_c):
            iy0 = stride_c_y * iy
            iyp = stride_c_y * ((iy + 1) % ny_c)
            
            i1y0 = stride_f_y * 2 * iy
            i1yp = i1y0 + stride_f_y
            
            for iz in range(nz_c):
                iz0 = stride_c_z * iz
                izp = stride_c_z * ((iz + 1) % nz_c)
                
                i1z0 = stride_f_z * 2 * iz
                i1zp = i1z0 + stride_f_z
                
                ii0 = ix0 + iy0 + iz0
                
                # The rest of the computation (IF/ELSE) is unchanged...
                if i1xo_global % 2 == 0:
                    v1h[i1x0 + i1y0 + i1z0] = v2h[ii0]
                    val_y = (v2h[ii0] + v2h[ix0 + iyp + iz0]) * 0.5
                    v1h[i1x0 + i1yp + i1z0] = val_y
                    val_z = (v2h[ii0] + v2h[ix0 + iy0 + izp]) * 0.5
                    v1h[i1x0 + i1y0 + i1zp] = val_z
                    val_yz = (v2h[ii0] + v2h[ix0 + iyp + iz0] + 
                              v2h[ix0 + iy0 + izp] + v2h[ix0 + iyp + izp]) * 0.25
                    v1h[i1x0 + i1yp + i1zp] = val_yz
                else:
                    v1h[i1x0 + i1y0 + i1z0] = (v2h[ii0] + v2h[ixp + iy0 + iz0]) * 0.5
                    v1h[i1x0 + i1yp + i1z0] = (v2h[ii0] + v2h[ixp + iy0 + iz0] + 
                                               v2h[ix0 + iyp + iz0] + v2h[ixp + iyp + iz0]) * 0.25
                    v1h[i1x0 + i1y0 + i1zp] = (v2h[ii0] + v2h[ixp + iy0 + iz0] + 
                                               v2h[ix0 + iy0 + izp] + v2h[ixp + iy0 + izp]) * 0.25
                    term1 = v2h[ii0] + v2h[ixp + iy0 + iz0]
                    term2 = v2h[ix0 + iyp + iz0] + v2h[ix0 + iy0 + izp]
                    term3 = v2h[ixp + iyp + iz0] + v2h[ixp + iy0 + izp]
                    term4 = v2h[ix0 + iyp + izp] + v2h[ixp + iyp + izp]
                    v1h[i1x0 + i1yp + i1zp] = (term1 + term2 + term3 + term4) * 0.125

###############################################
#### reduce the fine grid to coarse grid ####
###############################################



@njit(fastmath=True, parallel=True, nogil=True, cache=True)
def reduce_jit(v1h: np.ndarray, v2h: np.ndarray, nmesh_dims_fine: np.ndarray, 
               localnmeshx_c: int, offsetx_c: int):
    """
    Performs restriction (Fine -> Coarse) using Full Weighting.
    Strides are required to access the 3D grid in 1D flattened array.
    It supports periodic boundary conditions.
    It allows parallelization over the coarse X dimension for future MPI use.

    Parameters:
    -----------
    v1h : array 1D float
        Fine grid (Input).
    v2h : array 1D float
        Coarse grid (Output).
    nmesh_dims_fine : array-like
        Global dimensions of the FINE grid [Nx, Ny, Nz].
    localnmeshx_c : int
        Local X dimension of the Coarse slice (output).
    offsetx_c : int
        Global X offset of the Coarse slice.
    """
    
    # 1. Fine dimensions (global input)
    nx_f = nmesh_dims_fine[0]
    ny_f = nmesh_dims_fine[1]
    nz_f = nmesh_dims_fine[2]

    # 2. Coarse dimensions (local/global output)
    ny_c = ny_f // 2
    nz_c = nz_f // 2

    # 3. Strides (memory steps)
    # For the fine grid (input)
    stride_f_z = 1
    stride_f_y = nz_f
    stride_f_yz = ny_f * nz_f

    # For the coarse grid (output)
    stride_c_z = 1
    stride_c_y = nz_c
    stride_c_yz = ny_c * nz_c
    
    # Parallel loop over the coarse X
    for ix_c in prange(localnmeshx_c):

        # Global coarse X index
        global_ix_c = ix_c + offsetx_c

        # Compute X indices on the fine grid
        # Coarse node 'i' maps to fine node '2i'
        ix_f_0 = (global_ix_c * 2) % nx_f
        ix_f_p = (ix_f_0 + 1) % nx_f           # +1 (Plus)
        ix_f_m = (ix_f_0 - 1 + nx_f) % nx_f    # -1 (Minus) with wrap-around

        # Flat offsets for X
        off_x_0 = ix_f_0 * stride_f_yz
        off_x_p = ix_f_p * stride_f_yz
        off_x_m = ix_f_m * stride_f_yz

        # Base output offset
        out_idx_x = ix_c * stride_c_yz

        for iy_c in range(ny_c):
            # Compute fine Y indices
            iy_f_0 = (iy_c * 2)
            iy_f_p = (iy_f_0 + 1) % ny_f
            iy_f_m = (iy_f_0 - 1 + ny_f) % ny_f

            # Flat offsets for Y
            off_y_0 = iy_f_0 * stride_f_y
            off_y_p = iy_f_p * stride_f_y
            off_y_m = iy_f_m * stride_f_y

            out_idx_xy = out_idx_x + (iy_c * stride_c_y)

            for iz_c in range(nz_c):
                # Compute fine Z indices
                iz_f_0 = (iz_c * 2)
                iz_f_p = (iz_f_0 + 1) % nz_f
                iz_f_m = (iz_f_0 - 1 + nz_f) % nz_f

                # Flat offsets for Z
                off_z_0 = iz_f_0 * stride_f_z
                off_z_p = iz_f_p * stride_f_z
                off_z_m = iz_f_m * stride_f_z

                # Final output index
                ii_out = out_idx_xy + iz_c

                # --- WEIGHTED SUM (3x3x3 KERNEL) ---

                # 1. Center (weight 8) - coordinates (0, 0, 0)
                val = 8.0 * v1h[off_x_0 + off_y_0 + off_z_0]

                # 2. Faces (weight 4) - 6 neighbors
                # (±1, 0, 0), (0, ±1, 0), (0, 0, ±1)
                val += 4.0 * (
                    v1h[off_x_p + off_y_0 + off_z_0] + # X+
                    v1h[off_x_m + off_y_0 + off_z_0] + # X-
                    v1h[off_x_0 + off_y_p + off_z_0] + # Y+
                    v1h[off_x_0 + off_y_m + off_z_0] + # Y-
                    v1h[off_x_0 + off_y_0 + off_z_p] + # Z+
                    v1h[off_x_0 + off_y_0 + off_z_m]   # Z-
                )
                
                # 3. Edges (weight 2) - 12 neighbors
                # Combinations of two ±1
                val += 2.0 * (
                    # XY plane (Z=0)
                    v1h[off_x_p + off_y_p + off_z_0] + v1h[off_x_m + off_y_p + off_z_0] +
                    v1h[off_x_p + off_y_m + off_z_0] + v1h[off_x_m + off_y_m + off_z_0] +
                    # XZ plane (Y=0)
                    v1h[off_x_p + off_y_0 + off_z_p] + v1h[off_x_m + off_y_0 + off_z_p] +
                    v1h[off_x_p + off_y_0 + off_z_m] + v1h[off_x_m + off_y_0 + off_z_m] +
                    # YZ plane (X=0)
                    v1h[off_x_0 + off_y_p + off_z_p] + v1h[off_x_0 + off_y_m + off_z_p] +
                    v1h[off_x_0 + off_y_p + off_z_m] + v1h[off_x_0 + off_y_m + off_z_m]
                )

                # 4. Vertices (weight 1) - 8 neighbors
                # Combinations of three ±1
                val += (
                    v1h[off_x_p + off_y_p + off_z_p] + v1h[off_x_m + off_y_p + off_z_p] +
                    v1h[off_x_p + off_y_m + off_z_p] + v1h[off_x_m + off_y_m + off_z_p] +
                    v1h[off_x_p + off_y_p + off_z_m] + v1h[off_x_m + off_y_p + off_z_m] +
                    v1h[off_x_p + off_y_m + off_z_m] + v1h[off_x_m + off_y_m + off_z_m]
                )
                
                v2h[ii_out] = val / 64.0


###############################################
#### jacobi iteration on grid ################
###############################################

@njit(fastmath=True, parallel=True, nogil=True, cache=True) 
def jacobi_jit(v: np.ndarray, f: np.ndarray, jac_buffer: np.ndarray, nmesh_dims: np.ndarray, 
            localnmeshx: int, offsetx: int, boxsize: np.ndarray, boxcenter: np.ndarray, 
            beta: float, damping_factor: float, los_vector: np.ndarray, use_plane_parallel: bool):
    
    nx_global = nmesh_dims[0]
    ny = nmesh_dims[1]
    nz = nmesh_dims[2]

    grid_dtype = v.dtype
    grid_type_func = v.dtype.type
    
    # Pre-allocations for geometry
    cellsize    = np.empty(3, dtype=grid_dtype)
    cellsize2   = np.empty(3, dtype=grid_dtype)
    icellsize2  = np.empty(3, dtype=grid_dtype)
    min_corner_mesh = np.empty(3, dtype=grid_dtype)
    losn        = np.empty(3, dtype=grid_dtype)
    
    for i in range(3):
        cellsize[i] = boxsize[i] / nmesh_dims[i]
        cellsize2[i] = cellsize[i] * cellsize[i]
        icellsize2[i] = 1.0 / cellsize2[i]
        min_corner_mesh[i] = (boxcenter[i] - boxsize[i] / 2.0) / cellsize[i]
        if use_plane_parallel:
            losn[i] = los_vector[i] / cellsize[i]

    stride_z = 1
    stride_y = nz
    stride_yz = ny * nz 

    for ix_local in prange(localnmeshx):
        # 1. Global X coordinate for physical position (LOS)
        ix_global = ix_local + offsetx
        px = losn[0] if use_plane_parallel else (ix_global + min_corner_mesh[0])
        
        # 2. FIXED: Local X coordinates for safe memory access within the slice
        ix_p_local = (ix_local + 1) % localnmeshx
        ix_m_local = (ix_local - 1 + localnmeshx) % localnmeshx
        
        ix0_idx = ix_local * stride_yz
        ixp_idx = ix_p_local * stride_yz
        ixm_idx = ix_m_local * stride_yz

        for iy in range(ny):
            py = losn[1] if use_plane_parallel else (iy + min_corner_mesh[1])
            
            iy0_idx = iy * stride_y
            iyp_idx = ((iy + 1) % ny) * stride_y
            iym_idx = ((iy - 1 + ny) % ny) * stride_y

            for iz in range(nz):
                pz = losn[2] if use_plane_parallel else (iz + min_corner_mesh[2])
                
                iz0_idx = iz * stride_z
                izp_idx = ((iz + 1) % nz) * stride_z
                izm_idx = ((iz - 1 + nz) % nz) * stride_z
                
                ii = ix0_idx + iy0_idx + iz0_idx
                
                # Equation coefficients
                denom = (cellsize2[0]*px*px + cellsize2[1]*py*py + cellsize2[2]*pz*pz)
                g = 0.0 if denom == 0.0 else beta / denom
                
                gpx2 = icellsize2[0] + g * px * px
                gpy2 = icellsize2[1] + g * py * py
                gpz2 = icellsize2[2] + g * pz * pz
                
                # Second derivative neighbor terms
                val_xp = v[ixp_idx + iy0_idx + iz0_idx]
                val_xm = v[ixm_idx + iy0_idx + iz0_idx]
                term_x = gpx2 * (val_xp + val_xm)
                
                val_yp = v[ix0_idx + iyp_idx + iz0_idx]
                val_ym = v[ix0_idx + iym_idx + iz0_idx]
                term_y = gpy2 * (val_yp + val_ym)
                
                val_zp = v[ix0_idx + iy0_idx + izp_idx]
                val_zm = v[ix0_idx + iy0_idx + izm_idx]
                term_z = gpz2 * (val_zp + val_zm)
                
                
                res = -f[ii] + term_x + term_y + term_z 
                
                # Cross terms (Asymmetric lines-of-sight coupling)
                if beta != 0.0:
                    cross_xy = px * py * (v[ixp_idx + iyp_idx + iz0_idx] + v[ixm_idx + iym_idx + iz0_idx] 
                                        - v[ixm_idx + iyp_idx + iz0_idx] - v[ixp_idx + iym_idx + iz0_idx])
                    cross_xz = px * pz * (v[ixp_idx + iy0_idx + izp_idx] + v[ixm_idx + iy0_idx + izm_idx] 
                                        - v[ixm_idx + iy0_idx + izp_idx] - v[ixp_idx + iy0_idx + izm_idx])
                    cross_yz = py * pz * (v[ix0_idx + iyp_idx + izp_idx] + v[ix0_idx + iym_idx + izm_idx] 
                                        - v[ix0_idx + iym_idx + izp_idx] - v[ix0_idx + iyp_idx + izm_idx])
                    res += (g * 0.5) * (cross_xy + cross_xz + cross_yz)

                # ==========================================================
                # ADDED: Wide-Angle First Derivative Corrections
                # ==========================================================
                if not use_plane_parallel:
                    res += g * (px * (val_xp - val_xm) + 
                                py * (val_yp - val_ym) + 
                                pz * (val_zp - val_zm))
                # ==========================================================

                # Diagonal normalization step
                diag = 2.0 * (gpx2 + gpy2 + gpz2)
                jac_buffer[ii] = grid_type_func(0.0) if diag == 0.0 else grid_type_func(res / diag)

    # Final damped Jacobi update
    for i in prange(localnmeshx * stride_yz):
        v[i] = (1.0 - damping_factor) * v[i] + damping_factor * jac_buffer[i]

###############################################
#### compute residual on grid ################
###############################################

@njit(fastmath=True, parallel=True, nogil=True, cache=True)
def residual_jit(v: np.ndarray, f: np.ndarray, res_out: np.ndarray, nmesh_dims: np.ndarray,
                 localnmeshx: int, offsetx: int, boxsize: np.ndarray, boxcenter: np.ndarray, 
                 beta: float, los_vector: np.ndarray, use_plane_parallel: bool):
    """
    Computes the residual r = f - L(v) on the grid for the modified differential equation.
    \[ \nabla \cdot \left( \mathbf{G} \cdot \nabla v \right) = f \]
    Parameters:
    -----------
    v : array 1D float
        Grid of input (flattened).
    f : array 1D float
        Source grid (flattened). In our case, f = -delta.
    res_out : array 1D float
        Output residual grid (flattened).
    nmesh_dims : array-like
        Global grid dimensions [Nx, Ny, Nz].
    localnmeshx : int
        Local X dimension of the slice (for MPI).
    offsetx : int
        Global X offset of the slice (for MPI).
    boxsize : array-like
        Physical box dimensions [Lx, Ly, Lz].
    boxcenter : array-like
        Physical box center [Cx, Cy, Cz].
    beta : float
        Modification factor for the G tensor.
    los_vector : array-like
        Line-of-sight vector (for plane-parallel approximation only, ignored otherwise).
    use_plane_parallel : bool
        If True, uses the plane-parallel approximation. 
    """
    nx_global = nmesh_dims[0]
    ny = nmesh_dims[1]
    nz = nmesh_dims[2]

    grid_dtype = v.dtype
    grid_type_func = v.dtype.type
    
    # --- GEOMETRIC SETUP ---
    cellsize = np.empty(3, dtype=grid_dtype)
    cellsize2 = np.empty(3, dtype=grid_dtype)
    icellsize2 = np.empty(3, dtype=grid_dtype)
    min_corner_mesh = np.empty(3, dtype=grid_dtype)
    losn = np.empty(3, dtype=grid_dtype)
    
    for i in range(3):
        cellsize[i] = boxsize[i] / nmesh_dims[i]
        cellsize2[i] = cellsize[i] * cellsize[i]
        icellsize2[i] = 1.0 / cellsize2[i]
        min_corner_mesh[i] = (boxcenter[i] - boxsize[i] / 2.0) / cellsize[i]
        if use_plane_parallel:
            losn[i] = los_vector[i] / cellsize[i]

    stride_z = 1
    stride_y = nz
    stride_yz = ny * nz 

    for ix_local in prange(localnmeshx):
        ix_global = ix_local + offsetx
        ix0_idx = ix_local * stride_yz
        
        # X indices
        ix_p_global = (ix_global + 1) % nx_global
        ix_m_global = (ix_global - 1 + nx_global) % nx_global
        ixp_idx = ix_p_global * stride_yz
        ixm_idx = ix_m_global * stride_yz
        
        px = losn[0] if use_plane_parallel else (ix_global + min_corner_mesh[0])

        for iy in range(ny):
            iy0_idx = iy * stride_y
            iyp_idx = ((iy + 1) % ny) * stride_y
            iym_idx = ((iy - 1 + ny) % ny) * stride_y
            py = losn[1] if use_plane_parallel else (iy + min_corner_mesh[1])

            for iz in range(nz):
                iz0_idx = iz * stride_z
                izp_idx = ((iz + 1) % nz) * stride_z
                izm_idx = ((iz - 1 + nz) % nz) * stride_z
                pz = losn[2] if use_plane_parallel else (iz + min_corner_mesh[2])
                
                ii = ix0_idx + iy0_idx + iz0_idx
                
                # Coefficients
                denom = (cellsize2[0]*px*px + cellsize2[1]*py*py + cellsize2[2]*pz*pz)
                g = grid_type_func(0.0) if denom == 0.0 else grid_type_func( beta / denom )
                
                gpx2 = icellsize2[0] + g * px * px
                gpy2 = icellsize2[1] + g * py * py
                gpz2 = icellsize2[2] + g * pz * pz
                
                # Sum of neighbors
                val_xp = v[ixp_idx + iy0_idx + iz0_idx]
                val_xm = v[ixm_idx + iy0_idx + iz0_idx]
                term_x = gpx2 * (val_xp + val_xm)
                
                val_yp = v[ix0_idx + iyp_idx + iz0_idx]
                val_ym = v[ix0_idx + iym_idx + iz0_idx]
                term_y = gpy2 * (val_yp + val_ym)
                
                val_zp = v[ix0_idx + iy0_idx + izp_idx]
                val_zm = v[ix0_idx + iy0_idx + izm_idx]
                term_z = gpz2 * (val_zp + val_zm)
                
                sum_neighbors = term_x + term_y + term_z
                
                # Cross terms (optional)
                if beta != 0.0:
                     cross_xy = px * py * (v[ixp_idx+iyp_idx+iz0_idx] + v[ixm_idx+iym_idx+iz0_idx]
                                         - v[ixm_idx+iyp_idx+iz0_idx] - v[ixp_idx+iym_idx+iz0_idx])
                     cross_xz = px * pz * (v[ixp_idx+iy0_idx+izp_idx] + v[ixm_idx+iy0_idx+izm_idx] 
                                         - v[ixm_idx+iy0_idx+izp_idx] - v[ixp_idx+iy0_idx+izm_idx])
                     cross_yz = py * pz * (v[ix0_idx+iyp_idx+izp_idx] + v[ix0_idx+iym_idx+izm_idx] 
                                         - v[ix0_idx+iym_idx+izp_idx] - v[ix0_idx+iyp_idx+izm_idx])
                     sum_neighbors += (g * 0.5) * (cross_xy + cross_xz + cross_yz)

                diag = grid_type_func(2.0) * (gpx2 + gpy2 + gpz2)
                
                # --- CORRECT RESIDUAL COMPUTATION ---
                # r = f - L(v) = f - (Neighbors - diag*v)
                # r = f - Neighbors + diag*v
                res_out[ii] = f[ii] - sum_neighbors + (diag * v[ii])

                if not use_plane_parallel:
                    res_out[ii] -= g * (px * (val_xp - val_xm) + 
                                        py * (val_yp - val_ym) + 
                                        pz * (val_zp - val_zm))

########################################################
#### multicolor gauss seidel - alternative to full MG ####
########################################################

@njit(fastmath=True, parallel=True, nogil=True, cache=True) 
def multicolor_gs_jit(v: np.ndarray, f: np.ndarray, nmesh_dims: np.ndarray, 
                      localnmeshx: int, offsetx: int, boxsize: np.ndarray, boxcenter: np.ndarray, 
                      beta: float, los_vector: np.ndarray, use_plane_parallel: bool, 
                      target_color: int):
    
    nx_global = nmesh_dims[0]
    ny = nmesh_dims[1]
    nz = nmesh_dims[2]
    
    grid_dtype = v.dtype
    
    cellsize    = np.empty(3, dtype=grid_dtype)
    cellsize2   = np.empty(3, dtype=grid_dtype)
    icellsize2  = np.empty(3, dtype=grid_dtype)
    min_corner_mesh = np.empty(3, dtype=grid_dtype)
    losn        = np.empty(3, dtype=grid_dtype)
    
    for i in range(3):
        cellsize[i] = boxsize[i] / nmesh_dims[i]
        cellsize2[i] = cellsize[i] * cellsize[i]
        icellsize2[i] = 1.0 / cellsize2[i]
        min_corner_mesh[i] = (boxcenter[i] - boxsize[i] / 2.0) / cellsize[i]
        if use_plane_parallel:
            losn[i] = los_vector[i] / cellsize[i]

    stride_z = 1
    stride_y = nz
    stride_yz = ny * nz 

    # --- COMPUTE PARITIES AND START INDICES ---
    # Decode target_color (0-7) into the original parities (0 or 1)
    target_z_parity = (target_color // 4) % 2
    target_y_parity = (target_color // 2) % 2
    target_x_parity = target_color % 2

    # The local X index (ix_local) must align with the global X parity.
    # If the offset parity matches the target, start at 0, otherwise at 1.
    start_x = 0 if (offsetx % 2 == target_x_parity) else 1

    # Calculate how many points satisfy the condition
    # This is equivalent to range(start_x, localnmeshx, 2)
    num_iterations = (localnmeshx - start_x + 1) // 2

    # Loop with step 2 over all axes
    for i in prange(num_iterations):
        # Map the parallel index 'i' back to your required grid index
        ix_local = start_x + (i * 2)
        ix_global = ix_local + offsetx
        
        px = losn[0] if use_plane_parallel else (ix_global + min_corner_mesh[0])
        ix_p_local = (ix_local + 1) % localnmeshx
        ix_m_local = (ix_local - 1 + localnmeshx) % localnmeshx
        
        ix0_idx = ix_local * stride_yz
        ixp_idx = ix_p_local * stride_yz
        ixm_idx = ix_m_local * stride_yz

        for iy in range(target_y_parity, ny, 2):
            py = losn[1] if use_plane_parallel else (iy + min_corner_mesh[1])
            iy0_idx = iy * stride_y
            iyp_idx = ((iy + 1) % ny) * stride_y
            iym_idx = ((iy - 1 + ny) % ny) * stride_y

            for iz in range(target_z_parity, nz, 2):
                pz = losn[2] if use_plane_parallel else (iz + min_corner_mesh[2])
                
                iz0_idx = iz * stride_z
                izp_idx = ((iz + 1) % nz) * stride_z
                izm_idx = ((iz - 1 + nz) % nz) * stride_z
                
                ii = ix0_idx + iy0_idx + iz0_idx
                
                denom = (cellsize2[0]*px*px + cellsize2[1]*py*py + cellsize2[2]*pz*pz)
                g = 0.0 if denom == 0.0 else beta / denom
                
                gpx2 = icellsize2[0] + g * px * px
                gpy2 = icellsize2[1] + g * py * py
                gpz2 = icellsize2[2] + g * pz * pz
                
                val_xp = v[ixp_idx + iy0_idx + iz0_idx]
                val_xm = v[ixm_idx + iy0_idx + iz0_idx]
                term_x = gpx2 * (val_xp + val_xm)
                
                val_yp = v[ix0_idx + iyp_idx + iz0_idx]
                val_ym = v[ix0_idx + iym_idx + iz0_idx]
                term_y = gpy2 * (val_yp + val_ym)
                
                val_zp = v[ix0_idx + iy0_idx + izp_idx]
                val_zm = v[ix0_idx + iy0_idx + izm_idx]
                term_z = gpz2 * (val_zp + val_zm)
                
                res = -f[ii] + term_x + term_y + term_z 
                
                if beta != 0.0:
                    cross_xy = px * py * (v[ixp_idx + iyp_idx + iz0_idx] + v[ixm_idx + iym_idx + iz0_idx] 
                                        - v[ixm_idx + iyp_idx + iz0_idx] - v[ixp_idx + iym_idx + iz0_idx])
                    cross_xz = px * pz * (v[ixp_idx + iy0_idx + izp_idx] + v[ixm_idx + iy0_idx + izm_idx] 
                                        - v[ixm_idx + iy0_idx + izp_idx] - v[ixp_idx + iy0_idx + izm_idx])
                    cross_yz = py * pz * (v[ix0_idx + iyp_idx + izp_idx] + v[ix0_idx + iym_idx + izm_idx] 
                                        - v[ix0_idx + iym_idx + izp_idx] - v[ix0_idx + iyp_idx + izm_idx])
                    res += (g * 0.5) * (cross_xy + cross_xz + cross_yz)

                if not use_plane_parallel:
                    res += g * (px * (val_xp - val_xm) + 
                                py * (val_yp - val_ym) + 
                                pz * (val_zp - val_zm))

                diag = 2.0 * (gpx2 + gpy2 + gpz2)
                
                if diag != 0.0:
                    v[ii] = res / diag

