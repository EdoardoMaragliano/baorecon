# multigrid_lib.py

import numpy as np
from numba import njit, prange
from zeldareco.utils.loggers import setup_logger

logger = setup_logger(__name__)

##############################################
#### Prolong the coarse grid to fine grid ####
##############################################

@njit(fastmath=True, parallel=True, nogil=True, cache=True)
def prolong_jit(v2h: np.ndarray, v1h: np.ndarray, nmesh_dims: np.ndarray, localnmeshx: int, offsetx: int):

    """
    Prolongation (Interpolazione).
    v2h: Coarse (Input)
    v1h: Fine (Output)
    nmesh_dims: Global dimensions of the FINE grid [Nx, Ny, Nz]
    """
    # --- CORREZIONE: Deriviamo le dimensioni Coarse da quelle Fine ---
    nx_f = nmesh_dims[0]
    ny_f = nmesh_dims[1]
    nz_f = nmesh_dims[2]
    
    # Coarse dimensions are half of Fine dimensions
    nx_c = nx_f // 2
    ny_c = ny_f // 2
    nz_c = nz_f // 2

    
    # Calcolo Strides (come prima)
    stride_c_z = 1
    stride_c_y = nz_c
    stride_c_yz = ny_c * nz_c
    
    stride_f_z = 1
    stride_f_y = 2 * nz_c
    stride_f_yz = (2 * ny_c) * (2 * nz_c)
    
    # Loop parallelo
    for i1x in prange(localnmeshx):
        i1x0 = stride_f_yz * i1x
        i1xo_global = i1x + offsetx
        
        # --- CORREZIONE LOGICA PERIODICITÀ ---
        # Usiamo nx_c (dimensione globale) per il wrapping corretto
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
                
                # Il resto del calcolo (IF/ELSE) rimane identico...
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
    
    # 1. Dimensioni Fine (Input Globali)
    nx_f = nmesh_dims_fine[0]
    ny_f = nmesh_dims_fine[1]
    nz_f = nmesh_dims_fine[2]
    
    # 2. Dimensioni Coarse (Output Locali/Globali)
    ny_c = ny_f // 2
    nz_c = nz_f // 2
    
    # 3. Strides (Passi in memoria)
    # Per la griglia Fine (Input)
    stride_f_z = 1
    stride_f_y = nz_f
    stride_f_yz = ny_f * nz_f
    
    # Per la griglia Coarse (Output)
    stride_c_z = 1
    stride_c_y = nz_c
    stride_c_yz = ny_c * nz_c
    
    # Loop Parallelo sulla X Coarse
    for ix_c in prange(localnmeshx_c):
        
        # Indice X Globale Coarse
        global_ix_c = ix_c + offsetx_c
        
        # Calcolo indici X sulla griglia Fine
        # Il nodo coarse 'i' corrisponde al nodo fine '2i'
        ix_f_0 = (global_ix_c * 2) % nx_f
        ix_f_p = (ix_f_0 + 1) % nx_f           # +1 (Plus)
        ix_f_m = (ix_f_0 - 1 + nx_f) % nx_f    # -1 (Minus) con wrap-around
        
        # Offset piatti per X
        off_x_0 = ix_f_0 * stride_f_yz
        off_x_p = ix_f_p * stride_f_yz
        off_x_m = ix_f_m * stride_f_yz
        
        # Offset output base
        out_idx_x = ix_c * stride_c_yz
        
        for iy_c in range(ny_c):
            # Calcolo indici Y Fine
            iy_f_0 = (iy_c * 2)
            iy_f_p = (iy_f_0 + 1) % ny_f
            iy_f_m = (iy_f_0 - 1 + ny_f) % ny_f
            
            # Offset piatti per Y
            off_y_0 = iy_f_0 * stride_f_y
            off_y_p = iy_f_p * stride_f_y
            off_y_m = iy_f_m * stride_f_y
            
            out_idx_xy = out_idx_x + (iy_c * stride_c_y)

            for iz_c in range(nz_c):
                # Calcolo indici Z Fine
                iz_f_0 = (iz_c * 2)
                iz_f_p = (iz_f_0 + 1) % nz_f
                iz_f_m = (iz_f_0 - 1 + nz_f) % nz_f
                
                # Offset piatti per Z
                off_z_0 = iz_f_0 * stride_f_z
                off_z_p = iz_f_p * stride_f_z
                off_z_m = iz_f_m * stride_f_z
                
                # Indice Output finale
                ii_out = out_idx_xy + iz_c
                
                # --- SOMMA PESATA (KERNEL 3x3x3) ---
                
                # 1. Centro (Peso 8) - Coordinate (0, 0, 0)
                val = 8.0 * v1h[off_x_0 + off_y_0 + off_z_0]
                
                # 2. Facce (Peso 4) - 6 vicini
                # (±1, 0, 0), (0, ±1, 0), (0, 0, ±1)
                val += 4.0 * (
                    v1h[off_x_p + off_y_0 + off_z_0] + # X+
                    v1h[off_x_m + off_y_0 + off_z_0] + # X-
                    v1h[off_x_0 + off_y_p + off_z_0] + # Y+
                    v1h[off_x_0 + off_y_m + off_z_0] + # Y-
                    v1h[off_x_0 + off_y_0 + off_z_p] + # Z+
                    v1h[off_x_0 + off_y_0 + off_z_m]   # Z-
                )
                
                # 3. Spigoli (Peso 2) - 12 vicini
                # Combinazioni di due ±1
                val += 2.0 * (
                    # Piano XY (Z=0)
                    v1h[off_x_p + off_y_p + off_z_0] + v1h[off_x_m + off_y_p + off_z_0] +
                    v1h[off_x_p + off_y_m + off_z_0] + v1h[off_x_m + off_y_m + off_z_0] +
                    # Piano XZ (Y=0)
                    v1h[off_x_p + off_y_0 + off_z_p] + v1h[off_x_m + off_y_0 + off_z_p] +
                    v1h[off_x_p + off_y_0 + off_z_m] + v1h[off_x_m + off_y_0 + off_z_m] +
                    # Piano YZ (X=0)
                    v1h[off_x_0 + off_y_p + off_z_p] + v1h[off_x_0 + off_y_m + off_z_p] +
                    v1h[off_x_0 + off_y_p + off_z_m] + v1h[off_x_0 + off_y_m + off_z_m]
                )
                
                # 4. Vertici (Peso 1) - 8 vicini
                # Combinazioni di tre ±1
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
            beta: float, damping_factor: float, los_vector: np.ndarray, use_plane_parallel: bool, dtype=np.float64):
    
    nx_global = nmesh_dims[0]
    ny = nmesh_dims[1]
    nz = nmesh_dims[2]
    
    # Pre-allocations for geometry
    cellsize    = np.empty(3, dtype=dtype)
    cellsize2   = np.empty(3, dtype=dtype)
    icellsize2  = np.empty(3, dtype=dtype)
    min_corner_mesh = np.empty(3, dtype=dtype)
    losn        = np.empty(3, dtype=dtype)
    
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
                jac_buffer[ii] = 0.0 if diag == 0.0 else res / diag

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
    
    # --- SETUP GEOMETRICO (Invariato) ---
    cellsize = np.empty(3, dtype=np.float64)
    cellsize2 = np.empty(3, dtype=np.float64)
    icellsize2 = np.empty(3, dtype=np.float64)
    min_corner_mesh = np.empty(3, dtype=np.float64)
    losn = np.empty(3, dtype=np.float64)
    
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
        
        # Indici X
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
                
                # Coefficienti
                denom = (cellsize2[0]*px*px + cellsize2[1]*py*py + cellsize2[2]*pz*pz)
                g = 0.0 if denom == 0.0 else beta / denom
                
                gpx2 = icellsize2[0] + g * px * px
                gpy2 = icellsize2[1] + g * py * py
                gpz2 = icellsize2[2] + g * pz * pz
                
                # Somma dei Vicini
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
                
                # Cross terms (opzionale)
                if beta != 0.0:
                     cross_xy = px * py * (v[ixp_idx+iyp_idx+iz0_idx] + v[ixm_idx+iym_idx+iz0_idx] 
                                         - v[ixm_idx+iyp_idx+iz0_idx] - v[ixp_idx+iym_idx+iz0_idx])
                     cross_xz = px * pz * (v[ixp_idx+iy0_idx+izp_idx] + v[ixm_idx+iy0_idx+izm_idx] 
                                         - v[ixm_idx+iy0_idx+izp_idx] - v[ixp_idx+iy0_idx+izm_idx])
                     cross_yz = py * pz * (v[ix0_idx+iyp_idx+izp_idx] + v[ix0_idx+iym_idx+izm_idx] 
                                         - v[ix0_idx+iym_idx+izp_idx] - v[ix0_idx+iyp_idx+izm_idx])
                     sum_neighbors += (g * 0.5) * (cross_xy + cross_xz + cross_yz)

                diag = 2.0 * (gpx2 + gpy2 + gpz2)
                
                # --- CALCOLO RESIDUO CORRETTO ---
                # r = f - L(v) = f - (Neighbors - diag*v)
                # r = f - Neighbors + diag*v
                res_out[ii] = f[ii] - sum_neighbors + (diag * v[ii])

                if not use_plane_parallel:
                    res_out[ii] -= g * (px * (val_xp - val_xm) + 
                                        py * (val_yp - val_ym) + 
                                        pz * (val_zp - val_zm))

########################################################
#### interpolate potential at particle positions    ####
########################################################

## these are not used right now
## they should replace the current python interpolation in the MultigridSolver class for better performance. 

@njit(fastmath=True, parallel=True, cache=True, nogil=True)
def interpolate_potential_jit_cic(
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

    This function follows a 'prolong-style' interface, modifying the output buffer in-place.

    Parameters
    ----------
    mesh : ndarray (Nx, Ny, Nz)
        The 3D grid containing the gravitational potential.
    positions : ndarray (N_particles, 3)
        The Cartesian positions of the particles.
    shifts : ndarray (N_particles, 3)
        OUTPUT BUFFER. This array is filled with the calculated displacement vectors.
    nmesh_dims : ndarray (3,)
        Global grid dimensions [Nx, Ny, Nz].
    offset_x : int
        Global X-offset (for MPI compatibility). Typically set to 0 for single-node execution.
    boxsize : ndarray (3,)
        Physical dimensions of the box [Lx, Ly, Lz].
    """
    
    # 1. Setup Geometria e Costanti
    nx = int(nmesh_dims[0])
    ny = int(nmesh_dims[1])
    nz = int(nmesh_dims[2])
    n_particles = positions.shape[0]

    # Gestione boxsize (scalare vs array)
    bs = boxsize
    '''np.empty(3, dtype=np.float64)
    if np.isscalar(boxsize):
        bs[:] = boxsize
    else:
        for k in range(3):
            bs[k] = boxsize.flat[k] # .flat per sicurezza se boxsize è strano'''

    # Fattori inversi per convertire pos -> indice
    inv_dx = nx / bs[0]
    inv_dy = ny / bs[1]
    inv_dz = nz / bs[2]

    # Fattori di normalizzazione finale (Derivata centrata -> diviso 2*dx)
    # cellsize = 2.0 * L / N
    # factor = 1 / cellsize = N / (2.0 * L)
    norm_x = nx / (2.0 * bs[0])
    norm_y = ny / (2.0 * bs[1])
    norm_z = nz / (2.0 * bs[2])

    # 2. Loop Parallelo sulle Particelle
    for ii in prange(n_particles):
        
        # --- A. Trova Cella e Pesi ---
        
        # Coordinate in unità griglia
        gx = positions[ii, 0] * inv_dx
        gy = positions[ii, 1] * inv_dy
        gz = positions[ii, 2] * inv_dz
        
        # Indice "basso" della cella (cast a int)
        ix0 = int(gx)
        iy0 = int(gy)
        iz0 = int(gz)
        
        # Pesi CIC (frazione interna alla cella)
        dx = gx - ix0
        dy = gy - iy0
        dz = gz - iz0
        
        # Se offset_x != 0 (MPI), bisognerebbe sottrarlo qui a ix0 per accedere a 'mesh' locale.
        # Assumiamo single-node per ora:
        ix0_loc = ix0 - offset_x

        # --- B. Calcolo Indici Vicini (Periodici) ---
        # Ci servono: i-1 (m), i (0), i+1 (p), i+2 (pp)
        
        # X
        ix_m  = (ix0_loc - 1 + nx) % nx
        ix_0  = ix0_loc % nx
        ix_p  = (ix0_loc + 1) % nx
        ix_pp = (ix0_loc + 2) % nx # Serve per il "plus plus" del codice C
        
        # Y
        iy_m  = (iy0 - 1 + ny) % ny
        iy_0  = iy0 % ny
        iy_p  = (iy0 + 1) % ny
        iy_pp = (iy0 + 2) % ny
        
        # Z
        iz_m  = (iz0 - 1 + nz) % nz
        iz_0  = iz0 % nz
        iz_p  = (iz0 + 1) % nz
        iz_pp = (iz0 + 2) % nz

        # Accumulatori
        px = 0.0
        py = 0.0
        pz = 0.0
        
        # Variabile temporanea per il peso
        wt = 0.0

        # --- C. Somma Pesata (Replica esatta logica C) ---
        
        # 1. Vertice 000 (Sinistra-Basso-Dietro)
        wt = (1.0 - dx) * (1.0 - dy) * (1.0 - dz)
        px -= mesh[ix_0, iy_0, iz_0] * wt
        py += (mesh[ix_0, iy_p, iz_0] - mesh[ix_0, iy_m, iz_0]) * wt
        pz += (mesh[ix_0, iy_0, iz_p] - mesh[ix_0, iy_0, iz_m]) * wt

        # 2. Vertice 010 (Sinistra-Alto-Dietro)
        wt = (1.0 - dx) * dy * (1.0 - dz)
        px -= mesh[ix_0, iy_p, iz_0] * wt
        py += (mesh[ix_0, iy_pp, iz_0] - mesh[ix_0, iy_0, iz_0]) * wt # Nota: iy_pp
        pz += (mesh[ix_0, iy_p, iz_p] - mesh[ix_0, iy_p, iz_m]) * wt

        # 3. Vertice 001 (Sinistra-Basso-Avanti)
        wt = (1.0 - dx) * (1.0 - dy) * dz
        px -= mesh[ix_0, iy_0, iz_p] * wt
        py += (mesh[ix_0, iy_p, iz_p] - mesh[ix_0, iy_m, iz_p]) * wt
        pz += (mesh[ix_0, iy_0, iz_pp] - mesh[ix_0, iy_0, iz_0]) * wt # Nota: iz_pp

        # 4. Vertice 011 (Sinistra-Alto-Avanti)
        wt = (1.0 - dx) * dy * dz
        px -= mesh[ix_0, iy_p, iz_p] * wt
        py += (mesh[ix_0, iy_pp, iz_p] - mesh[ix_0, iy_0, iz_p]) * wt
        pz += (mesh[ix_0, iy_p, iz_pp] - mesh[ix_0, iy_p, iz_0]) * wt

        # 5. Vertice 100 (Destra-Basso-Dietro) -> Qui PX cambia segno (+)
        wt = dx * (1.0 - dy) * (1.0 - dz)
        px += mesh[ix_p, iy_0, iz_0] * wt
        py += (mesh[ix_p, iy_p, iz_0] - mesh[ix_p, iy_m, iz_0]) * wt
        pz += (mesh[ix_p, iy_0, iz_p] - mesh[ix_p, iy_0, iz_m]) * wt

        # 6. Vertice 110 (Destra-Alto-Dietro)
        wt = dx * dy * (1.0 - dz)
        px += mesh[ix_p, iy_p, iz_0] * wt
        py += (mesh[ix_p, iy_pp, iz_0] - mesh[ix_p, iy_0, iz_0]) * wt
        pz += (mesh[ix_p, iy_p, iz_p] - mesh[ix_p, iy_p, iz_m]) * wt

        # 7. Vertice 101 (Destra-Basso-Avanti)
        wt = dx * (1.0 - dy) * dz
        px += mesh[ix_p, iy_0, iz_p] * wt
        py += (mesh[ix_p, iy_p, iz_p] - mesh[ix_p, iy_m, iz_p]) * wt
        pz += (mesh[ix_p, iy_0, iz_pp] - mesh[ix_p, iy_0, iz_0]) * wt

        # 8. Vertice 111 (Destra-Alto-Avanti)
        wt = dx * dy * dz
        px += mesh[ix_p, iy_p, iz_p] * wt
        py += (mesh[ix_p, iy_pp, iz_p] - mesh[ix_p, iy_0, iz_p]) * wt
        pz += (mesh[ix_p, iy_p, iz_pp] - mesh[ix_p, iy_p, iz_0]) * wt

        # --- D. Scrittura Output Normalizzato ---
        shifts[ii, 0] = px * norm_x
        shifts[ii, 1] = py * norm_y
        shifts[ii, 2] = pz * norm_z


@njit(fastmath=True, parallel=True, cache=True, nogil=True)
def interpolate_potential_tsc_jit(
                mesh: np.ndarray, 
                positions: np.ndarray, 
                shifts: np.ndarray, 
                nmesh_dims: np.ndarray, 
                boxsize: np.ndarray,
                ) -> None:
    """
    Interpolates the gravitational potential 'mesh' at particle 'positions' using the Triangular Shaped Cloud (TSC) scheme.
    Calculates displacement vectors at particle positions and writes the results directly into the 'shifts' buffer.

    This function follows a 'prolong-style' interface, modifying the output buffer in-place.

    Parameters
    ----------
    mesh : ndarray (Nx, Ny, Nz)
        The 3D grid containing the gravitational potential.
    positions : ndarray (N_particles, 3)
        The Cartesian positions of the particles.
    shifts : ndarray (N_particles, 3)
        OUTPUT BUFFER. This array is filled with the calculated displacement vectors.
    nmesh_dims : ndarray (3,)
        Global grid dimensions [Nx, Ny, Nz].
    offset_x : int
        Global X-offset (for MPI compatibility). Typically set to 0 for single-node execution.
    boxsize : ndarray (3,)
        Physical dimensions of the box [Lx, Ly, Lz].
    """
    
    nx, ny, nz = nmesh_dims
    n_particles = positions.shape[0]
    inv_cell = nmesh_dims / boxsize
    norm = nmesh_dims / (2.0 * boxsize) # Per derivata centrata

    for ii in prange(n_particles):
        # 1. Trova il nodo più vicino (nearest grid point)
        # Usiamo round() per centrare il TSC sul nodo più vicino
        gx = positions[ii, 0] * inv_cell[0]
        gy = positions[ii, 1] * inv_cell[1]
        gz = positions[ii, 2] * inv_cell[2]
        
        ix_c = int(round(gx))
        iy_c = int(round(gy))
        iz_c = int(round(gz))
        
        # Distanze dal centro (da -0.5 a 0.5)
        dx = gx - ix_c
        dy = gy - iy_c
        dz = gz - iz_c
        
        # 2. Calcola i pesi TSC 1D
        # Asse X
        wx = (0.5*(0.5-dx)**2, 0.75-dx**2, 0.5*(0.5+dx)**2)
        # Asse Y
        wy = (0.5*(0.5-dy)**2, 0.75-dy**2, 0.5*(0.5+dy)**2)
        # Asse Z
        wz = (0.5*(0.5-dz)**2, 0.75-dz**2, 0.5*(0.5+dz)**2)

        # 3. Loop 3x3x3 sui 27 vicini
        px = 0.0; py = 0.0; pz = 0.0
        
        for i in range(3):
            # Indici X (m, c, p) con PBC
            ix = (ix_c - 1 + i) % nx
            # Pre-calcoliamo i vicini per la derivata centrale su X
            ix_m, ix_p = (ix - 1) % nx, (ix + 1) % nx
            
            for j in range(3):
                iy = (iy_c - 1 + j) % ny
                iy_m, iy_p = (iy - 1) % ny, (iy + 1) % ny
                
                for k in range(3):
                    iz = (iz_c - 1 + k) % nz
                    iz_m, iz_p = (iz - 1) % nz, (iz + 1) % nz
                    
                    # Peso 3D combinato
                    weight = wx[i] * wy[j] * wz[k]
                    
                    # Accumulo gradienti (Method A inside Method B)
                    px += (mesh[ix_p, iy, iz] - mesh[ix_m, iy, iz]) * weight
                    py += (mesh[ix, iy_p, iz] - mesh[ix, iy_m, iz]) * weight
                    pz += (mesh[ix, iy, iz_p] - mesh[ix, iy, iz_m]) * weight

        # 4. Scrittura con segno meno (Psi = -grad_phi)
        shifts[ii, 0] = -px * norm[0]
        shifts[ii, 1] = -py * norm[1]
        shifts[ii, 2] = -pz * norm[2]