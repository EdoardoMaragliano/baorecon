import numpy as np
import scipy.fft as sfft
import time
from numba import njit, prange
from scipy.interpolate import interpn
from zeldareco.utils.loggers import setup_logger
logger = setup_logger(__name__)

def project_vector_field(vector_field1: np.ndarray, vector_field2: np.ndarray) -> np.ndarray:
        """
        Compute the projection of a vector field along another vector field, at each point in the grid.

        Parameters
        ----------
        vector_field1 : array_like
            Array of shape (N, N, N, 3) containing the vector field on the mesh.
        vector_field2 : array_like
            Array of shape (N, N, N, 3) containing the vector field along which to project.

        Returns
        -------
        projection : array_like
            Array of shape (N, N, N, 3) containing the projection of the vector field along another vector field.
        """

        # normalize the vector field along which to project
        norm = np.linalg.norm(vector_field2, axis=-1, keepdims=False)

        # avoid division by zero
        mask = norm > 0

        # normalized direction field
        unit_vector_field2 = np.zeros_like(vector_field2)
        unit_vector_field2[mask] = vector_field2[mask] / norm[mask, np.newaxis]

        # dot product between the two vector fields
        v_dot_r = np.sum(vector_field1 * unit_vector_field2, axis=-1, keepdims=True)

        # projection of the vector field along the other vector field
        v_parallel = v_dot_r * unit_vector_field2
        
        return v_parallel


def interpolate_vector_field_np(field:np.ndarray, pos:np.ndarray, boxsize:float, nmesh:int) -> np.ndarray:
    """
    Interpolate a vector field defined on the mesh at the given positions.

    Parameters:
    -----------
    field: array
        (N, 3) array containing the vector field on mesh to interpolate
    pos: array
        (N, 3) array containing the positions where to interpolate the field

    Returns:
    --------
    interp_field: array
        (N, 3) array containing the interpolated vector field at the given positions
    """

    # get the component of the field on the grid
    field_x = field[...,0] 
    field_y = field[...,1] 
    field_z = field[...,2]

    # edges of the cells
    x = np.linspace(0, boxsize, nmesh+1)
    y = np.linspace(0, boxsize, nmesh+1)
    z = np.linspace(0, boxsize, nmesh+1)

    # get the mean points of the cells
    x = (x[:-1] + x[1:]) / 2
    y = (y[:-1] + y[1:]) / 2
    z = (z[:-1] + z[1:]) / 2

    # interpolate the field at the tracers positions
    field_interp_x = interpn((x, y, z), field_x, pos, bounds_error=False, fill_value=None)
    field_interp_y = interpn((x, y, z), field_y, pos, bounds_error=False, fill_value=None)
    field_interp_z = interpn((x, y, z), field_z, pos, bounds_error=False, fill_value=None)

    interp_field = np.stack([field_interp_x, field_interp_y, field_interp_z], axis=-1)

    return interp_field


#######################################################
###### DIVERGENCE OF A VECTOR FIELD ON THE MESH ######
#######################################################

def divergence_FFT(vector_field:np.ndarray, kmesh:np.ndarray) -> np.ndarray:
    """
    Compute the divergence of a vector field using the Fourier Transform.
    Parameters
    ----------
    vector_field : ndarray
        Vector field of shape (N,N,N,3) in configuration space.
    kmesh : ndarray
        Wavevector mesh of shape (Nx, Ny, Nz//2 + 1, 3). Note the reduced
        dimension on the last spatial axis due to rfftn!

        
    Returns
    -------
    div : ndarray
        Divergence of the vector field.
    """
    
    if vector_field.shape[-1] != 3:
        raise ValueError("The last dimension of vector_field must be of size 3 representing vector components.")

    
    kx, ky, kz = kmesh[..., 0], kmesh[..., 1], kmesh[..., 2]
    
    v_k = sfft.rfftn(vector_field, axes=(0,1,2), workers=-1)
    div_k = v_k[..., 0] * (1j * kx) +  v_k[..., 1] * (1j * ky) + v_k[..., 2] * (1j * kz)

    divergence_from_fourier = sfft.irfftn(div_k, axes=(0,1,2), workers=-1)
    return divergence_from_fourier


####################################
####### JITTED METHODS ############
####################################

from numba import njit, prange

@njit(parallel=True, fastmath=True)
def project_vector_field_jit(vector_field, los_versor, out):
    """
    Project the vector field along the line-of-sight (LOS) into a preallocated array.

    Parameters
    ----------
    vector_field : ndarray, shape (Nx, Ny, Nz, 3)
        Input vector field to project.
    los_versor : ndarray, shape (Nx, Ny, Nz, 3)
        Unit vectors defining the LOS direction at each mesh point.
    out : ndarray, shape (Nx, Ny, Nz, 3)
        Preallocated array to write the projected field into.

    Returns
    -------
    out : ndarray
        The projected vector field.
    """

    nx, ny, nz, _ = vector_field.shape
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                dot = (vector_field[i,j,k,0]*los_versor[i,j,k,0] +
                       vector_field[i,j,k,1]*los_versor[i,j,k,1] +
                       vector_field[i,j,k,2]*los_versor[i,j,k,2])
                
                for c in range(3):
                    out[i,j,k,c] = dot * los_versor[i,j,k,c]
                    
    return out

@njit(parallel=True, fastmath=True)
def interpolate_cic_vector(pos, field, boxsize, pbc=True, dtype=np.float32):
    """
    Trilinear CIC interpolation for vector fields.
    
    Parameters
    ----------
    pos : (N,3) array
        Particle positions
    field : (nmesh, nmesh, nmesh, 3) array
        Grid field (vector field)
    boxsize : float
        Box size
    pbc : bool
        Apply periodic boundary conditions
    dtype : numpy dtype
        Output dtype

    Returns
    -------
    out : (N,3) array
        Interpolated values at particle positions
    """
    N = pos.shape[0]
    nmesh = field.shape[0]
    out = np.zeros((N,3), dtype=dtype)
    cell_size = boxsize / nmesh
    
    for idx in prange(N):
        fx = pos[idx, 0] / cell_size
        fy = pos[idx, 1] / cell_size
        fz = pos[idx, 2] / cell_size

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
            i0 = min(i0, nmesh-1)
            j0 = min(j0, nmesh-1)
            k0 = min(k0, nmesh-1)
            i1 = min(i0+1, nmesh-1)
            j1 = min(j0+1, nmesh-1)
            k1 = min(k0+1, nmesh-1)

        for c in range(3):
            c000 = field[i0,j0,k0,c]
            c100 = field[i1,j0,k0,c]
            c010 = field[i0,j1,k0,c]
            c001 = field[i0,j0,k1,c]
            c101 = field[i1,j0,k1,c]
            c011 = field[i0,j1,k1,c]
            c110 = field[i1,j1,k0,c]
            c111 = field[i1,j1,k1,c]

            c00 = c000*(1-tx) + c100*tx
            c01 = c001*(1-tx) + c101*tx
            c10 = c010*(1-tx) + c110*tx
            c11 = c011*(1-tx) + c111*tx

            c0 = c00*(1-ty) + c10*ty
            c1 = c01*(1-ty) + c11*ty

            out[idx, c] = c0*(1-tz) + c1*tz

    return out


@njit(inline='always')
def tsc_weight(dx):
    """Helper function to calculate TSC weights."""
    dx = abs(dx)
    if dx < 0.5:
        return 0.75 - dx*dx
    elif dx < 1.5:
        return 0.5 * (1.5 - dx)**2
    else:
        return 0.0

@njit(parallel=True, fastmath=True)
def interpolate_tsc_vector(pos, field, boxsize, pbc=True, dtype=np.float32):
    N = pos.shape[0]
    nmesh = field.shape[0]
    out = np.zeros((N, 3), dtype=dtype)
    cell_size = boxsize / nmesh
    
    for idx in prange(N):
        fx = pos[idx, 0] / cell_size
        fy = pos[idx, 1] / cell_size
        fz = pos[idx, 2] / cell_size

        # TSC is anchored to the NEAREST grid point
        i0 = int(np.floor(fx + 0.5))
        j0 = int(np.floor(fy + 0.5))
        k0 = int(np.floor(fz + 0.5))

        # We iterate over the 3x3x3 cloud (-1, 0, +1 relative to nearest point)
        for di in range(-1, 2):
            wx = tsc_weight(fx - (i0 + di))
            if wx == 0.0: continue
            
            i_idx = i0 + di
            if pbc: i_idx = i_idx % nmesh
            else: i_idx = min(max(i_idx, 0), nmesh-1)

            for dj in range(-1, 2):
                wy = tsc_weight(fy - (j0 + dj))
                if wy == 0.0: continue
                
                j_idx = j0 + dj
                if pbc: j_idx = j_idx % nmesh
                else: j_idx = min(max(j_idx, 0), nmesh-1)

                for dk in range(-1, 2):
                    wz = tsc_weight(fz - (k0 + dk))
                    if wz == 0.0: continue
                    
                    k_idx = k0 + dk
                    if pbc: k_idx = k_idx % nmesh
                    else: k_idx = min(max(k_idx, 0), nmesh-1)

                    w_tot = wx * wy * wz
                    
                    for c in range(3):
                        out[idx, c] += field[i_idx, j_idx, k_idx, c] * w_tot

    return out

def interpolate_vector_field(pos:np.ndarray, field:np.ndarray, boxsize:float, MAS:str='CIC', pbc:bool=True, dtype=np.float32) -> np.ndarray:
    """
    Interpolate a vector field defined on the mesh at the given positions using specified mass assignment scheme.
    Wraps the appropriate jitted interpolation function based on the chosen MAS.

    Parameters:
    -----------
    pos: array
        (N, 3) array containing the positions where to interpolate the field
    field: array
        (nmesh, nmesh, nmesh, 3) array containing the vector field on mesh to interpolate
    boxsize: float
        Box size of the mesh
    MAS: str
        Mass Assignment Scheme to use ('CIC' or 'TSC')
    pbc: bool
        Whether to apply periodic boundary conditions
    dtype: numpy dtype
        Output dtype

    Returns:
    --------
    interp_field: array
        (N, 3) array containing the interpolated vector field at the given positions
    """

    start_time = time.time()

    if MAS == 'np':
        logger.debug("Using numpy interpolation for vector field. Deprecated")
        interp_field = interpolate_vector_field_np(field, pos, boxsize, field.shape[0])
    elif MAS == 'CIC':
        interp_field = interpolate_cic_vector(pos, field, boxsize, pbc=pbc, dtype=dtype)
    elif MAS == 'TSC':
        interp_field = interpolate_tsc_vector(pos, field, boxsize, pbc=pbc, dtype=dtype)
    else:
        raise ValueError("MAS must be one of 'CIC' or 'TSC'")
    end_time = time.time()
    logger.debug(f"{MAS} interpolation took {end_time - start_time:.4f} seconds.")

    return interp_field


@njit(parallel=True, fastmath=True, nogil=True)
def divergence_finite_diff_jit(vector_field, cell_size):
    """
    Compute the divergence of a vector field using finite difference.
    Parameters
    ----------
    vector_field : ndarray
        Vector field of shape (N,N,N,3).
        
    cell_size : float
        Size of each cell in the grid.
    """
    nx, ny, nz, _ = vector_field.shape
    div = np.zeros((nx, ny, nz), dtype=vector_field.dtype)
    
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                div_x = (vector_field[min(i+1,nx-1), j, k, 0] - vector_field[max(i-1,0), j, k, 0]) / (2*cell_size)
                div_y = (vector_field[i, min(j+1,ny-1), k, 1] - vector_field[i, max(j-1,0), k, 1]) / (2*cell_size)
                div_z = (vector_field[i, j, min(k+1,nz-1), 2] - vector_field[i, j, max(k-1,0), 2]) / (2*cell_size)
                div[i,j,k] = div_x + div_y + div_z
    
    return div

#########################################
####### SMOOTHING KERNEL ###############
#########################################

def smoothed_field(field_on_mesh:np.ndarray, mesh, smoothing_radius:float, pbc:bool=True, mode:str='reflect') -> np.ndarray:
    """
    Compute the smoothed density field. If pbc is True, use FFT-based Gaussian smoothing.
    Otherwise, use scipy's gaussian_filter with specified mode.
    Parameters
    ----------
    field_on_mesh : array_like
        Array of shape (N, N, N) containing the density field on the mesh.
    mesh : Mesh object  
        Mesh object containing mesh properties.
    smoothing_radius : float
        Smoothing radius in configuration space.
    pbc : bool, optional
        Whether to apply periodic boundary conditions. Default is True.
    mode : str, optional
        Mode to use for scipy's gaussian_filter when pbc is False. Default is 'reflect'. Other options include 'constant', 'nearest', 'mirror', 'wrap'.
    Returns
    -------
    smoothed_density : array_like
        Array of shape (N, N, N) containing the smoothed density field in configuration space.
    """

    ##if pbc:
    #logger.debug('Applying FFT-based Gaussian smoothing with PBC.')
    delta_k = sfft.rfftn(field_on_mesh)
    S_k = _gaussian_kernel(mesh, smoothing_radius)
    sm_delta_k = S_k * delta_k
    return sfft.irfftn(sm_delta_k, s=field_on_mesh.shape, workers=-1)
    '''else:
        logger.debug('Applying real-space Gaussian smoothing without PBC. Mode: {}'.format(mode))
        from scipy.ndimage import gaussian_filter
        dx = mesh.cell_size
        sigma_vox = smoothing_radius / dx
        return gaussian_filter(field_on_mesh, sigma=sigma_vox, mode=mode)'''
    
def _gaussian_kernel(mesh, smoothing_radius:float) -> np.ndarray:
    """
    Return a Gaussian kernel S(k) = exp(-0.5 * (k^2) * R^2) in Fourier space.
        
    Returns
    -------
    kernel : ndarray
        Gaussian kernel array of shape (N, N, N).
    """

    logger.debug(f'kmesh shape in _gaussian_kernel is  {mesh.kmesh.shape}')
    if mesh.kmesh.shape[-1] != 3:
        raise ValueError("The last dimension of kmesh must be of size 3 representing wavevector components.")
        
    kernel =  np.exp(-0.5 * np.sum(mesh.kmesh**2, axis=-1) * smoothing_radius**2)
    return kernel
    