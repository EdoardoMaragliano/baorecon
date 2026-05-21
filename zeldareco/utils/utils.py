# utils.py

from .wrapper import xp

"""
A set of functions to compute the divergence of a vector field, the Gaussian kernel, the Kaiser RSD correction factor, and the multipoles of the power spectrum.

"""

def divergence(vector_field, cell_size, algo='FFT'):
    """
    Compute the divergence of a vector field using the specified algorithm.

    Parameters
    ----------
    vector_field : ndarray
        Vector field of shape (N, N, N, 3).
        
    cell_size : float
        Size of each cell in the grid.
        
    algo : str, optional
        Algorithm to compute the divergence. Options are 'finite_diff', 'finite_diff_periodic', and 'FFT'.
        
    Returns
    -------
    div : ndarray
        Divergence of the vector field.
    """
    
    if algo == 'finite_diff':
        div = divergence_finite_diff(vector_field, cell_size)
    elif algo == 'finite_diff_periodic':
        div = divergence_periodic(vector_field, cell_size)
    elif algo == 'FFT':
        div = FFTdivergence(vector_field, cell_size)
    else:
        raise ValueError("Invalid algorithm. Options are 'finite_diff', 'periodic' and 'FFT'.")
        
    return div


def divergence_finite_diff(vector_field, cell_size):
    """
    Compute the divergence of a vector field using finite differences.
    
    Parameters
    ----------
    vector_field : ndarray
        Vector field of shape (N, N, N, 3).
        
    cell_size : float
        Size of each cell in the grid.
        
    Returns
    -------
    div : ndarray
        Divergence of the vector field.
    """
    
    if vector_field.shape[-1] != 3:
        raise ValueError("The last dimension of vector_field must be of size 3 representing vector components.")
        
    div = (xp.gradient(vector_field[..., 0], cell_size, axis=0) +
           xp.gradient(vector_field[..., 1], cell_size, axis=1) +
           xp.gradient(vector_field[..., 2], cell_size, axis=2))
    
    return div


def divergence_periodic(vector_field, cell_size):
    """
    Compute the divergence of a vector field with periodic boundary conditions.
    
    Parameters
    ----------
    vector_field : ndarray
        Vector field of shape (N, N, N, 3).
        
    cell_size : float
        Size of each cell in the grid.
        
    Returns
    -------
    div : ndarray
        Divergence of the vector field with periodic boundary conditions.
    """
    
    if vector_field.shape[-1] != 3:
        raise ValueError("The last dimension of vector_field must be of size 3 representing vector components.")
    
    # Compute finite difference with periodic boundary conditions
    div_x = (xp.roll(vector_field[..., 0], -1, axis=0) - xp.roll(vector_field[..., 0], 1, axis=0)) / (2 * cell_size)
    div_y = (xp.roll(vector_field[..., 1], -1, axis=1) - xp.roll(vector_field[..., 1], 1, axis=1)) / (2 * cell_size)
    div_z = (xp.roll(vector_field[..., 2], -1, axis=2) - xp.roll(vector_field[..., 2], 1, axis=2)) / (2 * cell_size)
    
    # Sum the components to get the divergence
    div = div_x + div_y + div_z
    
    return div


def FFTdivergence(vector_field, cell_size):
    """
    Compute the divergence of a vector field using the Fourier Transform.
    
    Parameters
    ----------
    vector_field : ndarray
        Vector field of shape (N, N, N, 3) in configuration space.
        
    cell_size : float
        Size of each cell in the grid.
        
    Returns
    -------
    div : ndarray
        Divergence of the vector field.
    """
    
    if vector_field.shape[-1] != 3:
        raise ValueError("The last dimension of vector_field must be of size 3 representing vector components.")
        
    kx = xp.fft.fftfreq(vector_field.shape[0], d=cell_size) * 2 * xp.pi
    ky = xp.fft.fftfreq(vector_field.shape[1], d=cell_size) * 2 * xp.pi
    kz = xp.fft.rfftfreq(vector_field.shape[2], d=cell_size) * 2 * xp.pi
    
    kx, ky, kz = xp.meshgrid(kx, ky, kz, indexing='ij')
    
    v_k = xp.fft.rfftn(vector_field, axes=(0, 1, 2))
    div_k = v_k[..., 0] * (1j * kx) + v_k[..., 1] * (1j * ky) + v_k[..., 2] * (1j * kz)
    
    divergence_from_fourier = xp.fft.irfftn(div_k)
    
    return divergence_from_fourier


def _gaussian_kernel(kmesh, smoothing_radius=15):
    """
    Return a Gaussian kernel function with the given smoothing radius.
    The kernel is computed on a mesh of wavevector, of shape (N,N,N,3)
    Parameters
    ----------
    kmesh : ndarray
        Mesh of wavevectors of shape (N,N,N,3)
    smoothing_radius : float, optional
        Smoothing radius of the Gaussian kernel, in Mpc/h.
        
    Returns
    -------
    kernel : ndarray
        Gaussian kernel array of shape (N, N, N).
    """

    print('kmesh shape in _gaussian_kernel is ', kmesh.shape)
    if kmesh.shape[-1] != 3:
        raise ValueError("The last dimension of kmesh must be of size 3 representing wavevector components.")
        
    kernel =  xp.exp(-0.5 * xp.sum(kmesh**2, axis=-1) * smoothing_radius**2)

    return kernel

def _KaiserRSD(f, b, mu):
    """
    Return the Kaiser RSD correction factor.
    Parameters
    ----------
    f : float
        Growth rate parameter.
    mu : float
        Direction of the line of sight.
        
    Returns
    -------
    kaiser : ndarray
        Kaiser RSD correction factor.
    """
    
    kaiser = 1 + f/b * mu**2

    return kaiser

def pk_multipoles_kaiser(pk_real_space, beta):
    """
    Compute the multipoles of the power spectrum in redshift space using the Kaiser approximation.

    Parameters
    ----------
    pk_real_space : ndarray
        Real-space power spectrum.
    beta : float
        Growth rate parameter.
        
    Returns
    -------
    pk0 : ndarray
        Monopole of the redshift-space power spectrum.
    pk2 : ndarray
        Quadrupole of the redshift-space power spectrum.
    pk4 : ndarray   
        Hexadecapole of the redshift-space power spectrum.
    """
    
    pk0 = (1+2*beta/3+beta**2/5)*pk_real_space
    pk2 = (4*beta/3+4*beta**2/7)*pk_real_space
    pk4 = 8*beta**2/35*pk_real_space

    return pk0, pk2, pk4


def rho_to_delta(rho, mean_density):
    """
    Compute the overdensity field from the input density field.
    Parameters
    ----------
    rho : ndarray
        Density field.
    mean_density : float
        Mean density of the field.
        
    Returns
    -------
    delta : ndarray
        Overdensity field.
    """
    
    delta = rho / mean_density - 1.0
    
    return delta

def delta_to_rho(delta, mean_density):
    """
    Compute the density field from the input overdensity field.
    Parameters
    ----------
    delta : ndarray
        Overdensity field.
    mean_density : float
        Mean density of the field.
        
    Returns
    -------
    rho : ndarray
        Density field.
    """
    
    rho = (delta + 1.0) * mean_density
    
    return rho

def effective_smoothing_radius(cell_size, gaussian_kernel_radius):
    """
    Compute the resulting effective smoothing radius of the Gaussian kernel in Fourier space when applied to a density field on a grid.
    LambdaCDM is assumed.

    Parameters
    ----------
    cell_size : float
        Size of each cell in the grid.
    gaussian_kernel_radius : float
        Smoothing radius of the Gaussian kernel, in Mpc/h.
        
    Returns
    -------
    k : float
        Resulting effective smoothing radius in Fourier space.
    """
    
    R_eff = xp.sqrt(gaussian_kernel_radius**2 - (0.64 * cell_size)**2)
    
    return R_eff

def gaussian_smoothing_radius_from_effective(cell_size, effective_smoothing_radius):
    """
    Compute the Gaussian smoothing radius from the effective smoothing radius in Fourier space.
    LambdaCDM is assumed.

    Parameters
    ----------
    cell_size : float
        Size of each cell in the grid.
    effective_smoothing_radius : float
        Effective smoothing radius in Fourier space.
        
    Returns
    -------
    R : float
        Gaussian smoothing radius in configuration space.
    """
    
    R = xp.sqrt(effective_smoothing_radius**2 + (0.64 * cell_size)**2)
    
    return R


def split_box_into_eight(density_on_grid):
    """
    Split a density field on a grid into eight sub-boxes.
    Parameters
    ----------
    density_on_grid : ndarray
        Density field on a grid.
        
    Returns
    -------
    sub_boxes : list
        List of eight sub-boxes.
    """
    N = density_on_grid.shape[0]
    new_edge = N // 2

    sub_boxes = [density_on_grid[i:new_edge+i, j:new_edge+j, k:new_edge+k]
                 for i in (0, new_edge)
                 for j in (0, new_edge)
                 for k in (0, new_edge)]
                
    return sub_boxes

def periodic_distance(pos1, pos2, boxsize):
    diff = pos2-pos1 - xp.round((pos2-pos1)/boxsize)*boxsize
    return xp.linalg.norm(diff, axis=-1)