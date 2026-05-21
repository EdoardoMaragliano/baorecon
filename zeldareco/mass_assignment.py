# mass_assignment.py

import numpy as np
from numba import njit, prange, set_num_threads, get_num_threads, get_thread_id
import time
from zeldareco.utils.loggers import setup_logger
logger = setup_logger(__name__)

def _mass_assignment_info():
    info = """
    Mass Assignment Schemes implemented:
    - NGP (Nearest Grid Point)
    - CIC (Cloud-In-Cell)
    - TSC (Triangular Shaped Cloud)

    Each method has its own function for mass assignment from particles to grid.
    The main wrapper function 'mass_assignment' allows selection of the method.

    Additionally, grid-to-particle interpolation functions are provided for each scheme.
    """
    logger.info(info)
 

#############################
# NGP (vectorized)
#############################
def ngp_ma(pos: np.ndarray, 
            boxsize: float, 
            nmesh: int, 
            weights: np.ndarray = None, 
            pbc: bool = True, 
            dtype=np.float64, 
            verbose: bool=False
           ) -> np.ndarray:
    """
    Compute and return the interpolated density field of a cubic distribution 
    of particles using weightses.

    Parameters:
    - pos: np.ndarray
        Positions of the particles, shape (N, coord), where N is the number of particles
        and coord is the number of spatial dimensions.
    - boxsize: float
        Size of the simulation box.
    - nmesh: int
        Number of grid cells along each axis (assumes cubic grid).
    - weights: np.ndarray
        Masses associated with each particle, shape (N,).
    - pbc: bool
        Periodic boundary conditions flag.

    Returns:
    - np.ndarray
        The interpolated density field (3D grid) of shape (nmesh, nmesh, nmesh).
    """

    if verbose:
        start_time = time.time()
        logger.info("Starting NGP mass assignment...")

    # Assign default weights if not provided
    if weights is None:
        weights = np.ones(pos.shape[0], dtype=dtype)
        logger.info("weights not provided. Assuming all particles have unit mass.")

    if pos.shape[0] != weights.shape[0]:
        raise ValueError("pos and weights must have the same number of particles")
    if pos.shape[1] != 3:
        raise ValueError("pos must have shape (N, 3)")
    if pos.any() < 0 or pos.any() > boxsize:
        raise ValueError("pos must be in the range [0, boxsize]")
    
    # Initialize the density field to zero
    density_field = np.zeros((nmesh, nmesh, nmesh), dtype=dtype)

    # Calculate the inverse of the cell size
    inv_cell_size = nmesh / boxsize

    # Compute indices for all particles at once
    indices = np.floor(pos * inv_cell_size + 0.5).astype(int)

    if pbc:
        indices = np.mod(indices, nmesh)

    # Flatten to 1D for accumulation
    flat_idx = np.ravel_multi_index(indices.T, (nmesh, nmesh, nmesh))
    np.add.at(density_field.ravel(), flat_idx, weights)

    if verbose:
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"NGP mass assignment completed in %f seconds." % elapsed_time)

    return density_field


#############################
# CIC (Numba JIT) not parallel
#############################
@njit(parallel=False, nopython=True, fastmath=True)
def cic_ma_numba(pos: np.ndarray, 
                 boxsize: float, 
                 n_cell: int, 
                 weights: np.ndarray, 
                 pbc: bool = True, 
                 dtype=np.float64
                ) -> np.ndarray:
    """
    CIC (Cloud-In-Cell) interpolator for computing density on a grid.

    This function takes in the positions and weightses of particles, as well as the shape of the grid,
    and computes the density on the grid using the CIC interpolation method.

    Parameters:
        pos (array-like): Array of particle positions of shape (N, 3).
        boxsize (float): Size of the simulation box.
        n_cell (int): Number of grid cells along one dimension.
        weights (array-like): Array of particle weights of shape (N,). Default is None.
    Returns:
        array-like: The density on the grid.
    
    Raises:
        ValueError: If the number of particles in pos and weights do not match.
        ValueError: If the shape of pos is not (N, 3).
        ValueError: If any particle is outside the range [0, boxsize].

    """

    # Initialize an empty density grid
    density = np.zeros((n_cell, n_cell, n_cell), dtype=dtype)  

    # Find the number of particles and dimensions
    particles = pos.shape[0]  # number of particles
    coord = pos.shape[1]  # number of dimensions (3D)
    
    # Calcola il reciproco della dimensione della cella
    inv_cell_size = n_cell / boxsize

    # Initialise arrays to store the weights and indices
    u       = np.ones(3, dtype=dtype)
    d       = np.ones(3, dtype=dtype)
    index_u = np.zeros(3, dtype=np.int64)
    index_d = np.zeros(3, dtype=np.int64)
    
    # Loop over all particles
    for i in range(particles):

        # For each dimension, calculate the fractional and complementary distance
        for axis in range(coord):
            dist                = pos[i, axis] * inv_cell_size                  # Normalized distance
            u[axis]             = dist - int(dist)                              # Fractional part
            d[axis]             = 1.0 - u[axis]                                 # Complementary part
            if pbc:
                index_d[axis]   = int(dist) % n_cell                  # index of the lower cell (cyclic)
                index_u[axis]   = (index_d[axis] + 1) % n_cell        # index of the upper cell (cyclic)
            else:
                index_d[axis]   = min(int(dist), n_cell - 1)
                index_u[axis]   = min(index_d[axis] + 1, n_cell - 1)


        # Update the density grid
        weight = weights[i]
        density[index_d[0], index_d[1], index_d[2]] += d[0] * d[1] * d[2] * weight
        density[index_d[0], index_d[1], index_u[2]] += d[0] * d[1] * u[2] * weight
        density[index_d[0], index_u[1], index_d[2]] += d[0] * u[1] * d[2] * weight
        density[index_d[0], index_u[1], index_u[2]] += d[0] * u[1] * u[2] * weight
        density[index_u[0], index_d[1], index_d[2]] += u[0] * d[1] * d[2] * weight
        density[index_u[0], index_d[1], index_u[2]] += u[0] * d[1] * u[2] * weight
        density[index_u[0], index_u[1], index_d[2]] += u[0] * u[1] * d[2] * weight
        density[index_u[0], index_u[1], index_u[2]] += u[0] * u[1] * u[2] * weight

    return density

##############################
# CIC parallel safe version
###############################

@njit(parallel=True, nopython=True, fastmath=True)
def cic_interpolation_safe(pos, boxsize, nmesh, weights, pbc=True, dtype=np.float64) -> np.ndarray:
    """
    Interpolazione CIC thread-safe con pesi e periodicità.
    
    pos: array (N,3)
    boxsize: dimensione della scatola
    nmesh: dimensione della griglia cubica
    weights: array di pesi dei punti
    pbc: True se si vogliono condizioni periodiche
    """
    
    N = pos.shape[0]
    cell_size = boxsize / nmesh

    # Prepare local grids for each thread
    nthreads = get_num_threads()
    local_grids = np.zeros((nthreads, nmesh, nmesh, nmesh), dtype=dtype)

    for p in prange(N):
        tid = get_thread_id()
        x, y, z = pos[p]
        mass = weights[p]

        gx = x / cell_size
        gy = y / cell_size
        gz = z / cell_size

        i = int(np.floor(gx))
        j = int(np.floor(gy))
        k = int(np.floor(gz))

        dx = gx - i
        dy = gy - j
        dz = gz - k

        wx = np.array([1 - dx, dx], dtype=dtype)
        wy = np.array([1 - dy, dy], dtype=dtype)
        wz = np.array([1 - dz, dz], dtype=dtype)

        for ii in range(2):
            for jj in range(2):
                for kk in range(2):
                    weight = wx[ii] * wy[jj] * wz[kk] * mass
                    ni = i + ii
                    nj = j + jj
                    nk = k + kk

                    if pbc:
                        ni = ni % nmesh
                        nj = nj % nmesh
                        nk = nk % nmesh
                    else:
                        if ni < 0 or ni >= nmesh or nj < 0 or nj >= nmesh or nk < 0 or nk >= nmesh:
                            continue

                    local_grids[tid, ni, nj, nk] += weight

    density_grid = np.zeros((nmesh, nmesh, nmesh), dtype=dtype)
    for t in range(nthreads):
        density_grid += local_grids[t]

    return density_grid


def cic_ma(pos: np.ndarray, 
           boxsize: float, 
           nmesh: int, 
           weights: np.ndarray = None, 
           pbc: bool = True, 
           dtype=np.float64, 
           verbose: bool=False,
           parallel: bool = False,
           ) -> np.ndarray:

    """
    Wrapper function for CIC mass assignment.
    Parameters:
    - pos: np.ndarray
        Positions of the particles, shape (N, coord), where N is the number of particles
        and coord is the number of spatial dimensions.
    - boxsize: float
        Size of the simulation box.
    - nmesh: int
        Number of grid cells along each axis (assumes cubic grid).
    - weights: np.ndarray
        Weights associated with each particle, shape (N,). Can be masses or other weights.
    - pbc: bool
        Periodic boundary conditions flag. 
    Returns:
        np.ndarray: The updated density grid.
    """

    if verbose:
        start_time = time.time()
        logger.info("Starting CIC mass assignment...")

    if weights is None:
        weights = np.ones(pos.shape[0], dtype=dtype)
        logger.info("weights not provided. Assuming all particles have unit weight.")
    if pos.shape[0] != weights.shape[0]:
        raise ValueError("pos and weights must have the same number of particles")
    if pos.shape[1] != 3:
        raise ValueError("pos must have shape (N, 3)")

    if pbc:
        if (pos < 0).any() or (pos > boxsize).any():
            import warnings
            warnings.warn("Particles are outside the range [0, boxsize]. PBC will be applied.")
    else:
        if (pos < 0).any() or (pos > boxsize).any():
            raise ValueError(f"pos must be in the range [0, boxsize]. Got min {pos.min()} and max {pos.max()}. boxsize: {boxsize}.")
    if not parallel:
        field = cic_ma_numba(pos.astype(dtype), boxsize, nmesh, weights.astype(dtype), pbc, dtype=dtype)
        logger.info("Using single-threaded CIC interpolation.\n")
    else:
        logger.info("Using %d threads for CIC interpolation.\n" % get_num_threads())
        field = cic_interpolation_chunks(pos=pos.astype(dtype), boxsize=boxsize, nmesh=nmesh, weights=weights.astype(dtype), pbc=pbc)

    if verbose:
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"CIC mass assignment completed in %f seconds." % elapsed_time)

    return field



#############################
# TSC (Numba JIT) not parallel
#############################
@njit(parallel=False, nopython=True, fastmath=True) 
def tsc_ma_numba(pos: np.ndarray,
                 boxsize: float, 
                 n_cell: int, 
                 weights: np.ndarray = None, 
                 pbc: bool = True, 
                 dtype=np.float64, 
                ) -> np.ndarray:
    """
    Perform Triangular Shaped Cloud weighting (TSC) to compute a density grid.

    Parameters:
        pos (np.ndarray): Particle positions, shape (N_particles, N_dim).
        boxsize (float): Size of the simulation box.
        n_cell (float): The initial value for the grid density.
        weights (np.ndarray): Masses for each particle, shape (N_particles,).
        pbc (bool): Periodic boundary conditions flag.
        dtype (np.dtype): Data type for the density grid.

    Returns:
        np.ndarray: The updated density grid.
    """

    # Initialize the density grid
    density_grid = np.zeros((n_cell, n_cell, n_cell), dtype=dtype)

    particles = pos.shape[0]                        # Number of particles
    coord = pos.shape[1]                            # Number of spatial dimensions
    inv_cell_size = n_cell / boxsize

    # Initialize temporary arrays
    tsc_w = np.ones((3, 3), dtype=dtype)           # Weights matrix
    index = np.zeros((3, 3), dtype=np.int64)        # Grid indices

    # Loop over all particles
    for i in range(particles):

        # Loop over each coordinate axis
        for axis in range(coord):
            dist = pos[i, axis] * inv_cell_size         # Normalized distance       
            minimum = np.int64(np.floor(dist - 1.5))    # Lower bound of the 3x3x3 cube

            for j in range(3):  # Each particle contributes to a 3x3x3 cube

                # Compute the grid index and apply periodic boundary conditions
                index[axis, j] = (minimum + j + 1 + n_cell) % n_cell  

                # Compute the distance from the grid point          
                diff = np.abs(minimum + j + 1 - dist)                              

                # Compute the TSC weights    
                if diff < 0.5:
                    tsc_w[axis, j] = 0.75 - diff**2
                elif diff < 1.5:
                    tsc_w[axis, j] = 0.5 * (1.5 - diff)**2
                else:
                    tsc_w[axis, j] = 0.0

        # Accumulate density into the grid
        for l in range(3):
            for m in range(3):
                for n in range(3):
                    density_grid[index[0, l], index[1, m], index[2, n]] += (
                        tsc_w[0, l] * tsc_w[1, m] * tsc_w[2, n] * weights[i]
                    )


    return density_grid

#####################################
#### TSC parallel safe version ####
######################################

@njit(inline='always')
def tsc_weight(dx):
    """
    Triangular-shaped cloud (TSC) weight function.
    dx should be in [0, 1.5]
    """
    dx = abs(dx)
    if dx < 0.5:
        return 0.75 - dx*dx
    elif dx < 1.5:
        return 0.5 * (1.5 - dx)**2
    else:
        return 0.0

@njit(inline='always')
def clip(val, min_val, max_val):
    if val < min_val:
        return min_val
    elif val > max_val:
        return max_val
    else:
        return val

@njit(parallel=True, nopython=True, fastmath=True)
def tsc_interpolation_safe(pos, boxsize, nmesh, weights, pbc=True, dtype=np.float64):
    """
    Thread-safe, bias-free TSC particle->grid assignment.
    Ensures exact mass conservation per particle and handles boundaries correctly.
    """

    raise RuntimeError("Using CIC interpolation (thread-safe version). Deprecated. Yields small biases. Use parallel = False.\n")

    N = pos.shape[0]
    cell_size = boxsize / nmesh
    nthreads = get_num_threads()
    local_grids = np.zeros((nthreads, nmesh, nmesh, nmesh), dtype=dtype)

    for p in prange(N):
        tid = get_thread_id()
        x, y, z = pos[p]
        mass = dtype(weights[p])

        # Wrap positions for PBC
        if pbc:
            x = x % boxsize
            y = y % boxsize
            z = z % boxsize

        # Normalized grid coordinates
        gx = x / cell_size
        gy = y / cell_size
        gz = z / cell_size

        # Index of nearest grid point
        i0 = int(np.floor(gx))
        j0 = int(np.floor(gy))
        k0 = int(np.floor(gz))

        # Compute weights and normalize to ensure sum = 1
        wsum = 0.0
        w_array = np.zeros((3, 3, 3), dtype=dtype)  # store weights
        for di in range(-1, 2):
            dx = clip(gx - (i0 + di), 0.0, 1.5)
            wx = tsc_weight(dx)
            for dj in range(-1, 2):
                dy = clip(gy - (j0 + dj), 0.0, 1.5)
                wy = tsc_weight(dy)
                for dk in range(-1, 2):
                    dz = clip(gz - (k0 + dk), 0.0, 1.5)
                    wz = tsc_weight(dz)
                    w = wx * wy * wz
                    w_array[di+1, dj+1, dk+1] = w
                    wsum += w

        # Distribute mass with exact normalization
        for di in range(-1, 2):
            ii = i0 + di
            if pbc:
                ii_wr = ii % nmesh
            else:
                if ii < 0 or ii >= nmesh:
                    continue
                ii_wr = ii
            for dj in range(-1, 2):
                jj = j0 + dj
                if pbc:
                    jj_wr = jj % nmesh
                else:
                    if jj < 0 or jj >= nmesh:
                        continue
                    jj_wr = jj
                for dk in range(-1, 2):
                    kk = k0 + dk
                    if pbc:
                        kk_wr = kk % nmesh
                    else:
                        if kk < 0 or kk >= nmesh:
                            continue
                        kk_wr = kk
                    w = w_array[di+1, dj+1, dk+1] / wsum  # normalized
                    local_grids[tid, ii_wr, jj_wr, kk_wr] += mass * w

    # Sum thread-local grids
    density_grid = np.zeros((nmesh, nmesh, nmesh), dtype=dtype)
    for t in range(nthreads):
        density_grid += local_grids[t]

    return density_grid



def tsc_ma(pos: np.ndarray, 
           boxsize: float, 
           nmesh: int, 
           weights: np.ndarray = None, 
           pbc: bool = True, 
           dtype=np.float64, 
           verbose: bool=False,
           parallel: bool = False,
          ) -> np.ndarray:
    """
    Wrapper function for TSC mass assignment.
    Parameters:
    - pos: np.ndarray
        Positions of the particles, shape (N, coord), where N is the number of particles
        and coord is the number of spatial dimensions.
    - boxsize: float
        Size of the simulation box.
    - nmesh: int
        Number of grid cells along each axis (assumes cubic grid).
    - weights: np.ndarray
        Masses associated with each particle, shape (N,).
    - pbc: bool
        Periodic boundary conditions flag. 
    - dtype: np.dtype
        Data type for the density grid.
    - verbose: bool
        Verbosity flag.
    Returns:
        np.ndarray: The updated density grid.
    """
    if verbose:
        start_time = time.time()
        print("Starting TSC mass assignment...")

     # Assign default weights if not provided
    if weights is None:
        weights = np.ones(pos.shape[0], dtype=dtype)

    if pos.shape[0] != weights.shape[0]:
        raise ValueError("pos and weights must have the same number of particles")
    if pos.shape[1] != 3:
        raise ValueError("pos must have shape (N, 3)")

    if pbc:
        if (pos < 0).any() or (pos > boxsize).any():
            import warnings
            warnings.warn("Particles are outside the range [0, boxsize]. Results may be incorrect.")
    else:
        if (pos < 0).any() or (pos > boxsize).any():
            raise ValueError(f"pos must be in the range [0, boxsize]. Got min {pos.min()} and max {pos.max()}. boxsize: {boxsize}.")
    if not parallel:
        logger.info("Using single-threaded TSC interpolation.\n")
        field = tsc_ma_numba(pos.astype(dtype), boxsize, nmesh, weights.astype(dtype), pbc=pbc, dtype=dtype)
    else:
        logger.info("Using %d threads for TSC interpolation.\n" % get_num_threads())
        field = tsc_interpolation_chunks(pos.astype(dtype), boxsize, nmesh, weights.astype(dtype), pbc=pbc, dtype=dtype)
    if verbose:
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"TSC mass assignment completed in %f seconds." % elapsed_time)

    return field


##############################################
# General Wrapper for mass assignment
###############################################
def mass_assignment(pos: np.ndarray, 
                    boxsize: float, 
                    nmesh: int, 
                    weights: np.ndarray = None, 
                    method: str = 'CIC', 
                    pbc: bool = True, 
                    dtype=np.float64, 
                    verbose: bool = False,
                    parallel: bool = False
                    )  -> np.ndarray:
    """
    General mass assignment function that selects the method.
    Parameters:
    - pos: np.ndarray
        Positions of the particles, shape (N, coord), where N is the number of particles
        and coord is the number of spatial dimensions.
    - boxsize: float
        Size of the simulation box.
    - nmesh: int
        Number of grid cells along each axis (assumes cubic grid).
    - weights: np.ndarray
        Masses associated with each particle, shape (N,).
    - method: str
        Mass assignment method: 'NGP', 'CIC', or 'TSC'.
    - pbc: bool
        Periodic boundary conditions flag.
    - dtype: np.dtype
        Data type for the density grid.
    - verbose: bool
        Verbosity flag.
    - parallel: bool
        Parallel execution flag.
    Returns:
        np.ndarray: The updated density grid.
    """

    if method.upper() == 'NGP':
        return ngp_ma(pos, boxsize, nmesh, weights, pbc, dtype=dtype, verbose=verbose)
    elif method.upper() == 'CIC':
        return cic_ma(pos, boxsize, nmesh, weights, pbc, dtype=dtype, verbose=verbose, parallel=parallel)
    elif method.upper() == 'TSC':
        return tsc_ma(pos, boxsize, nmesh, weights, pbc, dtype=dtype, verbose=verbose, parallel=parallel)
    else:
        raise ValueError(f"Invalid method: {method}. Choose 'NGP', 'CIC', or 'TSC'")

#############################
# Chuncked MAS methods
#############################
import numpy as np
from numba import njit, prange

@njit(parallel=True, fastmath=True)
def cic_interpolation_chunks(pos, boxsize, nmesh, weights, num_chunks=16, pbc=True, dtype=np.float32):
    """
    Optimized Cloud-in-Cell (CIC) interpolation for multi-core CPUs.
    
    Uses a map-reduce strategy via 'num_chunks' to avoid race conditions and 
    False Sharing. Set 'num_chunks' equal to your physical CPU cores for best 
    performance, or lower if you need to conserve RAM.
    
    Parameters
    ----------
    pos : ndarray
        Array of particle positions, shape (N, 3).
    boxsize : float
        The physical size of the simulation box.
    nmesh : int
        The number of grid cells along one axis.
    weights : ndarray
        Array of particle weights/masses, shape (N,).
    num_chunks : int, optional
        Number of independent grid copies to create (default is 16).
    pbc : bool, optional
        Whether to apply Periodic Boundary Conditions (default is True).
    dtype : numpy.dtype, optional
        Data type for the grid (default is float32 to save RAM).
        
    Returns
    -------
    density_grid : ndarray
        The 3D mass/density grid of shape (nmesh, nmesh, nmesh).
    """
    N = pos.shape[0]
    cell_size = boxsize / nmesh

    # 1. Create local grids based on CHUNKS (not thread_ids)
    # This avoids Numba's dynamic scheduling race conditions
    local_grids = np.zeros((num_chunks, nmesh, nmesh, nmesh), dtype=dtype)
    chunk_size = (N + num_chunks - 1) // num_chunks

    # 2. Parallelize over CHUNKS
    for c in prange(num_chunks):
        start = c * chunk_size
        end = min(start + chunk_size, N)
        
        # Each chunk has its own private grid to write to safely
        grid = local_grids[c]
        
        for p in range(start, end):
            mass = weights[p]
            
            gx = pos[p, 0] / cell_size
            gy = pos[p, 1] / cell_size
            gz = pos[p, 2] / cell_size

            # Base cell indices
            i = int(np.floor(gx))
            j = int(np.floor(gy))
            k = int(np.floor(gz))

            # Distances from the lower cell boundary
            dx = gx - i
            dy = gy - j
            dz = gz - k
            
            tx = 1.0 - dx
            ty = 1.0 - dy
            tz = 1.0 - dz

            # 3. EXTREME UNROLLING 
            # Zero memory allocations (no np.array calls) inside the hot loop.
            # We calculate the 8 vertex weights immediately.
            w000 = tx * ty * tz * mass
            w001 = tx * ty * dz * mass
            w010 = tx * dy * tz * mass
            w011 = tx * dy * dz * mass
            w100 = dx * ty * tz * mass
            w101 = dx * ty * dz * mass
            w110 = dx * dy * tz * mass
            w111 = dx * dy * dz * mass

            # Neighbor indices
            i1 = i + 1
            j1 = j + 1
            k1 = k + 1

            if pbc:
                # Wrap indices around the box
                i = i % nmesh; i1 = i1 % nmesh
                j = j % nmesh; j1 = j1 % nmesh
                k = k % nmesh; k1 = k1 % nmesh
                
                grid[i, j, k]    += w000
                grid[i, j, k1]   += w001
                grid[i, j1, k]   += w010
                grid[i, j1, k1]  += w011
                grid[i1, j, k]   += w100
                grid[i1, j, k1]  += w101
                grid[i1, j1, k]  += w110
                grid[i1, j1, k1] += w111
            else:
                # If no PBC, strictly check boundaries for each vertex
                if 0 <= i < nmesh and 0 <= j < nmesh and 0 <= k < nmesh:   grid[i, j, k] += w000
                if 0 <= i < nmesh and 0 <= j < nmesh and 0 <= k1 < nmesh:  grid[i, j, k1] += w001
                if 0 <= i < nmesh and 0 <= j1 < nmesh and 0 <= k < nmesh:  grid[i, j1, k] += w010
                if 0 <= i < nmesh and 0 <= j1 < nmesh and 0 <= k1 < nmesh: grid[i, j1, k1] += w011
                if 0 <= i1 < nmesh and 0 <= j < nmesh and 0 <= k < nmesh:  grid[i1, j, k] += w100
                if 0 <= i1 < nmesh and 0 <= j < nmesh and 0 <= k1 < nmesh: grid[i1, j, k1] += w101
                if 0 <= i1 < nmesh and 0 <= j1 < nmesh and 0 <= k < nmesh: grid[i1, j1, k] += w110
                if 0 <= i1 < nmesh and 0 <= j1 < nmesh and 0 <= k1 < nmesh: grid[i1, j1, k1] += w111

    # Fast, vectorized final reduction outside the parallel region
    density_grid = np.sum(local_grids, axis=0)
    
    return density_grid


@njit(parallel=True, fastmath=True)
def tsc_interpolation_chunks(pos, boxsize, nmesh, weights, num_chunks=16, pbc=True, dtype=np.float32):
    """
    Optimized Triangular Shaped Cloud (TSC) interpolation for multi-core CPUs.
    
    Spreads particle mass over 27 adjacent cells (3x3x3 grid) using exact analytic
    weights to guarantee strict mass conservation without manual normalization.
    Uses a map-reduce chunking strategy to prevent False Sharing and race conditions.
    
    Parameters
    ----------
    pos : ndarray
        Array of particle positions, shape (N, 3).
    boxsize : float
        The physical size of the simulation box.
    nmesh : int
        The number of grid cells along one axis.
    weights : ndarray
        Array of particle weights/masses, shape (N,).
    num_chunks : int, optional
        Number of independent grid copies to create (default is 16). 
        Set this close to your CPU core count.
    pbc : bool, optional
        Whether to apply Periodic Boundary Conditions (default is True).
    dtype : numpy.dtype, optional
        Data type for the grid (default is float32 to save RAM).
        
    Returns
    -------
    density_grid : ndarray
        The 3D mass/density grid of shape (nmesh, nmesh, nmesh).
    """
    N = pos.shape[0]
    cell_size = boxsize / nmesh

    # 1. Create local grids based on CHUNKS to guarantee thread safety
    local_grids = np.zeros((num_chunks, nmesh, nmesh, nmesh), dtype=dtype)
    chunk_size = (N + num_chunks - 1) // num_chunks

    # 2. Parallelize safely over CHUNKS
    for c in prange(num_chunks):
        start = c * chunk_size
        end = min(start + chunk_size, N)
        
        # Private grid for this specific chunk
        grid = local_grids[c]
        
        for p in range(start, end):
            mass = weights[p]
            
            # Normalized grid coordinates
            gx = pos[p, 0] / cell_size
            gy = pos[p, 1] / cell_size
            gz = pos[p, 2] / cell_size

            # Nearest Grid Point (NGP) indices
            # By adding 0.5 and flooring, we find the closest cell center
            ix = int(np.floor(gx + 0.5))
            iy = int(np.floor(gy + 0.5))
            iz = int(np.floor(gz + 0.5))

            # Distance from the Nearest Grid Point (always between -0.5 and +0.5)
            dx = gx - ix
            dy = gy - iy
            dz = gz - iz

            # 3. EXACT ANALYTIC TSC WEIGHTS
            # We use Tuples instead of np.array() to guarantee zero memory 
            # allocation overhead inside the hot loop.
            # Index 0 is neighbor -1, Index 1 is NGP (0), Index 2 is neighbor +1
            wx = (0.5 * (0.5 - dx)**2, 0.75 - dx**2, 0.5 * (0.5 + dx)**2)
            wy = (0.5 * (0.5 - dy)**2, 0.75 - dy**2, 0.5 * (0.5 + dy)**2)
            wz = (0.5 * (0.5 - dz)**2, 0.75 - dz**2, 0.5 * (0.5 + dz)**2)

            # 4. MASS ASSIGNMENT (27 cells)
            # We separate PBC logic outside the inner loops so the compiler 
            # can aggressively vectorize the math without branch prediction stalls.
            if pbc:
                for di in range(3):
                    cx = (ix + di - 1) % nmesh
                    wx_mass = wx[di] * mass
                    for dj in range(3):
                        cy = (iy + dj - 1) % nmesh
                        wxy_mass = wx_mass * wy[dj]
                        for dk in range(3):
                            cz = (iz + dk - 1) % nmesh
                            grid[cx, cy, cz] += wxy_mass * wz[dk]
            else:
                for di in range(3):
                    cx = ix + di - 1
                    # Skip if out of bounds
                    if cx < 0 or cx >= nmesh: continue
                    wx_mass = wx[di] * mass
                    
                    for dj in range(3):
                        cy = iy + dj - 1
                        if cy < 0 or cy >= nmesh: continue
                        wxy_mass = wx_mass * wy[dj]
                        
                        for dk in range(3):
                            cz = iz + dk - 1
                            if cz < 0 or cz >= nmesh: continue
                            grid[cx, cy, cz] += wxy_mass * wz[dk]

    # Fast, vectorized final reduction outside the parallel region
    density_grid = np.sum(local_grids, axis=0)
    
    return density_grid

######################################################################################
###################### GRID TO PARTICLES MASS ASSIGNMENT #############################
######################################################################################

def interpolate_grid_to_particles(pos: np.ndarray,
                                    grid: np.ndarray,
                                    boxsize: float,
                                    method: str = 'CIC',
                                    pbc: bool = True,
                                    dtype=np.float64
                                     ) -> np.ndarray:
        """
        Interpolate grid values to particle positions using specified mass assignment scheme.
    
        Parameters:
        - pos: np.ndarray
            Positions of the particles, shape (N, 3).
        - grid: np.ndarray
            3D grid of values to interpolate from.
        - boxsize: float
            Size of the simulation box.
        - method: str
            Interpolation method: 'NGP', 'CIC', or 'TSC'.
        - pbc: bool
            Periodic boundary conditions flag.
        - dtype: np.dtype
            Data type for computations.
    
        Returns:
        - np.ndarray
            Interpolated values at particle positions, shape (N,).
        """
    
        nmesh = grid.shape[0]
        inv_cell_size = nmesh / boxsize
        interpolated_values = np.zeros(pos.shape[0], dtype=dtype)
    
        if method.upper() == 'NGP':
            interpolated_values = grid_to_particle_ngp(
                pos.astype(dtype),
                grid.astype(dtype), 
                boxsize, 
                pbc, 
                dtype=dtype
            )
    
        elif method.upper() == 'CIC':
            interpolated_values = grid_to_particle_cic(
                pos.astype(dtype),
                grid.astype(dtype), 
                boxsize, 
                pbc, 
                dtype=dtype
            )

        elif method.upper() == 'TSC':
            interpolated_values = grid_to_particle_tsc(
                pos.astype(dtype),
                grid.astype(dtype), 
                boxsize, 
                pbc, 
                dtype=dtype
            )
    

        else:
            raise ValueError(f"Invalid method: {method}. Choose 'NGP', 'CIC', or 'TSC'")
        
        return interpolated_values

#############################
# Grid-to-Particle Methods
#############################

@njit(parallel=False, nopython=True, fastmath=True)
def grid_to_particle_ngp(pos, grid, boxsize, pbc=True, dtype=np.float64):
    """
    NGP interpolation: nearest grid point.
    
    Parameters:
        pos: (N,3) array of particle positions
        grid: (n_cell,n_cell,n_cell) array
        boxsize: box size
        pbc: bool, apply periodic boundary conditions
    Returns:
        values at particle positions
    """
    n_cell = grid.shape[0]
    N = pos.shape[0]
    inv_cell_size = n_cell / boxsize
    out = np.zeros(N, dtype=dtype)
    
    index = np.zeros(3, dtype=np.int64)
    
    for i in range(N):
        for axis in range(3):
            idx = int(round(pos[i, axis] * inv_cell_size))
            if pbc:
                index[axis] = idx % n_cell
            else:
                index[axis] = min(max(idx,0), n_cell-1)
        out[i] = grid[index[0], index[1], index[2]]
    
    return out

@njit(parallel=False, nopython=True, fastmath=True)
def grid_to_particle_cic(pos, grid, boxsize, pbc=True, dtype=np.float64):
    """
    CIC interpolation: trilinear interpolation.
    """
    n_cell = grid.shape[0]
    N = pos.shape[0]
    inv_cell_size = n_cell / boxsize
    out = np.zeros(N, dtype=dtype)
    
    u = np.ones(3, dtype=dtype)
    d = np.ones(3, dtype=dtype)
    index_u = np.zeros(3, dtype=np.int64)
    index_d = np.zeros(3, dtype=np.int64)
    
    for i in range(N):
        for axis in range(3):
            dist = pos[i, axis] * inv_cell_size
            u[axis] = dist - int(dist)
            d[axis] = 1.0 - u[axis]
            if pbc:
                index_d[axis] = int(dist) % n_cell
                index_u[axis] = (index_d[axis] + 1) % n_cell
            else:
                index_d[axis] = min(int(dist), n_cell - 1)
                index_u[axis] = min(index_d[axis] + 1, n_cell - 1)
        
        # trilinear interpolation
        out[i] = (d[0]*d[1]*d[2]*grid[index_d[0], index_d[1], index_d[2]] +
                  d[0]*d[1]*u[2]*grid[index_d[0], index_d[1], index_u[2]] +
                  d[0]*u[1]*d[2]*grid[index_d[0], index_u[1], index_d[2]] +
                  d[0]*u[1]*u[2]*grid[index_d[0], index_u[1], index_u[2]] +
                  u[0]*d[1]*d[2]*grid[index_u[0], index_d[1], index_d[2]] +
                  u[0]*d[1]*u[2]*grid[index_u[0], index_d[1], index_u[2]] +
                  u[0]*u[1]*d[2]*grid[index_u[0], index_u[1], index_d[2]] +
                  u[0]*u[1]*u[2]*grid[index_u[0], index_u[1], index_u[2]])
    return out

'''@njit
def tsc_weight(d):
    d = abs(d)
    if d < 0.5:
        return 0.75 - d*d
    elif d < 1.5:
        return 0.5*(1.5 - d)**2
    else:
        return 0.0'''

@njit(parallel=False, nopython=True, fastmath=True)
def grid_to_particle_tsc(pos, grid, boxsize, pbc=True, dtype=np.float64):
    """
    TSC interpolation: cubic (3x3x3) interpolation
    """
    n_cell = grid.shape[0]
    N = pos.shape[0]
    inv_cell_size = n_cell / boxsize
    out = np.zeros(N, dtype=dtype)
    
    for p in range(N):
        x, y, z = pos[p] * inv_cell_size
        i0, j0, k0 = int(np.floor(x)), int(np.floor(y)), int(np.floor(z))
        val = 0.0
        for di in range(-1,2):
            for dj in range(-1,2):
                for dk in range(-1,2):
                    ii, jj, kk = i0+di, j0+dj, k0+dk
                    if pbc:
                        ii %= n_cell
                        jj %= n_cell
                        kk %= n_cell
                    else:
                        if ii<0 or ii>=n_cell or jj<0 or jj>=n_cell or kk<0 or kk>=n_cell:
                            continue
                    wx = tsc_weight(x - ii)
                    wy = tsc_weight(y - jj)
                    wz = tsc_weight(z - kk)
                    val += grid[ii,jj,kk]*wx*wy*wz
        out[p] = val
    return out
