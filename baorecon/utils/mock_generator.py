import numpy as np
from baorecon.utils.loggers import setup_logger
logger = setup_logger(__name__)

def _gaussian_field_fourier(pk_grid, volume, n_cell, rng=np.random):
    """
    Draw a Gaussian random field in Fourier space with power spectrum ``pk_grid``.

    The field is built as the FFT of real white noise, ``W_k = rfftn(w)`` with
    ``w ~ N(0, 1)`` per cell, scaled by ``sqrt(P(k) V / N^3)``. Being the transform
    of a real field, ``W_k`` is Hermitian by construction, including on the
    self-conjugate planes ``k_z = 0`` and ``k_z = Nyquist`` where the previous
    amplitude-and-phase drawing left ``delta(k)`` and ``delta(-k)`` independent and
    ``irfftn`` silently halved their power. ``<|W_k|^2> = N^3`` for every mode, so
    ``<|delta_k|^2> = V P(k)``, with Rayleigh-distributed moduli as a complex
    Gaussian mode requires.

    Parameters
    ----------
    pk_grid : ndarray, shape (n, n, n//2 + 1)
        Power spectrum evaluated on the ``rfftn`` grid, in the convention
        ``<|delta_k|^2> = V P(k)``. Negative or NaN entries are set to zero.
    volume : float
        Box volume ``side**3``.
    n_cell : int
        Number of cells per side.
    rng : numpy random generator, optional
        Source of the white noise: the global ``np.random`` module (default) or a
        ``np.random.Generator``.

    Returns
    -------
    delta_k : ndarray, complex, shape (n, n, n//2 + 1)
        Hermitian Fourier coefficients with the zero mode set to 0.
    """
    pk_grid = np.nan_to_num(np.asarray(pk_grid, dtype=float), nan=0.0)
    pk_grid = np.clip(pk_grid, 0.0, None)
    white = rng.normal(size=(n_cell, n_cell, n_cell))
    delta_k = np.fft.rfftn(white) * np.sqrt(pk_grid * volume / n_cell**3)
    delta_k[0, 0, 0] = 0.0
    return delta_k


def generate_gaussian_map(pk_func, side, spacing, seed=None):

    """
    Generates a Gaussian random field from an input power spectrum in a given volume.

    Parameters:
    -----------
    pk_func : function
        A function that takes an array of wavenumbers (k) and returns the power spectrum P(k) at those wavenumbers.
    side : float
        The physical size of the side of the cubic volume.
    spacing : float
        The grid spacing for the Fourier transform. Determines the resolution of the grid.
    seed : int or None, optional
        Seed for the white noise behind the field. ``None`` (default) draws from
        NumPy's global generator, so ``np.random.seed`` keeps working; an integer
        uses a private ``np.random.default_rng(seed)`` and makes the field reproducible.

    Returns:
    --------
    delta_k : ndarray
        The generated Gaussian random field in Fourier space.
    delta_x : ndarray
        The generated Gaussian random field in real space.

    Example:
    --------
    >>> def pk_func(k):
    >>>     return k**-3  # Example power spectrum function
    >>> side = 1.0
    >>> spacing = 0.1
    >>> delta_k, delta_x = generate_gaussian_map(pk_func, side, spacing)
    """

    # set the volume
    volume = side**3

    n_cell = int(side//spacing)

    # set the Fourier grid

    kx = np.fft.fftfreq(n_cell, spacing)*np.pi*2
    ky = np.fft.fftfreq(n_cell, spacing)*np.pi*2
    kz = np.fft.rfftfreq(n_cell, spacing)*np.pi*2

    #kx = np.fft.fftshift(kx)
    #ky = np.fft.fftshift(ky)

    KX, KY, KZ = np.meshgrid(kx,ky,kz, indexing='ij')

    knorm = np.sqrt(KX**2+KY**2+KZ**2)
    pks = pk_func(knorm)

    # Hermitian Gaussian field: FFT of real white noise scaled by sqrt(P V / N^3)
    rng = np.random if seed is None else np.random.default_rng(seed)
    delta_k = _gaussian_field_fourier(pks, volume, n_cell, rng)

    delta_x = np.fft.irfftn(delta_k, s=(n_cell, n_cell, n_cell), norm='backward')/spacing**3

    return delta_k, delta_x


def generate_lognormal_map(pk_func, side, spacing, seed=None):
    """
    Generates a lognormal random field from an input power spectrum in a given volume.
    
    Parameters:
    -----------
    pk_func : function
        A function that takes an array of wavenumbers (k) and returns the power spectrum P(k) at those wavenumbers.
        side : float
        The physical size of the side of the cubic volume.
        spacing : float
        The grid spacing for the Fourier transform. Determines the resolution of the grid.
        seed : int or None, optional
        Seed for the white noise behind the field. ``None`` (default) draws from
        NumPy's global generator, so ``np.random.seed`` keeps working; an integer
        uses a private ``np.random.default_rng(seed)`` and makes the field reproducible.
        
        Returns:
        --------
        delta_x : ndarray
        The generated lognormal density field in real space.
        
        Example:
        --------
        >>> def pk_func(k):
        >>>     return k**-3  # Example power spectrum function
        >>> side = 1.0
        >>> spacing = 0.1
        >>> delta_x = generate_lognormal_map(pk_func, side, spacing)
        """
    
    # set the volume
    volume = side**3

    n_cell = int(side//spacing)

    # set the Fourier grid
    kx = np.fft.fftfreq(n_cell, spacing)*np.pi*2
    ky = np.fft.fftfreq(n_cell, spacing)*np.pi*2
    kz = np.fft.rfftfreq(n_cell, spacing)*np.pi*2
        
    KX, KY, KZ = np.meshgrid(kx,ky,kz, indexing='ij')

    # get the power spectrum on the grid
    knorm = np.sqrt(KX**2+KY**2+KZ**2)
    pks = pk_func(knorm)

    # get 2PCF on the grid
    xi = np.fft.irfftn(pks) / spacing**3

    if np.any(xi <= -1):
        raise ValueError("Invalid input: xi contains elements <= -1")
    # generate gaussian 2PCF
    xi_g = np.log(1+xi)

    # get PS from xi_g. The transform of log(1 + xi) is real up to round-off and is
    # not guaranteed to be positive everywhere: negative values are clipped to zero
    # (standard lognormal prescription; the old code took sqrt of a complex array).
    pk_g = np.clip((np.fft.rfftn(xi_g) * spacing**3).real, 0.0, None)

    # generate G(k), a Hermitian gaussian field in Fourier space with power spectrum pk_g
    rng = np.random if seed is None else np.random.default_rng(seed)
    G_k = _gaussian_field_fourier(pk_g, volume, n_cell, rng)

    # Compute G(x), the gaussian field in real space
    G_x = np.fft.irfftn(G_k, s=(n_cell, n_cell, n_cell), norm='backward')/spacing**3

    print('mean of G_x is = ', np.mean(G_x))
    # get delta from G_x using lognormal transform

    # compute variance of G_x: E[e^G_x] = e^(mu+var(G_x)/2)
    var_G = np.var(G_x)
    print('variance of G_x is = ', var_G)
    # subtract var(G_x)/2 to get E[G_x]=e^mu
    delta_x = np.exp(G_x-var_G/2)-1

    return delta_x


def poisson_sample_from_map(delta_x, side, spacing, N_objects, seed=666):
    """
    Generates a Poisson realization of a density field from an input density contrast field.

    Parameters:
    -----------
    delta_x : ndarray
        A 3D array representing the density contrast field.
    side : float
        The physical size of the simulation box (e.g., in Mpc/h).
    spacing : float
        The spatial resolution of the grid (e.g., in Mpc/h).
    N_objects : int
        The total number of objects to sample.
    seed : int, optional
        The random seed to use for reproducibility. Default is 666.

    Returns:
    --------
    points : ndarray
        An array of shape `(N_objects, 3)` containing the positions of the sampled objects.

    Example:
    --------
    >>> delta_x = np.random.randn(64, 64, 64)
    >>> side = 100.0
    >>> spacing = 1.5625
    >>> N_objects = 1000
    >>> points = poisson_sample_map(delta_x, side, spacing, N_objects)
    """

    # Set the random state
    np.random.seed(seed)
    # compute grid quantities
    Volume = side**3

    n_bar = N_objects/Volume
    cell_volume = spacing**3

    n_cell = int(side//spacing)

    # Ensure delta_x has the correct shape
    if delta_x.shape != (n_cell, n_cell, n_cell):
        raise ValueError(f"delta_x should have shape ({n_cell}, {n_cell}, {n_cell})")

    # from delta to density
    n_x = n_bar * (1 + delta_x)

    # from number density to mean number (not necessarily integer) of objects per cell

    Nobj_x = n_x * cell_volume

    # Poisson realization: integer number of obj to generate in each cell

    Nobj_x_sample = np.random.poisson(Nobj_x)

    # Generate points
    points = []
    for i in range(n_cell):
        x_min, x_max = i*spacing, (i+1)*spacing
        for j in range(n_cell):
            y_min, y_max = j*spacing, (j+1)*spacing
            for k in range(n_cell):
                z_min, z_max = k*spacing, (k+1)*spacing

                if Nobj_x_sample[i,j,k] > 0 :
                    points.extend(np.random.uniform(size=(Nobj_x_sample[i,j,k], 3), low=[x_min,y_min,z_min], high=[x_max,y_max, z_max]))

    print('total number of objects = ', len(points), 'out of ', N_objects)
    return np.array(points)