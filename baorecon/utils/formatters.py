import numpy as np
from baorecon.utils.loggers import setup_logger
logger = setup_logger(__name__)

def _check_weights(weights):
    if weights.ndim != 1:
        raise ValueError("weights must be a 1D array")
    if (weights < 0).any():
        raise ValueError("weights must be non-negative")

def format_weights(weights, size, dtype=np.float32) -> np.ndarray:
    """
    Format weights array. If weights is None, create an array of ones of given size.
    Format the weights to the specified dtype.

    Parameters
    ----------
    weights : np.ndarray or None
        Array of weights of shape (N,). If None, an array of ones will be created.
    size : int
        Size of the weights array to create if weights is None.
    dtype : data-type, optional
        Desired data-type for the weights array. Default is np.float32.
    Returns
    -------
    np.ndarray
        Formatted weights array of shape (N,) with specified dtype.
    Raises
    -------
    ValueError
        If weights is not a 1D array or contains negative values.
    """
    if weights is None:
        logger.debug("weights is None, creating an array of ones")
        weights = np.ones(size, dtype=dtype)
    else:
        _check_weights(weights)

    return weights.astype(dtype)

def format_rectype(reconstruction_type: str) -> str:
    """
    Format and validate the reconstruction type string.

    Parameters
    ----------
    reconstruction_type : str
        The reconstruction type string to format and validate.

    Returns
    -------
    str
        The formatted reconstruction type string.

    Raises
    -------
    ValueError
        If the reconstruction type is not recognized.
    """
    valid_types = {"rec-sym", "rec-iso"}
    rec_type = reconstruction_type.strip().lower()
    if rec_type not in valid_types:
        raise ValueError(f"Invalid reconstruction type '{reconstruction_type}'. Valid types are: {valid_types}")
    return rec_type

def format_rsd_space(rsd_space: str) -> str:
    """
    Format and validate the RSD space string.

    Accepts variations like "real", "realspace", "real_space" and
    "redshift", "redshiftspace", "redshift_space" and returns
    canonical strings "RealSpace" or "RedshiftSpace".

    Parameters
    ----------
    rsd_space : str
        The RSD space string to format and validate.

    Returns
    -------
    str
        Canonical RSD space string: "RealSpace" or "RedshiftSpace".

    Raises
    -------
    ValueError
        If the RSD space is not recognized.
    """
    mapping = {
        "real": "RealSpace",
        "realspace": "RealSpace",
        "real_space": "RealSpace",
        "redshift": "RedshiftSpace",
        "redshiftspace": "RedshiftSpace",
        "redshift_space": "RedshiftSpace",
    }

    key = rsd_space.strip().replace(" ", "").lower()
    if key not in mapping:
        raise ValueError(f"Invalid RSD space '{rsd_space}'. Valid options are: {list(mapping.keys())}")
    return mapping[key]

def format_boxcentre(boxcentre, dtype=np.float32) -> np.ndarray:
    """
    Format the box centre. If None, set to the center of the box.

    Parameters
    ----------
    box_centre : np.ndarray 
        Array of shape (3,) representing the box centre.
    dtype : data-type, optional
        Desired data-type for the box centre array. Default is np.float32.
    Returns
    -------
    np.ndarray
        Formatted box centre array of shape (3,) with specified dtype.
    Raises
    -------
    ValueError
        If box_centre is not of shape (3,).
    """

    boxcentre = np.asarray(boxcentre, dtype=dtype)
    if boxcentre.shape != (3,):
        raise ValueError("boxcentre must be of shape (3,)")
    return boxcentre

def format_mas(MAS: str) -> str:
    """
    Validate the mass assignment scheme (MAS) string.

    Parameters
    ----------
    MAS : str
        The mass assignment scheme string to validate.

    Returns
    -------
    str
        The validated mass assignment scheme string.

    Raises
    -------
    ValueError
        If the MAS is not recognized.
    """
    valid_MAS = ['NGP', 'CIC', 'TSC', 'PCS']
    MAS = MAS.strip().upper()
    if MAS not in valid_MAS:
        raise ValueError(f"Invalid MAS '{MAS}'. Valid options are: {valid_MAS}")
    return MAS

# Odd multipliers ``k`` allowed in a multigrid-friendly mesh size ``k * 2**p``.
# Restricting to small odd factors keeps the per-axis number of factors of two
# large (good full-coarsening depth) while still offering a fine-grained set of
# admissible sizes.
_MULTIGRID_FACTORS = (1, 3, 5, 7)

# Minimum power-of-two exponent, so the smallest admissible size is ``1*2**2 = 4``
# (matching MIN_COARSE in the multigrid solver).
_MULTIGRID_MIN_EXP = 2


def format_nmesh(nmesh, dtype=np.float32) -> np.ndarray:
    """
    Format the number of mesh cells to a length-3 integer array.

    A scalar is broadcast to a cubic grid ``np.full(3, scalar)``; an array-like
    of shape (3,) is kept as a per-axis (anisotropic) number of cells.

    Parameters
    ----------
    nmesh : int or array-like
        Number of grid cells (scalar for a cubic grid, or shape (3,)).

    Returns
    -------
    np.ndarray
        Number of cells, integer array of shape (3,).

    Raises
    -------
    ValueError
        If ``nmesh`` does not have a scalar or (3,) shape, is not integer-valued,
        or contains non-positive values.
    """
    n = np.asarray(nmesh, dtype=dtype)
    if n.ndim == 0:
        n = np.full(3, n)
    elif n.shape != (3,):
        raise ValueError(f"nmesh must be a scalar or an array-like of shape (3,). Got shape {n.shape}.")

    n_int = n.astype(np.int32)
    if not np.all(np.asarray(n, dtype=dtype) == n_int):
        raise ValueError(f"nmesh must contain integer values. Got {nmesh}.")
    if np.any(n_int <= 0):
        raise ValueError(f"nmesh must contain positive integers. Got {nmesh}.")
    return n_int


def round_to_multigrid_friendly(n: int) -> int:
    """
    Round ``n`` up to the nearest *multigrid-friendly* integer.

    A multigrid-friendly size has the form ``k * 2**p`` with ``k`` in
    ``{1, 3, 5, 7}`` and ``p >= 2`` (smallest admissible value 4). Such sizes
    coarsen well under full (factor-of-two) multigrid coarsening: they carry
    many factors of two while still allowing a dense set of values.

    Parameters
    ----------
    n : int
        Positive integer to round up.

    Returns
    -------
    int
        Smallest ``m >= n`` of the form ``k * 2**p`` with ``k in {1,3,5,7}`` and ``p >= 2``.

    Raises
    ------
    ValueError
        If ``n`` is not a positive integer.
    """
    n = int(n)
    if n <= 0:
        raise ValueError(f"n must be a positive integer. Got {n}.")

    best = None
    for k in _MULTIGRID_FACTORS:
        p = _MULTIGRID_MIN_EXP
        while k * (2 ** p) < n:
            p += 1
        candidate = k * (2 ** p)
        if best is None or candidate < best:
            best = candidate
    return best


def nmesh_boxsize_from_cellsize(extent: np.ndarray, cellsize: float, dtype=np.float32) -> tuple:
    """
    Derive a per-axis ``(nmesh, boxsize)`` from a target (isotropic) cell size.

    For each axis the cell count covers the requested extent, then is rounded up
    to the nearest multigrid-friendly size ``k * 2**p`` (``k in {1,3,5,7}``,
    ``p >= 2``) so the grid coarsens well under full multigrid coarsening (such
    sizes are also even, satisfying the rfft-based solvers). The box grows to
    ``boxsize = nmesh * cellsize >= extent``.

    Parameters
    ----------
    extent : array-like of shape (3,)
        Per-axis extent to be covered (e.g. the padded span of the positions).
    cellsize : float
        Desired isotropic size of a single grid cell.

    Returns
    -------
    nmesh : np.ndarray
        Per-axis cell count, multigrid-friendly integer array of shape (3,).
    boxsize : np.ndarray
        Per-axis box size, float array of shape (3,), equal to ``nmesh * cellsize``.

    Raises
    -------
    ValueError
        If ``extent`` is not of shape (3,) or ``cellsize`` is not positive.
    """
    extent = np.asarray(extent, dtype=dtype)
    if extent.shape != (3,):
        raise ValueError(f"extent must be an array-like of shape (3,). Got shape {extent.shape}.")

    cellsize = float(cellsize)
    if not np.isfinite(cellsize) or cellsize <= 0:
        raise ValueError(f"cellsize must be a positive finite number. Got {cellsize}.")

    nmesh = np.ceil(extent / cellsize).astype(np.int32)
    nmesh = np.array([round_to_multigrid_friendly(n) for n in nmesh], dtype=np.int32)
    boxsize = (nmesh.astype(np.float64) * cellsize).astype(dtype)

    logger.info(
        "Derived grid from cellsize=%.6g: nmesh=%s (multigrid-friendly), boxsize=%s (extent=%s)",
        cellsize,
        np.array2string(nmesh, separator=', '),
        np.array2string(boxsize, precision=4, separator=', '),
        np.array2string(extent, precision=4, separator=', '),
    )
    return nmesh, boxsize


def check_boxsize(box_size: np.ndarray, positions: np.ndarray) -> bool:
    """
    Check that the box size is big enough to contain the whole positions array.

    Parameters
    ----------
    box_size : np.ndarray
        Size of the simulation box.
    positions : np.ndarray
        Array of positions to check against the box size.

    Returns
    -------
    bool : True if box size is sufficient, False otherwise.
    """
    max_sep = positions.max(axis=0) - positions.min(axis=0)

    #check boxsize
    if (np.asarray(box_size) <= max_sep).any():
        raise ValueError(f"Box size {box_size} is too small to contain the positions with maximum separation {max_sep}.")
        return False
    return True
    
def format_boxsize(boxsize, positions: np.ndarray, pbc: bool) -> np.ndarray:
    """
    Normalize the box size to a per-axis array of shape (3,) and validate it.

    A scalar is broadcast to ``[L, L, L]``. When ``pbc`` is False the box is
    checked against the positions (must contain them on every axis).

    Parameters
    ----------
    boxsize : float or array-like
        Size of the simulation box (scalar for a cubic box, or shape (3,)).
    positions : np.ndarray
        Array of positions to check against the box size (when ``pbc`` is False).
    pbc : bool
        Whether periodic boundary conditions are applied. If True, the box size
        is not checked against positions.

    Returns
    -------
    np.ndarray
        The formatted per-axis box size, shape (3,).

    Raises
    -------
    ValueError
        If the box size has a bad shape, or is too small to contain the positions.
    """
    arr = np.asarray(boxsize, dtype=np.float32)
    if arr.ndim == 0:
        box = np.full(3, float(arr), dtype=np.float32)
    elif arr.shape == (3,):
        box = arr
    else:
        raise ValueError(f"boxsize must be a scalar or a length-3 array-like, got shape {arr.shape}.")

    if not pbc:
        if not check_boxsize(box, positions):
            raise ValueError(
                f"Provided box size {box} is too small to contain the positions. "
                "Provide a larger box size or set it to None to infer from positions."
            )
    return box

def set_boxsize_from_positions(positions: np.ndarray, padding: float) -> np.ndarray:
    """
    Set the per-axis box size from the position extent, with optional padding.

    Parameters
    ----------
    positions : np.ndarray (N, 3)
        Array of positions to determine the box size.
    padding : float, optional
        Additional padding added to every axis.

    Returns
    -------
    np.ndarray
        The computed per-axis box size, shape (3,).
    """
    max_sep = positions.max(axis=0) - positions.min(axis=0)
    box_size = (max_sep + padding).astype(np.float32)
    logger.info(
        f"Setting box size to {np.array2string(box_size, precision=4, separator=', ')} "
        f"based on per-axis separation in positions with padding {padding}."
    )
    return box_size

def set_boxcentre_from_positions(positions: np.ndarray, dtype=np.float32) -> np.ndarray:
    """
    Set the box centre from the midpoint of the position range.

    Parameters
    ----------
    positions : np.ndarray (N, 3)
        Array of positions to determine the box centre.
    dtype : data-type, optional
        Desired data-type for the box centre array. Default is np.float32.

    Returns
    -------
    np.ndarray
        The computed box centre.
    """
    min_corner = positions.min(axis=0)
    max_corner = positions.max(axis=0)
    boxcentre = 0.5 * (min_corner + max_corner)
    logger.info(f"Setting box centre to {boxcentre} from position range")
    return np.asarray(boxcentre, dtype=dtype)

def survey_to_box_frame(positions: np.ndarray, min_corner: np.ndarray, boxsize: float, 
                        pbc: bool = True, dtype=np.float32) -> np.ndarray:
    """
    Transform positions from survey frame to box frame [0, boxsize].

    Parameters
    ----------
    positions : np.ndarray
        Positions in survey frame, shape (N, 3)
    min_corner : np.ndarray
        Lower corner of the box in survey frame, shape (3,)
    boxsize : float
        Size of the box
    pbc : bool
        Whether to apply periodic boundary conditions. Default is True.
    dtype : data-type, optional
        Desired data-type for output. Default is np.float32.

    Returns
    -------
    np.ndarray
        Positions in box frame [0, boxsize], shape (N, 3)

    Raises
    ------
    ValueError
        If positions are outside [0, boxsize] when pbc=False.
    """
    pos = np.asarray(positions, dtype=dtype)
    pos_shifted = pos - min_corner
    if pbc:
        pos_shifted = np.mod(pos_shifted, boxsize)
    else:
        # validate bounds
        if (pos_shifted < 0).any() or (pos_shifted > boxsize).any():
            raise ValueError(
                f"Positions in box frame must be inside [0, boxsize] when pbc=False. "
                f"Got min {pos_shifted.min()}, max {pos_shifted.max()}, boxsize {boxsize}."
            )
    return pos_shifted.astype(dtype)

def format_padding(padding:float, pbc: bool) -> float:
    """
    Format the padding value. If padding is None, set to 0.0. If pbc is True, padding must be 0.0.

    Parameters
    ----------
    padding : float or None
        The padding value to format. If None, it will be set to 0.0.
    pbc : bool
        Whether periodic boundary conditions are applied. If True, padding must be 0.0.

    Returns
    -------
    float
        The formatted padding value.

    Raises
    -------
    ValueError
        If pbc is True and padding is not 0.0.
    """
    if padding is None:
        return 0.0
    if pbc and padding != 0.0:
        raise ValueError("Padding must be 0.0 when periodic boundary conditions are applied.")
    return float(padding)
