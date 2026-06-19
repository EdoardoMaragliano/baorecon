import numpy as np
from zeldareco.utils.loggers import setup_logger
logger = setup_logger(__name__)

def _check_positions(pos: np.ndarray, box_size: np.ndarray, pbc: bool) -> None:
    """
    Check that positions are valid, i.e., have correct shape and are within the box.
    Parameters
    ----------
    pos : np.ndarray
        Array of shape (N, 3) containing the positions.
    box_size : float
        Size of the simulation box.
    pbc : bool
        Whether to apply periodic boundary conditions."""
    # pos: (N, 3)
    if pos.ndim != 2:
        raise ValueError("pos must be a 2D array of shape (N, 3)")
    if pos.shape[1] != 3:
        raise ValueError("pos must have shape (N, 3)")
    
    # Check positions are within the box
    if not pbc:
        if (pos < 0).any() or (pos > box_size).any():
            raise ValueError(f"pos must be in the range [0, box_size]. Got min {pos.min()} and max {pos.max()}. box_size: {box_size}.")


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

    return np.asarray(weights, dtype=dtype)

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
    
def format_boxsize(boxsize: np.ndarray, positions: np.ndarray, pbc: bool) -> float:
    """
    Checks if the provided boxsize is big enough to contain the positions.
    
    Parameters
    ----------
    box_size : np.ndarray or None
        Size of the simulation box. If None, it will be set from positions.
    positions : np.ndarray or None
        Array of positions to check against the box size. 
    pbc : bool
        Whether periodic boundary conditions are applied. If True, box size is not checked against positions, since they are effectively wrapped.

    Returns
    -------
    float
        The formatted box size.

    Raises
    -------
    ValueError
        If the provided box size is too small to contain the positions.
    """
    if not pbc:
        if not check_boxsize(boxsize, positions):
            raise ValueError(f"Provided box size {boxsize} is too small to contain the positions. Please provide a larger box size or set it to None to infer from positions.")
        return np.asarray(boxsize, dtype=np.float32).max()  # ensure it's a scalar float
    else:
        boxsize = np.asarray(boxsize, dtype=np.float32).max()  # ensure it's a scalar float
        return boxsize

def set_boxsize_from_positions(positions: np.ndarray, padding: float) -> np.ndarray:
    """
    Set the box size from the maximum separation in the positions array, with optional padding.

    Parameters
    ----------
    positions : np.ndarray (N, 3)
        Array of positions to determine the box size.
    padding : float, optional
        Additional padding to add to the box size. 

    Returns
    -------
    np.ndarray
        The computed box size.
    """
    max_sep = positions.max(axis=0) - positions.min(axis=0)
    
    side_length = max_sep.max() + padding  # use the maximum dimension for a cubic box   
    box_size = np.full(3, side_length, dtype=np.float32) 
    logger.info(f"Rectangular box not yet supported. Setting box size to {np.array2string(box_size, precision=4, separator=', ')} based on maximum separation in positions with padding {padding}.")    

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
    pos = np.asarray(positions, dtype=np.float64)
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
