import numpy as np
from functools import lru_cache
from typing import Optional, Tuple, Sequence, Union
from scipy.interpolate import interp1d

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import Planck18, FlatLambdaCDM

from baorecon.utils.loggers import setup_logger
logger = setup_logger(__name__)


def create_cosmology(
    H0: float = 67.11,
    Om0: float = 0.3175,
    Ob0: float = 0.049,
    Tcmb0: float = 2.7255,
    Mnu: Optional[Tuple[float, ...]] = None,
    name: Optional[str] = None,
) -> FlatLambdaCDM:
    """
    Create a custom flat LCDM cosmology instance.
    """
    if H0 <= 0:
        raise ValueError("H0 must be positive. Got {0}.".format(H0))
    if not (0 < Om0 < 1):
        raise ValueError("Om0 must be in (0, 1). Got {0}.".format(Om0))

    # Build kwargs with only non-None values
    kwargs = {"H0": H0, "Om0": Om0}
    kwargs["Ob0"] = Ob0
    kwargs["Tcmb0"] = Tcmb0
    if Mnu is not None:
        kwargs["Mnu"] = Mnu
    if name is not None:
        kwargs["name"] = name

    cosmo = FlatLambdaCDM(**kwargs)
    logger.debug(
        "Created cosmology: H0=%.2f, Om0=%.4f, Ob0=%.3f, name=%s", 
        H0, Om0, Ob0, name
    )
    return cosmo


def _as_1d_array(values: Sequence, name: str) -> np.ndarray:
    """Convert input to 1D float32 array with validation."""
    arr = np.asarray(values).reshape(-1)
    if arr.size == 0:
        logger.warning("%s is empty.", name)
        raise ValueError("{0} must not be empty.".format(name))
    return arr


def _check_same_length(a: np.ndarray, b: np.ndarray, c: np.ndarray, names: Tuple[str, str, str]) -> None:
    """Validate that three arrays have the same length."""
    if not (a.size == b.size == c.size):
        logger.warning(
            "%s, %s, %s must have the same length. Got %d, %d, %d.",
            names[0], names[1], names[2], a.size, b.size, c.size
        )
        raise ValueError(
            "{0}, {1}, {2} must have the same length. Got {3}, {4}, {5}.".format(
                names[0], names[1], names[2], a.size, b.size, c.size
            )
        )


#@lru_cache(maxsize=16)
def _get_cosmology_grids(cosmo: FlatLambdaCDM, z_max: float, num_points: int = 2000):
    """
    Precomputes a highly dense redshift-to-distance grid. 
    Rounds up z_max to stable values to maximize cache hits.
    """
    z_cache_max = float(np.ceil(z_max)) * 1.1
    z_cache_max = max(1.0, z_cache_max) 
    
    z_grid = np.linspace(0.0, z_cache_max, num_points)
    d_grid = cosmo.comoving_distance(z_grid).to_value(u.Mpc)
    return z_grid, d_grid

def radec_z_to_xyz(
    ra: Sequence,
    dec: Sequence,
    redshift: Sequence,
    cosmo: Optional[FlatLambdaCDM] = None,
    ra_dec_unit: str = "deg",
    frame: str = "icrs",
    distance_unit: str = "Mpc/h",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert RA, DEC, redshift to Cartesian coordinates x, y, z.
    """

    if ra_dec_unit not in ["deg", "rad"]:
        raise ValueError("ra_dec_unit must be 'deg' or 'rad'.")
        
    if distance_unit not in ["Mpc", "Mpc/h"]:
        raise ValueError("distance_unit must be 'Mpc' or 'Mpc/h'.")

    ra_arr: np.ndarray = _as_1d_array(ra, "ra")
    dec_arr = _as_1d_array(dec, "dec")
    z_arr = _as_1d_array(redshift, "redshift")
    _check_same_length(ra_arr, dec_arr, z_arr, ("ra", "dec", "redshift"))

    n_coords = ra_arr.size

    if np.any(z_arr < 0):
        raise ValueError("redshift must be >= 0.")

    if cosmo is None:
        cosmo = Planck18
    
    dt = ra.dtype
    one = dt.type(1.0)
    h = dt.type(cosmo.h)
    
    # Compute distances
    dist_mpc = comoving_distance(z_arr, cosmo)
    xyz_scale = h if distance_unit == "Mpc/h" else one
    dist_out = dist_mpc * (h if distance_unit == "Mpc/h" else one)

    # Convert to radians
    if ra_dec_unit == "deg":
        ra_rad = np.radians(ra_arr)
        dec_rad = np.radians(dec_arr)
    else:
        ra_rad, dec_rad = ra_arr, dec_arr

    # Vectorized Cartesian conversion
    cos_dec = np.cos(dec_rad)
    xyz = np.empty((n_coords, 3), dtype=ra_arr.dtype)
    r_scaled = np.multiply(dist_mpc, xyz_scale) 
    r_xy = np.multiply(r_scaled, cos_dec)       

    ## x coord
    np.cos(ra_rad, out=xyz[:, 0])
    np.multiply(xyz[:, 0], r_xy, out=xyz[:, 0])

    ## y coord
    np.sin(ra_rad, out=xyz[:, 1])
    np.multiply(xyz[:, 1], r_xy, out=xyz[:, 1])

    ## z coord
    np.sin(dec_rad, out=xyz[:, 2])
    np.multiply(xyz[:, 2], r_scaled, out=xyz[:, 2])
    
    logger.debug("Coordinate conversion to XYZ complete.")
    return xyz, dist_out


def xyz_to_radec_z(
    xyz: Sequence,
    cosmo: Optional[FlatLambdaCDM] = None,
    ra_dec_unit: str = "deg",
    frame: str = "icrs",
    distance_unit: str = "Mpc/h",
    z_atol: float = 1e-8,  
    z_max: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert Cartesian coordinates x, y, z to RA, DEC, redshift.
    """
    xyz_arr: np.ndarray = np.asarray(xyz)
    if xyz_arr.size == 0:
        raise ValueError("xyz must not be empty.")
    if xyz_arr.ndim != 2 or xyz_arr.shape[1] != 3:
        raise ValueError("xyz must have shape (N, 3).")
    if ra_dec_unit not in ["deg", "rad"]:
        raise ValueError("ra_dec_unit must be 'deg' or 'rad'.")
    if distance_unit not in ["Mpc", "Mpc/h"]:
        raise ValueError("distance_unit must be 'Mpc' or 'Mpc/h'.")

    
    n_coords = xyz_arr.shape[0]
    if cosmo is None:
        cosmo = Planck18

    dt = xyz_arr.dtype
    one = dt.type(1.0)
    h = dt.type(cosmo.h)

    # Scale for the spherical computation
    xyz_scale = one / h if distance_unit == "Mpc/h" else one
    x, y, zc = xyz_arr[:, 0] * xyz_scale, xyz_arr[:, 1] * xyz_scale, xyz_arr[:, 2] * xyz_scale

    # Compute spherical coordinates
    r_xy = np.sqrt(x**2 + y**2)
    dist_mpc = np.sqrt(r_xy**2 + zc**2)
    
    lon = np.arctan2(y, x)
    lat = np.arcsin(zc / dist_mpc)

    if ra_dec_unit == "deg":
        c_360 = dt.type(360.0)
        ra = np.degrees(lon) % c_360
        dec = np.degrees(lat)
    else:
        c_2pi = dt.type(2.0 * np.pi)
        ra = lon % c_2pi
        dec = lat

    dist_out = dist_mpc * (h if distance_unit == "Mpc/h" else one)

    # Redshift interpolation
    redshift = distance_to_redshift(dist_mpc, cosmo)
    
    return ra, dec, redshift, dist_out


def comoving_distance(redshift: Union[float, np.ndarray], cosmo: Optional[FlatLambdaCDM] = None) -> np.ndarray:
    """Compute comoving distance for given redshift(s) using ultra-fast grid lookup."""
    if cosmo is None:
        cosmo = Planck18

    z_arr = np.asarray(redshift)
    if np.any(z_arr < 0):
        raise ValueError("redshift must be >= 0.")

    z_grid, d_grid = _get_cosmology_grids(cosmo, z_max=float(np.max(z_arr)))
    
    # Use cubic interpolation instead of linear for tight precision
    f = interp1d(z_grid, d_grid, kind='cubic', fill_value="extrapolate")
    return f(z_arr).astype(z_arr.dtype, copy=False)


def distance_to_redshift(
    comoving_distance: Union[float, np.ndarray], 
    cosmo: Optional[FlatLambdaCDM] = None,
    z_max: float = 5.0
) -> np.ndarray:
    """Compute redshift for given comoving distance(s) using ultra-fast grid lookup."""
    if cosmo is None:
        cosmo = Planck18

    d_arr = np.asarray(comoving_distance)
    if np.any(d_arr < 0):
        raise ValueError("comoving_distance must be >= 0.")

    # Fix: Ensure z_max dictates grid creation, NOT the physical distance (d_arr)
    z_grid, d_grid = _get_cosmology_grids(cosmo, z_max=z_max)
    
    # Use cubic interpolation instead of linear for tight precision
    f = interp1d(d_grid, z_grid, kind='cubic', fill_value="extrapolate")
    return f(d_arr).astype(d_arr.dtype, copy=False)