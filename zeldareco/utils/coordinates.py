import numpy as np
from typing import Optional, Tuple, Sequence, Union
from scipy.interpolate import interp1d

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import Planck18, FlatLambdaCDM

from zeldareco.utils.loggers import setup_logger
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
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
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


def _get_distance_to_redshift_interpolator(cosmo: FlatLambdaCDM, z_max: float = 5.0, num_points: int = 1000) -> interp1d:
    """
    Helper function to create a fast, vectorized interpolator for converting 
    comoving distance back to redshift.
    """
    logger.debug(
        "Building fast distance-to-redshift interpolator (z_max=%.2f, points=%d)", 
        z_max, num_points
    )
    
    # Create a dense redshift grid (adding a 10% buffer to z_max to safely handle edge cases)
    z_grid = np.linspace(0.0, z_max * 1.1, num_points)
    
    # Compute the corresponding comoving distances in Mpc
    d_grid = cosmo.comoving_distance(z_grid).to_value(u.Mpc)
    
    # Create and return the interpolator (Distance -> Redshift)
    return interp1d(d_grid, z_grid, kind='cubic', bounds_error=False, fill_value="extrapolate")

'''
def radec_z_to_xyz(
    ra: Sequence,
    dec: Sequence,
    redshift: Sequence,
    cosmo: Optional[FlatLambdaCDM] = None,
    ra_dec_unit: str = "deg",
    frame: str = "icrs",
    distance_unit: str = "Mpc",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert RA, DEC, redshift to Cartesian coordinates x, y, z.
    """
    ra_arr: np.ndarray = _as_1d_array(ra, "ra")
    dec_arr = _as_1d_array(dec, "dec")
    z_arr = _as_1d_array(redshift, "redshift")
    _check_same_length(ra_arr, dec_arr, z_arr, ("ra", "dec", "redshift"))

    n_coords = ra_arr.size
    logger.info("Converting %d coordinates from RA/DEC/Z to XYZ (%s)...", n_coords, distance_unit)

    if np.any(z_arr < 0):
        logger.error("Negative redshifts detected in input array.")
        raise ValueError("redshift must be >= 0.")

    if cosmo is None:
        cosmo = Planck18
    
    logger.debug("Using cosmology: %s (H0=%.2f, Om0=%.4f)", cosmo.name, cosmo.H0.value, cosmo.Om0)

    if ra_dec_unit == "deg":
        ra_q = ra_arr * u.deg
        dec_q = dec_arr * u.deg
    elif ra_dec_unit == "rad":
        ra_q = ra_arr * u.rad
        dec_q = dec_arr * u.rad
    else:
        raise ValueError("ra_dec_unit must be 'deg' or 'rad'.")

    logger.debug("Computing comoving distances from redshifts...")
    dist_mpc = cosmo.comoving_distance(z_arr).to_value(u.Mpc)

    if distance_unit == "Mpc":
        dist_out = dist_mpc
        xyz_scale = 1.0
        sky_distance = dist_mpc
    elif distance_unit == "Mpc/h":
        dist_out = dist_mpc * cosmo.h
        xyz_scale = cosmo.h
        sky_distance = dist_mpc
    else:
        raise ValueError("distance_unit must be 'Mpc' or 'Mpc/h'.")

    logger.debug("Applying SkyCoord spherical-to-cartesian transformation...")
    sc = SkyCoord(ra=ra_q, dec=dec_q, distance=sky_distance * u.Mpc, frame=frame)
    xyz = np.column_stack(
        (
            sc.cartesian.x.to_value(u.Mpc) * xyz_scale,
            sc.cartesian.y.to_value(u.Mpc) * xyz_scale,
            sc.cartesian.z.to_value(u.Mpc) * xyz_scale,
        )
    )
    
    logger.debug("Coordinate conversion to XYZ complete.")
    return xyz, dist_out


def xyz_to_radec_z(
    xyz: Sequence,
    cosmo: Optional[FlatLambdaCDM] = None,
    ra_dec_unit: str = "deg",
    frame: str = "icrs",
    distance_unit: str = "Mpc",
    z_atol: float = 1e-8,  
    z_max: float = 10.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert Cartesian coordinates x, y, z to RA, DEC, redshift.
    """
    xyz_arr: np.ndarray = np.asarray(xyz, dtype=np.float32)
    if xyz_arr.ndim != 2 or xyz_arr.shape[1] != 3:
        raise ValueError("xyz must have shape (N, 3).")
    
    n_coords = xyz_arr.shape[0]
    if n_coords == 0:
        logger.warning("xyz array is empty.")
        raise ValueError("xyz must not be empty.")

    logger.info("Converting %d coordinates from XYZ (%s) to RA/DEC/Z...", n_coords, distance_unit)

    if cosmo is None:
        cosmo = Planck18
        
    logger.debug("Using cosmology: %s (H0=%.2f, Om0=%.4f)", cosmo.name, cosmo.H0.value, cosmo.Om0)

    x = xyz_arr[:, 0]
    y = xyz_arr[:, 1]
    zc = xyz_arr[:, 2]

    if distance_unit == "Mpc":
        xyz_scale = 1.0
    elif distance_unit == "Mpc/h":
        xyz_scale = 1.0 / cosmo.h
    else:
        raise ValueError("distance_unit must be 'Mpc' or 'Mpc/h'.")

    logger.debug("Applying SkyCoord cartesian-to-spherical transformation...")
    sc = SkyCoord(
        x=x * xyz_scale * u.Mpc,
        y=y * xyz_scale * u.Mpc,
        z=zc * xyz_scale * u.Mpc,
        representation_type="cartesian",
        frame=frame,
    )

    if ra_dec_unit == "deg":
        ra = sc.spherical.lon.to_value(u.deg)
        dec = sc.spherical.lat.to_value(u.deg)
    elif ra_dec_unit == "rad":
        ra = sc.spherical.lon.to_value(u.rad)
        dec = sc.spherical.lat.to_value(u.rad)
    else:
        raise ValueError("ra_dec_unit must be 'deg' or 'rad'.")

    dist = sc.spherical.distance.to_value(u.Mpc)
    dist_mpc = dist
    
    if distance_unit == "Mpc":
        dist_out = dist_mpc
    else:
        dist_out = dist_mpc * cosmo.h

    # Fast vectorized conversion using the helper interpolator
    logger.debug("Interpolating distances to extract redshifts...")
    z_interp = _get_distance_to_redshift_interpolator(cosmo, z_max=z_max)
    redshift = z_interp(dist_mpc).astype(np.float32)
    
    logger.debug("Coordinate conversion to RA/DEC/Z complete.")
    return ra, dec, redshift, dist_out
'''

def radec_z_to_xyz(
    ra: Sequence,
    dec: Sequence,
    redshift: Sequence,
    cosmo: Optional[FlatLambdaCDM] = None,
    ra_dec_unit: str = "deg",
    frame: str = "icrs",
    distance_unit: str = "Mpc",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert RA, DEC, redshift to Cartesian coordinates x, y, z.
    """
    ra_arr: np.ndarray = _as_1d_array(ra, "ra")
    dec_arr = _as_1d_array(dec, "dec")
    z_arr = _as_1d_array(redshift, "redshift")
    _check_same_length(ra_arr, dec_arr, z_arr, ("ra", "dec", "redshift"))

    n_coords = ra_arr.size
    logger.info("Converting %d coordinates from RA/DEC/Z to XYZ (%s)...", n_coords, distance_unit)

    if np.any(z_arr < 0):
        raise ValueError("redshift must be >= 0.")

    if cosmo is None:
        cosmo = Planck18
    
    # Calcolo distanze
    dist_mpc = cosmo.comoving_distance(z_arr).to_value(u.Mpc)
    xyz_scale = cosmo.h if distance_unit == "Mpc/h" else 1.0
    dist_out = dist_mpc * (cosmo.h if distance_unit == "Mpc/h" else 1.0)

    # Conversione in radianti
    if ra_dec_unit == "deg":
        ra_rad = np.radians(ra_arr)
        dec_rad = np.radians(dec_arr)
    else:
        ra_rad, dec_rad = ra_arr, dec_arr

    # Conversione cartesiana vettorializzata 
    cos_dec = np.cos(dec_rad)
    xyz = np.empty((n_coords, 3), dtype=np.float32)
    xyz[:, 0] = dist_mpc * cos_dec * np.cos(ra_rad) * xyz_scale
    xyz[:, 1] = dist_mpc * cos_dec * np.sin(ra_rad) * xyz_scale
    xyz[:, 2] = dist_mpc * np.sin(dec_rad) * xyz_scale
    
    logger.debug("Coordinate conversion to XYZ complete.")
    return xyz, dist_out


def xyz_to_radec_z(
    xyz: Sequence,
    cosmo: Optional[FlatLambdaCDM] = None,
    ra_dec_unit: str = "deg",
    frame: str = "icrs",
    distance_unit: str = "Mpc",
    z_atol: float = 1e-8,  
    z_max: float = 10.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert Cartesian coordinates x, y, z to RA, DEC, redshift.
    """
    xyz_arr: np.ndarray = np.asarray(xyz, dtype=np.float32)
    if xyz_arr.ndim != 2 or xyz_arr.shape[1] != 3:
        raise ValueError("xyz must have shape (N, 3).")
    
    n_coords = xyz_arr.shape[0]
    if cosmo is None:
        cosmo = Planck18

    # Scala per il calcolo sferico
    xyz_scale = 1.0 / cosmo.h if distance_unit == "Mpc/h" else 1.0
    x, y, zc = xyz_arr[:, 0] * xyz_scale, xyz_arr[:, 1] * xyz_scale, xyz_arr[:, 2] * xyz_scale

    # Calcolo coordinate sferiche
    r_xy = np.sqrt(x**2 + y**2)
    dist_mpc = np.sqrt(r_xy**2 + zc**2)
    
    lon = np.arctan2(y, x)
    lat = np.arcsin(zc / dist_mpc)

    if ra_dec_unit == "deg":
        ra = np.degrees(lon) % 360
        dec = np.degrees(lat)
    else:
        ra = lon % (2 * np.pi)
        dec = lat

    dist_out = dist_mpc * (cosmo.h if distance_unit == "Mpc/h" else 1.0)

    # Interpolazione redshift
    z_interp = _get_distance_to_redshift_interpolator(cosmo, z_max=z_max)
    redshift = z_interp(dist_mpc).astype(np.float32)
    
    return ra, dec, redshift, dist_out

def comoving_distance(redshift: Union[float, np.ndarray], cosmo: Optional[FlatLambdaCDM] = None) -> np.ndarray:
    """
    Compute comoving distance for given redshift(s) and cosmology.
    """
    if cosmo is None:
        cosmo = Planck18

    z_arr = np.asarray(redshift, dtype=np.float32)
    if np.any(z_arr < 0):
        logger.error("Negative redshifts detected in comoving_distance calculation.")
        raise ValueError("redshift must be >= 0.")

    logger.debug("Computing comoving distance for %d redshift value(s)...", z_arr.size)
    dist_mpc = cosmo.comoving_distance(z_arr).to_value(u.Mpc)
    return dist_mpc * cosmo.h