import numpy as np
import pytest

astropy = pytest.importorskip("astropy")

from baorecon.utils.coordinates import radec_z_to_xyz, xyz_to_radec_z, create_cosmology


def _angle_diff_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute minimal angular difference in degrees, robust to 0/360 wrap."""
    return np.abs((a - b + 180.0) % 360.0 - 180.0)


def _angle_diff_rad(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute minimal angular difference in radians, robust to 0/2pi wrap."""
    return np.abs((a - b + np.pi) % (2.0 * np.pi) - np.pi)


DTYPES = [np.float32, np.float64]

# Round-trip closure tolerances per working precision. float32 has ~1e-7 relative
# machine precision, so angles (in degrees) close only to ~1e-5 -- we allow 1e-4;
# float64 keeps the original tight bounds.
_ROUNDTRIP_TOL = {
    np.dtype(np.float32): {"ang": 1e-4, "z": 5e-5, "dist": 2e-3},
    np.dtype(np.float64): {"ang": 1e-9, "z": 5e-6, "dist": 1e-6},
}
_CUSTOM_COSMO_TOL = {
    np.dtype(np.float32): {"ang": 1e-4, "z": 5e-5, "dist": 2e-3},
    np.dtype(np.float64): {"ang": 1e-9, "z": 1e-5, "dist": 1e-3},
}


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "ra_dec_unit, distance_unit, seed",
    [("deg", "Mpc", 42), ("rad", "Mpc", 7), ("deg", "Mpc/h", 11)],
)
def test_radec_xyz_roundtrip(dtype, ra_dec_unit, distance_unit, seed) -> None:
    """RA/DEC/z -> xyz -> RA/DEC/z closes and preserves the input dtype."""
    rng = np.random.RandomState(seed)
    n = 40
    if ra_dec_unit == "deg":
        ra = rng.uniform(0.0, 360.0, n).astype(dtype)
        dec = rng.uniform(-80.0, 80.0, n).astype(dtype)
        ang_diff = _angle_diff_deg
    else:
        ra = rng.uniform(0.0, 2.0 * np.pi, n).astype(dtype)
        dec = rng.uniform(-0.4 * np.pi, 0.4 * np.pi, n).astype(dtype)
        ang_diff = _angle_diff_rad
    redshift = rng.uniform(0.02, 1.0, n).astype(dtype)

    xyz, dist = radec_z_to_xyz(
        ra=ra, dec=dec, redshift=redshift,
        ra_dec_unit=ra_dec_unit, distance_unit=distance_unit,
    )
    ra2, dec2, redshift2, dist2 = xyz_to_radec_z(
        xyz=xyz, ra_dec_unit=ra_dec_unit, distance_unit=distance_unit,
        z_atol=1e-8, z_max=5.0,
    )

    # dtype is preserved through the whole round trip.
    assert xyz.dtype == dtype and dist.dtype == dtype
    assert ra2.dtype == dtype and dec2.dtype == dtype
    assert redshift2.dtype == dtype and dist2.dtype == dtype

    assert xyz.shape == (n, 3)
    assert dist.shape == (n,)
    assert np.isfinite(xyz).all()
    assert np.isfinite(dist).all()

    tol = _ROUNDTRIP_TOL[np.dtype(dtype)]
    assert np.max(ang_diff(ra, ra2)) < tol["ang"]
    assert np.max(np.abs(dec - dec2)) < tol["ang"]
    assert np.max(np.abs(redshift - redshift2)) < tol["z"]
    assert np.allclose(dist, dist2, rtol=0.0, atol=tol["dist"])


def test_radec_z_to_xyz_negative_redshift_raises() -> None:
    """Test that negative redshift raises ValueError."""
    ra = np.array([10.0, 20.0])
    dec = np.array([5.0, -3.0])
    redshift = np.array([0.1, -0.2])

    with pytest.raises(ValueError, match="redshift"):
        radec_z_to_xyz(ra, dec, redshift)


def test_radec_z_to_xyz_length_mismatch_raises() -> None:
    """Test that mismatched array lengths raise ValueError."""
    ra = np.array([10.0, 20.0, 30.0])
    dec = np.array([5.0, -3.0])
    redshift = np.array([0.1, 0.2])

    with pytest.raises(ValueError, match="same length"):
        radec_z_to_xyz(ra, dec, redshift)


def test_radec_z_to_xyz_empty_raises() -> None:
    """Test that empty input raises ValueError."""
    ra = np.array([])
    dec = np.array([])
    redshift = np.array([])

    with pytest.raises(ValueError, match="must not be empty"):
        radec_z_to_xyz(ra, dec, redshift)


def test_radec_z_to_xyz_invalid_ra_dec_unit() -> None:
    """Test that invalid ra_dec_unit raises ValueError."""
    ra = np.array([10.0])
    dec = np.array([5.0])
    redshift = np.array([0.1])

    with pytest.raises(ValueError, match="ra_dec_unit"):
        radec_z_to_xyz(ra, dec, redshift, ra_dec_unit="gon")


def test_radec_z_to_xyz_invalid_distance_unit() -> None:
    """Test that invalid distance_unit raises ValueError."""
    ra = np.array([10.0])
    dec = np.array([5.0])
    redshift = np.array([0.1])

    with pytest.raises(ValueError, match="distance_unit"):
        radec_z_to_xyz(ra, dec, redshift, distance_unit="parsec")


def test_xyz_to_radec_z_shape_validation() -> None:
    """Test that invalid xyz shape raises ValueError."""
    xyz_bad = np.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="shape"):
        xyz_to_radec_z(xyz_bad)


def test_xyz_to_radec_z_empty_validation() -> None:
    """Test that empty xyz raises ValueError."""
    xyz_empty = np.zeros((0, 3))

    with pytest.raises(ValueError, match="must not be empty"):
        xyz_to_radec_z(xyz_empty)


def test_xyz_to_radec_z_invalid_ra_dec_unit() -> None:
    """Test that invalid ra_dec_unit raises ValueError."""
    xyz = np.array([[10.0, 20.0, 30.0]])

    with pytest.raises(ValueError, match="ra_dec_unit"):
        xyz_to_radec_z(xyz, ra_dec_unit="minutes")


def test_xyz_to_radec_z_invalid_distance_unit() -> None:
    """Test that invalid distance_unit raises ValueError."""
    xyz = np.array([[10.0, 20.0, 30.0]])

    with pytest.raises(ValueError, match="distance_unit"):
        xyz_to_radec_z(xyz, distance_unit="lightyear")


@pytest.mark.parametrize("dtype", DTYPES)
def test_radec_z_to_xyz_preserves_dtype(dtype) -> None:
    """radec_z_to_xyz returns arrays with the same dtype as its inputs."""
    ra = np.array([10.0, 20.0], dtype=dtype)
    dec = np.array([5.0, -3.0], dtype=dtype)
    redshift = np.array([0.1, 0.2], dtype=dtype)

    xyz, dist = radec_z_to_xyz(ra, dec, redshift)

    assert xyz.dtype == dtype
    assert dist.dtype == dtype


@pytest.mark.parametrize("dtype", DTYPES)
def test_xyz_to_radec_z_preserves_dtype(dtype) -> None:
    """xyz_to_radec_z returns arrays with the same dtype as its input."""
    xyz = np.array([[10.0, 20.0, 30.0], [15.0, 25.0, 35.0]], dtype=dtype)

    ra, dec, redshift, dist = xyz_to_radec_z(xyz)

    assert ra.dtype == dtype
    assert dec.dtype == dtype
    assert redshift.dtype == dtype
    assert dist.dtype == dtype


def test_create_cosmology_default() -> None:
    """Test creating cosmology with default (Planck 2018) parameters."""
    cosmo = create_cosmology()

    assert cosmo is not None
    assert cosmo.H0.value == 67.11
    assert cosmo.Om0 == 0.3175
    assert cosmo.Ob0 == 0.049
    assert cosmo.Tcmb0.value == 2.7255


def test_create_cosmology_custom() -> None:
    """Test creating cosmology with custom parameters."""
    H0_test = 70.0
    Om0_test = 0.3
    Ob0_test = 0.05

    cosmo = create_cosmology(H0=H0_test, Om0=Om0_test, Ob0=Ob0_test)

    assert cosmo.H0.value == H0_test
    assert cosmo.Om0 == Om0_test
    assert cosmo.Ob0 == Ob0_test


def test_create_cosmology_with_name() -> None:
    """Test creating cosmology with custom name."""
    name_test = "MyCustomCosmo"
    cosmo = create_cosmology(name=name_test)

    assert cosmo.name == name_test


def test_create_cosmology_negative_H0_raises() -> None:
    """Test that negative H0 raises ValueError."""
    with pytest.raises(ValueError, match="H0 must be positive"):
        create_cosmology(H0=-70.0)


def test_create_cosmology_zero_H0_raises() -> None:
    """Test that zero H0 raises ValueError."""
    with pytest.raises(ValueError, match="H0 must be positive"):
        create_cosmology(H0=0.0)


def test_create_cosmology_invalid_Om0_too_small() -> None:
    """Test that Om0 <= 0 raises ValueError."""
    with pytest.raises(ValueError, match="Om0 must be in"):
        create_cosmology(Om0=0.0)


def test_create_cosmology_invalid_Om0_too_large() -> None:
    """Test that Om0 >= 1 raises ValueError."""
    with pytest.raises(ValueError, match="Om0 must be in"):
        create_cosmology(Om0=1.0)


@pytest.mark.parametrize("dtype", DTYPES)
def test_radec_xyz_roundtrip_with_custom_cosmology(dtype) -> None:
    """Round-trip conversion using a custom cosmology, for both precisions."""
    cosmo = create_cosmology(H0=70.0, Om0=0.3)

    rng = np.random.RandomState(123)
    n = 20
    ra = rng.uniform(0.0, 360.0, n).astype(dtype)
    dec = rng.uniform(-80.0, 80.0, n).astype(dtype)
    redshift = rng.uniform(0.01, 1.0, n).astype(dtype)

    xyz, dist = radec_z_to_xyz(
        ra=ra, dec=dec, redshift=redshift, cosmo=cosmo,
        ra_dec_unit="deg", distance_unit="Mpc",
    )
    ra2, dec2, redshift2, dist2 = xyz_to_radec_z(
        xyz=xyz, cosmo=cosmo, ra_dec_unit="deg", distance_unit="Mpc",
        z_atol=1e-8, z_max=5.0,
    )

    assert xyz.dtype == dtype and dist.dtype == dtype
    assert ra2.dtype == dtype and dec2.dtype == dtype
    assert redshift2.dtype == dtype and dist2.dtype == dtype

    tol = _CUSTOM_COSMO_TOL[np.dtype(dtype)]
    assert np.max(_angle_diff_deg(ra, ra2)) < tol["ang"]
    assert np.max(np.abs(dec - dec2)) < tol["ang"]
    assert np.max(np.abs(redshift - redshift2)) < tol["z"]
    assert np.allclose(dist, dist2, rtol=0.0, atol=tol["dist"])