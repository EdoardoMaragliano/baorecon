"""Tests for the cone-aligned reference frame (:mod:`baorecon.utils.frames`)."""

import numpy as np
import pytest

from baorecon.utils.frames import ConeFrame


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def unit_vectors(ra_deg, dec_deg):
    """(RA, DEC) in degrees -> unit vectors of shape (N, 3)."""
    ra, dec = np.radians(ra_deg), np.radians(dec_deg)
    return np.stack(
        [np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)], axis=-1
    )


def to_angles(positions):
    """Cartesian positions -> (RA, DEC) in degrees, RA in [0, 360)."""
    x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
    ra = np.degrees(np.arctan2(y, x)) % 360.0
    dec = np.degrees(np.arcsin(np.clip(z / np.linalg.norm(positions, axis=1), -1.0, 1.0)))
    return ra, dec


def angle_between(a, b):
    """Angle in degrees between two vectors."""
    a = np.asarray(a, dtype=np.float64) / np.linalg.norm(a)
    b = np.asarray(b, dtype=np.float64) / np.linalg.norm(b)
    return float(np.degrees(np.arccos(np.clip(a @ b, -1.0, 1.0))))


def ra_diff(a, b):
    """Minimal RA difference in degrees, robust to the 0/360 wrap."""
    return np.abs((a - b + 180.0) % 360.0 - 180.0)


def patch(ra_c=170.0, dec_c=10.0, size=40.0, n=20_000, seed=0, dtype=np.float64):
    """A square-ish footprint plus radial distances: (ra, dec, positions)."""
    rng = np.random.default_rng(seed)
    ra = rng.uniform(ra_c - size / 2, ra_c + size / 2, n) % 360.0
    dec = rng.uniform(dec_c - size / 2, dec_c + size / 2, n)
    r = rng.uniform(1000.0, 3000.0, n)          # comoving distances, Mpc/h
    pos = (unit_vectors(ra, dec) * r[:, None]).astype(dtype)
    return ra, dec, pos


DIRECTIONS = [
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),          # already aligned
    (0.0, 0.0, -1.0),         # antipodal
    (0.3, -0.5, 0.81),
    (-0.7, 0.1, -0.2),
]


# ---------------------------------------------------------------------------
# the matrix is a rotation, and it does what it claims
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("direction", DIRECTIONS)
def test_matrix_is_a_proper_rotation(direction):
    frame = ConeFrame.from_direction(np.array(direction), dtype=np.float64)
    R = frame.matrix
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-14)
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-14)


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_maps_its_axis_onto_z(direction):
    frame = ConeFrame.from_direction(np.array(direction), dtype=np.float64)
    np.testing.assert_allclose(
        frame.matrix @ frame.mean_direction, [0.0, 0.0, 1.0], atol=1e-14
    )


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_inverse_is_the_transpose(direction):
    frame = ConeFrame.from_direction(np.array(direction), dtype=np.float64)
    np.testing.assert_array_equal(frame.inverse, frame.matrix.T)


def test_direction_need_not_be_normalised():
    a = ConeFrame.from_direction(np.array([0.3, -0.5, 0.81]), dtype=np.float64)
    b = ConeFrame.from_direction(np.array([3.0, -5.0, 8.1]), dtype=np.float64)
    np.testing.assert_allclose(a.matrix, b.matrix, atol=1e-15)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_roundtrip_returns_the_original(dtype):
    _, _, pos = patch(dtype=dtype)
    frame = ConeFrame.from_positions(pos)
    back = frame.unrotate(frame.rotate(pos))
    tol = 2e-2 if dtype == np.float32 else 1e-9   # Mpc/h on ~3000 Mpc/h positions
    np.testing.assert_allclose(back, pos, atol=tol)


def test_rotation_centres_the_cone_on_z():
    """The whole point: the cone ends up around z, within its own opening angle."""
    ra, dec, pos = patch(ra_c=60.0, dec_c=-40.0, size=40.0)
    frame = ConeFrame.from_positions(pos)
    rotated = frame.rotate(pos)

    z = np.array([0.0, 0.0, 1.0])
    before = max(angle_between(p, z) for p in pos[:2000])
    after = max(angle_between(p, z) for p in rotated[:2000])

    assert before > 100.0            # off-axis cone, far from z
    assert after < 30.0              # centred: only the cone's own radius is left
    assert after == pytest.approx(frame.angular_radius(pos), abs=1.0)


# ---------------------------------------------------------------------------
# the two constructors are the same axis by two routes
# ---------------------------------------------------------------------------

def test_from_angles_and_from_positions_agree():
    """The comoving distance cancels on normalisation, so the axis is identical."""
    ra, dec, pos = patch()
    assert angle_between(
        ConeFrame.from_angles(ra, dec, dtype=np.float64).mean_direction,
        ConeFrame.from_positions(pos, dtype=np.float64).mean_direction,
    ) < 1e-10


def test_axis_does_not_depend_on_the_distances():
    """Same directions, wildly different radii -> same axis."""
    ra, dec, _ = patch()
    rng = np.random.default_rng(7)
    near = unit_vectors(ra, dec) * rng.uniform(10.0, 20.0, ra.size)[:, None]
    far = unit_vectors(ra, dec) * rng.uniform(5000.0, 9000.0, ra.size)[:, None]
    assert angle_between(
        ConeFrame.from_positions(near).mean_direction,
        ConeFrame.from_positions(far).mean_direction,
    ) < 1e-10


def test_raw_position_mean_would_be_distance_biased():
    """Guards the normalise-then-sum rule by showing the naive alternative differs.

    Averaging positions instead of directions weights every object by its
    comoving distance. With a lopsided n(z) that tilts the axis measurably, so
    the two must NOT agree -- if they ever do, this test is no longer proving
    anything and the kernel should be re-checked.
    """
    rng = np.random.default_rng(3)
    n = 50_000
    ra = rng.uniform(150.0, 190.0, n)
    dec = rng.uniform(-20.0, 20.0, n)
    # Distance correlated with position on the sky: the far half sits to one side.
    r = np.where(ra > 170.0, rng.uniform(4000.0, 5000.0, n), rng.uniform(500.0, 900.0, n))
    pos = unit_vectors(ra, dec) * r[:, None]

    proper = ConeFrame.from_positions(pos).mean_direction
    naive = pos.mean(axis=0)
    assert angle_between(proper, naive) > 1.0


# ---------------------------------------------------------------------------
# float64 accumulation (regression guard)
# ---------------------------------------------------------------------------

def test_direction_sum_accumulates_in_float64():
    """float32 input must not degrade the axis.

    Summing float32 unit vectors along axis 0 of an (N, 3) array does not get
    numpy's pairwise summation, and the axis drifts by a large angle at survey
    scale. The kernels promote to float64 per element; this pins that down by
    comparing against an exact float64 reference. float32 accumulation would
    miss by several orders of magnitude more than the tolerance.
    """
    rng = np.random.default_rng(11)
    n = 2_000_000
    ra = rng.uniform(150.0, 190.0, n)
    dec = rng.uniform(-20.0, 20.0, n)
    pos32 = (unit_vectors(ra, dec) * rng.uniform(1000.0, 3000.0, n)[:, None]).astype(np.float32)

    # Reference: float64 throughout, accumulated in blocks over the contiguous axis.
    acc = np.zeros(3, dtype=np.float64)
    for start in range(0, n, 250_000):
        block = pos32[start:start + 250_000].astype(np.float64)
        block /= np.linalg.norm(block, axis=1, keepdims=True)
        acc += block.sum(axis=0)

    # dtype=float64 stores the axis at full precision, so the assertion measures
    # the accumulator and not the storage: float32 storage alone costs ~1e-5 deg.
    frame = ConeFrame.from_positions(pos32, dtype=np.float64)
    assert angle_between(frame.mean_direction, acc) < 1e-8


def test_axis_survives_the_ra_zero_wrap():
    """A footprint straddling RA=0 must not throw the axis to the far side.

    The vector mean has no wrap to get wrong; a per-coordinate midpoint of RA
    would land ~180 degrees away, which is what motivates the vector mean.
    """
    rng = np.random.default_rng(5)
    ra = np.concatenate([rng.uniform(340.0, 360.0, 50_000), rng.uniform(0.0, 20.0, 50_000)])
    dec = rng.uniform(-20.0, 20.0, 100_000)
    frame = ConeFrame.from_angles(ra, dec)

    assert angle_between(frame.mean_direction, unit_vectors(0.0, 0.0)) < 1.0
    pos = unit_vectors(ra, dec) * 2000.0
    assert frame.angular_radius(pos) < 35.0

    midpoint_axis = unit_vectors(0.5 * (ra.min() + ra.max()), 0.5 * (dec.min() + dec.max()))
    assert angle_between(frame.mean_direction, midpoint_axis) > 170.0


# ---------------------------------------------------------------------------
# positions and displacements transform alike
# ---------------------------------------------------------------------------

def test_displacements_transform_like_positions():
    """A pure rotation is linear, so Psi can be rotated directly or by difference."""
    _, _, pos = patch()
    rng = np.random.default_rng(2)
    shifted = pos + rng.normal(0.0, 10.0, pos.shape)
    frame = ConeFrame.from_positions(pos)

    psi_then_rotate = frame.rotate(shifted - pos)
    rotate_then_diff = frame.rotate(shifted) - frame.rotate(pos)
    np.testing.assert_allclose(psi_then_rotate, rotate_then_diff, atol=1e-9)


def test_rotation_preserves_norms_and_angles():
    _, _, pos = patch()
    frame = ConeFrame.from_positions(pos)
    rotated = frame.rotate(pos)
    np.testing.assert_allclose(
        np.linalg.norm(rotated, axis=1), np.linalg.norm(pos, axis=1), rtol=1e-12
    )
    np.testing.assert_allclose(
        (rotated[:-1] * rotated[1:]).sum(axis=1), (pos[:-1] * pos[1:]).sum(axis=1), rtol=1e-10
    )


# ---------------------------------------------------------------------------
# in-place / out= / dtype
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_in_place_matches_out_of_place(dtype):
    _, _, pos = patch(dtype=dtype)
    frame = ConeFrame.from_positions(pos)

    fresh = frame.rotate(pos)
    work = pos.copy()
    returned = frame.rotate(work, out=work)

    assert returned is work
    np.testing.assert_array_equal(work, fresh)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_output_dtype_follows_the_input(dtype):
    _, _, pos = patch(dtype=dtype)
    assert ConeFrame.from_positions(pos).rotate(pos).dtype == dtype


def test_explicit_out_buffer():
    _, _, pos = patch()
    frame = ConeFrame.from_positions(pos)
    buf = np.empty_like(pos)
    assert frame.rotate(pos, out=buf) is buf
    np.testing.assert_array_equal(buf, frame.rotate(pos))


def test_out_buffer_must_match():
    _, _, pos = patch()
    frame = ConeFrame.from_positions(pos)
    with pytest.raises(ValueError, match="shape and dtype"):
        frame.rotate(pos, out=np.empty((pos.shape[0], 3), dtype=np.float32))


# ---------------------------------------------------------------------------
# angular radius
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "size, expected",
    [(5.0, 3.53), (10.0, 7.05), (20.0, 14.10), (40.0, 27.95)],
)
def test_angular_radius_tracks_the_footprint_size(size, expected):
    """It measures how big the survey is on the sky, nothing else."""
    _, _, pos = patch(dec_c=0.0, size=size, n=200_000, seed=1)
    radius = ConeFrame.from_positions(pos).angular_radius(pos)
    assert radius == pytest.approx(expected, abs=0.15)


def test_angular_radius_in_radians():
    _, _, pos = patch()
    frame = ConeFrame.from_positions(pos)
    assert frame.angular_radius(pos, degrees=False) == pytest.approx(
        np.radians(frame.angular_radius(pos))
    )


# ---------------------------------------------------------------------------
# serialisation
# ---------------------------------------------------------------------------

def test_dict_roundtrip():
    _, _, pos = patch()
    frame = ConeFrame.from_positions(pos)
    restored = ConeFrame.from_dict(frame.to_dict())
    np.testing.assert_allclose(restored.matrix, frame.matrix, atol=1e-15)
    np.testing.assert_allclose(restored.mean_direction, frame.mean_direction, atol=1e-15)


def test_to_dict_is_plain_python():
    import json
    _, _, pos = patch()
    payload = ConeFrame.from_positions(pos).to_dict()
    assert json.loads(json.dumps(payload)) == payload


# ---------------------------------------------------------------------------
# input validation
# ---------------------------------------------------------------------------

def test_rejects_empty_and_malformed_input():
    with pytest.raises(ValueError, match="empty"):
        ConeFrame.from_positions(np.empty((0, 3)))
    with pytest.raises(ValueError, match="empty"):
        ConeFrame.from_angles(np.array([]), np.array([]))
    with pytest.raises(ValueError, match=r"\(N, 3\)"):
        ConeFrame.from_positions(np.zeros((10, 2)))
    with pytest.raises(ValueError, match="match in shape"):
        ConeFrame.from_angles(np.zeros(10), np.zeros(11))
    with pytest.raises(ValueError, match="non-zero"):
        ConeFrame.from_direction(np.zeros(3))


def test_rejects_a_vanishing_direction_sum():
    """Antipodal pairs have no mean line of sight; better to say so than to guess."""
    pos = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="no mean line of sight"):
        ConeFrame.from_positions(pos)


def test_rejects_bad_shapes_at_construction():
    with pytest.raises(ValueError, match=r"shape \(3, 3\)"):
        ConeFrame(matrix=np.eye(2), mean_direction=np.array([0.0, 0.0, 1.0]))
    with pytest.raises(ValueError, match=r"shape \(3,\)"):
        ConeFrame(matrix=np.eye(3), mean_direction=np.zeros(2))


# ---------------------------------------------------------------------------
# parity with the ZAtools implementation this replaces
# ---------------------------------------------------------------------------

def test_matches_zatools_rotate_cone():
    """Same matrix as ZAtools.Rotate_cone, elementwise.

    The azimuthal orientation around the axis is free -- any rotation about z
    composed with a valid one is also valid -- so matching elementwise (rather
    than up to a spin about z) is only possible because both use the same
    theta/phi construction. That makes the migration provably a no-op.
    """
    zatools = pytest.importorskip("ZAtools.Irregular_geometry")
    ra, dec, _ = patch()
    expected, expected_inv = zatools.Rotate_cone(ra, dec)
    frame = ConeFrame.from_angles(ra, dec, dtype=np.float64)
    np.testing.assert_allclose(frame.matrix, expected, atol=1e-12)
    np.testing.assert_allclose(frame.inverse, expected_inv, atol=1e-12)


def test_matches_zatools_apply_rot_end_to_end():
    """Rotating positions lands on the same sky directions as rotating angles."""
    zatools = pytest.importorskip("ZAtools.Irregular_geometry")
    ra, dec, pos = patch()
    R, _ = zatools.Rotate_cone(ra, dec)
    ra_ref, dec_ref = zatools.Apply_rot(R, ra, dec, np.linalg.norm(pos, axis=1))

    ra_ours, dec_ours = to_angles(ConeFrame.from_angles(ra, dec, dtype=np.float64).rotate(pos))
    assert ra_diff(ra_ours, np.asarray(ra_ref) % 360.0).max() < 1e-9
    assert np.abs(dec_ours - dec_ref).max() < 1e-9


# ---------------------------------------------------------------------------
# working precision: settable, and never paid for by the catalogue
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_working_precision_is_settable(dtype):
    frame = ConeFrame.from_angles(*patch()[:2], dtype=dtype)
    assert frame.dtype == dtype
    assert frame.matrix.dtype == dtype
    assert frame.mean_direction.dtype == dtype


def test_from_angles_defaults_to_single_precision():
    """Matches the package default (`format_positions`, `Mesh`, `BAOReconstructor`)."""
    assert ConeFrame.from_angles(*patch()[:2]).dtype == np.float32


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_from_positions_follows_the_catalogue(dtype):
    """A frame built from a catalogue is ready to rotate it with no cast."""
    _, _, pos = patch(dtype=dtype)
    frame = ConeFrame.from_positions(pos)
    assert frame.dtype == dtype
    assert frame.rotate(pos).dtype == dtype


def test_rotate_refuses_a_mismatched_dtype():
    """Better a loud error than a silent full-size cast, or a silent precision change."""
    _, _, pos32 = patch(dtype=np.float32)
    frame = ConeFrame.from_positions(pos32)
    with pytest.raises(ValueError, match="does not match the frame's working precision"):
        frame.rotate(pos32.astype(np.float64))


def test_astype_moves_the_frame_not_the_catalogue():
    _, _, pos64 = patch(dtype=np.float64)
    frame32 = ConeFrame.from_angles(*patch()[:2], dtype=np.float32)
    moved = frame32.astype(np.float64)

    assert moved.dtype == np.float64
    assert frame32.dtype == np.float32          # frozen: the original is untouched
    assert moved.rotate(pos64).dtype == np.float64


def test_matrix_is_built_in_float64_then_cast_down():
    """Single precision is where the matrix is *stored*, not where it is computed."""
    direction = np.array([0.3, -0.5, 0.81])
    exact = ConeFrame.from_direction(direction, dtype=np.float64)
    single = ConeFrame.from_direction(direction, dtype=np.float32)
    np.testing.assert_allclose(single.matrix, exact.matrix.astype(np.float32), rtol=0, atol=0)


def test_rejects_non_floating_input():
    frame = ConeFrame.from_angles(*patch()[:2], dtype=np.float64)
    with pytest.raises(ValueError, match="floating-point"):
        ConeFrame.from_positions(np.ones((10, 3), dtype=np.int64))
    with pytest.raises(ValueError, match="floating-point"):
        frame.angular_radius(np.ones((10, 3), dtype=np.int64))
    with pytest.raises(ValueError, match="floating-point"):
        ConeFrame(matrix=np.eye(3, dtype=np.int64), mean_direction=np.zeros(3))


def test_dtype_survives_the_dict_roundtrip():
    """A float32 frame round-trips without the matrix being silently promoted."""
    frame = ConeFrame.from_angles(*patch()[:2], dtype=np.float32)
    restored = ConeFrame.from_dict(frame.to_dict())
    assert restored.dtype == np.float32
    np.testing.assert_array_equal(restored.matrix, frame.matrix)


def test_reads_strided_arrays_without_copying():
    """A view must not be quietly materialised on the way into the kernels."""
    _, _, pos = patch(n=200_000, dtype=np.float32)
    view = pos[::2]
    assert not view.flags["C_CONTIGUOUS"]

    frame = ConeFrame.from_positions(view)
    contiguous = ConeFrame.from_positions(np.ascontiguousarray(view))
    assert angle_between(frame.mean_direction, contiguous.mean_direction) < 1e-4
    np.testing.assert_allclose(
        frame.rotate(view), frame.rotate(np.ascontiguousarray(view)), rtol=0, atol=0
    )


def test_in_place_rotation_allocates_nothing():
    """The reason out=a exists: a 50M-row random must not need a second buffer."""
    import tracemalloc

    _, _, pos = patch(n=400_000, dtype=np.float32)
    frame = ConeFrame.from_positions(pos)
    frame.rotate(pos, out=pos)                     # warm the JIT before measuring

    tracemalloc.start()
    frame.rotate(pos, out=pos)
    _, peak_inplace = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    tracemalloc.start()
    frame.rotate(pos)
    _, peak_fresh = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert peak_inplace < 64 * 1024                # a few objects, not an array
    assert peak_fresh > pos.nbytes / 2             # the allocating path really does
