"""Cone-aligned reference frame for survey-shaped catalogues.

A lightcone points wherever the survey happens to look, so its objects sit far
off the coordinate axes: for a patch centred at, say, ``DEC = -40`` the farthest
tracer is ~150 degrees away from the ``z`` axis. Rotating the catalogue so that
the survey's mean line of sight lies *along* ``z`` leaves only the cone's own
angular radius (the survey's size on the sky, typically tens of degrees), which
no choice of axis can reduce.

The rotation buys a **compact bounding box**: a cone centred on ``z`` has a much
smaller Cartesian extent than the same cone sitting off-axis, so the same mesh
resolves it with fewer cells, or the same ``nmesh`` resolves it more finely. It
does *not* change the physics of a local (radial) line of sight, which is
equivariant under rotation: rotating positions rotates each tracer's line of
sight with them.

The frame is a **pure rotation about the origin**, which is what makes it safe to
apply to positions and to displacement fields alike -- both transform as ``v ->
R v``. Contrast :func:`~baorecon.utils.formatters.survey_to_box_frame`, which
also *translates*: applying that one to a displacement would be wrong.

Two constructors give the same axis by two routes, and the equality is exact
rather than approximate, because ``(x, y, z) / |(x, y, z)| = n(ra, dec)`` -- the
comoving distance cancels, so the axis does not depend on the cosmology:

* :meth:`ConeFrame.from_angles` -- from RA/DEC, for the ``ra_dec_z`` input path.
* :meth:`ConeFrame.from_positions` -- from Cartesian positions, the only route
  available for catalogues that are already Cartesian.

Precision
---------
The frame carries a working ``dtype`` like the rest of the package, defaulting to
``float32``, and **never casts the catalogue**: :meth:`ConeFrame.rotate` refuses a
mismatched input rather than silently upcasting a multi-hundred-megabyte array,
and adapts the 3x3 matrix instead. Nothing here copies the input either -- the
kernels read strided arrays directly, so passing a view does not quietly
materialise it.

Two precisions are deliberately distinct, and conflating them is a bug in each
direction:

* *Storage* precision -- the matrix, and the arrays it is applied to. Settable,
  and matched to the pipeline's working dtype.
* *Accumulation* precision -- the direction sum, always float64. This is three
  scalars, so it costs no memory, and it is not optional: summing 50 million
  float32 unit vectors along axis 0 of an ``(N, 3)`` array skips numpy's pairwise
  summation and put the recovered axis 23 degrees off in testing, with nothing
  failing loudly -- the rotation still "works", it is simply pointed the wrong
  way. The float64 accumulator feeds a float64 matrix construction, which is only
  then cast down to the working dtype.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
from numba import njit, prange

from baorecon.utils.loggers import setup_logger

logger = setup_logger(__name__)

__all__ = ["ConeFrame"]


# ============================================================================
# Kernels
# ============================================================================
# The direction kernels accumulate in float64 whatever the input dtype: see the
# module docstring for why this is load-bearing rather than defensive.

@njit(parallel=True, fastmath=True, cache=True)
def _direction_sum_from_positions(positions):
    """Sum of the per-object unit vectors of ``positions``, in float64."""
    sx = 0.0
    sy = 0.0
    sz = 0.0
    for i in prange(positions.shape[0]):
        x = np.float64(positions[i, 0])
        y = np.float64(positions[i, 1])
        z = np.float64(positions[i, 2])
        r = np.sqrt(x * x + y * y + z * z)
        if r > 0.0:
            sx += x / r
            sy += y / r
            sz += z / r
    return sx, sy, sz


@njit(parallel=True, fastmath=True, cache=True)
def _direction_sum_from_angles(ra_deg, dec_deg):
    """Sum of the unit vectors of the (RA, DEC) directions, in float64."""
    sx = 0.0
    sy = 0.0
    sz = 0.0
    d2r = np.pi / 180.0
    for i in prange(ra_deg.shape[0]):
        ra = np.float64(ra_deg[i]) * d2r
        dec = np.float64(dec_deg[i]) * d2r
        cos_dec = np.cos(dec)
        sx += cos_dec * np.cos(ra)
        sy += cos_dec * np.sin(ra)
        sz += np.sin(dec)
    return sx, sy, sz


@njit(parallel=True, fastmath=True, cache=True)
def _rotate_inplace(a, R):
    """Overwrite ``a`` with ``a @ R.T``; the three components are read first."""
    r00, r01, r02 = R[0, 0], R[0, 1], R[0, 2]
    r10, r11, r12 = R[1, 0], R[1, 1], R[1, 2]
    r20, r21, r22 = R[2, 0], R[2, 1], R[2, 2]
    for i in prange(a.shape[0]):
        x = a[i, 0]
        y = a[i, 1]
        z = a[i, 2]
        a[i, 0] = r00 * x + r01 * y + r02 * z
        a[i, 1] = r10 * x + r11 * y + r12 * z
        a[i, 2] = r20 * x + r21 * y + r22 * z


@njit(parallel=True, fastmath=True, cache=True)
def _rotate_into(a, R, out):
    """Write ``a @ R.T`` into ``out``."""
    r00, r01, r02 = R[0, 0], R[0, 1], R[0, 2]
    r10, r11, r12 = R[1, 0], R[1, 1], R[1, 2]
    r20, r21, r22 = R[2, 0], R[2, 1], R[2, 2]
    for i in prange(a.shape[0]):
        x = a[i, 0]
        y = a[i, 1]
        z = a[i, 2]
        out[i, 0] = r00 * x + r01 * y + r02 * z
        out[i, 1] = r10 * x + r11 * y + r12 * z
        out[i, 2] = r20 * x + r21 * y + r22 * z


@njit(parallel=True, fastmath=True, cache=True)
def _min_cos_deviation(positions, ax, ay, az):
    """Smallest ``cos`` between ``positions`` and the axis (i.e. widest angle)."""
    smallest = 1.0
    for i in prange(positions.shape[0]):
        x = np.float64(positions[i, 0])
        y = np.float64(positions[i, 1])
        z = np.float64(positions[i, 2])
        r = np.sqrt(x * x + y * y + z * z)
        if r > 0.0:
            smallest = min(smallest, (x * ax + y * ay + z * az) / r)
    return smallest


# ============================================================================
# Frame
# ============================================================================

def _rotation_onto_z(direction: np.ndarray) -> np.ndarray:
    """Rotation mapping the unit vector ``direction`` onto ``(0, 0, 1)``.

    Built as ``R_y(theta) @ R_z(phi)`` with the polar/azimuthal angles read
    straight off the unit vector, skipping an ``arccos``/``arctan2`` round trip.
    This fixes the azimuthal convention, which is otherwise free: any rotation
    about ``z`` composed with this one also maps ``direction`` to ``z``. The
    convention matches ZAtools' ``align_vectors`` so the two agree elementwise.

    The degenerate cases need no special handling: when ``direction`` is
    ``(0, 0, +1)`` the formula collapses to the identity, and when it is
    ``(0, 0, -1)`` to a 180-degree rotation about ``y`` -- a proper rotation in
    both cases.
    """
    nx, ny, nz = direction
    sin_theta = float(np.hypot(nx, ny))
    cos_theta = float(nz)
    if sin_theta > 0.0:
        cos_phi = float(nx) / sin_theta
        sin_phi = float(ny) / sin_theta
    else:
        # (Anti)parallel to z: the azimuth is arbitrary, pick phi = 0.
        cos_phi, sin_phi = 1.0, 0.0

    return np.array(
        [
            [cos_theta * cos_phi, cos_theta * sin_phi, -sin_theta],
            [-sin_phi, cos_phi, 0.0],
            [sin_theta * cos_phi, sin_theta * sin_phi, cos_theta],
        ],
        dtype=np.float64,
    )


@dataclass(frozen=True)
class ConeFrame:
    """A rotation aligning a survey cone's mean line of sight with the ``z`` axis.

    Attributes
    ----------
    matrix : np.ndarray
        ``(3, 3)`` rotation taking survey-frame coordinates to cone-frame ones.
        Its dtype is the frame's working precision.
    mean_direction : np.ndarray
        ``(3,)`` unit vector of the survey's mean line of sight, in the *survey*
        frame. In the cone frame it is ``(0, 0, 1)`` by construction, so only
        this one is stored.

    Notes
    -----
    Instances are immutable and cheap to pass around, which is what lets one
    frame be shared across redshift bins so every bin is reconstructed in the
    same geometry rather than each picking its own axis.
    """

    matrix: np.ndarray
    mean_direction: np.ndarray

    def __post_init__(self) -> None:
        if np.shape(self.matrix) != (3, 3):
            raise ValueError(f"matrix must have shape (3, 3), got {np.shape(self.matrix)}")
        if np.shape(self.mean_direction) != (3,):
            raise ValueError(
                f"mean_direction must have shape (3,), got {np.shape(self.mean_direction)}"
            )
        if np.asarray(self.matrix).dtype.kind != "f":
            raise ValueError(
                f"matrix must be a floating-point array, got {np.asarray(self.matrix).dtype}"
            )

    @property
    def dtype(self) -> np.dtype:
        """The frame's working precision: the dtype it rotates arrays at."""
        return np.asarray(self.matrix).dtype

    def astype(self, dtype) -> "ConeFrame":
        """The same frame at another working precision.

        Cheap -- it re-casts nine numbers -- and the way to line a frame up with
        a catalogue whose precision differs, since :meth:`rotate` will not cast
        the catalogue itself.
        """
        return ConeFrame(
            matrix=np.asarray(self.matrix).astype(dtype),
            mean_direction=np.asarray(self.mean_direction).astype(dtype),
        )

    # -- constructors --------------------------------------------------------
    @classmethod
    def from_direction(cls, direction: np.ndarray, dtype=np.float32) -> "ConeFrame":
        """Build the frame from an explicit axis (need not be normalised).

        The escape hatch that keeps the choice of axis out of the code: pass the
        mean direction, the centre of the smallest enclosing cone, or an axis
        shared across bins, and the frame is built the same way regardless.

        The matrix is always *computed* in float64 and only then cast to
        ``dtype``: building it at single precision would spend the error budget
        of nine numbers on every position it is later applied to.
        """
        n = np.asarray(direction, dtype=np.float64).reshape(3)
        norm = np.linalg.norm(n)
        if not np.isfinite(norm) or norm == 0.0:
            raise ValueError("direction must be a finite, non-zero vector")
        n = n / norm
        working = np.dtype(dtype)
        if working.kind != "f":
            raise ValueError(f"dtype must be a floating-point type, got {working}")
        return cls(
            matrix=_rotation_onto_z(n).astype(working),
            mean_direction=n.astype(working),
        )

    @classmethod
    def from_angles(cls, ra_deg: np.ndarray, dec_deg: np.ndarray, dtype=np.float32) -> "ConeFrame":
        """Build the frame from sky coordinates, in degrees.

        Uses the vector mean of the directions, which is weighted by the number
        of objects -- computed from the *randoms* it therefore follows the survey
        selection function. It is also the only formulation that survives the
        ``RA = 0`` wrap: a per-coordinate midpoint of a footprint straddling zero
        lands on the opposite side of the sky.
        """
        ra = np.asarray(ra_deg).ravel()
        dec = np.asarray(dec_deg).ravel()
        if ra.shape != dec.shape:
            raise ValueError(f"ra_deg and dec_deg must match in shape: {ra.shape} vs {dec.shape}")
        if ra.size == 0:
            raise ValueError("cannot build a cone frame from an empty catalogue")
        return cls._from_direction_sum(_direction_sum_from_angles(ra, dec), dtype)

    @classmethod
    def from_positions(cls, positions: np.ndarray, dtype=None) -> "ConeFrame":
        """Build the frame from Cartesian positions of shape ``(N, 3)``.

        ``dtype`` defaults to the precision of ``positions``, so a frame built
        from a catalogue is ready to rotate that catalogue with no cast.

        Each position is normalised *before* being summed. Averaging the raw
        positions instead would weight every object by its comoving distance and
        tilt the axis toward the far end of the lightcone; after normalisation
        the distance cancels exactly, so this agrees with :meth:`from_angles` to
        round-off. The array is read in place, strided or not: nothing is copied.
        """
        pos = np.asarray(positions)
        if pos.ndim != 2 or pos.shape[1] != 3:
            raise ValueError(f"positions must have shape (N, 3), got {pos.shape}")
        if pos.shape[0] == 0:
            raise ValueError("cannot build a cone frame from an empty catalogue")
        if pos.dtype.kind != "f":
            raise ValueError(f"positions must be a floating-point array, got {pos.dtype}")
        return cls._from_direction_sum(
            _direction_sum_from_positions(pos), pos.dtype if dtype is None else dtype
        )

    @classmethod
    def _from_direction_sum(cls, components, dtype) -> "ConeFrame":
        total = np.array(components, dtype=np.float64)
        norm = np.linalg.norm(total)
        if norm == 0.0:
            raise ValueError(
                "the direction sum vanishes: the catalogue has no mean line of "
                "sight (objects spread symmetrically over the whole sky)"
            )
        frame = cls.from_direction(total, dtype=dtype)
        logger.info(
            "Cone frame built at %s: mean direction (%.6f, %.6f, %.6f)",
            frame.dtype, *np.asarray(frame.mean_direction, dtype=np.float64),
        )
        return frame

    # -- geometry ------------------------------------------------------------
    @property
    def inverse(self) -> np.ndarray:
        """Cone frame back to survey frame. A rotation's inverse *is* its transpose.

        Derived rather than stored, so the two can never drift apart.
        """
        return np.asarray(self.matrix).T

    def rotate(self, a: np.ndarray, out: Optional[np.ndarray] = None) -> np.ndarray:
        """Survey frame -> cone frame.

        Applies to positions and to displacement fields alike: the frame is a
        pure rotation about the origin, so points and vectors transform the same
        way. Pass ``out=a`` to rotate in place, which for a 50-million-row random
        catalogue avoids a second multi-hundred-megabyte buffer.
        """
        return self._apply(a, np.asarray(self.matrix), out)

    def unrotate(self, a: np.ndarray, out: Optional[np.ndarray] = None) -> np.ndarray:
        """Cone frame -> survey frame. The exact inverse of :meth:`rotate`."""
        return self._apply(a, self.inverse, out)

    def _apply(self, a: np.ndarray, R: np.ndarray, out: Optional[np.ndarray]) -> np.ndarray:
        arr = np.asarray(a)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(f"expected an (N, 3) array, got {arr.shape}")
        if arr.dtype != self.dtype:
            raise ValueError(
                f"array dtype {arr.dtype} does not match the frame's working "
                f"precision {self.dtype}. Rotating it would either cast the "
                f"array -- allocating a full copy -- or silently change the "
                f"pipeline's precision. Use frame.astype({arr.dtype}) to move "
                f"the frame (nine numbers) instead."
            )

        if out is None:
            out = np.empty_like(arr)
        elif out.shape != arr.shape or out.dtype != arr.dtype:
            raise ValueError(
                f"out must match the input in shape and dtype: "
                f"{out.shape}/{out.dtype} vs {arr.shape}/{arr.dtype}"
            )

        # R already carries the working dtype, and the kernels read strided
        # arrays directly, so neither the input nor the matrix is copied here.
        if out is arr:
            _rotate_inplace(out, R)
        else:
            _rotate_into(arr, R, out)
        return out

    def angular_radius(self, positions: np.ndarray, degrees: bool = True) -> float:
        """Widest angle between the frame's axis and any of ``positions``.

        The cone's half-opening angle: a property of how large the survey is on
        the sky, not of the frame's quality -- no axis can shrink it. Reported
        because it is the scale at which a fixed-axis (plane-parallel) line of
        sight stops being a good approximation; with a local line of sight it is
        merely descriptive.

        Accepts any floating precision: the angle is accumulated in float64
        regardless, and the input is read in place.
        """
        pos = np.asarray(positions)
        if pos.ndim != 2 or pos.shape[1] != 3:
            raise ValueError(f"positions must have shape (N, 3), got {pos.shape}")
        if pos.shape[0] == 0:
            raise ValueError("cannot measure an angular radius of an empty catalogue")
        if pos.dtype.kind != "f":
            raise ValueError(f"positions must be a floating-point array, got {pos.dtype}")
        n = np.asarray(self.mean_direction, dtype=np.float64)
        cos_max = _min_cos_deviation(pos, n[0], n[1], n[2])
        angle = float(np.arccos(np.clip(cos_max, -1.0, 1.0)))
        return np.degrees(angle) if degrees else angle

    # -- serialisation -------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Plain-Python view, for the run metadata.

        Saved reconstruction grids live in the *cone* frame whatever happens, so
        without the matrix alongside them they cannot be put back on the sky. The
        values are written at full precision and the working dtype recorded
        separately, so a float32 frame round-trips without losing the matrix.
        """
        return {
            "matrix": np.asarray(self.matrix, dtype=float).tolist(),
            "mean_direction": np.asarray(self.mean_direction, dtype=float).tolist(),
            "dtype": np.dtype(self.dtype).name,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ConeFrame":
        """Rebuild a frame from :meth:`to_dict`."""
        working = np.dtype(payload.get("dtype", np.float64))
        return cls(
            matrix=np.asarray(payload["matrix"], dtype=working),
            mean_direction=np.asarray(payload["mean_direction"], dtype=working),
        )

    def __repr__(self) -> str:
        x, y, z = np.asarray(self.mean_direction, dtype=float)
        return f"ConeFrame(mean_direction=({x:.6f}, {y:.6f}, {z:.6f}), dtype={np.dtype(self.dtype).name})"
