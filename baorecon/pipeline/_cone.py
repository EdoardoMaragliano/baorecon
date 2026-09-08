"""Shared cone-alignment wiring for the two pipeline implementations.

:mod:`~baorecon.pipeline.bao_pipeline` and
:mod:`~baorecon.pipeline.bao_pipeline_interactive` are parallel implementations
that differ only in when they release intermediate arrays, so the alignment
steps live here instead of being written twice and drifting apart.

The rotation belongs in the coordinate layer rather than in
:class:`~baorecon.reconstruction.bao_reconstructor.BAOReconstructor`:
``convert_to_xyz``/``convert_back`` are already the symmetric pair that puts the
catalogue into the frame the solver works in and takes it back out, and keeping
it there leaves the reconstructor's ``boxsize`` and ``los`` unambiguous -- they
keep meaning what they meant, in the one frame the reconstructor ever sees.
"""

from typing import Optional

import numpy as np

from baorecon.utils.frames import ConeFrame
from baorecon.utils.loggers import setup_logger

logger = setup_logger(__name__)

__all__ = ["build_frame", "rotate_all", "unrotate_all"]


def _as_radec(direction) -> tuple:
    """A unit vector as ``(RA, DEC)`` in degrees."""
    x, y, z = np.asarray(direction, dtype=np.float64)
    ra = np.degrees(np.arctan2(y, x)) % 360.0
    dec = np.degrees(np.arcsin(np.clip(z, -1.0, 1.0)))
    return float(ra), float(dec)


def build_frame(
    dtype,
    ra: Optional[np.ndarray] = None,
    dec: Optional[np.ndarray] = None,
    positions: Optional[np.ndarray] = None,
) -> ConeFrame:
    """Build the cone frame from the *randoms*, by whichever route is available.

    Angles when the run has them (cheaper: no ``(N, 3)`` to walk), positions
    otherwise -- which is the only route for a catalogue that arrived already in
    Cartesian coordinates. The two give the same axis to round-off.

    Always the randoms, never the data: their angular density traces the survey
    selection function, and the geometry then does not move with the data's
    cosmic variance. Data and randoms must share one frame regardless.
    """
    if ra is not None and dec is not None:
        frame = ConeFrame.from_angles(ra, dec, dtype=dtype)
    elif positions is not None:
        frame = ConeFrame.from_positions(positions, dtype=dtype)
    else:
        raise ValueError("build_frame needs either (ra, dec) or positions")

    centre_ra, centre_dec = _as_radec(frame.mean_direction)
    logger.info(
        "Cone alignment ON: rotating the survey's mean line of sight onto the z axis "
        "(footprint centre RA = %.4f deg, DEC = %+.4f deg).",
        centre_ra, centre_dec,
    )
    logger.debug("Footprint centre before rotation: RA = %.6f deg, DEC = %+.6f deg",
                 centre_ra, centre_dec)

    # After the rotation the axis is +z, which in RA/DEC is the *pole*: DEC = +90
    # and RA degenerate, since the footprint then wraps all the way around it.
    # (The centre does not land on RA/DEC = (0, 0) -- that would be the +x axis.)
    # So the DEC is the half of the check worth printing; in place of the
    # meaningless RA, report how far off the axis actually lands.
    # Done in float64 on a single 3-vector, whatever the frame's working dtype,
    # and measured off the transverse component rather than through arccos of the
    # z one. Both matter: arccos is ill-conditioned near 1, so a float32-rounded
    # axis would be amplified into a ~0.02 deg residual for an alignment that is
    # in fact perfect to the stored precision.
    matrix = np.asarray(frame.matrix, dtype=np.float64)
    aligned = matrix @ np.asarray(frame.mean_direction, dtype=np.float64)
    aligned /= np.linalg.norm(aligned)
    _, dec_after = _as_radec(aligned)
    residual = float(np.degrees(np.arctan2(np.hypot(aligned[0], aligned[1]), aligned[2])))
    logger.debug(
        "Footprint centre after rotation: DEC = %+.6f deg (the z axis; RA is degenerate "
        "at the pole), residual misalignment %.2e deg", dec_after, residual,
    )
    return frame


def rotate_all(frame: ConeFrame, *arrays: Optional[np.ndarray]) -> None:
    """Survey frame -> cone frame, in place. ``None`` entries are skipped."""
    for array in arrays:
        if array is not None:
            frame.rotate(array, out=array)


def unrotate_all(frame: ConeFrame, *arrays: Optional[np.ndarray]) -> None:
    """Cone frame -> survey frame, in place. ``None`` entries are skipped."""
    for array in arrays:
        if array is not None:
            frame.unrotate(array, out=array)
