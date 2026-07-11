"""Mass assignment (particles <-> mesh).

``assign`` paints particles onto a grid; ``readout`` interpolates a grid back
to particle positions. Both take a :class:`~baorecon.mesh.mesh.Mesh` and
dispatch to the CPU/GPU kernels.
"""

from baorecon.mas._interface import (
    CUPY_AVAILABLE,
    HALO_WIDTH,
    assign,
    read_field_at,
    readout,
)
from baorecon.utils.loggers import setup_logger

logger = setup_logger(__name__)

__all__ = ["assign", "readout", "read_field_at", "HALO_WIDTH", "CUPY_AVAILABLE",
           "_mass_assignment_info"]


def _mass_assignment_info():
    """Log a short description of the available mass-assignment schemes."""
    logger.info(
        """
    Mass Assignment Schemes implemented:
    - NGP (Nearest Grid Point)
    - CIC (Cloud-In-Cell)
    - TSC (Triangular Shaped Cloud)

    `assign` paints particles onto the mesh; `readout` interpolates a grid
    back to particle positions.
    """
    )
