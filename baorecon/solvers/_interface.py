"""Abstract solver interface.

:class:`PoissonSolver` is the common base for the FFT and multigrid
displacement solvers. It provides lazy caching of ``potential`` and
``displacement``.
"""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

try:
    import cupy as cp
    from numba import cuda
    CUPY_AVAILABLE = cuda.is_available()
except ImportError:
    CUPY_AVAILABLE = False

from baorecon.utils.loggers import setup_logger

logger = setup_logger(__name__)


def _as_array(arr):
    """Like ``np.asarray`` but passes cupy arrays through (no host transfer)."""
    if CUPY_AVAILABLE and isinstance(arr, cp.ndarray):
        return arr
    return np.asarray(arr)


class PoissonSolver(ABC):
    """Base class producing a potential and a displacement field from delta."""

    def __init__(self, delta_on_mesh, mesh, f=None, bias=1.0, RSDspace="RealSpace", pbc=False) -> None:
        self.delta_on_mesh = _as_array(delta_on_mesh)
        self.mesh = mesh
        # Whether the box is periodic: the read-out then wraps the coordinates
        # into [0, boxsize) and the stencil, as survey_to_box_frame + kernels did.
        self._pbc = bool(pbc)
        self.f = f
        self.bias = bias
        self.beta = (self.f / self.bias) if (self.f is not None and self.bias != 0) else 0.0
        self.RSDspace = RSDspace

        self._potential: Optional[np.ndarray] = None
        self._displacement: Optional[np.ndarray] = None

    @property
    def potential(self):
        if self._potential is None:
            logger.debug("Computing potential via _compute_potential_mesh()")
            self._compute_potential_mesh()
        return self._potential

    @property
    def displacement(self):
        if self._displacement is None:
            logger.debug("Computing displacement via _compute_displacement_mesh()")
            self._compute_displacement_mesh()
        return self._displacement

    @abstractmethod
    def _compute_displacement_mesh(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def _compute_potential_mesh(self) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def read_displacement_at(self, positions: np.ndarray, mas: str = "CIC",
                             survey_frame: bool = False) -> np.ndarray:
        """Read the displacement Psi at ``positions``, returning a host ``(N, 3)`` array.

        ``positions`` are in the box frame, or in the survey frame when
        ``survey_frame`` is set: the solver then moves them to the box frame
        inside the read-out kernel, without building a box-frame copy.
        """
        raise NotImplementedError

    def _origin(self, positions: np.ndarray, survey_frame: bool):
        """The box's lower corner for the read-out kernels (see :mod:`baorecon.mas.cpu`);
        ``None`` when ``positions`` are already in the box frame."""
        if not survey_frame:
            return None
        return np.asarray(self.mesh.min_corner, dtype=np.asarray(positions).dtype)
    
