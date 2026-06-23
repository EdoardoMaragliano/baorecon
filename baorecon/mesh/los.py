"""Line-of-sight (LOS) strategies.

The line-of-sight is modelled as a strategy object that knows how to project a
vector field onto the parallel (LOS) direction at every mesh point. Two
concrete strategies are provided:

* :class:`FixedAxisLOS` -- a global plane-parallel LOS along one Cartesian
  axis. It stores only the axis index and allocates no arrays: the projection
  simply keeps the chosen component and zeros the others.
* :class:`LocalLOS` -- a per-cell radial LOS pointing away from the observer.
  The radial versor field is computed lazily and cached.

The reconstructor instantiates the appropriate strategy and injects it into the
solvers.
"""

from abc import ABC, abstractmethod

import numpy as np
from numba import njit, prange

from baorecon.field_ops import project_vector_field
from baorecon.utils.loggers import setup_logger

logger = setup_logger(__name__)

try:
    import cupy as cp
    from numba import cuda
    CUPY_AVAILABLE = cuda.is_available()
except ImportError:
    CUPY_AVAILABLE = False


def _get_array_module(*arrays):
    """Return the array module (cupy or numpy) matching the given arrays."""
    if CUPY_AVAILABLE:
        for arr in arrays:
            if isinstance(arr, cp.ndarray):
                return cp
    return np


@njit(parallel=True, fastmath=True)
def project_vector_field_jit(vector_field, los_versor, out):
    """Project ``vector_field`` onto ``los_versor`` at every mesh point.

    Parameters
    ----------
    vector_field : ndarray, shape (Nx, Ny, Nz, 3)
        Input vector field to project.
    los_versor : ndarray, shape (Nx, Ny, Nz, 3)
        Unit vectors defining the LOS direction at each mesh point.
    out : ndarray, shape (Nx, Ny, Nz, 3)
        Pre-allocated output buffer (may alias ``vector_field``).

    Returns
    -------
    out : ndarray
        The projected vector field.
    """
    nx, ny, nz, _ = vector_field.shape
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                dot = (vector_field[i, j, k, 0] * los_versor[i, j, k, 0] +
                       vector_field[i, j, k, 1] * los_versor[i, j, k, 1] +
                       vector_field[i, j, k, 2] * los_versor[i, j, k, 2])
                for c in range(3):
                    out[i, j, k, c] = dot * los_versor[i, j, k, c]
    return out


class LOSStrategy(ABC):
    """Abstract line-of-sight strategy."""

    @abstractmethod
    def project_parallel(self, vector_field: np.ndarray, out: np.ndarray = None) -> np.ndarray:
        """Return the component of ``vector_field`` parallel to the LOS."""
        raise NotImplementedError


class FixedAxisLOS(LOSStrategy):
    """Global plane-parallel LOS along a single Cartesian axis.

    Stores only the axis index; allocates no versor arrays. The projection
    keeps the component along ``axis`` and zeros the other two.
    """

    def __init__(self, axis: int) -> None:
        if axis not in (0, 1, 2):
            raise ValueError("axis must be 0 (x), 1 (y) or 2 (z)")
        self.axis = int(axis)

    @property
    def direction(self) -> np.ndarray:
        """Unit vector along the LOS axis, shape (3,)."""
        e = np.zeros(3, dtype=np.float32)
        e[self.axis] = 1.0
        return e

    def project_parallel(self, vector_field: np.ndarray, out: np.ndarray = None) -> np.ndarray:
        xp = _get_array_module(vector_field)
        if out is None:
            out = xp.empty_like(vector_field)
        # Keep only the LOS-axis component; zero the others. Copy the kept axis
        # first so this is correct even when out aliases vector_field.
        kept = vector_field[..., self.axis].copy()
        out[...] = 0
        out[..., self.axis] = kept
        return out


class LocalLOS(LOSStrategy):
    """Per-cell radial line-of-sight pointing away from the observer.

    The radial versor field is computed lazily from the box geometry and cached
    (on host, and on device for the GPU path).
    """

    def __init__(self, boxcentre, min_corner, boxsize, nmesh, device: str = "cpu") -> None:
        self.boxcentre = np.asarray(boxcentre, dtype=np.float32)
        self.min_corner = np.asarray(min_corner, dtype=np.float32)
        self.boxsize = np.broadcast_to(np.asarray(boxsize, dtype=np.float32), (3,)).copy()
        nmesh = np.asarray(nmesh)
        if nmesh.ndim == 0:
            nmesh = np.full(3, int(nmesh), dtype=np.int32)
        self.nmesh = nmesh.astype(np.int32)
        self.device = device

        self._radial_versor = None
        self._radial_versor_dev = None

    @property
    def radial_versor(self) -> np.ndarray:
        """Unit radial vector at every mesh point, shape (Nx, Ny, Nz, 3)."""
        if self._radial_versor is None:
            nx, ny, nz = (int(n) for n in self.nmesh)
            x = np.linspace(0, self.boxsize[0], nx, endpoint=False, dtype=np.float32)[:, None, None]
            y = np.linspace(0, self.boxsize[1], ny, endpoint=False, dtype=np.float32)[None, :, None]
            z = np.linspace(0, self.boxsize[2], nz, endpoint=False, dtype=np.float32)[None, None, :]
            
            # Survey-frame coordinate of each mesh point.
            x += self.min_corner[0]
            y += self.min_corner[1]
            z += self.min_corner[2]

            # compute norm of the separation vector
            norm = x**2 + y**2 + z **2
            np.sqrt(norm, out=norm)

            # compute 1/norm, store in norm
            mask = norm > 0
            np.divide(1.0, norm, out=norm, where=mask)

            # pre-allocate versor field for los
            versor = np.empty((nx, ny, nz, 3), dtype=np.float32)
            np.multiply(x, norm, out=versor[..., 0])
            np.multiply(y, norm, out=versor[..., 1])
            np.multiply(z, norm, out=versor[..., 2])

            if not mask.all():
                versor[~mask] = 0.0
                
            self._radial_versor = versor
        return self._radial_versor

    def project_parallel(self, vector_field: np.ndarray, out: np.ndarray = None) -> np.ndarray:
        xp = _get_array_module(vector_field)
        if out is None:
            out = xp.empty_like(vector_field)

        if xp is np:
            return project_vector_field_jit(vector_field, self.radial_versor, out)

        # GPU path: upload and cache the versor once, project natively on-device.
        if self._radial_versor_dev is None:
            self._radial_versor_dev = xp.asarray(self.radial_versor)
        return project_vector_field(vector_field, self._radial_versor_dev, out=out)
