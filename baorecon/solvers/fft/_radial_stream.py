"""Streamed radial (``LocalLOS``) projection kernels, versor evaluated on the fly.

Both CPU FFT backends -- the default ``scipy.fft`` path
(:class:`~baorecon.solvers.fft.cpu.FFTSolverCPU`) and the opt-in in-place pyfftw
path (:mod:`baorecon.solvers.fft._pyfftw_cpu`) -- project the potential gradient
onto a per-cell radial line of sight (LOS). Rather than materialising the full
3-vector versor grid (``LocalLOS.radial_versor``, shape ``(Nx, Ny, Nz, 3)``) and
the full gradient field, they stream one gradient component at a time and
evaluate the unit versor cell by cell::

    n_hat(cell) = coord / |coord|,   coord_a = min_corner[a] + idx_a * cell_size[a]

The radial projection of a gradient field is then a two-pass reduction:

1. **accumulate** the LOS-parallel magnitude ``s = grad . n_hat`` one gradient
   component at a time -- :func:`project_grad_onto_los`;
2. **scatter** it back to the parallel vector field ``parallel_a = s * n_hat_a``
   one component at a time -- :func:`reconstruct_parallel_vector`,

so the ``(Nx, Ny, Nz, 3)`` gradient / versor / parallel-field grids are never
held in memory at once.

These kernels are pure numpy/numba (no pyfftw), so the scipy path can use them
without pulling in the optional pyfftw dependency. All geometry is passed as
plain float scalars (``min_*`` / ``cell_*``) so the compiled kernels never touch
the Python LOS object; the versor is single precision (``float32``), matching
the cached ``LocalLOS.radial_versor`` this path replaces.
"""

import numpy as np
from numba import njit, prange


@njit(fastmath=True, inline="always")
def get_los_versor_component(
    coord_x: float, coord_y: float, coord_z: float, axis: int
) -> float:
    """Return the ``axis`` component of the radial unit versor at one cell.

    ``n_hat = coord / |coord|`` with ``coord = (coord_x, coord_y, coord_z)`` the
    survey-frame position of the cell. The cell at the observer (``|coord| = 0``)
    has no defined direction and yields ``0`` on every axis, matching
    ``LocalLOS.radial_versor`` (which zeroes the ``norm == 0`` cells).

    Parameters
    ----------
    coord_x, coord_y, coord_z : float
        Survey-frame Cartesian coordinates of the cell.
    axis : int
        Component to return: ``0`` (x), ``1`` (y) or ``2`` (z).

    Returns
    -------
    float
        The ``axis`` component of the unit versor, ``float32`` precision.
    """
    radius_sq = coord_x * coord_x + coord_y * coord_y + coord_z * coord_z
    if radius_sq <= 0:
        return np.float32(0.0)
    axis_coord = coord_x if axis == 0 else (coord_y if axis == 1 else coord_z)
    return axis_coord / np.sqrt(radius_sq)


@njit(parallel=True, fastmath=True, cache=True)
def project_grad_onto_los(
    parallel_magnitude: np.ndarray,
    gradient_component: np.ndarray,
    axis: int,
    min_x: float,
    min_y: float,
    min_z: float,
    cell_x: float,
    cell_y: float,
    cell_z: float,
    initialize: bool,
) -> None:
    """Accumulate one gradient component into the LOS-parallel magnitude.

    Adds ``gradient_component * n_hat_axis`` cell by cell to ``parallel_magnitude``
    (the running scalar field ``s = grad . n_hat``), with the radial versor
    evaluated on the fly. Called once per axis; ``initialize=True`` writes the
    first (x) pass so no separate zeroing of ``parallel_magnitude`` is needed.

    Parameters
    ----------
    parallel_magnitude : np.ndarray
        Real ``(Nx, Ny, Nz)`` output, written in place. Overwritten when
        ``initialize`` is ``True``, otherwise incremented.
    gradient_component : np.ndarray
        Real ``(Nx, Ny, Nz)`` grid holding the ``axis`` gradient component.
    axis : int
        Which gradient / versor component this pass contributes (``0``/``1``/``2``).
    min_x, min_y, min_z : float
        Survey-frame coordinate of the ``(0, 0, 0)`` cell (``LocalLOS.min_corner``).
    cell_x, cell_y, cell_z : float
        Cell size along each axis (``boxsize / nmesh``).
    initialize : bool
        ``True`` on the first (``axis == 0``) pass to initialise the accumulator;
        ``False`` to add to it.

    Returns
    -------
    None
        ``parallel_magnitude`` is modified in place.
    """
    nx, ny, nz = parallel_magnitude.shape
    for i in prange(nx):
        coord_x = min_x + i * cell_x
        for j in range(ny):
            coord_y = min_y + j * cell_y
            for k in range(nz):
                coord_z = min_z + k * cell_z
                versor = get_los_versor_component(coord_x, coord_y, coord_z, axis)
                contribution = gradient_component[i, j, k] * versor
                if initialize:
                    parallel_magnitude[i, j, k] = contribution
                else:
                    parallel_magnitude[i, j, k] += contribution


@njit(parallel=True, fastmath=True, cache=True)
def reconstruct_parallel_vector(
    parallel_component: np.ndarray,
    parallel_magnitude: np.ndarray,
    axis: int,
    min_x: float,
    min_y: float,
    min_z: float,
    cell_x: float,
    cell_y: float,
    cell_z: float,
) -> None:
    """Scatter the parallel magnitude into one component of the parallel field.

    Writes ``parallel_component = parallel_magnitude * n_hat_axis`` cell by cell
    (the ``axis`` component of ``parallel = (grad . n_hat) n_hat``), with the
    radial versor evaluated on the fly. Called once per axis; the caller then
    transforms the single component into the divergence accumulator.

    Parameters
    ----------
    parallel_component : np.ndarray
        Real ``(Nx, Ny, Nz)`` output, fully overwritten with the ``axis``
        component of the parallel vector field.
    parallel_magnitude : np.ndarray
        Real ``(Nx, Ny, Nz)`` LOS-parallel magnitude ``s = grad . n_hat``.
    axis : int
        Which versor / output component to produce (``0``/``1``/``2``).
    min_x, min_y, min_z : float
        Survey-frame coordinate of the ``(0, 0, 0)`` cell (``LocalLOS.min_corner``).
    cell_x, cell_y, cell_z : float
        Cell size along each axis (``boxsize / nmesh``).

    Returns
    -------
    None
        ``parallel_component`` is modified in place.
    """
    nx, ny, nz = parallel_component.shape
    for i in prange(nx):
        coord_x = min_x + i * cell_x
        for j in range(ny):
            coord_y = min_y + j * cell_y
            for k in range(nz):
                coord_z = min_z + k * cell_z
                versor = get_los_versor_component(coord_x, coord_y, coord_z, axis)
                parallel_component[i, j, k] = parallel_magnitude[i, j, k] * versor
