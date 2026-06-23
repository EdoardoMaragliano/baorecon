"""High-level functional API for BAO reconstruction."""

from typing import Optional, Tuple

import numpy as np

from baorecon.reconstruction.bao_reconstructor import BAOReconstructor


def reconstruct_positions(
    data_pos: np.ndarray,
    random_pos: np.ndarray,
    f: float,
    bias: float,
    nmesh: int = 256,
    smoothing: float = 15,
    los: Optional[str] = None,
    device: str = "cpu",
    n_iterations: int = 3,
    cellsize: Optional[float] = None,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """Reconstruct (shift) data and random positions in one call.

    Parameters
    ----------
    data_pos, random_pos : ndarray, shape (N, 3)
        Cartesian positions of the data and random catalogues.
    f : float
        Growth rate.
    bias : float
        Linear bias.
    nmesh : int, optional
        Mesh size per axis. Default 256.
    smoothing : float, optional
        Gaussian smoothing radius in Mpc/h. Default 15.
    los : {None, 'x', 'y', 'z'}, optional
        Line of sight. ``None`` uses the local (radial) line of sight.
    device : {'cpu', 'gpu'}, optional
        Compute backend. Default 'cpu'.
    n_iterations : int, optional
        Iterations for the redshift-space iterative solver. Default 3.
    cellsize : float, optional
        Target (isotropic) cell size. Mutually exclusive with ``nmesh``; when
        given, the per-axis grid is derived from the catalogue extent and
        ``nmesh`` is ignored.
    **kwargs
        Forwarded to :class:`~baorecon.reconstruction.bao_reconstructor.BAOReconstructor`.

    Returns
    -------
    (shifted_data, shifted_random) : tuple of ndarray
        Reconstructed positions.
    """
    reconstructor = BAOReconstructor(
        data_pos=data_pos,
        random_pos=random_pos,
        f=f,
        bias=bias,
        nmesh=None if cellsize is not None else nmesh,
        cellsize=cellsize,
        R_sm=smoothing,
        los=los,
        device=device,
        n_iterations=n_iterations,
        **kwargs,
    )
    return reconstructor.run_reconstruction()
