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
    solver_type: str = "multigrid",
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
    solver_type : {'multigrid', 'ifft'}, optional
        Which Poisson solver to use. Default 'multigrid' -- note that the example
        parfiles configure 'ifft', so a call here and a pipeline run do not pick
        the same solver unless one of them says so explicitly.
    cellsize : float, optional
        Target (isotropic) cell size. Mutually exclusive with ``nmesh``; when
        given, the per-axis grid is derived from the catalogue extent and
        ``nmesh`` is ignored.
    **kwargs
        Forwarded to :class:`~baorecon.reconstruction.bao_reconstructor.BAOReconstructor`
        as constructor arguments -- ``bias`` aside, that is where ``rectype``,
        ``RSDspace``, ``boxsize``, ``pbc`` and the rest go.

        ``n_iterations`` (iterations of the redshift-space iterative FFT solver,
        3 by convention) is among them and reaches the solver. Other solver
        options travel in the ``solver_args`` dict, which also overrides
        ``n_iterations`` if it sets it::

            reconstruct_positions(..., solver_type="ifft",
                                  solver_args={"smoother": "mcgs"})

        A keyword that is not one of the reconstructor's parameters raises
        ``TypeError`` rather than being silently dropped, and the message points
        at where it probably belongs. ``align_cone`` is the one worth naming: it
        belongs to :class:`~baorecon.pipeline.bao_pipeline.ReconstructionPipeline`
        and has no meaning here, since this function reconstructs the positions it
        is given, in the frame it is given them in.

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
        solver_type=solver_type,
        **kwargs,
    )
    return reconstructor.run_reconstruction()
