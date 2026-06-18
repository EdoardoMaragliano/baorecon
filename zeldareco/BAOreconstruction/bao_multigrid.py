# bao_multigrid.py

"""
BAO Reconstruction using Multigrid method (backward-compatible wrapper).

This module provides the legacy BAOMultigridReconstruction interface which now delegates
to the unified BAOReconstructor with solver_type='multigrid'.

For new code, use BAOReconstructor directly or import BAOMultigridReconstruction for
backward compatibility with existing scripts.
"""

import numpy as np
from zeldareco.BAOreconstruction.bao_reconstructor import BAOReconstructor
from zeldareco.utils.loggers import setup_logger

logger = setup_logger(__name__)


class BAOMultigridReconstruction(BAOReconstructor):
    """
    Backward-compatible wrapper for BAO Reconstruction using Multigrid solver.

    This class now delegates all functionality to BAOReconstructor with
    solver_type='multigrid'. It maintains the original constructor signature
    for full backward compatibility.

    All parameters are forwarded to BAOReconstructor; see its documentation
    for parameter details.
    """

    def __init__(
        self,
        data_pos: np.ndarray,
        random_pos: np.ndarray,
        data_weights: np.ndarray = None,
        random_weights: np.ndarray = None,
        data_ids: np.ndarray = None,
        RSDspace: str = "RedshiftSpace",
        nmesh: int = 256,
        boxsize: float = None,
        boxcentre: np.ndarray = None,
        padding: float = 0.01,
        los: str = None,
        R_sm: float = 15,
        pbc: bool = False,
        rectype: str = "rec-sym",
        threshold_randoms: float = 0.01,
        f: float = 0.88,
        bias: float = 1.0,
        MAS: str = "CIC",
        dtype=np.float32,
        damping_factor: float = 0.4,
        jacobi_iterations: int = 5,
        vcycle_iterations: int = 6,
    ):
        """
        Initialize the Multigrid BAO Reconstructor.

        This delegates to the parent `BAOReconstructor` while explicitly setting
        the solver type to use a multigrid approach.

        Parameters
        ----------
        data_pos : np.ndarray
            Array of shape (N, 3) containing the Cartesian coordinates of the data catalog.
        random_pos : np.ndarray
            Array of shape (M, 3) containing the Cartesian coordinates of the random catalog.
        data_weights : np.ndarray, optional
            Array of shape (N,) containing weights for the data catalog. Default is None.
        random_weights : np.ndarray, optional
            Array of shape (M,) containing weights for the random catalog. Default is None.
        data_ids : np.ndarray, optional
            Array of shape (N,) containing unique identifiers for the data objects. Default is None.
        RSDspace : str, optional
            Space in which the input data is provided. Typically "RedshiftSpace" or "RealSpace".
            Default is "RedshiftSpace".
        nmesh : int, optional
            Number of cells per box side for the density mesh. Default is 256.
        boxsize : float, optional
            Physical size of the bounding box. If None, it is estimated from the data extent 
            plus the `padding`. Default is None.
        boxcentre : np.ndarray, optional
            Array of shape (3,) defining the center of the bounding box. If None, it is estimated 
            from the data. Default is None.
        padding : float, optional
            Fractional padding added to the bounding box if `boxsize` is not provided. 
            Default is 0.01.
        los : str, optional
            Line-of-sight direction/convention. Default is None.
        R_sm : float, optional
            Smoothing scale (radius) applied to the density field, typically in Mpc/h. 
            Default is 15.0.
        pbc : bool, optional
            Whether to use periodic boundary conditions. Default is False.
        rectype : str, optional
            Reconstruction convention to apply (e.g., "rec-sym", "rec-iso"). Default is "rec-sym".
        threshold_randoms : float, optional
            Density threshold to mask out regions with low random catalog density. Default is 0.01.
        f : float, optional
            Logarithmic growth rate of structure. Default is 0.88.
        bias : float, optional
            Linear galaxy bias parameter. Default is 1.0.
        MAS : str, optional
            Mass Assignment Scheme used to project particles to the mesh 
            (e.g., "CIC", "TSC", "NGP"). Default is "CIC".
        dtype : type, optional
            Data type for the mesh arrays. Default is np.float32.
        damping_factor : float, optional
            Damping parameter for the multigrid Jacobi smoother. Default is 0.4.
        jacobi_iterations : int, optional
            Number of Jacobi smoothing sweeps per multigrid level. Default is 5.
        vcycle_iterations : int, optional
            Number of complete V-cycles to perform in the multigrid solver. Default is 6.
        """

        # Delegate to parent BAOReconstructor with solver_type='multigrid'
        # Pass multigrid-specific parameters via solver_kwargs
        super().__init__(
            data_pos=data_pos,
            random_pos=random_pos,
            data_weights=data_weights,
            random_weights=random_weights,
            data_ids=data_ids,
            RSDspace=RSDspace,
            nmesh=nmesh,
            boxsize=boxsize,
            boxcentre=boxcentre,
            padding=padding,
            los=los,
            R_sm=R_sm,
            pbc=pbc,
            rectype=rectype,
            f=f,
            bias=bias,
            MAS=MAS,
            dtype=dtype,
            threshold_randoms=threshold_randoms,
            solver_type="multigrid",
            # Multigrid-specific parameters passed to solver
            n_smooth=jacobi_iterations,
            v_cycles=vcycle_iterations,
            damping=damping_factor,
        )
