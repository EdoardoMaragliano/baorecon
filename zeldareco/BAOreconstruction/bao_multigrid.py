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
        RSDspace: str = "RealSpace",
        nmesh: int = 256,
        boxsize: float = None,
        boxcentre: np.ndarray = None,
        padding: float = 0.01,
        los: str = "z",
        R_sm: float = 15,
        pbc: bool = False,
        rectype: str = "rec-sym",
        threshold_randoms: float = 0.7,
        f: float = 0.88,
        bias: float = 1.0,
        MAS: str = "CIC",
        dtype=np.float32,
        damping_factor: float = 0.4,
        jacobi_iterations: int = 5,
        vcycle_iterations: int = 6,
        mg_tolerance: float = 1e-6,
        mg_max_iterations: int = 10,
    ):
        """
        Initialize BAO Multigrid reconstruction (delegates to BAOReconstructor).

        Parameters
        ----------
        damping_factor : float, optional
            Damping for Jacobi iteration. Passed to solver_kwargs. Default is 0.4.
        jacobi_iterations : int, optional
            Jacobi steps per level. Passed to solver_kwargs. Default is 5.
        vcycle_iterations : int, optional
            V-cycles per level. Passed to solver_kwargs. Default is 6.
        mg_tolerance : float, optional
            Multigrid convergence tolerance. Passed to solver_kwargs. Default is 1e-6.
        mg_max_iterations : int, optional
            Max multigrid iterations. Passed to solver_kwargs. Default is 10.
        All other parameters: see BAOReconstructor documentation.
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
