# bao_ifft.py

"""
BAO Reconstruction using iFFT solver (backward-compatible wrapper).

This module provides the legacy BAOiFFTreconstruction interface which now delegates
to the unified BAOReconstructor with solver_type='ifft'.

For new code, use BAOReconstructor directly or import BAOiFFTreconstruction for
backward compatibility with existing scripts.
"""

import numpy as np
from zeldareco.BAOreconstruction.bao_reconstructor import BAOReconstructor
from zeldareco.utils.loggers import setup_logger

logger = setup_logger(__name__)


class BAOiFFTreconstruction(BAOReconstructor):
    """
    Backward-compatible wrapper for BAO Reconstruction using iFFT solver.

    This class now delegates all functionality to BAOReconstructor with
    solver_type='ifft'. It maintains the original constructor signature
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
        n_iterations: int = 3,
        f: float = 0.88,
        bias: float = 1.0,
        MAS: str = "CIC",
        dtype=np.float32,
        param_file: str = None,
    ):
        """
        Initialize BAO iFFT reconstruction (delegates to BAOReconstructor).

        Parameters
        ----------
        param_file : str, optional
            (Deprecated) Configuration file. Not implemented. Default is None.
        All other parameters: see BAOReconstructor documentation.
        """
        if param_file is not None:
            raise NotImplementedError("param_file functionality is not implemented.")

        # Delegate to parent BAOReconstructor with solver_type='ifft'
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
            solver_type="ifft",
            n_iterations=n_iterations,
        )
