"""Computational backend selection for the FFT solvers.

Holds the small :class:`FFTBackend` container and the :func:`get_fft_backend`
factory used to pick between the CPU (numpy/scipy) and GPU (CuPy) backends.
``CUPY_AVAILABLE`` is the single source of truth for GPU availability across
the package.
"""

import numpy as np
from scipy import fft as sfft
from dataclasses import dataclass
from typing import Callable

from baorecon.utils.loggers import setup_logger

logger = setup_logger(__name__)

# --- GPU library import and availability check ---
try:
    import cupy as cp
    from numba import cuda
    CUPY_AVAILABLE = cuda.is_available()
    if not CUPY_AVAILABLE:
        logger.warning("CuPy is installed, but no CUDA-enabled GPU was detected by Numba.")
except ImportError:
    CUPY_AVAILABLE = False
    logger.debug("CuPy or Numba not found. GPU backend will be unavailable.")


@dataclass
class FFTBackend:
    """A container for the backend-specific FFT modules and transfer helpers."""

    xp: object       # numpy or cupy module
    fft: object      # scipy.fft or cupy.fft module
    to_device: Callable
    to_host: Callable


def get_fft_backend(device: str = "cpu") -> FFTBackend:
    """Factory returning the computational backend for the FFT solver."""
    if device == "cpu":
        logger.info("Using CPU backend (numpy/scipy) for FFT solver.")
        return FFTBackend(xp=np, fft=sfft, to_device=np.asarray, to_host=np.asarray)
    elif device == "gpu":
        if not CUPY_AVAILABLE:
            raise RuntimeError(
                "GPU backend requested, but CuPy is not installed or not configured correctly."
            )
        logger.info("Using GPU backend (CuPy) for FFT solver.")
        return FFTBackend(xp=cp, fft=cp.fft, to_device=cp.asarray, to_host=cp.asnumpy)
    else:
        raise ValueError(f"Unsupported device '{device}'. Choose 'cpu' or 'gpu'.")
