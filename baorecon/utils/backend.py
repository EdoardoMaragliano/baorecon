"""Computational backend selection for the FFT solvers.

Holds the small :class:`FFTBackend` container and the :func:`get_fft_backend`
factory used to pick between the CPU (numpy/scipy) and GPU (CuPy) backends.
``CUPY_AVAILABLE`` is the single source of truth for GPU availability across
the package.
"""

import os

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

# --- pyfftw availability (optional in-place, low-memory CPU FFT backend) ---
# scipy remains the default CPU backend; pyfftw is opt-in via BAORECON_FFT=pyfftw.
# The in-place path transforms a single padded buffer instead of allocating a
# fresh output per transform, which roughly halves the CPU FFT working set.
try:
    import pyfftw  # noqa: F401
    PYFFTW_AVAILABLE = True
except ImportError:
    PYFFTW_AVAILABLE = False
    logger.debug("pyfftw not found. The in-place CPU FFT backend will be unavailable.")


def use_pyfftw() -> bool:
    """Return True when the CPU solver should use the in-place pyfftw backend.

    Enabled by setting the environment variable ``BAORECON_FFT=pyfftw`` (any
    case). Falls back to scipy when pyfftw is not installed, so this is always
    safe to query. Kept out of the reconstructor/pipeline so the choice needs no
    change to their call sites.
    """
    if os.environ.get("BAORECON_FFT", "scipy").lower() != "pyfftw":
        return False
    if not PYFFTW_AVAILABLE:
        logger.warning("BAORECON_FFT=pyfftw requested but pyfftw is not installed; using scipy.")
        return False
    return True


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
