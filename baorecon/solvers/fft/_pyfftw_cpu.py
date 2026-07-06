"""In-place pyfftw CPU FFT path for the iterative (Burden) displacement solver.

This is the opt-in low-memory backend (``BAORECON_FFT=pyfftw``). Where the scipy
path allocates a fresh complex/real array for every ``rfftn``/``irfftn`` (its
working set is the dominant term in the CPU peak), this path transforms a single
padded buffer *in place* and reuses it for every transform in the Burden
iteration and the final displacement build. Combined with the streamed LocalLOS
projection (the radial versor is evaluated on the fly, so no 3-vector versor or
gradient grid is materialised) it roughly halves the CPU peak.

Only the two line-of-sight strategies the reconstructor actually builds are
supported here -- a fixed axis (``FixedAxisLOS.axis``) and a radial
``LocalLOS`` (via ``radial_versor``); the caller falls back to scipy otherwise.
Results match the scipy path to float32 round-off. The potential recompute is
left on the scipy path (cold code, needs the already-computed displacement).
"""

import os
import pickle
from pathlib import Path

import numpy as np
import pyfftw

from baorecon.solvers.fft._common import build_inv_k2, prepare_k_components
from baorecon.solvers.fft._radial_stream import (
    project_grad_onto_los,
    reconstruct_parallel_vector,
)
from baorecon.utils.loggers import setup_logger

logger = setup_logger(__name__)

_IMAG_UNIT = np.complex64(1j)


# ---------------------------------------------------------------------------
# FFTW wisdom persistence: FFTW_MEASURE plans well but plans slowly. Persisting
# wisdom to disk makes planning a one-time cost across runs and subprocesses
# (the benchmarks spawn a fresh process per configuration).
# ---------------------------------------------------------------------------
_WISDOM_PATH = Path(
    os.environ.get("BAORECON_FFTW_WISDOM", Path.home() / ".cache" / "baorecon" / "fftw_wisdom.pkl")
)
_wisdom_loaded = False


def _load_wisdom() -> None:
    global _wisdom_loaded
    if _wisdom_loaded:
        return
    _wisdom_loaded = True
    try:
        with open(_WISDOM_PATH, "rb") as wisdom_file:
            pyfftw.import_wisdom(pickle.load(wisdom_file))
    except Exception:  # missing / stale / unreadable wisdom is non-fatal
        pass


def _save_wisdom() -> None:
    try:
        _WISDOM_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_WISDOM_PATH, "wb") as wisdom_file:
            pickle.dump(pyfftw.export_wisdom(), wisdom_file)
    except Exception:
        pass


def _threads() -> int:
    # Match scipy.fft's ``workers=-1`` (all cores) by default so the pyfftw path
    # is compared on equal footing; override with BAORECON_FFT_THREADS. OMP is
    # intentionally NOT consulted -- it throttles numba/BLAS, not the FFT.
    thread_override = os.environ.get("BAORECON_FFT_THREADS")
    if thread_override and thread_override.isdigit() and int(thread_override) > 0:
        return int(thread_override)
    return os.cpu_count() or 1
'''def _threads() -> int:
    """Determine the optimal number of FFTW threads with strict HPC guardrails.
    
    Hierarchy of checks:
    1. BAORECON_FFT_THREADS (Explicit package override)
    2. SLURM_CPUS_PER_TASK (Safe default on clusters)
    3. OMP_NUM_THREADS (Standard scientific python limit)
    4. os.cpu_count() (Fallback for local laptops/dedicated servers)
    """
    # 1. Override specifico dell'utente per PyFFTW
    val = os.environ.get("BAORECON_FFT_THREADS")
    if val and val.isdigit() and int(val) > 0:
        return int(val)
        
    # 2. Rilevamento SLURM: Quanti core mi ha assegnato realmente il cluster?
    val = os.environ.get("SLURM_CPUS_PER_TASK")
    if val and val.isdigit() and int(val) > 0:
        return int(val)
        
    # 3. Rilevamento OpenMP: Se l'utente ha messo un tetto globale, rispettiamolo
    val = os.environ.get("OMP_NUM_THREADS")
    if val and val.isdigit() and int(val) > 0:
        return int(val)
        
    # 4. Fallback (solo se non siamo su un cluster o l'utente non ha posto limiti)
    return os.cpu_count() or 1
'''

_PLAN_FLAGS = {
    "estimate": "FFTW_ESTIMATE",     # instant planning, slightly slower transforms
    "measure": "FFTW_MEASURE",       # slow first plan, fast transforms (cached via wisdom)
    "patient": "FFTW_PATIENT",
}


def _plan_flags():
    """Planning rigour (BAORECON_FFT_PLAN). Default ``estimate``: instant
    planning and, with all cores, transforms as fast as scipy here -- so the
    memory win comes for free. ``measure``/``patient`` plan slower (bounded by
    ``BAORECON_FFT_PLAN_TIMELIMIT`` and cached to disk as wisdom) for workloads
    where the extra transform speed pays off."""
    return (_PLAN_FLAGS.get(os.environ.get("BAORECON_FFT_PLAN", "estimate").lower(), "FFTW_ESTIMATE"),)


def _plan_timelimit() -> float:
    timelimit_env = os.environ.get("BAORECON_FFT_PLAN_TIMELIMIT", "15")
    try:
        return float(timelimit_env)
    except ValueError:
        return 15.0


# The streamed radial projection kernels (``project_grad_onto_los`` /
# ``reconstruct_parallel_vector``, with the unit versor evaluated on the fly -- no
# gradient or versor grid) are shared with the scipy path and live in
# ``_radial_stream``.


# ---------------------------------------------------------------------------
# In-place real FFT buffer + plans.
# ---------------------------------------------------------------------------
class _InPlaceRFFT:
    """One padded real buffer with in-place forward (r2c) and backward (c2r)
    plans over its full shape. ``real_view`` is the logical (nx,ny,nz) real view
    and ``complex_view`` the (nx,ny,nz//2+1) complex view of the *same* memory."""

    def __init__(self, shape, threads, flags=None):
        nx, ny, nz = (int(n) for n in shape)
        n_complex = nz // 2 + 1
        flags = flags or _plan_flags()
        plan_time_limit = _plan_timelimit()
        self.buffer = pyfftw.empty_aligned((nx, ny, 2 * n_complex), dtype="float32")
        self.real_view = self.buffer[:, :, :nz]
        self.complex_view = self.buffer.view("complex64")
        _load_wisdom()
        self.forward = pyfftw.FFTW(self.real_view, self.complex_view, axes=(0, 1, 2), direction="FFTW_FORWARD",
                                   flags=flags, threads=threads, planning_timelimit=plan_time_limit)
        # ``normalise_idft`` is applied on the backward call, matching scipy's 1/N^3.
        self.backward = pyfftw.FFTW(self.complex_view, self.real_view, axes=(0, 1, 2), direction="FFTW_BACKWARD",
                                    flags=flags, threads=threads, planning_timelimit=plan_time_limit)
        _save_wisdom()


def supported(los, n_iterations) -> bool:
    """Whether the in-place path can handle this LOS/iteration combination.

    NB: a radial LOS is detected via ``min_corner`` (a plain attribute), NOT via
    ``radial_versor`` -- the latter is a lazy property whose access would
    materialise the 3-vector versor grid this path exists to avoid.
    """
    if n_iterations == 0:
        return True  # RealSpace: only the final build, LOS is irrelevant
    if getattr(los, "axis", None) is not None:
        return True  # fixed-axis LOS
    # Radial (LocalLOS): needs the geometry; the versor is evaluated on the fly.
    return getattr(los, "min_corner", None) is not None


def displacement_inplace(delta, mesh, los, f, bias, beta, n_iterations):
    """Compute the displacement field with the in-place pyfftw path.

    Same physics as :meth:`FFTSolverCPU._compute_displacement_iterative_potential`
    (potential -> gradient -> LOS projection -> divergence -> Burden update, then
    psi = grad(phi)), but every transform reuses a single padded buffer. Returns an
    ``(N,N,N,3)`` float32 array (the layout the interpolation kernels require).
    """
    delta = np.ascontiguousarray(delta, dtype=np.float32)
    shape = delta.shape
    kx, ky, kz = prepare_k_components(mesh.cell_size, mesh.nmesh)
    k_broadcast = (kx[:, None, None], ky[None, :, None], kz[None, None, :])
    inv_k2_bias = build_inv_k2((kx, ky, kz), bias=bias)   # (N,N,nc) real

    axis = getattr(los, "axis", None)
    fft = _InPlaceRFFT(shape, threads=_threads())
    real_view, complex_view = fft.real_view, fft.complex_view

    real_view[...] = delta
    fft.forward()                   # complex_view = delta_k

    if n_iterations > 0 and axis is None:
        # Radial LOS: geometry + scratch for the parallel magnitude s and the
        # in-place divergence accumulation.
        min_corner = los.min_corner
        cell_size = los.cell_size
        los_magnitude = np.empty(shape, dtype=np.float32)       # s = grad . n_hat
        delta_k_saved = np.empty_like(complex_view)
        divergence_accum = np.empty_like(complex_view)

    for iteration in range(n_iterations):
        if axis is not None:
            k_axis = k_broadcast[axis]
            # Gradient of the potential along the LOS axis: grad_a = d_a phi.
            complex_view *= inv_k2_bias
            complex_view *= k_axis
            complex_view *= -_IMAG_UNIT
            fft.backward()          # real_view = grad_a
            # Divergence of the parallel field (single axis): d_a grad_a.
            fft.forward()           # complex_view = rfft(grad_a)
            complex_view *= k_axis
            complex_view *= _IMAG_UNIT
            fft.backward()          # real_view = correction
        else:
            # Project the gradient onto the radial LOS: accumulate the parallel
            # magnitude s = grad.n_hat one gradient component at a time.
            delta_k_saved[...] = complex_view   # preserve delta_k across the 3 components
            for component in range(3):
                complex_view[...] = delta_k_saved
                complex_view *= inv_k2_bias
                complex_view *= k_broadcast[component]
                complex_view *= -_IMAG_UNIT
                fft.backward()      # real_view = grad_i
                project_grad_onto_los(los_magnitude, real_view, component, min_corner[0], min_corner[1], min_corner[2], cell_size[0], cell_size[1], cell_size[2], component == 0)
            # Divergence of the parallel field s*n_hat, one component at a time.
            divergence_accum[...] = 0
            for component in range(3):
                reconstruct_parallel_vector(real_view, los_magnitude, component, min_corner[0], min_corner[1], min_corner[2], cell_size[0], cell_size[1], cell_size[2])
                fft.forward()       # complex_view = rfft(s * n_hat_j)
                complex_view *= k_broadcast[component]
                complex_view *= _IMAG_UNIT
                divergence_accum += complex_view
            complex_view[...] = divergence_accum
            fft.backward()          # real_view = correction

        # Burden update: reconstructed density = delta - f * div(parallel).
        real_view *= -f
        if iteration == 0:
            real_view /= (1 + beta)
        real_view += delta
        fft.forward()               # complex_view = delta_k for the next iteration / final build

    if n_iterations > 0 and axis is None:
        del los_magnitude, delta_k_saved, divergence_accum

    # Converged density -> displacement psi = grad(phi). delta_k is copied out of
    # the shared buffer once, since each component reloads it.
    delta_k = np.array(complex_view, copy=True)
    displacement = np.empty(shape + (3,), dtype=np.float32)
    for component in range(3):
        complex_view[...] = delta_k
        complex_view *= inv_k2_bias
        complex_view *= k_broadcast[component]
        complex_view *= _IMAG_UNIT
        fft.backward()
        displacement[..., component] = real_view
    return displacement
