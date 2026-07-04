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
from numba import njit, prange

from baorecon.solvers.fft._common import build_inv_k2, prepare_k_components
from baorecon.utils.loggers import setup_logger

logger = setup_logger(__name__)

_CJ = np.complex64(1j)


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
        with open(_WISDOM_PATH, "rb") as fh:
            pyfftw.import_wisdom(pickle.load(fh))
    except Exception:  # missing / stale / unreadable wisdom is non-fatal
        pass


def _save_wisdom() -> None:
    try:
        _WISDOM_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_WISDOM_PATH, "wb") as fh:
            pickle.dump(pyfftw.export_wisdom(), fh)
    except Exception:
        pass


def _threads() -> int:
    # Match scipy.fft's ``workers=-1`` (all cores) by default so the pyfftw path
    # is compared on equal footing; override with BAORECON_FFT_THREADS. OMP is
    # intentionally NOT consulted -- it throttles numba/BLAS, not the FFT.
    val = os.environ.get("BAORECON_FFT_THREADS")
    if val and val.isdigit() and int(val) > 0:
        return int(val)
    return os.cpu_count() or 1
"""
def _threads() -> int:
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
"""

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
    val = os.environ.get("BAORECON_FFT_PLAN_TIMELIMIT", "15")
    try:
        return float(val)
    except ValueError:
        return 15.0


# ---------------------------------------------------------------------------
# Numba kernels: radial (LocalLOS) projection with the unit versor evaluated on
# the fly (no gradient or versor grid, and no stored 1/norm field).
# n_hat(cell) = coord / |coord|, coord_a = min_corner[a] + idx_a * cell_size[a].
# ---------------------------------------------------------------------------
@njit(parallel=True, fastmath=True, cache=True)
def _proj_acc(s, g, axis, mcx, mcy, mcz, csx, csy, csz, first):
    """Accumulate the LOS-parallel magnitude s += g * n_hat_axis, streaming one
    gradient component g at a time (``first`` initialises s on the x pass)."""
    nx, ny, nz = s.shape
    for i in prange(nx):
        cx = mcx + i * csx
        for j in range(ny):
            cy = mcy + j * csy
            for k in range(nz):
                cz = mcz + k * csz
                n2 = cx * cx + cy * cy + cz * cz
                if n2 > 0:
                    coord = cx if axis == 0 else (cy if axis == 1 else cz)
                    nhat = coord / np.sqrt(n2)
                else:
                    nhat = np.float32(0.0)
                v = g[i, j, k] * nhat
                if first:
                    s[i, j, k] = v
                else:
                    s[i, j, k] += v


@njit(parallel=True, fastmath=True, cache=True)
def _scatter(out, s, axis, mcx, mcy, mcz, csx, csy, csz):
    """out = s * n_hat_axis, the axis component of the parallel vector field."""
    nx, ny, nz = out.shape
    for i in prange(nx):
        cx = mcx + i * csx
        for j in range(ny):
            cy = mcy + j * csy
            for k in range(nz):
                cz = mcz + k * csz
                n2 = cx * cx + cy * cy + cz * cz
                if n2 > 0:
                    coord = cx if axis == 0 else (cy if axis == 1 else cz)
                    nhat = coord / np.sqrt(n2)
                else:
                    nhat = np.float32(0.0)
                out[i, j, k] = s[i, j, k] * nhat


# ---------------------------------------------------------------------------
# In-place real FFT buffer + plans.
# ---------------------------------------------------------------------------
class _InPlaceRFFT:
    """One padded real buffer with in-place forward (r2c) and backward (c2r)
    plans over its full shape. ``r`` is the logical (nx,ny,nz) real view and
    ``c`` the (nx,ny,nz//2+1) complex view of the *same* memory."""

    def __init__(self, shape, threads, flags=None):
        nx, ny, nz = (int(n) for n in shape)
        nc = nz // 2 + 1
        flags = flags or _plan_flags()
        tl = _plan_timelimit()
        self.buf = pyfftw.empty_aligned((nx, ny, 2 * nc), dtype="float32")
        self.r = self.buf[:, :, :nz]
        self.c = self.buf.view("complex64")
        _load_wisdom()
        self.fwd = pyfftw.FFTW(self.r, self.c, axes=(0, 1, 2), direction="FFTW_FORWARD",
                               flags=flags, threads=threads, planning_timelimit=tl)
        # ``normalise_idft`` is applied on the backward call, matching scipy's 1/N^3.
        self.bwd = pyfftw.FFTW(self.c, self.r, axes=(0, 1, 2), direction="FFTW_BACKWARD",
                               flags=flags, threads=threads, planning_timelimit=tl)
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

    Mirrors :meth:`FFTSolverCPU._compute_displacement_iterative_potential` but
    transforms a single reused buffer. Returns an ``(N,N,N,3)`` float32 array
    (the layout the reconstructor's interpolation kernels require).
    """
    delta = np.ascontiguousarray(delta, dtype=np.float32)
    shape = delta.shape
    kx, ky, kz = prepare_k_components(mesh.cell_size, mesh.nmesh)
    kb = (kx[:, None, None], ky[None, :, None], kz[None, None, :])
    inv_k2_bias = build_inv_k2((kx, ky, kz), bias=bias)   # (N,N,nc) real

    axis = getattr(los, "axis", None)
    fft = _InPlaceRFFT(shape, threads=_threads())
    r, c = fft.r, fft.c

    r[...] = delta
    fft.fwd()                       # c = delta_k

    if n_iterations > 0 and axis is None:
        # LocalLOS: stream the radial projection with the versor evaluated on the
        # fly (no gradient/versor/1-norm grids). Geometry from the LOS strategy.
        mc = np.asarray(los.min_corner, dtype=np.float32)
        cs = (np.asarray(los.boxsize, dtype=np.float32) /
              np.asarray(los.nmesh, dtype=np.float32)).astype(np.float32)
        s = np.empty(shape, dtype=np.float32)
        dk_save = np.empty_like(c)
        div_acc = np.empty_like(c)

    for it in range(n_iterations):
        if axis is not None:
            ka = kb[axis]
            # grad_a = irfft(-i k_a inv_k2_bias delta_k)
            c *= inv_k2_bias
            c *= ka
            c *= -_CJ
            fft.bwd()               # r = grad_a
            # correction = irfft(i k_a rfft(grad_a))
            fft.fwd()               # c = rfft(grad_a)
            c *= ka
            c *= _CJ
            fft.bwd()               # r = correction
        else:
            dk_save[...] = c        # preserve delta_k across the 3 components
            for i in range(3):
                c[...] = dk_save
                c *= inv_k2_bias
                c *= kb[i]
                c *= -_CJ
                fft.bwd()           # r = grad_i
                _proj_acc(s, r, i, mc[0], mc[1], mc[2], cs[0], cs[1], cs[2], i == 0)
            div_acc[...] = 0
            for j in range(3):
                _scatter(r, s, j, mc[0], mc[1], mc[2], cs[0], cs[1], cs[2])
                fft.fwd()           # c = rfft(s * n_hat_j)
                c *= kb[j]
                c *= _CJ
                div_acc += c
            c[...] = div_acc
            fft.bwd()               # r = correction

        r *= -f
        if it == 0:
            r /= (1 + beta)
        r += delta
        fft.fwd()                   # c = delta_k for the next iteration / final build

    # Drop the LocalLOS scratch grids before the (3-grid) displacement build so
    # they do not stack with the output.
    if n_iterations > 0 and axis is None:
        del s, dk_save, div_acc

    # Final displacement build. delta_k must survive all three components, so it
    # is copied out once (1 grid) and reloaded per component.
    dk = np.array(c, copy=True)
    displacement = np.empty(shape + (3,), dtype=np.float32)
    for i in range(3):
        c[...] = dk
        c *= inv_k2_bias
        c *= kb[i]
        c *= _CJ
        fft.bwd()
        displacement[..., i] = r
    return displacement
