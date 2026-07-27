"""Shared helpers for the baorecon benchmark suite.

This module centralises everything the individual ``bench_*.py`` scripts need:

* mock-data generation (uniform positions / RA-DEC-z patches, unit weights),
* a single timing+memory measurement primitive that honours the project's
  benchmarking conventions (Numba/CUDA warmup, GPU synchronisation, peak resident
  set size via ``resource.getrusage`` for CPU memory, the CuPy memory pool for GPU
  memory),
* the parent/worker subprocess plumbing (:func:`spawn_worker` / :func:`run_worker`)
  that runs **each backend in its own Python process** so that ``ru_maxrss``
  measures one backend's peak memory without contamination from the others,
* CSV writing (with a commented provenance header) and terminal table printing.

Nothing here imports or mutates the ``baorecon`` package; the benchmark scripts
do that themselves (and only inside their worker subprocesses, never the parent).
"""

from __future__ import annotations

import csv
import gc
import io
import json
import os
import platform
import resource
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# --- Optional GPU stack ------------------------------------------------------
# We consider the GPU usable only when both CuPy imports *and* Numba sees a CUDA
# device, because that is exactly the condition under which baorecon enables its
# ``device='gpu'`` code paths.
try:
    import cupy as cp  # type: ignore

    try:
        from numba import cuda as _numba_cuda  # type: ignore

        GPU_AVAILABLE = bool(_numba_cuda.is_available())
    except Exception:  # pragma: no cover - numba present but cuda probing failed
        GPU_AVAILABLE = False
except Exception:  # pragma: no cover - cupy not installed
    cp = None  # type: ignore
    GPU_AVAILABLE = False


# Canonical column order for every CSV the suite produces.
#   memory_peak_mb  -- memory attributable to the step (RSS delta / GPU pool delta)
#   memory_total_mb -- absolute peak of the whole process (RSS / GPU pool usage)
#   vram_peak_mb    -- GPU device-memory peak (CuPy pool high-water); 0.0 on CPU
CSV_COLUMNS: Tuple[str, ...] = (
    "backend",
    "n_particles",
    "nmesh",
    "step",
    "time_mean",
    "time_std",
    "memory_peak_mb",
    "memory_total_mb",
    "vram_peak_mb",
)

# Default cubic box side (Mpc/h) used by the box-based benchmarks.
BOXSIZE = 1000.0

# Output directory for CSVs. Override with BAORECON_BENCH_RESULTS (absolute path,
# or a name relative to benchmarks/) to keep per-node runs from clobbering each
# other, e.g. BAORECON_BENCH_RESULTS=results_teogpu02.
_RESULTS_ENV = os.environ.get("BAORECON_BENCH_RESULTS", "results")
RESULTS_DIR = Path(_RESULTS_ENV)
if not RESULTS_DIR.is_absolute():
    RESULTS_DIR = Path(__file__).resolve().parent / RESULTS_DIR
FIGURES_DIR = Path(__file__).resolve().parent / "figures"
# Repo root (parent of benchmarks/); put on a worker's PYTHONPATH so it can import
# baorecon even from a plain checkout that was never ``pip install``-ed.
REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Provenance / system information
# ---------------------------------------------------------------------------
def _cpu_model() -> str:
    """Best-effort human-readable CPU model name."""
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine() or "unknown"


def _gpu_model() -> Optional[str]:
    """Name of CUDA device 0, or ``None`` when no GPU is available."""
    if not GPU_AVAILABLE:
        return None
    try:
        name = cp.cuda.runtime.getDeviceProperties(0)["name"]
        return name.decode() if isinstance(name, bytes) else str(name)
    except Exception:  # pragma: no cover - defensive
        return "unknown CUDA device"


def system_info() -> Dict[str, str]:
    """Collect the versions / hardware identifiers stamped on every CSV."""
    info: Dict[str, str] = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "cpu": _cpu_model(),
    }
    if cp is not None:
        info["cupy"] = cp.__version__
    info["gpu"] = _gpu_model() or "none"
    try:  # a useful extra; not required by the spec
        import pyrecon  # type: ignore

        info["pyrecon"] = pyrecon.__version__
    except Exception:
        pass
    return info


def _header_comment_lines(info: Dict[str, str]) -> List[str]:
    """Render system info as ``# key: value`` comment lines for a CSV header."""
    lines = ["# baorecon benchmark results"]
    lines.append("# generated: " + time.strftime("%Y-%m-%d %H:%M:%S"))
    for key, value in info.items():
        lines.append(f"# {key}: {value}")
    return lines


# ---------------------------------------------------------------------------
# Mock data
# ---------------------------------------------------------------------------
def gen_positions(n_particles: int, boxsize: float = BOXSIZE, seed: int = 42,
                  dtype=np.float32) -> Tuple[np.ndarray, np.ndarray]:
    """Uniform random positions in ``[0, boxsize)^3`` with unit weights.

    Returns ``(positions[n, 3], weights[n])``.
    """
    n = int(n_particles)
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0.0, boxsize, size=(n, 3)).astype(dtype, copy=False)
    weights = np.ones(n, dtype=dtype)
    return pos, weights


def gen_radec_z(n_particles: int, seed: int = 42,
                ra_range: Tuple[float, float] = (150.0, 200.0),
                dec_range: Tuple[float, float] = (-5.0, 35.0),
                z_range: Tuple[float, float] = (0.9, 1.1),
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Uniform RA/DEC/redshift mock over a rectangular sky patch + redshift slab."""
    n = int(n_particles)
    rng = np.random.default_rng(seed)
    ra = rng.uniform(*ra_range, size=n)
    dec = rng.uniform(*dec_range, size=n)
    z = rng.uniform(*z_range, size=n)
    return ra, dec, z


# ---------------------------------------------------------------------------
# Measurement primitive
# ---------------------------------------------------------------------------
@dataclass
class Measurement:
    time_mean: float
    time_std: float
    memory_peak_mb: float   # memory used by the step (peak RSS delta around it)
    memory_total_mb: float  # absolute peak RSS of the whole process
    vram_peak_mb: float = 0.0  # GPU device-memory peak (CuPy pool high-water); 0.0 on CPU

    def as_row(self, backend: str, n_particles: int, nmesh: int, step: str) -> Dict[str, object]:
        return {
            "backend": backend,
            "n_particles": int(n_particles),
            "nmesh": int(nmesh),
            "step": step,
            "time_mean": self.time_mean,
            "time_std": self.time_std,
            "memory_peak_mb": self.memory_peak_mb,
            "memory_total_mb": self.memory_total_mb,
            "vram_peak_mb": self.vram_peak_mb,
        }


def _sync(device: str) -> None:
    if device == "gpu" and cp is not None:
        cp.cuda.Stream.null.synchronize()


# Process peak-RSS baseline (KiB) against which CPU memory deltas are reported.
# It is set once per worker, after the heavy imports and mock-data setup, via
# :func:`reset_memory_baseline`; every ``measure`` call then reports memory growth
# relative to it. Because each backend runs in its own process (see
# :func:`spawn_worker`), this baseline -- and the ``ru_maxrss`` high-water mark it
# is compared against -- belong to that backend alone.
_RSS_BASELINE_KB: Optional[int] = None


def _ru_maxrss_kb() -> int:
    """Process peak resident set size in KiB (the unit of ``ru_maxrss`` on Linux)."""
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def reset_memory_baseline() -> int:
    """Record the peak-RSS baseline for subsequent :func:`measure` deltas.

    Call this once inside a worker, *after* importing the backend and generating
    the mock data but *before* the first timed step, so the reported per-step
    deltas reflect the benchmark's own working memory rather than the interpreter,
    libraries, or input arrays.
    """
    global _RSS_BASELINE_KB
    gc.collect()
    _RSS_BASELINE_KB = _ru_maxrss_kb()
    return _RSS_BASELINE_KB


def _memory_baseline_kb() -> int:
    if _RSS_BASELINE_KB is None:
        return reset_memory_baseline()
    return _RSS_BASELINE_KB


def measure(fn: Callable[[], object], repeats: int = 5, warmup: bool = True,
            device: str = "cpu") -> Measurement:
    """Time ``fn`` ``repeats`` times and measure its peak memory.

    * ``fn`` takes no arguments and returns its result.
    * A warmup call is issued first so Numba JIT / CUDA kernel compilation is not
      counted in the timings.
    * For ``device='gpu'`` the GPU is synchronised before every ``perf_counter``
      reading and memory comes from the CuPy default pool (delta + absolute).
    * Otherwise CPU memory comes from the process peak resident set size,
      ``resource.getrusage(RUSAGE_SELF).ru_maxrss`` (KiB on Linux). It is a
      monotonic high-water mark, so ``memory_peak_mb`` is its growth above the
      worker's :func:`reset_memory_baseline` (the memory this step required) and
      ``memory_total_mb`` is its absolute value (the total process peak). Running
      each backend in its own process keeps these readings uncontaminated.
    """
    if warmup:
        result = fn()
        _sync(device)
        del result

    times: List[float] = []
    for _ in range(repeats):
        _sync(device)
        start = time.perf_counter()
        result = fn()
        _sync(device)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        del result

    if device == "gpu" and cp is not None:
        pool = cp.get_default_memory_pool()
        pool.free_all_blocks()
        baseline = pool.used_bytes()
        result = fn()
        _sync(device)
        used_after = pool.used_bytes()
        # VRAM peak: the pool only returns memory to the device on
        # free_all_blocks(), so total_bytes() = used + cached-free tracks the
        # largest footprint the pool reached during the run -- the true peak of
        # CuPy allocations, unlike used_bytes() which only sees what is still
        # live at this instant. (Non-pool allocations -- cuFFT plan scratch,
        # numba device arrays -- are not pooled and thus not counted here.)
        vram_peak_mb = pool.total_bytes() / 1e6
        del result
        pool.free_all_blocks()
        memory_step_mb = (used_after - baseline) / 1e6
        memory_total_mb = used_after / 1e6
    else:
        # ru_maxrss is the process high-water mark, so the timed loop above has
        # already pushed it to this step's peak; compare it to the worker baseline.
        rss_after_kb = _ru_maxrss_kb()
        # ru_maxrss is in KiB on Linux -> divide by 1024 for MiB.
        memory_step_mb = max(0.0, (rss_after_kb - _memory_baseline_kb()) / 1024.0)
        memory_total_mb = rss_after_kb / 1024.0
        vram_peak_mb = 0.0

    return Measurement(
        time_mean=float(np.mean(times)),
        time_std=float(np.std(times)),
        memory_peak_mb=float(memory_step_mb),
        memory_total_mb=float(memory_total_mb),
        vram_peak_mb=float(vram_peak_mb),
    )


def safe_measure(fn: Callable[[], object], *, label: str, repeats: int = 5,
                 warmup: bool = True, device: str = "cpu") -> Optional[Measurement]:
    """Like :func:`measure` but converts failures (e.g. OOM) into a skip + ``None``.

    This keeps a single heavy configuration (a huge mesh that exhausts RAM/VRAM)
    from aborting the whole sweep. Note Numba/CUDA kernels are JIT-compiled on the
    per-call warmup inside :func:`measure`; in the per-backend worker process this
    fully primes the exact kernels that configuration uses, so no separate global
    warmup is needed.
    """
    try:
        return measure(fn, repeats=repeats, warmup=warmup, device=device)
    except Exception as exc:  # noqa: BLE001 - we genuinely want to swallow anything
        print(f"  [skip] {label}: {type(exc).__name__}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Parent / worker subprocess plumbing
# ---------------------------------------------------------------------------
# Each (backend, n_particles, nmesh) configuration is run in its own Python
# process so that ``ru_maxrss`` (a per-process high-water mark) measures one
# backend's peak memory without the others inflating it. The parent only spawns
# workers and aggregates their JSON output; it never imports baorecon/pyrecon.
WORKER_FLAG = "--worker"

# Workers emit their rows on a single stdout line prefixed with this marker, so
# the parent can pick the payload out from any logging the backend prints.
_RESULT_MARKER = "__BENCH_RESULT__ "


def is_worker(argv: Optional[Sequence[str]] = None) -> bool:
    """True when the current invocation is a measurement worker subprocess."""
    argv = sys.argv if argv is None else argv
    return WORKER_FLAG in argv


def run_worker(worker_fn: Callable[[Dict[str, object]], Sequence[Dict[str, object]]]) -> None:
    """Worker entry point: parse the spec, run ``worker_fn``, emit rows as JSON.

    ``worker_fn`` receives the spec dict (everything needed to reproduce one
    configuration: ``backend``, ``n_particles``, ``nmesh``, ``repeats``, …),
    imports its own backend, calls :func:`reset_memory_baseline` after those
    imports + mock-data setup, and returns a list of CSV-row dicts.
    """
    idx = sys.argv.index(WORKER_FLAG)
    spec = json.loads(sys.argv[idx + 1])
    rows = list(worker_fn(spec))
    sys.stdout.write(_RESULT_MARKER + json.dumps(rows) + "\n")
    sys.stdout.flush()


def spawn_worker(script: str, spec: Dict[str, object], *, label: str,
                 timeout: Optional[float] = None) -> List[Dict[str, object]]:
    """Run ``script`` in a fresh process for one spec and return the rows it emits.

    Failures (non-zero exit, an OOM kill, a timeout, or malformed output) are
    reported and turned into an empty list so a single heavy/failing backend does
    not abort the whole sweep.
    """
    cmd = [sys.executable, str(script), WORKER_FLAG, json.dumps(spec)]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(REPO_ROOT), env.get("PYTHONPATH", "")) if p)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                              env=env)
    except subprocess.TimeoutExpired:
        print(f"  [skip] {label}: timed out after {timeout:.0f}s")
        return []
    except Exception as exc:  # noqa: BLE001 - defensive
        print(f"  [skip] {label}: {type(exc).__name__}: {exc}")
        return []

    if proc.returncode != 0:
        tail = [ln for ln in proc.stderr.strip().splitlines() if ln.strip()]
        extra = f" -- {tail[-1]}" if tail else ""
        print(f"  [skip] {label}: worker exited with code {proc.returncode}{extra}")
        return []

    for line in proc.stdout.splitlines():
        if line.startswith(_RESULT_MARKER):
            try:
                return json.loads(line[len(_RESULT_MARKER):])
            except json.JSONDecodeError as exc:
                print(f"  [skip] {label}: malformed worker output ({exc})")
                return []
    print(f"  [skip] {label}: worker produced no result")
    return []


# ---------------------------------------------------------------------------
# Output: CSV + terminal table
# ---------------------------------------------------------------------------
def save_csv(path: Path, rows: Sequence[Dict[str, object]],
             info: Optional[Dict[str, str]] = None) -> None:
    """Write ``rows`` to ``path`` with a commented system-info header."""
    info = info or system_info()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        for line in _header_comment_lines(info):
            handle.write(line + "\n")
        writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in CSV_COLUMNS})
    print(f"\nSaved {len(rows)} rows -> {path}")


def print_table(rows: Sequence[Dict[str, object]], title: str = "") -> None:
    """Pretty-print benchmark rows as an aligned terminal table."""
    if title:
        print(f"\n=== {title} ===")
    if not rows:
        print("(no results)")
        return

    headers = list(CSV_COLUMNS)

    def fmt(col: str, value: object) -> str:
        if value == "" or value is None:
            return "-"
        if col in ("time_mean", "time_std"):
            return f"{float(value):.4e}"
        if col in ("memory_peak_mb", "memory_total_mb", "vram_peak_mb"):
            return f"{float(value):.1f}"
        if col == "n_particles":
            return f"{int(value):.0e}" if int(value) >= 1000 else str(int(value))
        return str(value)

    table = [[fmt(col, row.get(col, "")) for col in headers] for row in rows]
    widths = [max(len(h), *(len(r[i]) for r in table)) for i, h in enumerate(headers)]

    line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    print(line)
    print("  ".join("-" * widths[i] for i in range(len(headers))))
    for r in table:
        print("  ".join(r[i].ljust(widths[i]) for i in range(len(headers))))


def to_csv_buffer(rows: Sequence[Dict[str, object]], info: Dict[str, str]) -> str:
    """Return the CSV (header comments + table) as a string. Handy for previews."""
    buffer = io.StringIO()
    for line in _header_comment_lines(info):
        buffer.write(line + "\n")
    writer = csv.DictWriter(buffer, fieldnames=list(CSV_COLUMNS))
    writer.writeheader()
    for row in rows:
        writer.writerow({col: row.get(col, "") for col in CSV_COLUMNS})
    return buffer.getvalue()
