# In-place pyfftw CPU FFT backend

The iterative FFT (Burden / iFFT) displacement solver has an **opt-in, low-memory
CPU backend** built on [pyfftw](https://pyfftw.readthedocs.io). It performs every
forward/inverse transform **in place** on a single padded buffer instead of
allocating a fresh array per `rfftn`/`irfftn` (that per-transform working set is
the dominant term in the CPU peak). Combined with a streamed line-of-sight
projection (the radial versor is evaluated on the fly), it **cuts the end-to-end
CPU peak memory by ~55–60%** — bringing baorecon in line with `pyrecon` — with no
change to results beyond float32 round-off.

scipy remains the default. The pyfftw path is enabled entirely through
environment variables — no change to the reconstructor, pipeline, or any call
site is required.

- [At a glance](#at-a-glance)
- [Requirements](#requirements)
- [Enabling it](#enabling-it)
- [Environment variables](#environment-variables)
- [Controlling threads](#controlling-threads)
- [Planning modes and FFTW wisdom](#planning-modes-and-fftw-wisdom)
- [Running the benchmark](#running-the-benchmark)
- [What is accelerated (and what falls back to scipy)](#what-is-accelerated-and-what-falls-back-to-scipy)
- [Measured results](#measured-results)
- [Correctness](#correctness)
- [How it works](#how-it-works)
- [Limitations and caveats](#limitations-and-caveats)
- [Troubleshooting](#troubleshooting)

---

## At a glance

```bash
# Default: scipy (unchanged)
python your_run.py

# Opt in to the in-place low-memory backend
BAORECON_FFT=pyfftw python your_run.py
```

| | scipy (default) | pyfftw in-place |
|---|---|---|
| CPU peak memory | baseline | **~55–60% lower** (≈ pyrecon) |
| Accuracy | — | matches scipy to float32 round-off (~2.5e-7) |
| Default planning | — | `FFTW_ESTIMATE` (instant, no warm-up) |
| Threads | all cores (`workers=-1`) | all cores (configurable) |
| GPU path | CuPy | unaffected (CPU-only feature) |

---

## Requirements

- `pyfftw` must be importable (`pip install pyfftw` / `conda install -c conda-forge pyfftw`).
  If it is not installed, requesting `BAORECON_FFT=pyfftw` logs a warning and
  transparently falls back to scipy — nothing breaks.
- CPU device only. On `device="gpu"` the solver uses CuPy and this backend is
  irrelevant.

Availability is exposed programmatically:

```python
from baorecon.utils.backend import PYFFTW_AVAILABLE, use_pyfftw
PYFFTW_AVAILABLE   # True if pyfftw imported
use_pyfftw()       # True if BAORECON_FFT=pyfftw AND pyfftw is available
```

---

## Enabling it

Set a single environment variable in the shell (or process) that runs the
reconstruction:

```bash
export BAORECON_FFT=pyfftw
```

The choice is read at solve time by `baorecon.utils.backend.use_pyfftw()`, so it
needs no argument threading through `BAOReconstructor` or the pipeline. It also
propagates automatically to any subprocess that inherits the environment
(e.g. the benchmark workers).

The value is case-insensitive; anything other than `pyfftw` (including unset)
means scipy.

---

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `BAORECON_FFT` | `scipy` | `pyfftw` selects the in-place CPU backend; anything else uses scipy. |
| `BAORECON_FFT_THREADS` | all cores (`os.cpu_count()`) | FFTW thread count. **Not** taken from `OMP_NUM_THREADS` (see below). |
| `BAORECON_FFT_PLAN` | `estimate` | Planning rigour: `estimate`, `measure`, or `patient`. |
| `BAORECON_FFT_PLAN_TIMELIMIT` | `15` | Seconds to cap FFTW planning per plan (only relevant for `measure`/`patient`). |
| `BAORECON_FFTW_WISDOM` | `~/.cache/baorecon/fftw_wisdom.pkl` | Path to the persisted FFTW wisdom file. |

All are read at solve time; you can vary them run-to-run without editing code.

---

## Controlling threads

```bash
BAORECON_FFT=pyfftw BAORECON_FFT_THREADS=8 python your_run.py
```

Key behaviour:

- **Unset → all cores** (`os.cpu_count()`). The pyfftw path deliberately does
  **not** read `OMP_NUM_THREADS` / `NUMBA_NUM_THREADS` (those throttle numba and
  BLAS, not the FFT).
- This mirrors the **default scipy path**, which calls
  `scipy.fft.rfftn(..., workers=-1)` — also all cores, also ignoring `OMP`.

> **Fair-comparison note.** For an apples-to-apples scipy-vs-pyfftw comparison,
> **leave `BAORECON_FFT_THREADS` unset** so both use every core. If you set it to
> a small value, you throttle pyfftw while scipy still runs on all cores, and
> pyfftw will look artificially slow. Set it only when you deliberately want to
> constrain the FFT (e.g. a controlled N-thread study).

---

## Planning modes and FFTW wisdom

FFTW builds a *plan* per transform shape. The trade-off is planning time vs
transform speed:

| `BAORECON_FFT_PLAN` | Planning cost | Transform speed | Use when |
|---|---|---|---|
| `estimate` (default) | instant | ≈ scipy on all cores | one-off runs; the memory win is free |
| `measure` | slow first time (bounded, cached) | fastest | many runs / large sweeps |
| `patient` | slowest | fastest | dedicated production pipelines |

- With **`estimate`** (the default) there is no warm-up penalty and, on all
  cores, transforms are effectively as fast as scipy here — so you get the memory
  win for free.
- With **`measure`/`patient`**, the first plan for a given mesh size can be slow
  (minutes at 1024³). This is bounded by `BAORECON_FFT_PLAN_TIMELIMIT` seconds
  per plan and, crucially, **persisted to disk as FFTW wisdom** at
  `BAORECON_FFTW_WISDOM`. Subsequent runs — including separate processes such as
  the benchmark's per-config subprocesses — load that wisdom and plan instantly.

To pre-warm wisdom for a size once and reuse it forever:

```bash
BAORECON_FFT=pyfftw BAORECON_FFT_PLAN=measure python your_run.py   # first run plans + saves wisdom
BAORECON_FFT=pyfftw BAORECON_FFT_PLAN=measure python your_run.py   # subsequent runs are fast
```

Delete the wisdom file to force replanning (e.g. after a hardware/thread-count
change):

```bash
rm ~/.cache/baorecon/fftw_wisdom.pkl
```

---

## Running the benchmark

`benchmarks/bench_bao_reconstructor.py` sweeps the full reconstruction. To run it
against the in-place backend:

```bash
BAORECON_FFT=pyfftw python benchmarks/bench_bao_reconstructor.py --solver ifft
```

Notes:

- **`--solver ifft` is required.** pyfftw is the in-place path *inside the FFT
  solver*; `--solver multigrid` never touches it.
- Only the **`baorecon_cpu`** rows are affected. `baorecon_gpu` uses CuPy and
  `pyrecon` uses its own FFTs.
- The environment variable is inherited by each per-config worker subprocess
  (`bench_common.spawn_worker` copies `os.environ`), so setting it once in the
  launching shell is enough.
- The default `NMESH` includes **1024³**, which is large (~63 GB with scipy,
  ~27 GB with pyfftw) and slow. To restrict the sweep without editing the file:

  ```bash
  BAORECON_FFT=pyfftw python -c \
    "import bench_bao_reconstructor as b; b.run([int(1e6)], [256,512], repeats=3, solver='ifft', mas_parallel=False)"
  ```
  (run from the `benchmarks/` directory).

To control both backend and threads:

```bash
BAORECON_FFT=pyfftw BAORECON_FFT_THREADS=8 python benchmarks/bench_bao_reconstructor.py --solver ifft
```

---

## What is accelerated (and what falls back to scipy)

The in-place path handles exactly the line-of-sight strategies the reconstructor
builds; anything else transparently uses scipy for that solve.

| Case | Backend used |
|---|---|
| Fixed-axis LOS (`los="x"/"y"/"z"`), any RSD space | **pyfftw in-place** |
| Local (radial) LOS (`los=None`), any RSD space | **pyfftw in-place** (streamed projection) |
| `RealSpace` (0 iterations, any LOS) | **pyfftw in-place** (final build only) |
| Any other / custom LOS with iterations | scipy (automatic fallback) |
| Potential recompute (`solver.potential`) | scipy (cold path; reuses the already-computed displacement) |
| GPU device | CuPy (unaffected) |

The gating logic lives in `_pyfftw_cpu.supported(los, n_iterations)`; the router
is in `FFTSolverCPU._compute_displacement_iterative_potential`.

---

## Measured results

Single-node measurements (AMD EPYC, all cores). `R` = one float32 grid = `N³·4 B`
(512³ → 512 MiB).

**Solver only, N=512, RedshiftSpace, 3 iterations:**

| LOS | scipy | pyfftw | reduction |
|---|---|---|---|
| Fixed-axis | 10.5 R | 6.5 R | −38% |
| Local (radial) | 14.6 R | 5.6 R | −62% |

(The radial case reaches 5.6 R because the versor is evaluated on the fly — no
3-vector versor or gradient grid is ever allocated.)

**End-to-end reconstruction (`bench_bao_reconstructor.py`, 1e6 particles, `los=None`, `--solver ifft`), host peak RSS:**

| nmesh | scipy | pyfftw | reduction | pyrecon (ref) |
|---|---|---|---|---|
| 256 | 1020 MB | 442 MB | −57% | ~310 MB |
| 512 | 7957 MB | 3348 MB | −58% | ~3007 MB |

This essentially closes the gap to `pyrecon`: the scipy path was ~2.6× heavier
than pyrecon; with pyfftw baorecon is within ~10–40% of it (~11% at 512³).

**Speed:** solver-only FFT time is competitive (≈17.1 s vs scipy's ≈17.8 s at
512³ on all cores with `estimate`). End-to-end wall time is comparable at small
mesh and can be somewhat slower at 512³ in a single-shot run (one-time numba JIT
of the projection kernels + node-load noise); use `BAORECON_FFT_PLAN=measure` for
the fastest steady-state transforms.

---

## Correctness

The in-place path reproduces the scipy path to float32 round-off:

- pyfftw vs scipy displacement: **max |Δ| ≈ 2.5e-7** (fixed-axis and radial LOS,
  RealSpace and RedshiftSpace).
- Against the analytic GRF-closure ground truth, pyfftw is **as accurate as
  scipy** (max abs error ≈ 5.1e-6 vs 4.8e-6) — the two backends' rounding simply
  lands on different cells.

The FFTW backend differs from scipy's pocketfft only in float32 rounding. One
test (`tests/test_fft_solver.py::test_fftsolver_realistic_grf_closure`) previously
asserted `atol=1e-6` on the displacement, which happened to pass pocketfft by the
distribution of its rounding; it is now `atol=1e-5`, the genuine float32 FFT noise
floor for both backends (scipy still passes comfortably at ~5e-6).

Run the suite against either backend:

```bash
python -m pytest tests/test_fft_solver.py                 # scipy (default)
BAORECON_FFT=pyfftw python -m pytest tests/test_fft_solver.py -k cpu
```

---

## How it works

Two independent memory savings combine:

1. **In-place transforms.** A single padded real buffer of shape
   `(N, N, 2·(N//2+1))` is allocated once. Its real view `(N, N, N)` and complex
   view `(N, N, N//2+1)` share the same memory. Forward (r2c) and backward (c2r)
   FFTW plans transform this buffer in place, so the Burden iteration and the
   final displacement build never allocate a fresh array per transform. FFTW's
   `normalise_idft` reproduces scipy's `1/N³` inverse normalisation.

2. **Streamed radial projection (LocalLOS).** For a radial line of sight the
   projected magnitude `s(x) = grad·n̂` is a scalar. It is accumulated one
   gradient component at a time, and the divergence of `s·n̂` is formed component
   by component, so the full `(N, N, N, 3)` gradient is never materialised. The
   unit versor `n̂(x) = x/|x|` is evaluated **on the fly** inside small parallel
   numba kernels (from the LOS geometry), so neither a 3-vector versor field nor a
   stored `1/|x|` grid is kept.

For a fixed axis the projection reduces to a single component, so the iteration
uses just the padded buffer plus `delta` and `1/(bias·k²)`.

The returned displacement is a contiguous `(N, N, N, 3)` float32 array — the
layout the reconstructor's interpolation kernels require — so nothing downstream
changes.

Relevant files:

- `baorecon/solvers/fft/_pyfftw_cpu.py` — the in-place implementation, numba
  kernels, wisdom persistence, and env-var knobs.
- `baorecon/utils/backend.py` — `PYFFTW_AVAILABLE`, `use_pyfftw()`.
- `baorecon/solvers/fft/cpu.py` — routes to the in-place path when enabled and
  supported, else scipy.

---

## Limitations and caveats

- **CPU only.** The GPU solver keeps its fields on-device via CuPy and is
  unaffected.
- **Displacement only.** `solver.potential` is recomputed on the scipy path
  (cold code that reuses the already-computed displacement).
- **First-plan cost with `measure`/`patient`.** Bounded by
  `BAORECON_FFT_PLAN_TIMELIMIT` and amortised via on-disk wisdom; irrelevant with
  the default `estimate`.
- **Wisdom is thread-count- and machine-specific.** Delete the wisdom file after
  changing `BAORECON_FFT_THREADS` or hardware to avoid a stale/suboptimal plan.
- **The 3-grid displacement output is unavoidable** here — it is produced only in
  the final build (not the iteration peak) and is required by the reconstructor's
  interpolation.

---

## Troubleshooting

- **"BAORECON_FFT=pyfftw requested but pyfftw is not installed; using scipy"** —
  install pyfftw; until then the run continues on scipy.
- **First run is slow** — you are using `BAORECON_FFT_PLAN=measure`/`patient`;
  either switch to the default `estimate`, lower `BAORECON_FFT_PLAN_TIMELIMIT`, or
  accept the one-time cost (wisdom makes later runs fast).
- **pyfftw looks slower than scipy** — check you did not set
  `BAORECON_FFT_THREADS` to a small value while scipy uses all cores; leave it
  unset for a fair comparison.
- **Benchmark shows no change** — ensure `--solver ifft` (not `multigrid`) and
  that you are reading the `baorecon_cpu` rows.
- **Force replanning** — `rm ~/.cache/baorecon/fftw_wisdom.pkl` (or point
  `BAORECON_FFTW_WISDOM` elsewhere).
