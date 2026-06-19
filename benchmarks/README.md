# baorecon benchmarks

Profiling scripts that compare **baorecon** (the `zeldareco` package, CPU backend)
against **[pyrecon](https://github.com/cosmodesi/pyrecon)** on the core
reconstruction steps.

All benchmarks run on synthetic data only: uniform random positions in a cubic box
(or a uniform RA/DEC/z sky patch for the pipeline benchmark), with unit weights —
no real catalogs are needed.

## Layout

```
benchmarks/
├── bench_common.py            # shared helpers (mock data, timing, memory, subprocess plumbing, CSV/table)
├── bench_mass_assignment.py   # CIC / TSC mass assignment
├── bench_fft_solver.py        # iterative FFT solver (setup / solve / readout)
├── bench_pipeline.py          # end-to-end reconstruction
├── bench_scaling.py           # mesh-resolution scaling (baorecon CPU)
├── plot_results.py            # CSV -> figures
├── results/                   # CSV output (git-ignored except .gitkeep)
└── figures/                   # PDF output (git-ignored except .gitkeep)
```

## Requirements

- `zeldareco` (this repo) installed/importable
- `numpy`, `scipy`, `numba`
- `pyrecon` + `pmesh` (already available in the project environment)
- `matplotlib`, `pandas` (for `plot_results.py`)

## Running

From the repository root:

```bash
python benchmarks/bench_mass_assignment.py
python benchmarks/bench_fft_solver.py
python benchmarks/bench_pipeline.py
python benchmarks/bench_scaling.py
python benchmarks/plot_results.py        # after the above have written CSVs
```

Each `bench_*` script accepts:

- `--quick` — a tiny parameter set (small N and mesh) for a fast sanity check;
- `--repeats N` — number of timed repetitions (default **5**).

```bash
python benchmarks/bench_mass_assignment.py --quick      # fast smoke run
python benchmarks/bench_fft_solver.py --repeats 10
```

> **Heavy configurations.** The full sweeps include very demanding points:
> `1e8` particles (mass assignment / pipeline) and `nmesh=2048` (scaling). The
> `2048^3` FFT solve in particular needs tens of GB of RAM; configurations
> that run out of memory are caught, reported as `[skip]`, and left out of the CSV
> instead of aborting the whole sweep. Start with `--quick` to validate your setup.

## What each script measures

| Script | Sweep | Fixed | Backends | `step` values |
|---|---|---|---|---|
| `bench_mass_assignment.py` | N = 1e5…1e8 | nmesh = 256 | baorecon CPU, pyrecon | `CIC`, `TSC` |
| `bench_fft_solver.py` | nmesh = 256/512/1024 | N = 1e7 | baorecon CPU, pyrecon | `setup`, `solve`, `readout` |
| `bench_pipeline.py` | N = 1e5…1e7 | nmesh = 512 | baorecon CPU, pyrecon | `total` |
| `bench_scaling.py` | nmesh = 128…2048 | N = 1e7 | baorecon CPU | `mass_assignment`, `fft_solver` |

Backend ↔ implementation:

- **baorecon (CPU)** — `zeldareco.mass_assignment.mass_assignment` (Numba),
  `DensityManager` + `FFTSolver` (`RedshiftSpace` → iterative potential, 3
  iterations), and `BAOReconstructor.run_reconstruction` for the pipeline.
- **pyrecon** — `pmesh` `RealField.paint` with the matching resampler for mass
  assignment; `IterativeFFTReconstruction` (`assign_*` / `set_density_contrast`,
  `run(niterations=3)`, `read_shifts` / `read_shifted_positions`) for the solver
  and pipeline.

### Methodology

- **Process isolation.** Each `(backend, n_particles, nmesh)` configuration runs
  in its own Python subprocess: the parent process (which never imports zeldareco
  or pyrecon) launches one `python bench_*.py --worker <json-spec>` per
  configuration, and each worker prints its rows as JSON on stdout for the parent
  to aggregate into the CSV. This guarantees one backend's peak `ru_maxrss` is
  never inflated by another's.
- Times use `time.perf_counter`; each measurement is repeated 5× (default) and the
  CSV stores the **mean** and **standard deviation**.
- Numba kernels are JIT-compiled on `measure()`'s per-call warmup, run once
  inside each worker before its timed loop (and excluded from the reported
  timings). Because a worker runs a single configuration, that warmup primes
  exactly the kernels it then measures — no separate global warmup is needed.
- **Memory** is the process peak resident set size,
  `resource.getrusage(RUSAGE_SELF).ru_maxrss` (KiB on Linux, divided by 1024 for
  MiB). Each worker records a baseline right after its imports and mock-data setup
  (`reset_memory_baseline`); `memory_peak_mb` is the peak's growth above that
  baseline (the memory the step needed) and `memory_total_mb` is its absolute
  value (the process peak).

## Output

### CSV (`results/*.csv`)

Each file starts with a commented provenance header (Python, NumPy, CPU model, …)
followed by columns:

```
backend, n_particles, nmesh, step, time_mean, time_std, memory_peak_mb, memory_total_mb
```

`time_*` are in seconds. `memory_peak_mb` is the peak-RSS increase caused by the
step and `memory_total_mb` is the absolute process peak RSS, both in MiB. Read
them back with `pandas.read_csv(path, comment="#")`.

### Figures (`figures/*.pdf`)

`plot_results.py` reads the CSVs and writes:

1. `fig1_mass_assignment_time.pdf` — time vs N particles (CPU / pyrecon).
2. `fig2_fft_solver_time.pdf` — total FFT-solver time vs nmesh (CPU / pyrecon).
3. `fig3_pipeline_time.pdf` — end-to-end time vs N particles (CPU / pyrecon).
4. `fig5_memory.pdf` — peak-RSS increase per step vs N particles (CPU / pyrecon).

Missing backends are omitted from the curves.

## Notes & caveats on fairness

- For a like-for-like comparison the pmesh / pyrecon meshes use **float32**
  (`dtype='f4'`), matching baorecon's default precision.
- The FFT-solver and pipeline benchmarks use **data and randoms 1:1** for both
  backends.
- For the **pipeline**, all backends start from the same pre-computed Cartesian
  positions: the RA/DEC/z → XYZ coordinate conversion is done once, outside every
  timer, so it is excluded for baorecon and pyrecon alike. The baorecon timed
  region is `run_reconstruction`; the pyrecon one starts at the assignment phase.
- Memory uses `ru_maxrss`, the resident set size of the **whole process** and
  allocator-agnostic, so it captures pmesh/pyrecon allocations as well as
  baorecon's NumPy grids. Running each backend in its own subprocess keeps these
  readings independent — the earlier single-process design let one backend's peak
  contaminate the next (e.g. TSC right after CIC reporting a ~0 delta). It is still
  a monotonic high-water mark *within* a worker: when a worker times several steps
  (`setup`/`solve`/`readout`, or `mass_assignment`/`fft_solver`), each step's
  `memory_peak_mb` reflects the cumulative peak reached up to that point, so later
  steps include earlier ones' footprint. `memory_total_mb` is the absolute process
  peak.
