# baorecon benchmarks

Profiling scripts that compare **baorecon** (the `baorecon` package, CPU and GPU
backends) against **[pyrecon](https://github.com/cosmodesi/pyrecon)** on the core
reconstruction steps.

All benchmarks run on synthetic data only: uniform random positions in a cubic box
(or a uniform RA/DEC/z sky patch for the reconstruction and pipeline-class
benchmarks), with unit weights — no real catalogs are needed.

## Layout

```
benchmarks/
├── bench_common.py            # shared helpers (mock data, timing, memory, subprocess plumbing, CSV/table)
├── bench_mass_assignment.py   # CIC / TSC mass assignment
├── bench_fft_solver.py        # iterative FFT solver (setup / solve / readout)
├── bench_bao_reconstructor.py # end-to-end BAOReconstructor.run_reconstruction
├── bench_pipeline_class.py    # ReconstructionPipeline class (per-stage + total)
├── bench_scaling.py           # mesh-resolution scaling (baorecon CPU vs GPU)
├── plot_results.py            # CSV -> figures (mass assignment / fft / pipeline / speedup / memory)
├── plot_new.py                # CSV -> figures (reconstructor-focused: time + memory comparison)
├── results/                   # CSV output (git-ignored except .gitkeep)
└── figures/                   # PDF output (git-ignored except .gitkeep)
```

## Requirements

- `baorecon` (this repo) installed/importable
- `numpy`, `scipy`, `numba`
- `pyrecon` + `pmesh` (already available in the project environment)
- `matplotlib`, `pandas` (for `plot_results.py`)
- *optional:* `cupy` + a CUDA GPU. When no GPU is detected
  (`numba.cuda.is_available()` is `False`) the GPU rows are **silently skipped** —
  every script still runs and produces CPU + pyrecon results.

## Running

From the repository root:

```bash
python benchmarks/bench_mass_assignment.py
python benchmarks/bench_fft_solver.py
python benchmarks/bench_bao_reconstructor.py
python benchmarks/bench_pipeline_class.py
python benchmarks/bench_scaling.py
python benchmarks/plot_results.py        # after the above have written CSVs
```

Each `bench_*` script accepts:

- `--quick` — a tiny parameter set (small N and mesh) for a fast sanity check;
- `--repeats N` — number of timed repetitions (default **5**, except
  `bench_bao_reconstructor.py` and `bench_pipeline_class.py`, which default to **1**).

```bash
python benchmarks/bench_mass_assignment.py --quick      # fast smoke run
python benchmarks/bench_fft_solver.py --repeats 10
```

`bench_bao_reconstructor.py` additionally accepts:

- `--solver {multigrid,ifft}` — displacement solver (default `multigrid`);
- `--smoother {jacobi,mcgs}` — multigrid smoother, Jacobi V-cycle or multicolor
  Gauss–Seidel (default `jacobi`; ignored by `--solver ifft`);
- `--fft {scipy,pyfftw}` — CPU FFT backend for the ifft solver, wired through the
  `BAORECON_FFT` environment flag (default `scipy`; only affects the CPU ifft path,
  and warns when combined with multigrid / GPU). See
  [../docs/pyfftw_backend.md](../docs/pyfftw_backend.md);
- `--mas_parallel` — enable parallel mass assignment;
- `--skip_pyrecon` — drop the `pyrecon` backend from the sweep (baorecon-only run).

The `baorecon_gpu` backend is only included for `--solver ifft` (the multigrid
solver runs on CPU regardless of `device`), so a `--solver multigrid` run compares
`baorecon_cpu` against `pyrecon` only.

```bash
python benchmarks/bench_bao_reconstructor.py --quick --solver multigrid --smoother mcgs
python benchmarks/bench_bao_reconstructor.py --solver ifft --fft pyfftw
```

The output CSV filename encodes the variant, e.g.
`results/bao_reconstructor_multigrid_mcgs[_gpu].csv` or
`results/bao_reconstructor_ifft_pyfftw[_gpu].csv`.

> **Heavy configurations.** The full sweeps include very demanding points:
> `1e8` particles (mass assignment / pipeline) and `nmesh=2048` (scaling). The
> `2048^3` FFT solve in particular needs tens of GB of RAM (or VRAM); configurations
> that run out of memory are caught, reported as `[skip]`, and left out of the CSV
> instead of aborting the whole sweep. Start with `--quick` to validate your setup.

## What each script measures

| Script | Sweep | Fixed | Backends | `step` values |
|---|---|---|---|---|
| `bench_mass_assignment.py` | N = 1e5…1e8 | nmesh = 256 | baorecon CPU/GPU, pyrecon | `CIC`, `TSC` |
| `bench_fft_solver.py` | nmesh = 128/256/512 | N = 1e6 | baorecon CPU/GPU, pyrecon | `setup`, `solve`, `readout` |
| `bench_bao_reconstructor.py` | nmesh = 128/256/512/1024 | N = 1e6 | baorecon CPU/GPU, pyrecon | `total` |
| `bench_pipeline_class.py` | nmesh = 256/512/1024 | N = 1e6 | baorecon CPU/GPU | `load`, `to_xyz`, `reconstruct`, `convert_back`, `save`, `total` |
| `bench_scaling.py` | nmesh = 128…2048 | N = 1e6 | baorecon CPU/GPU | `mass_assignment`, `fft_solver` |

Backend ↔ implementation:

- **baorecon (CPU/GPU)** — `baorecon.mas.assign` (Numba / CUDA),
  `DensityManager` + `FFTSolverCPU`/`FFTSolverGPU` (`RedshiftSpace` → iterative
  potential, 3 iterations), and `BAOReconstructor.run_reconstruction` for the
  reconstruction benchmark.
- **pyrecon** — `pmesh` `RealField.paint` with the matching resampler for mass
  assignment; `IterativeFFTReconstruction` (`assign_*` / `set_density_contrast`,
  `run(niterations=3)`, `read_shifts` / `read_shifted_positions`) for the solver
  and reconstruction benchmarks.

**`bench_pipeline_class.py`** benchmarks the full `ReconstructionPipeline`
orchestrator (FITS read → RA/DEC/z→XYZ → reconstruction → XYZ→RA/DEC/z → FITS
write) — i.e. the catalog I/O and coordinate-conversion stages that
`bench_bao_reconstructor.py` deliberately excludes. It is **baorecon-only** and reports each stage
(`load` / `to_xyz` / `reconstruct` / `convert_back` / `save`) plus an end-to-end
`total` from the progressive-release `run()`. Every configuration is swept over
two save sets — catalogs only, and catalogs + grid potential/displacement —
encoded in the `backend` column (`baorecon_<device>_cat` / `_grids`); it writes
`results/pipeline_class_<solver>[_gpu].csv` (not yet rendered by
`plot_results.py`). On the GPU, `run()` releases the solver/delta grids mid-run,
so read GPU peak memory from the `reconstruct` stage rather than from `total`.

### Methodology

- **Process isolation.** Each `(backend, n_particles, nmesh)` configuration runs
  in its own Python subprocess: the parent process (which never imports baorecon
  or pyrecon) launches one `python bench_*.py --worker <json-spec>` per
  configuration, and each worker prints its rows as JSON on stdout for the parent
  to aggregate into the CSV. This guarantees one backend's peak `ru_maxrss` is
  never inflated by another's.
- Times use `time.perf_counter`; each measurement is repeated 5× (default) and the
  CSV stores the **mean** and **standard deviation**.
- Numba / CUDA kernels are JIT-compiled on `measure()`'s per-call warmup, run once
  inside each worker before its timed loop (and excluded from the reported
  timings). Because a worker runs a single configuration, that warmup primes
  exactly the kernels it then measures — no separate global warmup is needed.
- GPU work is synchronised with `cp.cuda.Stream.null.synchronize()` before each
  timer reading.
- **CPU memory** is the process peak resident set size,
  `resource.getrusage(RUSAGE_SELF).ru_maxrss` (KiB on Linux, divided by 1024 for
  MiB). Each worker records a baseline right after its imports and mock-data setup
  (`reset_memory_baseline`); `memory_peak_mb` is the peak's growth above that
  baseline (the memory the step needed) and `memory_total_mb` is its absolute
  value (the process peak). **GPU memory** is taken from the CuPy default pool:
  `memory_total_mb` uses `used_bytes()` (live at the end of the run), while the
  separate `vram_peak_mb` column uses `total_bytes()` — the pool high-water mark,
  which captures the true peak because the pool only releases blocks to the device
  on `free_all_blocks()`. Allocations outside the pool (cuFFT plan scratch, numba
  device arrays) are not counted.

## Output

### CSV (`results/*.csv`)

Each file starts with a commented provenance header (Python, NumPy, CuPy, CPU
model, GPU model, …) followed by columns:

```
backend, n_particles, nmesh, step, time_mean, time_std, memory_peak_mb, memory_total_mb, vram_peak_mb
```

`time_*` are in seconds. `memory_peak_mb` is the peak-RSS increase caused by the
step and `memory_total_mb` is the absolute process peak RSS, both in MiB (the GPU
backend instead reports the CuPy pool delta / usage in MB). `vram_peak_mb` is the
GPU device-memory peak — the CuPy pool high-water mark (`total_bytes()`) in MB —
and is `0.0` for CPU backends and pyrecon. Read them back with
`pandas.read_csv(path, comment="#")`.

### Figures (`figures/*.pdf`)

`plot_results.py` reads the CSVs and writes:

1. `fig1_mass_assignment_time.pdf` — time vs N particles (CPU / GPU / pyrecon).
2. `fig2_fft_gpu_solver_time.pdf` — total FFT-solver time (summed over `setup`/`solve`/`readout`)
   vs nmesh (CPU / GPU / pyrecon).
3. `fig3_pipeline_time_<solver>.pdf` — end-to-end time vs nmesh, one figure per solver
   type (CPU / GPU / pyrecon).
4. `fig4_gpu_speedup.pdf` — GPU/CPU speedup vs N (mass assignment) and vs nmesh
   (FFT-solver phase of `bench_scaling.py`).
5. `fig4_pipeline_memory_<solver>.pdf` — peak memory vs nmesh, one figure per solver type
   (CPU / GPU / pyrecon).

Missing backends (e.g. no GPU) are omitted from the curves; the speedup figure
shows a "GPU data not available" note instead of empty axes.

## Notes & caveats on fairness

- For a like-for-like comparison the pmesh / pyrecon meshes use **float32**
  (`dtype='f4'`), matching baorecon's default precision.
- The FFT-solver and pipeline benchmarks use **data and randoms 1:1** for both
  backends.
- For the **reconstruction benchmark** (`bench_bao_reconstructor.py`), all backends
  start from the same pre-computed Cartesian positions: the RA/DEC/z → XYZ
  coordinate conversion is done once, outside every timer, so it is excluded for
  baorecon and pyrecon alike. The baorecon timed region is `run_reconstruction`;
  the pyrecon one starts at the assignment phase. `bench_pipeline_class.py` is the
  complement: it *times* catalog I/O and both coordinate conversions as explicit
  stages, so they are deliberately included there.
- CPU memory uses `ru_maxrss`, the resident set size of the **whole process** and
  allocator-agnostic, so it captures pmesh/pyrecon allocations as well as
  baorecon's NumPy grids. Running each backend in its own subprocess keeps these
  readings independent — the earlier single-process design let one backend's peak
  contaminate the next (e.g. TSC right after CIC reporting a ~0 delta). It is still
  a monotonic high-water mark *within* a worker: when a worker times several steps
  (`setup`/`solve`/`readout`, or `mass_assignment`/`fft_solver`), each step's
  `memory_peak_mb` reflects the cumulative peak reached up to that point, so later
  steps include earlier ones' footprint. `memory_total_mb` is the absolute process
  peak.
