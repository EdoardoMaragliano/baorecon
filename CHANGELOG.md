# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.7.0] - 2026-07-27

### Added
- `read_displacement_at` is now implemented on the FFT solvers
  (`FFTSolverCPU`/`FFTSolverGPU`): it interpolates the spectral displacement grid
  at the requested positions and returns a host `(N, 3)` array, matching the
  multigrid solver. The GPU solver keeps the field on the device and brings only
  the per-particle result back to host.
- The FFT solvers accept `pbc` as a constructor argument, so the periodic-wrap
  behaviour of the displacement read-out is configured on the solver instead of
  the caller.
- `BAOReconstructor.interpolate_displacement` (real-space displacement at the
  tracer positions) and `BAOReconstructor.get_rsd_displacement` (the RSD
  contribution to the displacement) are now public, documented methods.

### Changed
- `BAOReconstructor` reads every solver out through the single
  `read_displacement_at` interface instead of branching on `solver_type`; the
  FFT/multigrid difference (interpolate a spectral Psi grid vs. differentiate the
  potential on the fly) is now hidden behind the solver.
- Mass-assignment-scheme (`MAS`) case handling is standardized through
  `format_mas` across the FFT and multigrid solvers.

## [0.6.0] - 2026-07-20

### Added
- `grid_density` pipeline output option: saves the overdensity field
  (`delta_on_mesh`) as a FITS image, alongside the existing `grid_potential` and
  `grid_displacement` grid outputs. Follows the same host-copy-then-release
  pattern on the GPU path, and is skipped (kept intact) when
  `reconstructor_object` is also requested.
- `ReconstructionPipelineInteractive` (exported as
  `baorecon.pipeline.ReconstructionPipelineInteractive`): a memory-retaining
  mirror of `ReconstructionPipeline` for interactive/notebook use, replacing the
  old, unexported `bao_pipeline_straight.py`. Every intermediate array (raw and
  Cartesian positions, reconstructed positions, solver grids) stays reachable on
  the object instead of being released mid-run.
- Packaging: adopted a PEP 621 `pyproject.toml` with `test`/`notebook`/`gpu`/`docs`
  optional-dependency extras and a version dynamically single-sourced from
  `baorecon.__version__`; `setup.py` and the `requirements/*.txt` files are
  removed. CI now installs via these extras and tests Python 3.10-3.12, and the
  Sphinx docs version is single-sourced the same way.

### Fixed
- `examples/run_bao_pipeline.py` now prints every path returned by
  `pipeline.run()` instead of assuming a fixed `(data_path, random_path)`
  two-tuple, which broke once `run()` started returning the full `saved_paths`
  dict.

## [0.5.0] - 2026-07-08

### Added
- Streamed radial (`LocalLOS`) projection promoted to shared infrastructure in
  `baorecon/solvers/fft/_radial_stream.py`: the per-cell radial versor
  `n̂ = x/|x|` is evaluated on the fly and the potential gradient is projected and
  scattered one component at a time, so the full `(N, N, N, 3)` gradient / versor /
  parallel-field grids are never held in memory at once. It is now used by the
  **default scipy CPU solver** and the **GPU solver** (via CuPy `ElementwiseKernel`
  twins), not only the opt-in pyfftw backend — lowering peak memory on the radial
  line of sight for every backend. Cross-implementation parity (numba vs numpy vs
  CuPy) is guarded by `tests/test_radial_kernels.py`.
- Benchmark: `bench_bao_reconstructor.py` gained `--smoother {jacobi,mcgs}`
  (multigrid smoother: Jacobi V-cycle or multicolor Gauss–Seidel) and
  `--fft {scipy,pyfftw}` (selects the CPU ifft FFT backend via `BAORECON_FFT`),
  and now warns when `--fft pyfftw` cannot take effect (multigrid / GPU).
- Benchmark: all CSVs gained a `vram_peak_mb` column — the CuPy memory-pool
  high-water mark (`total_bytes()`) on GPU, `0.0` on CPU / pyrecon — tracked once
  in the shared `bench_common.measure()` primitive.

### Changed
- `divergence_inplace` → `divergence_from_components` in
  `baorecon/solvers/fft/_common.py`: it now takes a `get_component(i)` callback so
  each gradient component can be synthesised lazily by the caller.
- `Mesh` stores `boxsize` / `boxcentre` (and the derived `cell_size` / `min_corner`)
  at the mesh's working precision (`dtype`) instead of pinning them to float32, so a
  float64 mesh keeps float64 geometry. The default float32 mode is unchanged, and
  `LocalLOS` still exposes a float32 `cell_size` so the on-the-fly radial versor
  matches the cached versor grid cell for cell in float32.
- Mass assignment / read-out now handle the non-periodic (`pbc=False`) box edge
  uniformly by **clamping** out-of-range stencil cells to the nearest boundary cell
  (mass-conserving) across every scheme (NGP / CIC / TSC), the serial and parallel
  CPU kernels, and the GPU kernels. Previously the parallel CPU CIC/TSC kernels and
  `tsc_read` *dropped* those contributions and the GPU kernels ignored `pbc` and
  always wrapped, so `pbc=False` results near the box edge change slightly: mass is
  now conserved and paint/read stay mutual adjoints at the boundary, identically on
  CPU and GPU.

### Fixed
- CPU mass assignment / read-out are now type-neutral instead of forcing float32:
  the interface allocates the grid at the mesh's working precision and casts
  positions / weights to match, so a `dtype=float64` run is honoured end-to-end
  rather than silently downcast during painting (the accumulation happened in
  float32 even when the surrounding pipeline was float64). The GPU path stays
  float32 by design.
- The iterative FFT (iFFT / Burden) CPU solver honours the working precision: the
  wavevectors (scipy and pyfftw paths) and the pyfftw in-place buffers / outputs now
  follow the delta dtype (float32/complex64 or float64/complex128), so a float64
  reconstruction stays float64 through the solver instead of round-tripping through
  float32. `prepare_k_components` also accepts a `numpy.dtype` instance, not only a
  type. (The radial `LocalLOS` versor geometry remains float32.)
- `tsc_assign_serial` silently ignored its `pbc` argument and always wrapped
  periodically; it now honours `pbc` (periodic wrap when true, boundary clamp when
  false), matching the other kernels and `tsc_read`.
- `cic_assign_serial` and `cic_read` used truncation (`int()`) for the cell index,
  which produced negative CIC weights for negative positions under PBC; they now use
  `floor`. Not reachable through the pipeline (positions are pre-wrapped to
  `[0, boxsize)`), but wrong for direct `assign` / `readout` calls.
- GPU mass assignment / read-out kernels now honour `pbc`. They previously took no
  `pbc` argument and **always wrapped periodically**, so a `device="gpu", pbc=False`
  run wrapped box-edge mass to the far side instead of clamping (diverging from the
  CPU kernels). The four `numba.cuda` kernels now thread `pbc` (wrap when true,
  boundary clamp when false) and use `floor` for the CIC cell index, matching
  `baorecon.mas.cpu`. `tests/test_density_manager.py` is parametrized over
  `pbc ∈ {True, False}` on both backends.
- Benchmark: the `bench_bao_reconstructor.py` `smoother` argument was passed as a
  bare kwarg that `BAOReconstructor(**kwargs)` silently swallowed and never reached
  the solver; it is now threaded through `solver_args`. GPU VRAM tracking also
  called a non-existent `MemoryPool.max_bytes()` (crashing the GPU worker) and
  stored the result in a column `save_csv` dropped; both are resolved.

## [0.4.0] - 2026-07-03

### Added
- Pluggable catalog I/O backends with a Parquet backend alongside FITS; the
  compute layer only ever sees NumPy arrays, decoupled from the on-disk format.
- Configurable working precision via `reconstruction.dtype` (default `float32`,
  `float64` to opt into double precision), propagated end-to-end from catalog
  load through reconstruction to the saved coordinate/displacement columns.
- `keep_cols` column pruning so only the columns the pipeline needs (plus any
  explicitly kept) are read; FITS column-subset reads use `fitsio` when
  available, otherwise prune in memory.

### Changed
- Catalogs are downcast to the working precision on load, and positions are held
  at that precision throughout the pipeline; the coordinate helpers now preserve
  their input dtype rather than forcing `float32`.
- Substantially reduced pipeline memory footprint: raw RA/DEC/z are released
  after coordinate conversion, and `run()` releases the heavy arrays
  progressively — potential/displacement grids are written and freed one at a
  time, and catalogues are written one tracer at a time with their position
  arrays dropped immediately after (on GPU the freed CuPy buffers are returned
  to the device memory pool).

### Fixed
- `bench_pipeline_class` reported roughly double the true total memory because it
  reused a previous pipeline allocation.
- `resolve_format` now raises clear errors for an unknown format or extension.

[Unreleased]: https://github.com/EdoardoMaragliano/baorecon/compare/v0.6.0...HEAD
[0.6.0]: https://github.com/EdoardoMaragliano/baorecon/releases/tag/v0.6.0
[0.5.0]: https://github.com/EdoardoMaragliano/baorecon/releases/tag/v0.5.0
[0.4.0]: https://github.com/EdoardoMaragliano/baorecon/releases/tag/v0.4.0
