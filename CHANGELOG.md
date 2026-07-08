# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
- CPU mass assignment / read-out now handle the non-periodic (`pbc=False`) box edge
  uniformly by **clamping** out-of-range stencil cells to the nearest boundary cell
  (mass-conserving) across every scheme (NGP / CIC / TSC) and both the serial and
  parallel kernels. Previously the parallel CIC/TSC kernels and `tsc_read` *dropped*
  those contributions, so `pbc=False` results near the box edge change slightly:
  mass is now conserved and paint/read stay mutual adjoints at the boundary.

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

[Unreleased]: https://github.com/EdoardoMaragliano/baorecon/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/EdoardoMaragliano/baorecon/releases/tag/v0.4.0
