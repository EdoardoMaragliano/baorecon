# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.4.0]: https://github.com/EdoardoMaragliano/baorecon/releases/tag/v0.4.0
