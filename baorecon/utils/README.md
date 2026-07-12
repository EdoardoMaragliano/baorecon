# Utils

This package groups small helper modules shared across the reconstruction pipeline.

## Responsibilities

- format user-facing inputs into a consistent internal representation
- provide logging setup and reusable utility helpers
- keep non-physics glue code separate from solver and mesh logic

## Main modules

- `formatters.py`: normalization/validation helpers for box size, box centre, weights,
  MAS, nmesh, reconstruction type, and RSD-space flags, plus grid-sizing helpers
  (`round_to_multigrid_friendly`, `nmesh_boxsize_from_cellsize`,
  `set_boxsize_from_positions`, `survey_to_box_frame`).
- `backend.py`: computational backend selection for the FFT solvers — the `FFTBackend`
  container and `get_fft_backend` factory that pick between the CPU (numpy/scipy) and
  GPU (CuPy) backends; `CUPY_AVAILABLE` is the single source of truth for GPU availability.
  Also exposes `PYFFTW_AVAILABLE` / `use_pyfftw()`, which gate the opt-in in-place CPU
  FFT backend on `BAORECON_FFT=pyfftw` (see [../../docs/pyfftw_backend.md](../../docs/pyfftw_backend.md)).
- `coordinates.py`: cosmology construction (`create_cosmology`) and RA/DEC/redshift ↔
  Cartesian conversions built on Astropy.
- `loggers.py`: logger configuration shared across modules (`setup_logger`).
- `utils.py`: general-purpose numerical helpers (rho/delta conversion, smoothing-radius
  conversions, periodic distances, box splitting).
- `mock_generator.py`: synthetic field/catalog generation (Gaussian and lognormal maps,
  Poisson sampling).

## Notes

- These helpers are meant to keep the public API consistent and reduce repeated input handling in the higher-level layers.