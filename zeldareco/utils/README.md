# utils

This package groups small helper modules shared across the reconstruction pipeline.

## Responsibilities

- format user-facing inputs into a consistent internal representation
- provide logging setup and reusable utility helpers
- keep non-physics glue code separate from solver and mesh logic

## Main modules

- `formatters.py`: normalization helpers for box size, positions, weights, MAS, and reconstruction flags
- `loggers.py`: logger configuration shared across modules
- `utils.py`: general-purpose utility functions
- `mock_generator.py`: helpers for synthetic catalog and field generation

## Notes

- These helpers are meant to keep the public API consistent and reduce repeated input handling in the higher-level layers.