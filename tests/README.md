# Tests

This folder contains both the current pytest suite and historical notebooks used to
validate the reconstruction pipeline.

## Pytest suite

These are the executable regression tests that should be run in CI or during development.

### Top-level tests

- `test_catalog_io.py` — catalog loading, column extraction, masking, and output-table building.
- `test_coordinates.py` — RA/DEC/z ↔ Cartesian round-trips and coordinate input validation.
- `test_density_manager.py` — density-field preparation, mass assignment, and PBC handling.
- `test_fft_solver.py` — `FFTSolverCPU`/`FFTSolverGPU` behaviour (analytic sine wave, GRF
  closure, mean conservation), parametrized over device.
- `test_field_ops.py` — vector-field projection, interpolation, divergence, and smoothing operators.
- `test_grid_sizing.py` — multigrid-friendly grid sizing, the cellsize entry point,
  anisotropic coarsening, and strict `nmesh` validation.
- `test_los.py` — `FixedAxisLOS` / `LocalLOS` strategies and `project_vector_field_jit`.
- `test_mesh.py` — `Mesh` geometry, validation, and lightweight (no large arrays) guarantees.
- `test_naming.py` — `baorecon/io/naming.py` `NamingTokenizer`.
- `test_rectangular_box.py` — rectangular/anisotropic box support across boxsize formatting,
  mesh, mass assignment, interpolation, and mass conservation.
- `test_solver_equivalence.py` — CPU vs GPU FFT solver agreement on a shared input field
  (real space and redshift space).

### Test packages

- `test_bao_reconstructor/test_bao_reconstruction.py` — the high-level `BAOReconstructor`
  pipeline (instantiation, box inference, delta mesh, potential/displacement, full
  reconstruction, interpolation, and a comparison vs `pyrecon`).
- `test_pipeline/test_bao_pipeline.py` — the YAML-driven `ReconstructionPipeline`, including
  saving the full set of configured outputs. 
- `tests_multigrid/test_multigrid_lib.py` — multigrid kernels and the FMG flow.

## Notes

- The pytest files are the executable regression tests; the notebooks are interactive references.
- GPU-dependent tests are skipped automatically when no CUDA GPU / CuPy is available.
