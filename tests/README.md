# Tests

This folder contains the pytest regression suite that should be run in CI or
during development. (Ad-hoc exploration notebooks live locally under the
git-ignored `local_tests/`; they are not part of the tracked suite.)

## Pytest suite

### Top-level tests

- `test_catalog_io.py` — catalog loading, column extraction, masking, and output-table building.
- `test_coordinates.py` — RA/DEC/z ↔ Cartesian round-trips and coordinate input validation.
- `test_density_manager.py` — density-field preparation, mass assignment, and PBC handling.
- `test_dtype_propagation.py` — float32/float64 working-precision propagation through
  the formatters, `DensityManager`, and `BAOReconstructor`.
- `test_fft_solver.py` — `FFTSolverCPU`/`FFTSolverGPU` behaviour (analytic sine wave, GRF
  closure, mean conservation), parametrized over device.
- `test_field_ops.py` — vector-field projection, interpolation, divergence, and smoothing operators.
- `test_grid_sizing.py` — multigrid-friendly grid sizing, the cellsize entry point,
  anisotropic coarsening, and strict `nmesh` validation.
- `test_los.py` — `FixedAxisLOS` / `LocalLOS` strategies and `project_vector_field_jit`.
- `test_mesh.py` — `Mesh` geometry, validation, and lightweight (no large arrays) guarantees.
- `test_mock_generator.py` — synthetic field generators in `utils/mock_generator.py`
  (Gaussian and lognormal maps, Poisson sampling).
- `test_naming.py` — `baorecon/io/naming.py` `NamingTokenizer`.
- `test_radial_kernels.py` — parity of the streamed radial-LOS projection kernels
  (`_radial_stream` numba vs a numpy reference, and vs the CuPy `ElementwiseKernel`
  twins in `fft/gpu.py` when a GPU is available).
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
- `tests_multigrid/test_smoother_consistency.py` — self-consistency between the `jacobi`
  and `mcgs` smoothers (agreement with each other and with the analytic FFT reference,
  convergence rate, stability under extra cycles).

## Notes

- GPU-dependent tests are skipped automatically when no CUDA GPU / CuPy is available.
