# Tests

This folder contains both historical notebooks and the current pytest suite used to validate the reconstruction pipeline.

## Notebook-based checks

- `test_mass_assignment_pyrecon.ipynb` compares the mass-assignment routines against `pyrecon`.
- `test_mesh.ipynb` exercises mesh helpers such as LOS handling and projections.
- `test_ifft/` contains the original iFFT comparison notebooks.
- `tests_multigrid/test_BAOmultigrid.ipynb` and `tests_multigrid/test_multigrid_lib.ipynb` contain the original multigrid checks.

## Pytest suite

- `test_density_manager.py` checks density-field preparation, mass assignment, and PBC handling.
- `test_solver_lazy.py` checks lazy solver initialization.
- `test_solver_equivalence.py` compares solver outputs on a shared input field.
- `test_bao_reconstruction.py` checks the high-level reconstruction pipeline.
- `tests_multigrid/test_multigrid_lib.py` covers the multigrid kernels and the FMG flow.

## Notes

- The notebooks are kept as interactive references and exploratory validation material.
- The pytest files are the executable regression tests that should be run in CI or during development.