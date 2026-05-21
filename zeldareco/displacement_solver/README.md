# displacement_solver

This package contains the Poisson and displacement solvers used by the reconstruction pipeline.

## Responsibilities

- expose solver objects with lazy `potential` and `displacement` properties
- keep the heavy numerical work in the JIT-compiled kernels from `multigrid_lib.py`
- provide interchangeable solver backends for the same mesh field

## Solvers

- `FFTSolver`: FFT-based solver for the displacement field
- `MultigridSolver`: multigrid-based solver built on top of the same solver interface
- `PoissonSolver`: shared lazy base class for Poisson-like solvers

## Design notes

- The solver layer operates on `delta_on_mesh` and `Mesh`.
- `FFTSolver` uses the mesh geometry directly in Fourier space.
- `MultigridSolver` uses the mesh metadata to configure the internal multigrid hierarchy, while the recursive kernels remain internal.
- The low-level kernels in `multigrid_lib.py` are the numerical core and should stay implementation-focused.