# mesh

This package defines the mesh geometry and the line-of-sight strategies used by
the solvers.

## Responsibilities

- represent the spatial grid geometry (box size, mesh size, cell size, corner)
- model the line-of-sight as an interchangeable strategy

## Modules

- `mesh.py` -- `Mesh`: a lightweight, geometry-only dataclass. It stores
  `boxsize`, `nmesh`, `boxcentre` and the derived `cell_size` and `min_corner`.
  Both `boxsize` and `nmesh` may be a scalar (cubic) or a length-3 array
  (rectangular / anisotropic). `Mesh` allocates **no** large arrays: wavevectors
  and real-space grids live in the solvers and the LOS strategies.
- `los.py` -- line-of-sight strategies with a common `project_parallel` method:
  - `FixedAxisLOS(axis)`: global plane-parallel LOS along `x`/`y`/`z`; stores
    only the axis index and allocates no arrays.
  - `LocalLOS(...)`: per-cell radial LOS; the radial versor field is computed
    lazily and cached. This module also holds the JIT `project_vector_field_jit`.

## Typical usage

Create a `Mesh` once from the reconstruction metadata and reuse it through the
solver pipeline. The reconstructor builds the appropriate LOS strategy
(`FixedAxisLOS` for `los='x'|'y'|'z'`, `LocalLOS` for `los=None`) and injects it
into the solver.

## Notes

- `Mesh` is intentionally lightweight; wavevectors are built on demand by
  `solvers/fft/_common.py` and never stored on the mesh.
- `nmesh` is validated by `format_nmesh` (rejects non-integer / non-positive /
  wrong-shape values).
