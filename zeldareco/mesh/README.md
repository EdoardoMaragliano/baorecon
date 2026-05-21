# mesh

This package defines the mesh geometry and the helpers used by the solvers.

## Responsibilities

- represent the spatial grid and its Fourier counterpart
- store box size, mesh size, and line-of-sight information
- provide derived geometric quantities such as `xmesh`, `kmesh`, and projection helpers

## Main class

- `Mesh`: lazy 3D mesh container used by the solver layer

## Typical usage

Create a `Mesh` once from the reconstruction metadata and reuse it through the solver pipeline.

## Notes

- Mesh creation is lazy: the spatial and Fourier grids are built only when requested.
- The class also stores LOS information so the solver layer can project vector fields consistently.