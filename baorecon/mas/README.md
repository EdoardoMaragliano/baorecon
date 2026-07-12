# MAS

Mass assignment: moving a quantity between particles and a regular mesh.
`assign` **paints** particles onto a fresh grid (particles → mesh); `readout`
**interpolates** a grid back to particle positions (mesh → particles). Both
support the NGP, CIC and TSC schemes on the CPU (numba) and GPU (numba.cuda).

## Responsibilities

- paint a weighted particle catalogue onto a density grid (`assign`)
- sample a scalar field defined on a grid at particle positions (`readout`)
- validate the inputs and dispatch to the CPU or GPU kernel for the requested scheme
- own the working precision and the periodic / non-periodic boundary policy

## Public API

```python
from baorecon.mas import assign, readout, CUPY_AVAILABLE

# particles -> grid
rho = assign(pos, weights, mesh, scheme="CIC", device="cpu", pbc=True, parallel=False)
# grid -> particles (scalar field)
vals = readout(grid, pos, mesh, scheme="CIC", device="cpu", pbc=True)
```

- `assign(pos, weights, mesh, ...)` → `(Nx, Ny, Nz)` grid (a CuPy array when
  `device="gpu"`). `pos` is `(N, 3)`; `weights` is `(N,)` or `None` (unit weights);
  `mesh` is a `baorecon.mesh.mesh.Mesh`.
- `readout(grid, pos, mesh, ...)` → `(N,)` array of interpolated values (CuPy on GPU).
- `CUPY_AVAILABLE` -- `True` when a CUDA GPU plus CuPy / Numba are usable.

## Modules

- `_interface.py` -- the only public surface. Validates `pos` / `weights` /
  `boxsize`, allocates the output grid, and dispatches to the CPU / GPU kernel.
  The kernels themselves do no validation and never allocate.
- `cpu.py` -- numba kernels. NGP is vectorised numpy; CIC and TSC each have a
  single-threaded (`*_serial`) and a multi-threaded (`*_chunks`, per-chunk private
  grids) painting variant, plus the `*_read` read-out kernels.
- `gpu.py` -- numba.cuda kernels: atomic-add painting and trilinear / TSC
  read-out. Guarded by `CUPY_AVAILABLE`; imported only when a GPU is present.

## Schemes

| scheme                          | stencil    | CPU | GPU |
| ------------------------------- | ---------- | --- | --- |
| `NGP` (Nearest Grid Point)      | 1 cell     | yes | —   |
| `CIC` (Cloud-In-Cell)           | 2³ cells   | yes | yes |
| `TSC` (Triangular Shaped Cloud) | 3³ cells   | yes | yes |

The GPU backend implements CIC and TSC only; an NGP request with `device="gpu"`
raises.

## Precision

- **The CPU path is type-neutral.** The grid, positions and weights follow the
  mesh's working precision (`mesh.dtype`), so a `float64` mesh is honoured
  end-to-end (`readout` follows the field's own floating dtype). Kernel temporaries
  follow the output array's dtype.
- **The GPU path is float32.** `device="gpu"` always works in — and returns —
  float32, regardless of `mesh.dtype`.

## Boundary handling

Set by `pbc` (default `True`) and applied identically across the CPU and GPU
backends — by every scheme, in the serial and parallel CPU kernels, and in paint
and read-out:

- `pbc=True` -- the stencil **wraps** periodically (`% nmesh`).
- `pbc=False` -- out-of-range stencil cells are **clamped** to the nearest boundary
  cell (`0` at the low edge, `nmesh - 1` at the high edge). This conserves mass
  (the contribution is folded onto the boundary rather than dropped) and keeps
  `readout` the exact adjoint of `assign` at the edge.

`assign` also checks that positions lie in `[0, boxsize]`: out-of-range input warns
when `pbc=True` and raises when `pbc=False`.

## Typical usage

`DensityManager.compute_delta` calls `assign` to paint the data and random
catalogues before building the overdensity field. `readout` samples a **scalar**
field; the reconstruction's **vector** displacement read-out uses the sibling
`field_ops.interpolate_vector_field` instead.

## Notes

- `boxsize` is read from `mesh.boxsize` as a length-3 array, so cubic and
  rectangular / anisotropic boxes share one code path.
- `parallel=True` (CPU CIC / TSC) allocates `num_chunks` (= 16) private full-size
  grids and reduces them: faster on many cores, at a proportional memory cost that
  doubles under `float64`. NGP and `readout` have no parallel variant.
- Kernels assume a C-contiguous, pre-zeroed output grid (guaranteed by the
  interface); they are pure and hold no state.
