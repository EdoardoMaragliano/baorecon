# Audit of the Multi-GPU Migration Plan

Audit of `docs/multigpu_migration_plan.md` against the code base, plus a memory-footprint
assessment of the current single-GPU path. The findings below drive the implementation on
`feat/multigpu`; each finding states what the plan claims, what the code actually does, and
what was changed.

## Verdict summary

The plan's architecture is sound and matches the code: slab (1-D, x-axis) decomposition,
ky-split k-space, CuPy + mpi4py + NCCL, ghost-zone MAS, `P = 1` as a zero-communication
special case. All the file/line anchors it cites are accurate (`solvers/fft/gpu.py:238`
displacement allocation, `gpu.py:277` `read_displacement_at` TODO, `mas/gpu.py:43` clamp,
`bao_reconstructor.py:133-134` device resolution, the kx-full / ky-full / kz-half layout in
`prepare_k_components`).

However, the plan has **two outright bugs** (the AllToAll pack/unpack code and the TSC halo
width), **one missing distributed operation** (the Gaussian smoothing FFT inside
`DensityManager.compute_delta`), and several places where the implementation can be simpler
than proposed. The current single-GPU solver is well optimized for memory but **not
maximal**: ~1.5 grid-equivalents of peak VRAM can be removed before distributing (details in
§3), which also simplifies the distributed port. These optimizations were applied first.

## 1. Errors found in the plan

### E1 — §5(a) `DistributedFFT` pack/unpack is wrong (verified numerically)

The plan's code mixes a *block* x-distribution with a *cyclic* unpack. In `rfftn`, the
receive side does

```python
recv.reshape(P, Nx_loc, Ny_loc, Nzh).transpose(1, 0, 2, 3).reshape(Nx, ...)
```

which assembles global x as `ix_local * P + src_rank` (cyclic), while the slabs are
distributed in contiguous blocks (`ix_global = src_rank * Nx_loc + ix_local`). The inverse
transform has the mirrored bug on the send side (`b.reshape(Nx_loc, P, ...)` where `b`'s
first axis is the full `Nx`). A numpy simulation of the plan's exact reshapes against
`np.fft.rfftn` gives max errors of ~1e2 (forward) and ~5 (inverse); the corrected scheme is
exact to machine precision:

* forward send: `a.reshape(Nx_loc, P, Ny_loc, Nzh).transpose(1,0,2,3)` (contiguous copy) —
  block per destination ky-rank (as in the plan);
* forward recv: `recv.reshape(P * Nx_loc, Ny_loc, Nzh)` — **no transpose**; source-rank
  blocks are already in global-x order;
* inverse send: `b.reshape(P, Nx_loc, Ny_loc, Nzh)` — already contiguous per destination
  x-rank, **no transpose**;
* inverse recv: `recv.reshape(P, Nx_loc, Ny_loc, Nzh).transpose(1,0,2,3).reshape(Nx_loc, Ny, Nzh)`
  — reassemble the ky-blocks (this is the plan's *forward* unpack, on the other side).

The implementation keeps pack/unpack as pure array-module-agnostic functions in
`baorecon/solvers/fft/_distributed_fft.py` so they are unit-tested on CPU with numpy against
`np.fft.rfftn`/`irfftn` (see `tests/test_distributed.py`), independent of NCCL.

### E2 — halo width `w = 1` is insufficient for TSC

The plan asserts "w = 1 suffices (CIC reaches +1; TSC's round-based stencil reaches ±1)".
For a particle owned by a rank via `gx_local ∈ [0, Nx_loc)`, the TSC kernels
(`mas/gpu.py:65`) use `ix_c = round(gx)`, so `gx just below Nx_loc` gives `ix_c = Nx_loc`
and a stencil cell at `Nx_loc + 1` — **two** planes beyond the owned region. The low side
needs 1 (`ix_c = 0` reaches `-1`). The implementation uses a per-scheme halo width:
`w = 1` for CIC, `w = 2` for TSC (symmetric, the extra low plane is one `Ny*Nz` plane of
waste). The same widths apply to the read-back halo.

### E3 — the Gaussian smoothing FFT is missing from the plan

`DensityManager.compute_delta` calls `smoothed_field` (`field_ops/_interface.py:107`) on
**both** painted grids — a full-grid `rfftn`/`irfftn` pair per catalogue on the GPU. Under
decomposition these are whole-mesh transforms and must route through the `DistributedFFT`,
with the separable 1-D Gaussian factors sliced to the local k-region (`sx` full over kx,
`sy` sliced to the rank's ky range, `sz` full). The plan's Phase 2 only mentions the
mean-density AllReduce. Omitting this would silently smooth each slab as if it were an
independent periodic box. Fixed: `smoothed_field` accepts an optional distributed FFT +
decomposition and slices `sy` accordingly.

### E4 — three scalars need reduction in `compute_delta`, not one

Besides the "mean density" reduction the plan lists, `compute_delta`
(`reconstruction/density.py:168-170`) needs global values for **(a)** `sum(data_rho)`,
**(b)** `sum(random_rho)` (both feed `alpha` *and* the random threshold) — the particle
count in the threshold (`len(self.random_pos_box)`) stays global as long as each rank keeps
the full catalogue and filters at paint time, which is the chosen design (single node: host
RAM is not the constrained resource, VRAM is; this also keeps the box-geometry derivation
in `_prepare_inputs` bit-identical across ranks with no Scatter step).

### E5 — the DC-mode fix needs no rank-awareness at all

The plan proposes making the `[0,0,0] → 1 → 0` DC fix in `build_inv_k2` "rank-aware". After
the single-GPU memory optimization below (§3.2), `1/(bias·k²)` is no longer materialized on
the GPU: a fused elementwise kernel evaluates it on the fly from the 1-D k-arrays and emits
zero where `k² == 0`. That test is *positional, not index-based*, so it is automatically
correct on every rank (only the rank owning `ky = 0` ever sees `k² = 0`). The index-based
fix remains only in the CPU path, which is not distributed.

### E6 — the versor kernels need no `global_x_offset` parameter

The plan proposes adding an offset parameter to the radial-LOS `ElementwiseKernel`s
(`solvers/fft/gpu.py:58-76`). Unnecessary: the kernels compute
`cx = min_x + ix * cell_x`, so passing `min_x_eff = min_x + x_offset * cell_x` gives the
global coordinate with **zero kernel changes**. The distributed solver just shifts the
scalar it already passes.

### E7 — kernel/NCCL stream synchronization is not addressed

MAS painting runs on numba.cuda's stream; NCCL halo exchange and the CuPy FFTs run on
CuPy's current stream. Without an explicit `cuda.synchronize()` (or event) between painting
and the halo exchange — and between filling halo buffers and reading them — the exchange can
ship stale data. The plan's "NCCL + numba.cuda coexistence" note covers device *binding*
only. The implementation synchronizes at the paint→exchange and exchange→read boundaries.

### E8 — smaller notes

* `inv_k2_bias` is called a "complex half-grid" in §1; it is a **real** float32 half-grid
  (~0.5 grid-equivalents, `solvers/fft/_common.py:65`). Only the accounting changes.
* `field_ops.divergence_FFT` (GPU) transforms the whole `(Nx,Ny,Nz,3)` field in one
  `rfftn(..., axes=(0,1,2))` — three complex half-grids at once. It is **not** on the GPU
  solver's hot path (the solver uses the streamed `divergence_from_components`), so
  distributing it is deprioritized; it now raises a clear error in distributed mode instead
  of silently producing a per-slab result.
* Consolidating the duplicated `CUPY_AVAILABLE` checks into the *new* `utils/distributed.py`
  (Phase 0) would make core modules depend on the distributed machinery. They are
  consolidated into `utils/backend.py` instead — which already documents itself as "the
  single source of truth for GPU availability" — and `utils/distributed.py` imports from it.
  There are seven duplicates, not four (also `reconstruction/density.py`,
  `reconstruction/bao_reconstructor.py`, `mesh/los.py`).
* The equal-block AllToAll requires `Nx % P == 0` and `Ny % P == 0`. As the plan suggests,
  the implementation *requires* divisibility (validated with a clear error at startup)
  rather than shipping an untested AllToAllv path.

## 2. Confirmed plan decisions (no change)

* Slab over pencil for P ≤ 8; one AllToAll per transform; `P ≤ Nx` cap is irrelevant here.
* Real space split on x (axis 0), k-space split on ky (axis 1); kz is the reduced rfft axis.
* Ghost-zone halo accumulate for painting (mass-conserving `Sendrecv`-add, PBC ring wrap,
  clamp at edges when `pbc=False`), copy-halos for read-back.
* Particles stay on their rank; catalogue distribution by x-domain.
* mpi4py for launch/host orchestration, NCCL (via `cupy.cuda.nccl`) for device collectives.
* Existing `FFTBackend` seam is where distribution is threaded; the Burden loop is unchanged.
* Verification strategy (P=2 vs P=1 agreement, mass conservation, FFT closure, existing
  GPU suites as the P=1 no-op proof).

## 3. Memory audit of the current single-GPU path

Units: one *grid-equivalent* `G = 4·N³` bytes (a full real float32 grid). A complex64
rfft half-grid `(Nx, Ny, Nz/2+1)` ≈ 1.0 G; a real float32 half-grid ≈ 0.5 G.

### 3.1 Where the VRAM goes (radial/LocalLOS Burden loop, the worst case)

Persistent across the loop (`solvers/fft/gpu.py:138-183`):

| array | shape | size |
|---|---|---|
| `delta_dev` | real grid | 1.0 G (aliases the delta grid, no copy — good) |
| `inv_k2_bias` | real half-grid | 0.5 G |
| `delta_k` | complex half-grid | 1.0 G |
| `temp_k_comp` | complex half-grid | 1.0 G |
| `scaled_k` | complex half-grid | 1.0 G |
| `s`, `proj_scratch` | real grids | 2.0 G |
| **persistent total** | | **6.5 G** |

Transients per inverse FFT: cuFFT c2r destroys its input, so CuPy copies it (+1.0 G), plus
the real output (+1.0 G) and the cached cuFFT plan work area (up to ~1.0 G for 3-D r2c/c2r).
Loop peak ≈ **8.5–9.5 G**. The final displacement stage holds `delta_dev + delta_k +
temp_k_comp + inv_k2_bias + (N³,3) displacement (3.0 G)` ≈ 6.5 G + FFT transients ≈
**8.5 G**. For 1024³ (G = 4 GiB) that is ~36–40 GB — consistent with the 27–63 GB range
quoted in `docs/pyfftw_backend.md`.

The existing design is already strong: the streamed radial projection avoids both the
`(N³,3)` versor grid and a `(N³,3)` gradient stack (saves 6 G vs the naive approach), the
complex scratch is reused, all k-space multiplies are in-place, and `delta_dev` is
read-only and un-copied.

### 3.2 Maximization plan (what was implemented first)

1. **Fuse the k-space gradient scaling into one elementwise kernel** — evaluate
   `out = (±i) · k_a / (bias·k²) · delta_k` on the fly from the three 1-D k-arrays
   (`_scale_component_k` in `solvers/fft/gpu.py`). Removes `inv_k2_bias` (−0.5 G) and
   `scaled_k` (−1.0 G) *and* cuts elementwise passes per radial iteration from 5 to 3.
   The same kernel serves the fixed-axis path, the final displacement build, and the
   potential (`bias=1`), so the GPU solver no longer materializes `1/k²` at all.
   **Loop peak drops ~1.5 G (≈ 6 GiB at 1024³, ~16%)**, and — because the kernel's DC
   handling is positional (E5) — it ports to the distributed ky-slab unchanged.
2. **Reuse the complex scratch in the fixed-axis path** — the plane-parallel branch
   allocated a fresh `corr_k` each iteration; it now reuses `temp_k_comp`.
3. `read_displacement_at` (the `gpu.py:277` TODO) is implemented against the
   device-resident displacement, so per-particle read-out needs no host round trip and no
   extra grid.

Documented but deliberately **not** done now:

* Streaming the final displacement per component (read out data+random shifts after each
  `irfftn`, never materializing `(N³,3)`) would save up to 2.0 G more, but
  `solver.displacement` is a public artifact (saved to FITS by the pipeline), so this
  needs an opt-in API; it is orthogonal to the multi-GPU work, which reduces the same 3.0 G
  by 1/P anyway.
* Capping/clearing the cuFFT plan cache trades speed for ~1 G; not worth it by default.

With (1)–(2), per-GPU peak ≈ `7 G / P + transients`, so the plan's ~1/P scaling target is
measured from a lower baseline.

## 4. Implementation deltas vs the plan's roadmap

The phase structure is kept, with these changes:

* **Phase 0**: `CUPY_AVAILABLE` consolidation targets `utils/backend.py` (E8);
  `utils/distributed.py` provides `SlabDecomp` (pure logic, CPU-tested), `DistEnv`
  (mpi4py + NCCL init, device binding for *both* CuPy and numba.cuda), and a
  thread-based `LoopbackComm` so all communication-adjacent logic is testable on a
  CPU-only host.
* **Phase 1**: `DistributedFFT` with the corrected transposes (E1); array-module agnostic
  core, NCCL only in the transport layer. The solver's versor kernels are reused via the
  `min_x` shift (E6); no rank-aware `build_inv_k2` is needed (E5).
* **Phase 2**: per-scheme halo widths (E2); offset-aware paint kernels that map global
  stencil cells into the extended local grid (PBC wrap lands in the halo images);
  explicit numba↔NCCL synchronization (E7); `compute_delta` reduces the two grid sums
  (E4); distributed `smoothed_field` (E3).
* **Phase 3**: read-back halos are allocated *with* the displacement grid in distributed
  mode (the solver writes into the interior view), so no post-hoc `(N³/P+…,3)` copy is
  needed to build the extended read grid.
* **Phase 4**: no Scatter — every rank loads/derives the full catalogue and geometry
  identically, filters its own particles at paint/read time, and the per-particle shifts
  are recombined with an in-place `Allreduce` (order-preserving, simple); rank 0 writes
  outputs. Grid saves gather slabs to rank 0.
* **Phase 5**: decomposition/FFT/halo/smoothing logic is verified on CPU with numpy +
  `LoopbackComm` (runs in this repo's CI without GPUs); the GPU/NCCL integration tests
  are provided under `mpirun` gates as the plan specifies.
