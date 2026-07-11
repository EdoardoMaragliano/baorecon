# Multi-GPU Migration Plan for `baorecon`

## Context

`baorecon` is strictly single-device today: GPU vs CPU is a `device` string (`"cpu"`/`"gpu"`) that
selects a backend, and every GPU array lives on the default CUDA device (device 0). The reconstruction
is bottlenecked by **single-GPU VRAM**: the FFT/Zel'dovich solver holds 5–10× a single grid
simultaneously (per `docs/pyfftw_backend.md`, a `1024³` solve is ≈27–63 GB and `2048³` needs "tens of
GB"), which caps the achievable `Nmesh` on one GPU. The goal is to distribute the mesh and the
FFT/MAS/field-ops workload across GPUs so that per-GPU memory scales as ~1/P, lifting the resolution
ceiling.

**Decisions locked in with the user:**
- **Target:** single node, 2–8 GPUs → **slab (1-D) domain decomposition**, NCCL over NVLink/PCIe.
- **Stack:** **CuPy (local compute) + mpi4py (launch/orchestration) + NCCL (GPU↔GPU collectives)** —
  a hand-rolled slab FFT, no cuFFTMp/NVSHMEM dependency.
- **MAS boundaries:** **ghost-zone halo exchange** (paint into local+halo, then accumulate halos into
  the owner rank; PBC wraps the ends). Particles stay on their rank.

The single-GPU path must remain the `P = 1` special case (zero communication), so nothing regresses for
existing users.

---

## 1. Code Analysis — data structures & bottlenecks

**Main data structures (all `(Nx, Ny, Nz)`-shaped, the memory drivers):**
- `Mesh` (`baorecon/mesh/mesh.py`) — geometry only (`nmesh`, `boxsize`, `boxcentre`, `cell_size`,
  `min_corner`, `dtype`); allocates no big arrays. This is where a *decomposition descriptor* (local
  shape + global offset per rank) must live.
- **Real grids** `(Nx, Ny, Nz)`: `delta_on_mesh`, `grad`, `correction`, radial `s`/`proj_scratch`, and
  the `(Nx, Ny, Nz, 3)` displacement (`solvers/fft/gpu.py:238`).
- **Complex half-grids** `(Nx, Ny, Nz//2+1)` (rfft): `delta_k`, `temp_k_comp`, `scaled_k`, `inv_k2_bias`
  (`solvers/fft/gpu.py:163-177`). These, plus FFT plan scratch, are the peak-VRAM offenders.

**Computational bottlenecks:**
- **The monolithic 3-D FFT** (`solvers/fft/gpu.py`) — a whole-grid `cupy.fft.rfftn`/`irfftn`, called
  many times inside the Burden iteration (`_compute_displacement_iterative_potential`). No partitioning
  exists; this is the crux of the migration.
- **MAS atomic-add painting** (`mas/gpu.py`) — `numba.cuda` CIC/TSC scatter into the grid;
  embarrassingly parallel per particle *except* at domain boundaries (the stencil crosses slabs).
- **Grid→particle read-back** — currently via `field_ops.interpolate_vector_field`
  (`field_ops/gpu.py:94-115`); the solver's `read_displacement_at` is a TODO (`solvers/fft/gpu.py:277`).

**Key layout fact that dictates the FFT decomposition:** `prepare_k_components`
(`solvers/fft/_common.py`) builds `kx=fftfreq(Nx)` (full), `ky=fftfreq(Ny)` (full),
`kz=rfftfreq(Nz)` (**half** — the Hermitian/reduced axis is z = axis 2). So the reduced axis is the
*last* axis; a slab FFT must split real space along **x (axis 0)** and end up split along
**ky (axis 1)** in k-space (details in §5).

---

## 2. Data Decomposition Strategy (slab / 1-D)

**Real space — split along x (axis 0).** Rank `r` of `P` owns `x ∈ [x0_r, x1_r)`, local grid
`(Nx_loc, Ny, Nz)` with `Nx_loc ≈ Nx/P` (block distribution; handle the `Nx % P` remainder by giving the
first `Nx % P` ranks one extra plane). Per-GPU real-grid memory drops ~P×.

**k-space — split along ky (axis 1).** After the forward transform (§5) the complex half-grids are
`(Nx, Ny_loc, Nz//2+1)`. The k-vectors become: `kx` full, **`ky` a local slice**, `kz` full.
`build_inv_k2` and every `k_bcast` product operate on the local ky slice — no communication to build them.
The DC-mode fix in `build_inv_k2` (`[0,0,0] → 1` then restore `0`) only fires on the rank that owns
`ky = 0` (rank 0 after transpose); make it rank-aware.

**Ghost zones / halos for MAS (chosen approach):**
- Allocate the local mesh with a halo of `w` planes on each x-side: `(Nx_loc + 2w, Ny, Nz)`.
  `w = 1` suffices (CIC reaches +1; TSC's `round`-based stencil reaches ±1). Size `w` to the stencil
  half-width.
- **Assign:** paint locally with the existing atomic-add kernels into the extended grid (particles stay
  on their rank; their stencil may touch the halo planes).
- **Accumulate halos:** `Sendrecv` each boundary halo plane to the neighbor and **add** it into that
  neighbor's owned edge plane (mass-conserving). **PBC** wraps rank `P-1`'s high halo into rank 0's low
  edge and vice versa (matches the current `% n` wrap in the kernels). With `pbc=False`, edge ranks
  clamp instead (mirrors `mas/gpu.py:43`).
- **Read-back halo:** the displacement grid is x-split; a particle near the high edge reads cells in the
  neighbor's slab. Before interpolation, **copy** (not add) `w` ghost planes from neighbors into the
  local displacement grid, then run the existing read kernels on the extended grid with an index offset.
  PBC wraps as above.

**LocalLOS (radial) under decomposition:** the on-the-fly versor kernels (`solvers/fft/gpu.py:40-79`)
recover the cell coordinate from the flat index assuming a full-Nx array. In distributed **real** space
the array is x-split, so add a `global_x_offset` (the rank's x-origin) and use
`ix_global = ix_local + global_x_offset` when forming `cx = min_x + ix_global*cell_x`. No communication —
just an index shift. `LocalLOS.radial_versor` (`mesh/los.py`) already avoids materializing a versor grid;
keep it that way.

**Why slab (not pencil):** for P ≤ 8 and `Nx ≥ 128`, slab thickness `Nx/P ≥ 16` is ample; slab needs
**one** AllToAll per transform vs pencil's two, and is far simpler. The decomposition descriptor
(below) is designed so pencil can be added later behind the same interface if multi-node is ever needed.

---

## 3. Communication & Stack

**Stack:** **mpi4py** (process launch, rank/world discovery, NCCL unique-id broadcast, host-side catalog
scatter/gather) + **CuPy** (all local compute, unchanged) + **NCCL via `cupy.cuda.nccl`** (all
GPU↔GPU collectives on device buffers, fastest over NVLink). Add `mpi4py` and (optionally) an
`nccl`-enabled `cupy` to `requirements/gpu.txt`. Bind each rank to its GPU at startup:
`cp.cuda.Device(rank % n_local_gpus).use()` **and** the matching `numba.cuda.select_device` (both CUDA
runtimes are in play — CuPy for FFT/field-ops, numba.cuda for MAS).

**Where cross-GPU communication is strictly necessary:**

| Operation | Collective | When |
|---|---|---|
| Distributed FFT transpose | **AllToAll** (grouped NCCL send/recv) | 1× per forward `rfftn`, 1× per inverse `irfftn` — the dominant cost (many per Burden iteration). |
| MAS halo accumulate | **Sendrecv** to ±x neighbors (add) | after each `assign` (paint). |
| Displacement read-halo | **Sendrecv** to ±x neighbors (copy) | before each grid→particle interpolation / `read_displacement_at`. |
| Mean density for `delta` | **AllReduce** (sum of total weight/count) | in `DensityManager.compute_delta`. |
| Convergence/DC handling | **AllReduce** (scalar), rank-local DC fix | Burden loop bookkeeping. |
| Catalog distribution | **Scatter** (mpi4py, host) | at load: partition particles by x-domain to owner ranks. |
| Output collection | **Gather** (mpi4py, host) | shifted catalog → rank 0 for writing (or parallel shard write). |

Everything is wrapped behind the existing `FFTBackend` seam so the solver's iterative loop barely
changes: `self.fft.rfftn/irfftn` become the distributed transforms; `self.backend.to_device/to_host`
become rank-aware.

---

## 4. Refactoring Roadmap

**Phase 0 — Foundation (no behavior change at P=1).**
- New `baorecon/utils/distributed.py`: `DistEnv` holding `rank`, `world_size`, mpi4py `comm`, the NCCL
  `NcclCommunicator`, the bound device, and a **`SlabDecomp`** descriptor (`global_shape`,
  `local_shape`, `x_offset`, neighbor ranks, `ky_offset`/`ky_local` for k-space). Provide
  `is_distributed` (False when `world_size==1`). Consolidate the four duplicated `CUPY_AVAILABLE` checks
  (`utils/backend.py`, `solvers/_interface.py`, `field_ops/_interface.py`, `mas/_interface.py`) to
  import from here.
- Extend `FFTBackend`/`get_fft_backend` (`utils/backend.py`) to accept a `DistEnv` and carry the
  decomposition; `to_device`/`to_host` respect the per-rank device. `P=1` returns today's behavior
  exactly.

**Phase 1 — Distributed FFT (the heart).**
- New `baorecon/solvers/fft/_distributed_fft.py`: a `DistributedFFT` exposing
  `rfftn(real_slab) -> ky-split complex` and `irfftn(ky-split complex, s) -> x-split real`, implemented
  as local `cupy.fft` + NCCL AllToAll transpose (see §5). At `P=1` it delegates straight to `cupy.fft`.
- Make `prepare_k_components`/`build_inv_k2` (`solvers/fft/_common.py`) **rank-aware**: return the local
  ky slice and a rank-aware DC fix. `divergence_from_components` already takes injected `rfftn`/`irfftn`
  and per-axis k arrays — feed it the distributed transforms and local k.
- `solvers/fft/gpu.py`: swap `self.fft.rfftn/irfftn` for the `DistributedFFT`; slice k-arrays locally;
  add `global_x_offset` to the versor ElementwiseKernels (`gpu.py:58-76`); allocate all grids at local
  shape. The Burden loop structure is unchanged.

**Phase 2 — Distributed MAS (ghost zones).**
- `mas/_interface.py` + `mas/gpu.py`: allocate the local mesh with `w` halo planes; paint with the
  existing kernels; add a `halo_exchange_add` (NCCL/`Sendrecv`) that accumulates halos into owners with
  PBC wrap / clamp. Keep kernels untouched aside from operating on the extended grid.
- `reconstruction/density.py`: `compute_delta` uses an **AllReduce** for the global mean density before
  forming `delta` (the local mean is wrong).

**Phase 3 — Distributed field-ops & read-back.**
- `field_ops/_interface.py` + `field_ops/gpu.py`: `interpolate_vector_field` fills the read-halo (copy)
  before the local read kernels; `divergence_FFT` (`field_ops/gpu.py:17`) routes through the
  `DistributedFFT`.
- Implement `FFTSolverGPU.read_displacement_at` (`solvers/fft/gpu.py:277`) using the halo-aware
  interpolation, keeping the displacement device-resident — closing the existing TODO for both single-
  and multi-GPU.

**Phase 4 — Pipeline / I/O / launch.**
- `reconstruction/bao_reconstructor.py`: thread `DistEnv` through solver/LOS/density construction
  (device is already resolved once here, at `bao_reconstructor.py:133-134`).
- `pipeline/bao_pipeline.py`: **Scatter** particles by x-domain at load, **Gather** shifted particles
  for output (or parallel shard writes). Launch via
  `mpirun -np P python examples/run_bao_pipeline.py config.yaml`; each rank binds its GPU. Add
  `reconstruction.n_gpus`/rank binding to the YAML config and CLI.

**Phase 5 — Tests & benchmarks.**
- Add distributed tests run under `mpirun -np 2`, asserting **P=2 vs P=1 agreement** within float32 tol
  for MAS mass-conservation, FFT closure, and end-to-end displacement. Reuse the `_to_host()` gather
  pattern from `tests/test_density_manager.py`.
- Close the **GPU field_ops test gap** (`tests/test_field_ops.py` is CPU-only).
- Extend `benchmarks/` with per-rank `vram_peak_mb` and weak/strong scaling (P=1,2,4,8) at
  `Nmesh ∈ {1024, 2048}` to confirm the ~1/P memory scaling.

---

## 5. Code Example

### (a) Slab distributed 3-D real FFT (CuPy local FFT + NCCL AllToAll transpose)

```python
# baorecon/solvers/fft/_distributed_fft.py  (conceptual)
import cupy as cp
from cupy.cuda import nccl

class DistributedFFT:
    """Slab (1-D) 3-D rFFT. Real space split on x (axis 0); k-space split on ky (axis 1)."""
    def __init__(self, env, global_shape):        # global_shape = (Nx, Ny, Nz)
        self.env, self.P = env, env.world_size
        self.Nx, self.Ny, self.Nz = global_shape
        self.Nzh = self.Nz // 2 + 1
        self.Nx_loc = self.Nx // self.P           # assume divisible; else block-remainder
        self.Ny_loc = self.Ny // self.P

    def _alltoall(self, send, recv):
        """Regular all-to-all of equal-size device blocks via grouped NCCL send/recv."""
        comm, rank = self.env.nccl_comm, self.env.rank
        cnt = send.size // self.P
        nccl.groupStart()
        for peer in range(self.P):
            comm.send(send[peer * cnt:].data.ptr, cnt, nccl.NCCL_FLOAT32, peer, cp.cuda.Stream.null.ptr)
            comm.recv(recv[peer * cnt:].data.ptr, cnt, nccl.NCCL_FLOAT32, peer, cp.cuda.Stream.null.ptr)
        nccl.groupEnd()

    def rfftn(self, real_slab):                    # (Nx_loc, Ny, Nz) real  ->  (Nx, Ny_loc, Nzh) complex
        if self.P == 1:
            return cp.fft.rfftn(real_slab)
        # 1) local transforms on the two non-split axes (y, z); z is the rfft axis.
        a = cp.fft.rfftn(real_slab, axes=(1, 2))   # (Nx_loc, Ny, Nzh) complex
        # 2) AllToAll transpose: x-split -> ky-split. Pack P blocks of (Nx_loc, Ny_loc, Nzh).
        send = a.reshape(self.Nx_loc, self.P, self.Ny_loc, self.Nzh).transpose(1, 0, 2, 3).ravel()
        recv = cp.empty_like(send)
        self._alltoall(send.view(cp.float32), recv.view(cp.float32))
        b = recv.reshape(self.P, self.Nx_loc, self.Ny_loc, self.Nzh) \
                .transpose(1, 0, 2, 3).reshape(self.Nx, self.Ny_loc, self.Nzh)
        # 3) local FFT along x (now contiguous on this rank).
        return cp.fft.fft(b, axis=0)               # (Nx, Ny_loc, Nzh) complex, ky-split

    def irfftn(self, kgrid, s):                    # inverse: reverse the three steps
        if self.P == 1:
            return cp.fft.irfftn(kgrid, s=s)
        b = cp.fft.ifft(kgrid, axis=0)             # inverse along x
        send = b.reshape(self.Nx_loc, self.P, self.Ny_loc, self.Nzh).transpose(1, 0, 2, 3).ravel()
        recv = cp.empty_like(send)
        self._alltoall(send.view(cp.float32), recv.view(cp.float32))   # ky-split -> x-split
        a = recv.reshape(self.P, self.Nx_loc, self.Ny_loc, self.Nzh) \
                .transpose(1, 0, 2, 3).reshape(self.Nx_loc, self.Ny, self.Nzh)
        return cp.fft.irfftn(a, axes=(1, 2), s=(s[1], s[2]))           # (Nx_loc, Ny, Nz) real
```

The solver's iterative loop stays as-is; only the transform calls and k-array slicing change:

```python
# in FFTSolverGPU._compute_displacement_iterative_potential (sketch of the diff)
dfft = DistributedFFT(self.env, self.mesh.shape)
kx_h, ky_h, kz_h = self._k_components()               # ky_h already sliced to this rank's ky range
kx, ky, kz = xp.asarray(kx_h), xp.asarray(ky_h), xp.asarray(kz_h)
inv_k2_bias = build_inv_k2((kx, ky, kz), bias=self.bias, dc_owner=self.env.owns_dc)  # rank-aware DC
delta_k = dfft.rfftn(delta_dev)                        # (Nx, Ny_loc, Nzh)
# ... identical Burden updates, using dfft.rfftn / dfft.irfftn and local k_bcast ...
```

### (b) MAS ghost-zone halo synchronization (paint → accumulate into owners)

```python
# baorecon/mas/_interface.py  (conceptual, after painting into the haloed local grid)
def halo_exchange_add(mesh_ext, env, w, pbc):
    """mesh_ext: (Nx_loc + 2w, Ny, Nz). Add each halo plane into the neighbor's owned edge."""
    if env.world_size == 1:                     # single GPU: fold halos locally (PBC) and return
        if pbc:
            mesh_ext[w:2*w]       += mesh_ext[-w:]        # high halo -> low edge
            mesh_ext[-2*w:-w]     += mesh_ext[:w]         # low  halo -> high edge
        return mesh_ext[w:-w]

    lo, hi = env.left_rank, env.right_rank      # PBC ring: rank-1, rank+1 (mod P)
    comm = env.nccl_comm
    low_halo  = cp.ascontiguousarray(mesh_ext[:w])       # to be added into left neighbor's high edge
    high_halo = cp.ascontiguousarray(mesh_ext[-w:])      # to be added into right neighbor's low edge
    recv_from_left  = cp.empty_like(high_halo)
    recv_from_right = cp.empty_like(low_halo)
    nccl.groupStart()
    comm.send(high_halo.data.ptr, high_halo.size, nccl.NCCL_FLOAT32, hi, cp.cuda.Stream.null.ptr)
    comm.recv(recv_from_left.data.ptr, recv_from_left.size, nccl.NCCL_FLOAT32, lo, cp.cuda.Stream.null.ptr)
    comm.send(low_halo.data.ptr, low_halo.size, nccl.NCCL_FLOAT32, lo, cp.cuda.Stream.null.ptr)
    comm.recv(recv_from_right.data.ptr, recv_from_right.size, nccl.NCCL_FLOAT32, hi, cp.cuda.Stream.null.ptr)
    nccl.groupEnd()
    owned = mesh_ext[w:-w]
    if pbc or not env.is_low_edge:  owned[:w]  += recv_from_left    # add neighbor's high halo
    if pbc or not env.is_high_edge: owned[-w:] += recv_from_right   # add neighbor's low  halo
    return owned                                 # (Nx_loc, Ny, Nz), fully accumulated
```

The read-back halo is the same pattern with **copy** instead of add (fill ghost planes, then run the
existing `read_cic`/`read_tsc` kernels on the extended displacement grid with an x-index offset).

---

## Verification

- **Unit (single GPU, P=1):** existing GPU-gated suites must pass unchanged —
  `tests/test_density_manager.py`, `tests/test_fft_solver.py`, `tests/test_radial_kernels.py`
  (`pytest -k gpu` on a CUDA host). This proves the `P=1` fast path is a true no-op.
- **Distributed correctness:** new tests under `mpirun -np 2 python -m pytest tests/test_distributed.py`:
  1. MAS mass conservation — total painted weight (AllReduce) equals `sum(weights)`, and P=2 grid
     (gathered) matches the P=1 grid within float32 tol, for CIC & TSC, `pbc ∈ {True, False}`.
  2. FFT closure — `irfftn(rfftn(x)) ≈ x`; and P=2 `delta_k`/displacement (gathered) match P=1.
  3. End-to-end — `BAOReconstructor` displacement (gathered) matches the single-GPU result within the
     GRF tolerance already used in `tests/test_fft_solver.py` (atol ~1e-5).
- **Close the GPU field_ops gap:** parametrize `tests/test_field_ops.py` over `device` with the
  `gpu_test` skip mark (currently CPU-only).
- **Memory scaling (the actual goal):** run `benchmarks/bench_scaling.py` extended for P=1,2,4,8 at
  `Nmesh ∈ {1024, 2048}`; confirm per-rank `vram_peak_mb` scales ~1/P and that a `2048³` solve that OOMs
  on one GPU completes on P=4/8.
- **End-to-end run:** `mpirun -np 4 python examples/run_bao_pipeline.py examples/bao_pipeline_example.yaml`
  (with `reconstruction.device: gpu`); verify the reconstructed catalog matches a single-GPU run and the
  pyrecon cross-check (`test_density_manager_vs_pyrecon`, correlation > 0.99).

## Notes / Risks
- **NCCL + numba.cuda coexistence:** both CUDA runtimes must bind the *same* device per rank
  (`cp.cuda.Device(...).use()` + `numba.cuda.select_device(...)`). Verify contexts agree at startup.
- **AllToAll dominates comm:** the Burden loop issues several transforms/iteration; batch/overlap where
  possible, but correctness first. Slab caps at `P ≤ Nx`; fine for ≤8 GPUs.
- **Remainder handling:** `Nx % P ≠ 0` and `Ny % P ≠ 0` need block-with-remainder distribution and an
  AllToAllv (unequal blocks) rather than the equal-block AllToAll shown; start by requiring divisibility,
  then generalize.
- **Precision:** MAS is float32-only on GPU today (`mas/_interface.py`); the transpose buffers above
  assume float32 (`NCCL_FLOAT32`). Keep that invariant or parametrize the NCCL dtype.
