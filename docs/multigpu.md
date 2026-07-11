# Multi-GPU reconstruction (slab decomposition)

`baorecon` can distribute the GPU reconstruction across the GPUs of a single
node. The mesh is split in contiguous x-slabs (one rank -- one GPU); per-GPU
memory for every grid scales as ~1/P, lifting the `Nmesh` ceiling set by a
single GPU's VRAM. Design background: `docs/multigpu_migration_plan.md`
(original plan) and `docs/multigpu_migration_audit.md` (audit + corrections
that this implementation follows).

## Requirements

* `device: gpu` prerequisites (CuPy matching your CUDA toolkit; see
  `docs/gpu_environment.md`),
* `mpi4py` on top of a working MPI (e.g. OpenMPI) -- `requirements/gpu.txt`,
* NCCL, which ships inside the `cupy-cuda12x` wheel (`cupy.cuda.nccl`),
* `Nx % P == 0` and `Ny % P == 0` (equal slab blocks); the run aborts with a
  clear error otherwise. `P` must not exceed the number of x-planes.

## Running

The YAML pipeline is unchanged -- only the launcher differs:

```bash
# single GPU (unchanged)
python examples/run_bao_pipeline.py examples/bao_pipeline_example.yaml

# 4 GPUs on one node
mpirun -np 4 python examples/run_bao_pipeline.py examples/bao_pipeline_example.yaml
```

With `reconstruction.device: gpu` and more than one MPI rank, each rank binds
GPU `rank % n_gpus` (for both CuPy and numba.cuda), owns an x-slab of the mesh,
and communicates over NCCL. Rank 0 writes all outputs (grid slabs are gathered
to it for FITS saves); the reconstructed catalogues are identical on every
rank. Launching with `-np 1` -- or without `mpirun` at all -- takes the
unchanged single-GPU code path everywhere.

The `BAOReconstructor` API accepts the environment directly:

```python
from baorecon.utils.distributed import auto_dist_env
from baorecon import BAOReconstructor

env = auto_dist_env()          # serial unless launched under mpirun
rec = BAOReconstructor(..., device="gpu", solver_type="ifft", dist=env)
data_rec, rand_rec = rec.run_reconstruction()   # full catalogues, every rank
```

## What is distributed, and how

| Piece | Decomposition | Communication |
|---|---|---|
| 3-D rFFT/irFFT (`DistributedFFT`) | real space x-split -> k-space ky-split | 1 NCCL AllToAll per transform |
| MAS painting | halo-extended x-slab (CIC w=1, TSC w=2) | halo accumulate to ±x neighbours |
| Gaussian smoothing | via `DistributedFFT`, local ky factor | the FFT's AllToAll |
| `delta` normalisation (`alpha`, threshold) | slab-local sums | 2 scalar AllReduce |
| Burden loop / displacement | all grids slab-local | only the FFTs communicate |
| Grid -> particle read-back | ghost planes filled by copy | halo copy from ±x neighbours |
| Catalogue shifts | ranks read their own particles | in-place AllReduce of disjoint rows |
| Grid FITS saves | slabs gathered to rank 0 | mpi4py gather |

Particles never migrate: every rank holds the full catalogue on the host
(so the derived box geometry is identical everywhere) and paints/reads only
the particles whose x-cell falls in its slab. Host RAM is not the constrained
resource -- VRAM is.

Limitations (current, by design -- see the audit):

* multi-GPU requires `solver_type: ifft` + `device: gpu` (multigrid and CPU
  solvers stay single-process);
* equal slabs only (no AllToAllv yet), single node (NCCL ring over
  NVLink/PCIe);
* the `reconstructor_object` pickle save is skipped in distributed mode;
* GPU MAS remains float32 (unchanged from single-GPU).

## Testing and benchmarking

* `pytest tests/test_distributed.py` -- CPU-only: decomposition, the FFT
  transpose layout, halo folds, distributed smoothing and reductions, all
  verified with numpy over an in-process loopback communicator (runs
  everywhere, no GPU/MPI needed).
* `mpirun -np 2 python -m pytest tests/test_distributed_gpu.py -q` -- the
  NCCL integration suite: P=2 vs P=1 agreement for MAS painting (CIC/TSC,
  pbc on/off, mass conservation), FFT forward/closure, the solver
  displacement (radial and plane-parallel LOS) and the end-to-end
  reconstructor.
* `mpirun -np P python benchmarks/bench_multigpu.py --nmesh 1024` -- per-rank
  `vram_peak_mb` and phase timings; run at P = 1, 2, 4, 8 to verify the ~1/P
  per-GPU memory scaling.
