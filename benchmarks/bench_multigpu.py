"""Multi-GPU scaling benchmark: per-rank VRAM peak and wall time.

Runs the distributed MAS paint + smoothing + iterative FFT displacement solve
on a Gaussian-random-field delta and reports, per rank, the peak CuPy memory
pool usage and phase timings. Run the same nmesh at several P to check the
~1/P per-GPU memory scaling that motivates the migration
(docs/multigpu_migration_plan.md):

    mpirun -np 1 python benchmarks/bench_multigpu.py --nmesh 512
    mpirun -np 2 python benchmarks/bench_multigpu.py --nmesh 512
    mpirun -np 4 python benchmarks/bench_multigpu.py --nmesh 512

P=1 (or launching without mpirun) runs the unchanged single-GPU path, so the
P=1 row is the true baseline.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nmesh", type=int, default=256)
    parser.add_argument("--boxsize", type=float, default=1000.0)
    parser.add_argument("--npart", type=int, default=1_000_000)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--los", choices=["local", "z"], default="local")
    args = parser.parse_args()

    import cupy as cp

    from baorecon.mas import assign
    from baorecon.mesh.los import FixedAxisLOS, LocalLOS
    from baorecon.mesh.mesh import Mesh
    from baorecon.solvers.fft import FFTSolverGPU
    from baorecon.utils.distributed import auto_dist_env

    env = auto_dist_env()
    mesh = Mesh(nmesh=args.nmesh, boxsize=args.boxsize,
                boxcentre=np.full(3, args.boxsize / 2))
    pool = cp.get_default_memory_pool()

    rng = np.random.default_rng(0)
    pos = rng.uniform(0, args.boxsize, size=(args.npart, 3)).astype(np.float32)

    timings, peaks = {}, {}

    def _phase(name, fn):
        pool.free_all_blocks()
        cp.cuda.Device().synchronize()
        t0 = time.perf_counter()
        out = fn()
        cp.cuda.Device().synchronize()
        timings[name] = time.perf_counter() - t0
        peaks[name] = pool.total_bytes() / 2**20
        return out

    grid = _phase("mass_assignment",
                  lambda: assign(pos, None, mesh, scheme="CIC", device="gpu",
                                 pbc=True, dist=env))
    del pos

    # GRF-ish delta from the painted grid (normalize to zero mean, small rms)
    mean = env.allreduce_sum(float(grid.sum())) / float(np.prod(mesh.shape))
    delta = (grid / mean - 1.0).astype(cp.float32)
    del grid

    if args.los == "local":
        los = LocalLOS(boxcentre=mesh.boxcentre, min_corner=mesh.min_corner,
                       boxsize=mesh.boxsize, nmesh=mesh.nmesh, device="gpu")
    else:
        los = FixedAxisLOS(2)

    def _solve():
        solver = FFTSolverGPU(delta_on_mesh=delta, mesh=mesh, los=los,
                              f=0.8, bias=1.5, RSDspace="RedshiftSpace",
                              n_iterations=args.iters, dist=env)
        return solver.displacement

    disp = _phase("fft_solver", _solve)
    del disp

    record = {
        "rank": env.rank,
        "world_size": env.world_size,
        "nmesh": args.nmesh,
        "timings_s": {k: round(v, 3) for k, v in timings.items()},
        "vram_peak_mb": {k: round(v, 1) for k, v in peaks.items()},
    }
    if env.is_distributed:
        records = env.comm.mpi.gather(record, root=0)
    else:
        records = [record]
    if env.rank == 0:
        print(json.dumps(records, indent=2))
    env.barrier()


if __name__ == "__main__":
    main()
