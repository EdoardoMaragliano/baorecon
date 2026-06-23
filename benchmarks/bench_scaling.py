"""Mesh-resolution scaling benchmark (baorecon CPU vs GPU).

Particle count is fixed at 1e7; the mesh is swept over 128, 256, 512, 1024, 2048.
For every mesh size two phases are measured for each backend:

* ``mass_assignment`` -- CIC paint of the data onto the mesh,
* ``fft_solver``      -- the iterative FFT displacement solve on a precomputed delta.

No pyrecon here -- this isolates how baorecon's own CPU and GPU paths scale with
resolution. Each ``(backend, nmesh)`` configuration runs in its own subprocess
(see ``bench_common.spawn_worker``) so that ``ru_maxrss`` measures one backend's
peak memory in isolation; the parent only launches workers and aggregates JSON.

The largest meshes are memory hungry (a 2048^3 complex displacement solve needs
tens of GB of RAM/VRAM); configurations that run out of memory are skipped and
recorded as missing rather than aborting the sweep.

Run::

    python benchmarks/bench_scaling.py
    python benchmarks/bench_scaling.py --quick
"""

from __future__ import annotations

# Number of threads to dedicate to the process
THREADS = "8"

import os

# 1. Numba threading
#os.environ["NUMBA_THREADING_LAYER"] = "tbb"
os.environ["NUMBA_NUM_THREADS"] = THREADS

# 2. NumPy backend threading (covers MKL, OpenBLAS, and standard OpenMP)
os.environ["OMP_NUM_THREADS"] = THREADS
os.environ["MKL_NUM_THREADS"] = THREADS
os.environ["OPENBLAS_NUM_THREADS"] = THREADS
os.environ["NUMEXPR_NUM_THREADS"] = THREADS

import argparse
import sys

import numpy as np

import bench_common as bc

N_PARTICLES = int(1e7)
NMESHES = [128, 256, 512, 1024, 2048]
N_ITERATIONS = 3
F_GROWTH = 0.8
BIAS = 2.0
R_SMOOTH = 15.0
SEED_DATA = 1
SEED_RANDOM = 2


# ---------------------------------------------------------------------------
# Worker side (runs inside a per-backend subprocess)
# ---------------------------------------------------------------------------
def worker(spec):
    from baorecon.reconstruction.density import DensityManager
    from baorecon.solvers.fft import FFTSolverCPU, FFTSolverGPU
    from baorecon.mesh.los import FixedAxisLOS
    from baorecon.mesh.mesh import Mesh
    from baorecon.mas import assign as mas_assign

    backend = spec["backend"]
    n = int(spec["n_particles"])
    nmesh = int(spec["nmesh"])
    repeats = int(spec["repeats"])
    device = "gpu" if backend.endswith("gpu") else "cpu"
    SolverCls = FFTSolverGPU if device == "gpu" else FFTSolverCPU

    data_pos, weights = bc.gen_positions(n, bc.BOXSIZE, seed=SEED_DATA)
    random_pos, _ = bc.gen_positions(n, bc.BOXSIZE, seed=SEED_RANDOM)
    bc.reset_memory_baseline()

    rows = []

    # --- mass assignment ---
    mesh_ma = Mesh(nmesh=nmesh, boxsize=bc.BOXSIZE, boxcentre=np.full(3, bc.BOXSIZE / 2.0))

    def assign():
        return mas_assign(data_pos, weights, mesh_ma, scheme="CIC", pbc=True, device=device)

    m = bc.safe_measure(assign, label=f"{backend} mass_assignment nmesh={nmesh}",
                        repeats=repeats, device=device)
    if m is not None:
        rows.append(m.as_row(backend, n, nmesh, "mass_assignment"))

    # --- fft solver (precompute a realistic delta once, guarded against OOM) ---
    try:
        boxcentre = np.full(3, bc.BOXSIZE / 2.0, dtype=np.float32)
        dm = DensityManager(
            data_pos=data_pos, random_pos=random_pos, nmesh=nmesh, boxsize=bc.BOXSIZE,
            boxcentre=boxcentre, padding=0.0, MAS="CIC", dtype=np.float32, pbc=True,
            smoothing_radius=R_SMOOTH, device=device,
        )
        delta = dm.compute_delta()
        mesh = dm.mesh
    except Exception as exc:  # noqa: BLE001
        print(f"  [skip] {backend} fft_solver nmesh={nmesh}: "
              f"could not build delta ({type(exc).__name__}: {exc})", file=sys.stderr)
        return rows

    def solve():
        solver = SolverCls(delta, mesh, los=FixedAxisLOS(2), f=F_GROWTH, bias=BIAS,
                           RSDspace="RedshiftSpace")
        return solver.displacement

    m = bc.safe_measure(solve, label=f"{backend} fft_solver nmesh={nmesh}",
                        repeats=repeats, device=device)
    if m is not None:
        rows.append(m.as_row(backend, n, nmesh, "fft_solver"))

    return rows


# ---------------------------------------------------------------------------
# Parent side (orchestration only -- no baorecon / pyrecon imports)
# ---------------------------------------------------------------------------
def run(n_particles, nmeshes, repeats):
    info = bc.system_info()
    rows = []
    print(f"Scaling benchmark | N={n_particles:.0e} | repeats={repeats}")
    print(f"GPU available: {bc.GPU_AVAILABLE}\n")

    backends = ["baorecon_cpu"]
    if bc.GPU_AVAILABLE:
        backends.append("baorecon_gpu")

    for nmesh in nmeshes:
        print(f"nmesh={nmesh}")
        for backend in backends:
            label = f"{backend} nmesh={nmesh}"
            spec = {"backend": backend, "n_particles": n_particles, "nmesh": nmesh,
                    "repeats": repeats}
            rows.extend(bc.spawn_worker(__file__, spec, label=label))

    bc.print_table(rows, title="Mesh-resolution scaling")
    bc.save_csv(bc.RESULTS_DIR / "scaling.csv", rows, info)
    return rows


def main():
    if bc.is_worker():
        bc.run_worker(worker)
        return

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                        help="small parameter set for a fast sanity run")
    parser.add_argument("--repeats", type=int, default=5,
                        help="number of timed repetitions (default 5)")
    args = parser.parse_args()

    if args.quick:
        run(int(1e5), [32, 64, 128], repeats=max(2, args.repeats))
    else:
        run(N_PARTICLES, NMESHES, repeats=args.repeats)


if __name__ == "__main__":
    main()
