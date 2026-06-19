"""End-to-end pipeline benchmark.

Measures the total wall-time and peak memory of a full reconstruction.

All backends are compared starting from the *same* pre-computed Cartesian
positions: the RA/DEC/z -> XYZ coordinate conversion is performed once (per
worker, outside every timer and before the memory baseline), so it is excluded
for baorecon and pyrecon alike.

* ``baorecon_cpu`` -- the timed region is ``BAOReconstructor.run_reconstruction``
  (density contrast, iterative FFT / multigrid solve, shifting data + randoms).
* ``pyrecon`` -- the timed region is ``assign_data`` / ``assign_randoms`` /
  ``set_density_contrast`` / ``run`` / ``read_shifted_positions``.

Particle count is swept over 1e5, 1e6, 1e7 (data and randoms 1:1); mesh fixed at
512^3.

Each ``(backend, n_particles)`` configuration runs in its own subprocess (see
``bench_common.spawn_worker``) so that ``ru_maxrss`` measures one backend's peak
memory in isolation. The parent only launches workers and aggregates their JSON
output; it never imports zeldareco or pyrecon.

Run::

    python benchmarks/bench_pipeline.py
    python benchmarks/bench_pipeline.py --quick
"""
from __future__ import annotations

import os

# Definisci il numero di thread che vuoi dedicare al processo
THREADS = "8"

# 1. Numba Threading
#os.environ["NUMBA_THREADING_LAYER"] = "tbb"
os.environ["NUMBA_NUM_THREADS"] = THREADS

# 2. NumPy backend Threading (copre MKL, OpenBLAS e OpenMP standard)
os.environ["OMP_NUM_THREADS"] = THREADS
os.environ["MKL_NUM_THREADS"] = THREADS
os.environ["OPENBLAS_NUM_THREADS"] = THREADS
os.environ["NUMEXPR_NUM_THREADS"] = THREADS


import argparse

import numpy as np

import bench_common as bc

import sys
sys.path.insert(0, '/home/emaragliano/Work/Projects/Dottorato/baorecon_main')

NMESH = [128, 256, 512, 1024]
N_PARTICLES = [int(1e6)]
N_ITERATIONS = 3
F_GROWTH = 0.8
BIAS = 2.0
R_SMOOTH = 15.0
PADDING = 0.02
SEED_DATA = 1
SEED_RANDOM = 2


# ---------------------------------------------------------------------------
# Worker side (runs inside a per-backend subprocess)
# ---------------------------------------------------------------------------
def _make_mock_xyz(n):
    """Deterministic RA/DEC/z mock converted to Cartesian (data, randoms)."""
    from zeldareco.utils.coordinates import create_cosmology, radec_z_to_xyz

    cosmo = create_cosmology()
    d_ra, d_dec, d_z = bc.gen_radec_z(n, seed=SEED_DATA)
    r_ra, r_dec, r_z = bc.gen_radec_z(n, seed=SEED_RANDOM)
    data_xyz, _ = radec_z_to_xyz(d_ra, d_dec, d_z, cosmo=cosmo)
    random_xyz, _ = radec_z_to_xyz(r_ra, r_dec, r_z, cosmo=cosmo)
    return data_xyz, random_xyz


def _worker_baorecon(n, nmesh, repeats, solver_type):
    from zeldareco.BAOreconstruction.bao_reconstructor import BAOReconstructor

    backend = "baorecon_cpu"
    data_xyz, random_xyz = _make_mock_xyz(n)
    weights_d = np.ones(len(data_xyz), dtype=np.float32)
    weights_r = np.ones(len(random_xyz), dtype=np.float32)

    # Cubic box derived exactly as pyrecon does, using the padding as a percentage
    # (2%) and without wasting RAM.
    lo = np.minimum(data_xyz.min(axis=0), random_xyz.min(axis=0))
    hi = np.maximum(data_xyz.max(axis=0), random_xyz.max(axis=0))

    boxcentre = (lo + hi) / 2.0
    boxsize = float((hi - lo).max()) * (1.0 + PADDING)

    bc.reset_memory_baseline()

    def run():
        recon = BAOReconstructor(
            data_pos=data_xyz, random_pos=random_xyz,
            data_weights=weights_d, random_weights=weights_r,
            RSDspace="RedshiftSpace", 
            nmesh=nmesh,
            boxsize=boxsize,           # explicit padded cubic box
            boxcentre=boxcentre,       # explicit box centre
            padding=0.0,               # ignored: the box is already padded
            los=None, R_sm=R_SMOOTH, pbc=False, rectype="rec-sym",
            f=F_GROWTH, bias=BIAS, MAS="CIC", dtype=np.float32,
            solver_type=solver_type, mas_parallel=True
        )
        return recon.run_reconstruction()

    m = bc.safe_measure(run, label=f"{backend} N={n:.0e}", repeats=repeats)
    return [m.as_row(backend, n, nmesh, "total")] if m is not None else []


def _worker_pyrecon(n, nmesh, repeats, solver_type):
    from pyrecon import MultiGridReconstruction, IterativeFFTReconstruction

    data_xyz, random_xyz = _make_mock_xyz(n)
    weights_d = np.ones(len(data_xyz), dtype=np.float32)
    weights_r = np.ones(len(random_xyz), dtype=np.float32)

    # Box derived once from the (already converted) cartesian positions.
    lo, hi = random_xyz.min(axis=0), random_xyz.max(axis=0)
    boxcenter = (lo + hi) / 2.0
    boxsize = float((hi - lo).max()) * (1.0 + PADDING)
    bc.reset_memory_baseline()

    def run():
        if solver_type == "ifft":
            recon = IterativeFFTReconstruction(
                f=F_GROWTH, bias=BIAS, los=None, nmesh=nmesh, boxsize=boxsize,
                boxcenter=boxcenter, dtype="f4",
            )
        elif solver_type == "multigrid":
            recon = MultiGridReconstruction(
                f=F_GROWTH, bias=BIAS, los=None, nmesh=nmesh, boxsize=boxsize,
                boxcenter=boxcenter, dtype="f4",
            )
        recon.assign_data(data_xyz, weights_d)
        recon.assign_randoms(random_xyz, weights_r)
        recon.set_density_contrast(smoothing_radius=R_SMOOTH)
        recon.run()
        # Shift data and randoms
        shifted_d = recon.read_shifted_positions(data_xyz)
        shifted_r = recon.read_shifted_positions(random_xyz)
        return shifted_d, shifted_r

    m = bc.safe_measure(run, label=f"pyrecon N={n:.0e}", repeats=repeats)
    return [m.as_row("pyrecon", n, nmesh, "total")] if m is not None else []


def worker(spec):
    backend = spec["backend"]
    n = int(spec["n_particles"])
    nmesh = int(spec["nmesh"])
    repeats = int(spec["repeats"])
    solver = str(spec["solver"])
    if backend == "pyrecon":
        return _worker_pyrecon(n, nmesh, repeats, solver)
    return _worker_baorecon(n, nmesh, repeats, solver)


# ---------------------------------------------------------------------------
# Parent side (orchestration only -- no zeldareco / pyrecon imports)
# ---------------------------------------------------------------------------
def run(n_particles, nmeshes, repeats, solver):
    info = bc.system_info()
    rows = []
    print(f"Pipeline (end-to-end) benchmark | solver={solver} | repeats={repeats}\n")

    backends = ["baorecon_cpu", "pyrecon"]

    for n in n_particles:
        print(f"N={n:.0e}")
        for nmesh in nmeshes: # <-- Nuovo ciclo per iterare su NMESH
            print(f"nmesh={nmesh}")
            for backend in backends:
                print(f"backend: {backend}")
                label = f"{backend} N={n:.0e} mesh={nmesh}"
                spec = {"backend": backend, "n_particles": n, "nmesh": nmesh,
                        "repeats": repeats, "solver": solver}
                rows.extend(bc.spawn_worker(__file__, spec, label=label))

    bc.print_table(rows, title="End-to-end pipeline")
    bc.save_csv(bc.RESULTS_DIR / "pipeline.csv", rows, info)
    return rows


def main():
    if bc.is_worker():
        bc.run_worker(worker)
        return

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                        help="small parameter set for a fast sanity run")
    parser.add_argument("--repeats", type=int, default=1,
                        help="number of timed repetitions (default 5)")
    parser.add_argument("--solver", type=str, choices=["multigrid", "ifft"],
                        default="multigrid", help="either multigrid or ifft (default ifft)")
    args = parser.parse_args()

    if args.quick:
        run([int(1e4), int(1e5)], [64], repeats=max(2, args.repeats), solver=args.solver)
    else:
        run(N_PARTICLES, NMESH, repeats=args.repeats, solver=args.solver)


if __name__ == "__main__":
    main()
