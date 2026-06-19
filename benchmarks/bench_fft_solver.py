"""Benchmark of the iterative FFT displacement solver (3 iterations).

For each backend the work is split into three timed phases:

* ``setup``   -- paint data + randoms and build the density contrast ``delta``,
* ``solve``   -- run the iterative FFT solve (3 iterations) for the displacement,
* ``readout`` -- interpolate the displacement field back at the data positions.

Backends:

* ``baorecon_cpu`` -- ``DensityManager`` + ``FFTSolver`` (RedshiftSpace ->
  iterative potential, 3 iterations),
* ``pyrecon`` -- ``IterativeFFTReconstruction`` (``assign_*`` + ``set_density_contrast``
  / ``run(niterations=3)`` / ``read_shifts``).

Particle count is fixed at 1e7 (data and randoms 1:1); the mesh is swept over
256, 512, 1024.

Each ``(backend, nmesh)`` configuration runs in its own subprocess (see
``bench_common.spawn_worker``) so that ``ru_maxrss`` measures one backend's peak
memory in isolation. The parent only launches workers and aggregates their JSON
output; it never imports zeldareco or pyrecon.

Run::

    python benchmarks/bench_fft_solver.py
    python benchmarks/bench_fft_solver.py --quick
"""

from __future__ import annotations

import argparse

import numpy as np

import bench_common as bc

N_PARTICLES = int(1e7)
NMESHES = [256, 512, 1024]
N_ITERATIONS = 3
F_GROWTH = 0.8
BIAS = 2.0
R_SMOOTH = 15.0
SEED_DATA = 1
SEED_RANDOM = 2


# ---------------------------------------------------------------------------
# Worker side (runs inside a per-backend subprocess)
# ---------------------------------------------------------------------------
def _worker_baorecon(n, nmesh, repeats):
    from zeldareco.BAOreconstruction.density_manager import DensityManager
    from zeldareco.displacement_solver.fft_solver import FFTSolver
    from zeldareco.mesh.field_ops import interpolate_vector_field

    backend = "baorecon_cpu"
    data_pos, weights_d = bc.gen_positions(n, bc.BOXSIZE, seed=SEED_DATA)
    random_pos, weights_r = bc.gen_positions(n, bc.BOXSIZE, seed=SEED_RANDOM)
    bc.reset_memory_baseline()

    boxcentre = np.full(3, bc.BOXSIZE / 2.0, dtype=np.float32)

    def _make_density_manager():
        # los='z' so the RedshiftSpace solve uses a fixed line of sight along z.
        return DensityManager(
            data_pos=data_pos, random_pos=random_pos, nmesh=nmesh,
            boxsize=bc.BOXSIZE, boxcentre=boxcentre, padding=0.0,
            MAS="CIC", dtype=np.float32, pbc=True,
            smoothing_radius=R_SMOOTH, los="z",
        )

    rows = []

    # --- setup / density contrast ---
    dm = _make_density_manager()
    m = bc.safe_measure(lambda: dm.compute_delta(),
                        label=f"{backend} setup nmesh={nmesh}", repeats=repeats)
    if m is not None:
        rows.append(m.as_row(backend, n, nmesh, "setup"))

    # Inputs shared by the next two phases.
    delta = dm.compute_delta()
    mesh = dm.mesh
    box_pos = dm.data_pos_box

    # --- solve (fresh solver each repeat so the cache does not short-circuit it) ---
    def solve():
        solver = FFTSolver(delta, mesh, f=F_GROWTH, bias=BIAS,
                           RSDspace="RedshiftSpace")
        return solver.displacement

    m = bc.safe_measure(solve, label=f"{backend} solve nmesh={nmesh}", repeats=repeats)
    if m is not None:
        rows.append(m.as_row(backend, n, nmesh, "solve"))

    # --- readout ---
    displacement = FFTSolver(delta, mesh, f=F_GROWTH, bias=BIAS,
                             RSDspace="RedshiftSpace").displacement

    def readout():
        return interpolate_vector_field(box_pos, displacement, mesh.boxsize,
                                        MAS="CIC", pbc=True)

    m = bc.safe_measure(readout, label=f"{backend} readout nmesh={nmesh}", repeats=repeats)
    if m is not None:
        rows.append(m.as_row(backend, n, nmesh, "readout"))

    return rows


def _worker_pyrecon(n, nmesh, repeats):
    from pyrecon import IterativeFFTReconstruction

    data_pos, weights_d = bc.gen_positions(n, bc.BOXSIZE, seed=SEED_DATA)
    random_pos, weights_r = bc.gen_positions(n, bc.BOXSIZE, seed=SEED_RANDOM)
    bc.reset_memory_baseline()

    def _new_recon():
        return IterativeFFTReconstruction(
            f=F_GROWTH, bias=BIAS, los=None, nmesh=nmesh, boxsize=bc.BOXSIZE,
            boxcenter=bc.BOXSIZE / 2.0, dtype="f4",
        )

    rows = []

    # --- setup / density contrast ---
    def setup():
        recon = _new_recon()
        recon.assign_data(data_pos, weights_d)
        recon.assign_randoms(random_pos, weights_r)
        recon.set_density_contrast(smoothing_radius=R_SMOOTH)
        return recon

    m = bc.safe_measure(setup, label=f"pyrecon setup nmesh={nmesh}", repeats=repeats)
    if m is not None:
        rows.append(m.as_row("pyrecon", n, nmesh, "setup"))

    # A density-contrast-ready reconstruction reused by solve/readout.
    base = setup()

    # --- solve: run() mutates state, so copy before each iteration ---
    def solve():
        recon = base.copy()
        recon.run(niterations=N_ITERATIONS)
        return recon

    m = bc.safe_measure(solve, label=f"pyrecon solve nmesh={nmesh}", repeats=repeats)
    if m is not None:
        rows.append(m.as_row("pyrecon", n, nmesh, "solve"))

    # --- readout ---
    solved = base.copy()
    solved.run(niterations=N_ITERATIONS)

    def readout():
        return solved.read_shifts(data_pos, field="disp+rsd")

    m = bc.safe_measure(readout, label=f"pyrecon readout nmesh={nmesh}", repeats=repeats)
    if m is not None:
        rows.append(m.as_row("pyrecon", n, nmesh, "readout"))

    return rows


def worker(spec):
    backend = spec["backend"]
    n = int(spec["n_particles"])
    nmesh = int(spec["nmesh"])
    repeats = int(spec["repeats"])
    if backend == "pyrecon":
        return _worker_pyrecon(n, nmesh, repeats)
    return _worker_baorecon(n, nmesh, repeats)


# ---------------------------------------------------------------------------
# Parent side (orchestration only -- no zeldareco / pyrecon imports)
# ---------------------------------------------------------------------------
def run(n_particles, nmeshes, repeats):
    info = bc.system_info()
    rows = []
    print(f"FFT solver benchmark | N={n_particles:.0e} | iterations={N_ITERATIONS} "
          f"| repeats={repeats}\n")

    backends = ["baorecon_cpu", "pyrecon"]

    for nmesh in nmeshes:
        print(f"nmesh={nmesh}")
        for backend in backends:
            label = f"{backend} nmesh={nmesh}"
            spec = {"backend": backend, "n_particles": n_particles, "nmesh": nmesh,
                    "repeats": repeats}
            rows.extend(bc.spawn_worker(__file__, spec, label=label))

    bc.print_table(rows, title="FFT solver (setup / solve / readout)")
    bc.save_csv(bc.RESULTS_DIR / "fft_solver.csv", rows, info)
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
        run(int(1e5), [32, 64], repeats=max(2, args.repeats))
    else:
        run(N_PARTICLES, NMESHES, repeats=args.repeats)


if __name__ == "__main__":
    main()
