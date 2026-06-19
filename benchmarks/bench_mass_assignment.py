"""Benchmark of CIC / TSC mass assignment.

Compares two backends painting the same uniform mock onto a fixed 256^3 mesh:

* ``baorecon_cpu`` -- ``zeldareco.mass_assignment.mass_assignment`` (Numba)
* ``pyrecon``      -- ``pmesh`` ``RealField.paint`` with the matching resampler

Particle count is swept over 1e5 ... 1e8; the mesh is fixed at 256^3.

Each ``(backend, n_particles, method)`` configuration runs in its own Python
subprocess (see ``bench_common.spawn_worker``) so that ``ru_maxrss`` measures one
backend's peak memory without the others inflating it. The parent process only
launches workers and aggregates their JSON output; it never imports zeldareco or
pyrecon.

Run::

    python benchmarks/bench_mass_assignment.py            # full sweep (heavy!)
    python benchmarks/bench_mass_assignment.py --quick    # tiny smoke run
"""

from __future__ import annotations

import argparse

import numpy as np

import bench_common as bc

NMESH = 256
N_PARTICLES = [int(5e5), int(1e6), int(5e6)]
METHODS = ["CIC", "TSC"]
SEED = 42

# pmesh resampler names matching baorecon's MAS schemes.
_PMESH_RESAMPLER = {"CIC": "cic", "TSC": "tsc"}


# ---------------------------------------------------------------------------
# Worker side (runs inside a per-backend subprocess)
# ---------------------------------------------------------------------------
def _worker_baorecon(n, nmesh, method, repeats):
    from zeldareco.mass_assignment import mass_assignment

    backend = "baorecon_cpu"
    pos, weights = bc.gen_positions(n, bc.BOXSIZE, seed=SEED)
    bc.reset_memory_baseline()

    def run():
        return mass_assignment(pos, bc.BOXSIZE, nmesh, weights=weights,
                               method=method, pbc=True, dtype=np.float32)

    m = bc.safe_measure(run, label=f"{backend} {method} N={n:.0e}", repeats=repeats)
    return [m.as_row(backend, n, nmesh, method)] if m is not None else []


def _worker_pyrecon(n, nmesh, method, repeats):
    from pmesh.pm import ParticleMesh

    pos, weights = bc.gen_positions(n, bc.BOXSIZE, seed=SEED)
    resampler = _PMESH_RESAMPLER[method]
    pm = ParticleMesh(BoxSize=bc.BOXSIZE, Nmesh=[nmesh] * 3,
                      dtype="f4", resampler=resampler)
    bc.reset_memory_baseline()

    def run():
        # paint -> a fresh RealField initialised to zero (hold defaults to False).
        return pm.paint(pos, mass=weights, resampler=resampler)

    m = bc.safe_measure(run, label=f"pyrecon {method} N={n:.0e}", repeats=repeats)
    return [m.as_row("pyrecon", n, nmesh, method)] if m is not None else []


def worker(spec):
    backend = spec["backend"]
    n = int(spec["n_particles"])
    nmesh = int(spec["nmesh"])
    method = spec["method"]
    repeats = int(spec["repeats"])
    if backend == "pyrecon":
        return _worker_pyrecon(n, nmesh, method, repeats)
    return _worker_baorecon(n, nmesh, method, repeats)


# ---------------------------------------------------------------------------
# Parent side (orchestration only -- no zeldareco / pyrecon imports)
# ---------------------------------------------------------------------------
def run(n_particles, methods, repeats):
    info = bc.system_info()
    rows = []
    print(f"Mass assignment benchmark | nmesh={NMESH} | repeats={repeats}\n")

    backends = ["baorecon_cpu", "pyrecon"]

    for n in n_particles:
        for method in methods:
            print(f"N={n:.0e} method={method}")
            for backend in backends:
                label = f"{backend} {method} N={n:.0e}"
                spec = {"backend": backend, "n_particles": n, "nmesh": NMESH,
                        "method": method, "repeats": repeats}
                rows.extend(bc.spawn_worker(__file__, spec, label=label))

    bc.print_table(rows, title="Mass assignment")
    bc.save_csv(bc.RESULTS_DIR / "mass_assignment.csv", rows, info)
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
        run([int(1e4), int(1e5)], ["CIC", "TSC"], repeats=max(2, args.repeats))
    else:
        run(N_PARTICLES, METHODS, repeats=args.repeats)


if __name__ == "__main__":
    main()
