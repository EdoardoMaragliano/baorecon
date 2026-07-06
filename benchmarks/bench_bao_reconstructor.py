"""End-to-end pipeline benchmark.

Measures the total wall-time and peak memory of a full reconstruction.

All backends are compared starting from the *same* pre-computed Cartesian
positions: the RA/DEC/z -> XYZ coordinate conversion is performed once (per
worker, outside every timer and before the memory baseline), so it is excluded
for baorecon and pyrecon alike.

* ``baorecon_cpu`` / ``baorecon_gpu`` -- the timed region is
  ``BAOReconstructor.run_reconstruction`` (density contrast, iterative FFT solve,
  shifting data + randoms).
* ``pyrecon`` -- the timed region is ``assign_data`` / ``assign_randoms`` /
  ``set_density_contrast`` / ``run`` / ``read_shifted_positions``.

Particle count is swept over 1e5, 1e6, 1e7 (data and randoms 1:1); mesh fixed at
512^3.

Each ``(backend, n_particles)`` configuration runs in its own subprocess (see
``bench_common.spawn_worker``) so that ``ru_maxrss`` measures one backend's peak
memory in isolation. The parent only launches workers and aggregates their JSON
output; it never imports baorecon or pyrecon.

Run::

    python benchmarks/bench_pipeline.py
    python benchmarks/bench_pipeline.py --quick
"""
from __future__ import annotations

import os

# Number of threads to dedicate to the process
THREADS = "8"

# 1. Numba threading
#os.environ["NUMBA_THREADING_LAYER"] = "tbb"
os.environ["NUMBA_NUM_THREADS"] = THREADS

# 2. NumPy backend threading (covers MKL, OpenBLAS, and standard OpenMP)
os.environ["OMP_NUM_THREADS"] = THREADS
os.environ["MKL_NUM_THREADS"] = THREADS
os.environ["OPENBLAS_NUM_THREADS"] = THREADS
os.environ["NUMEXPR_NUM_THREADS"] = THREADS
os.environ["BAORECON_FFT_THREADS"] = THREADS


import argparse

import numpy as np

import bench_common as bc

import sys
sys.path.insert(0, '/home/emaragliano/Work/Projects/Dottorato/baorecon')

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
    from baorecon.utils.coordinates import create_cosmology, radec_z_to_xyz

    cosmo = create_cosmology()
    d_ra, d_dec, d_z = bc.gen_radec_z(n, seed=SEED_DATA)
    r_ra, r_dec, r_z = bc.gen_radec_z(n, seed=SEED_RANDOM)
    data_xyz, _ = radec_z_to_xyz(d_ra, d_dec, d_z, cosmo=cosmo)
    random_xyz, _ = radec_z_to_xyz(r_ra, r_dec, r_z, cosmo=cosmo)
    return data_xyz, random_xyz

'''def _worker_baorecon(n, nmesh, device, repeats, solver_type, mas_parallel):
    from baorecon.reconstruction.bao_reconstructor import BAOReconstructor

    backend = f"baorecon_{device}"
    data_xyz, random_xyz = _make_mock_xyz(n)
    weights_d = np.ones(len(data_xyz), dtype=np.float32)
    weights_r = np.ones(len(random_xyz), dtype=np.float32)
    
    # 1. Compute the cubic box EXACTLY as pyrecon does,
    # using the padding as a percentage (2%) and without wasting RAM.
    lo = np.minimum(data_xyz.min(axis=0), random_xyz.min(axis=0))
    hi = np.maximum(data_xyz.max(axis=0), random_xyz.max(axis=0))
    
    boxcentre = (lo + hi) / 2.0
    boxsize = float((hi - lo).max()) * (1.0 + PADDING)

    bc.reset_memory_baseline()

    def run():
        recon = BAOReconstructor(
            data_pos=data_xyz, random_pos=random_xyz,
            data_weights=weights_d, random_weights=weights_r,
            RSDspace="RedshiftSpace", nmesh=nmesh, 
            boxsize=boxsize,           # Pass the padded cubic box explicitly
            boxcentre=boxcentre,       # Pass the box centre explicitly
            padding=0.0,               # Ignored: the box is already padded
            los=None, R_sm=R_SMOOTH, pbc=False, rectype="rec-sym", 
            f=F_GROWTH, bias=BIAS, MAS="CIC", dtype=np.float32, 
            solver_type=solver_type, device=device, mas_parallel=mas_parallel, smoother='jacobi'
        )
        return recon.run_reconstruction()

    m = bc.safe_measure(run, label=f"{backend} N={n:.0e}",
                        repeats=repeats, device=device)
    return [m.as_row(backend, n, nmesh, "total")] if m is not None else []
'''

def _worker_baorecon(n, nmesh, device, repeats, solver_type, mas_parallel, smoother):
    from baorecon.reconstruction.bao_reconstructor import BAOReconstructor

    backend = f"baorecon_{device}"
    data_xyz, random_xyz = _make_mock_xyz(n)
    weights_d = np.ones(len(data_xyz), dtype=np.float32)
    weights_r = np.ones(len(random_xyz), dtype=np.float32)

    lo = np.minimum(data_xyz.min(axis=0), random_xyz.min(axis=0))
    hi = np.maximum(data_xyz.max(axis=0), random_xyz.max(axis=0))

    boxcentre = (lo + hi) / 2.0
    boxsize = float((hi - lo).max()) * (1.0 + PADDING)

    bc.reset_memory_baseline()

    def run():
        recon = BAOReconstructor(
            data_pos=data_xyz, random_pos=random_xyz,
            data_weights=weights_d, random_weights=weights_r,
            RSDspace="RedshiftSpace", nmesh=nmesh,
            boxsize=boxsize,
            boxcentre=boxcentre,
            padding=0.0,
            los=None, R_sm=R_SMOOTH, pbc=False, rectype="rec-sym",
            f=F_GROWTH, bias=BIAS, MAS="CIC", dtype=np.float32,
            solver_type=solver_type, device=device, mas_parallel=mas_parallel,
            # The multigrid smoother is read from solver_args (not a bare kwarg,
            # which BAOReconstructor.__init__ would silently swallow). Ignored by
            # the ifft solver.
            solver_args={"smoother": smoother},
        )
        return recon.run_reconstruction()

    # bc.safe_measure captures the VRAM peak (CuPy pool high-water) for device=gpu
    # and reports 0.0 on cpu; it lands in the row via Measurement.as_row.
    m = bc.safe_measure(run, label=f"{backend} N={n:.0e}",
                        repeats=repeats, device=device)

    if m is None:
        return []

    return [m.as_row(backend, n, nmesh, "total")]

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
       

    m = bc.safe_measure(run, label=f"pyrecon N={n:.0e}",
                        repeats=repeats, device="cpu")

    if m is None:
        return []

    # pyrecon is CPU-only, so vram_peak_mb comes back 0.0 from as_row.
    return [m.as_row("pyrecon", n, nmesh, "total")]



def worker(spec):
    backend = spec["backend"]
    n = int(spec["n_particles"])
    nmesh = int(spec["nmesh"])
    repeats = int(spec["repeats"])
    solver = str(spec["solver"])
    smoother = str(spec.get("smoother", "jacobi"))
    fft = str(spec.get("fft", "scipy"))

    # Select the CPU FFT backend. use_pyfftw() reads BAORECON_FFT at runtime, so
    # setting it here (before run_reconstruction) is enough; it only affects the
    # CPU ifft path (multigrid and the GPU solver ignore it).
    os.environ["BAORECON_FFT"] = fft

    # Safely parse mas_parallel in case it comes back as a string from JSON subprocess args
    mas_parallel = spec.get("mas_parallel", False)
    if isinstance(mas_parallel, str):
        mas_parallel = mas_parallel.lower() == 'true'

    if backend == "pyrecon":
        return _worker_pyrecon(n, nmesh, repeats, solver)

    device = "gpu" if backend.endswith("gpu") else "cpu"
    return _worker_baorecon(n, nmesh, device, repeats, solver, mas_parallel, smoother)


# ---------------------------------------------------------------------------
# Parent side (orchestration only -- no baorecon / pyrecon imports)
# ---------------------------------------------------------------------------
def run(n_particles, nmeshes, repeats, solver, mas_parallel, smoother, fft):
    info = bc.system_info()
    # Stamp the run-constant knobs into the CSV provenance header (they are
    # single-valued per invocation, so they live here + in the filename rather
    # than as per-row columns).
    info["solver"] = solver
    info["smoother"] = smoother
    info["fft_backend"] = fft
    rows = []
    print('benchmark of refactor branch')
    print(f"Pipeline (end-to-end) benchmark | repeats={repeats}")
    print(f"Solver is {solver}")
    print(f"Smoother is {smoother} (multigrid only)")
    print(f"FFT backend is {fft} (cpu ifft only)")
    print(f"GPU available: {bc.GPU_AVAILABLE}\n")
    print(f"mas parallel is {mas_parallel}")

    backends = ["baorecon_cpu"]
    if bc.GPU_AVAILABLE and solver == "ifft":
        backends.append("baorecon_gpu")
    backends.append("pyrecon")

    # The FFT backend only affects the CPU ifft path; warn when it cannot take
    # effect so a 'pyfftw' request on multigrid/GPU is not silently ignored.
    if fft == "pyfftw" and (solver != "ifft" or bc.GPU_AVAILABLE):
        print(f"[warning] --fft {fft} only affects the CPU ifft solver; it is "
              f"ignored for solver={solver} and for GPU backends.")

    for n in n_particles:
        print(f"N={n:.0e}")
        for nmesh in nmeshes:  # iterate over NMESH
            print(f"nmesh={nmesh}")
            for backend in backends:
                print(f"backend: {backend}")
                label = f"{backend} N={n:.0e}"
                spec = {"backend": backend, "n_particles": n, "nmesh": nmesh,
                        "repeats": repeats, "solver": solver, "mas_parallel": mas_parallel,
                        "smoother": smoother, "fft": fft}
                rows.extend(bc.spawn_worker(__file__, spec, label=label))
                bc.print_table(rows, title="Intermediate data")
    bc.print_table(rows, title="End-to-end pipeline")

    # Encode the varied knobs in the filename so variants do not clobber each
    # other: smoother matters for multigrid, fft backend for ifft, _gpu for GPU.
    name = f"bao_reconstructor_{solver}"
    if solver == "multigrid":
        name += f"_{smoother}"
    elif solver == "ifft":
        name += f"_{fft}"
    #if bc.GPU_AVAILABLE:
    #    name += "_gpu"
    bc.save_csv(bc.RESULTS_DIR / f"{name}.csv", rows, info)
    return rows


def main():
    import sys

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                        help="small parameter set for a fast sanity run")
    parser.add_argument("--repeats", type=int, default=1,
                        help="number of timed repetitions (default 5)")
    parser.add_argument("--solver", type=str, choices=["multigrid", "ifft"],
                        default='multigrid', help="either multigrid or ifft")
    parser.add_argument("--smoother", type=str, choices=["jacobi", "mcgs"],
                        default="jacobi",
                        help="multigrid smoother: jacobi (Jacobi-smoothed V-cycle) "
                             "or mcgs (multicolor Gauss-Seidel). multigrid solver only.")
    parser.add_argument("--fft", type=str, choices=["scipy", "pyfftw"],
                        default="scipy",
                        help="CPU FFT backend (BAORECON_FFT); only affects the cpu ifft solver.")
    parser.add_argument("--mas_parallel", action="store_true",
                        help="Enable parallel mass assignment in baorecon")
    parser.add_argument('--debug', action="store_true")
    parser.add_argument("--worker", type=str, help="Internal use by bench_common")

    args = parser.parse_args()

    # Debug switch: run baorecon directly so errors surface on screen
    if args.debug:
        os.environ["BAORECON_FFT"] = args.fft
        if args.solver == 'multigrid':
            print("RUNNING BAORECON DIRECTLY TO SURFACE THE ERROR...")
            _worker_baorecon(n=int(1e4), nmesh=64, device="cpu", repeats=1,
                             solver_type="multigrid", mas_parallel=False, smoother=args.smoother)
            return
        else:
            print("RUNNING BAORECON DIRECTLY TO SURFACE THE ERROR...")
            _worker_baorecon(n=int(1e6), nmesh=128, device="gpu", repeats=1,
                             solver_type=args.solver, mas_parallel=False, smoother=args.smoother)
            return

    if bc.is_worker():
        bc.run_worker(worker)
        return


    if args.quick:
        run([int(1e4), int(1e5)], [64], repeats=max(2, args.repeats), solver=args.solver,
            mas_parallel=args.mas_parallel, smoother=args.smoother, fft=args.fft)
    else:
        run(N_PARTICLES, NMESH, repeats=args.repeats, solver=args.solver,
            mas_parallel=args.mas_parallel, smoother=args.smoother, fft=args.fft)


if __name__ == "__main__":
    main()