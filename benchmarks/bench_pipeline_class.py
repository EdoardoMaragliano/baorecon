"""End-to-end ``ReconstructionPipeline`` (orchestrator) benchmark.

Modificato per isolare completamente il calcolo 'total' in un sottoprocesso dedicato.
"""
from __future__ import annotations

import os

# Number of threads dedicated to the process (mirrors the other bench scripts).
THREADS = "8"
os.environ["NUMBA_NUM_THREADS"] = THREADS
os.environ["OMP_NUM_THREADS"] = THREADS
os.environ["MKL_NUM_THREADS"] = THREADS
os.environ["OPENBLAS_NUM_THREADS"] = THREADS
os.environ["NUMEXPR_NUM_THREADS"] = THREADS
os.environ["BAORECON_FFT_THREADS"] = THREADS

import argparse
import shutil
import tempfile

import numpy as np
import yaml

import bench_common as bc

import sys
sys.path.insert(0, '/home/emaragliano/Work/Projects/Dottorato/baorecon')

NMESH = [256, 512, 1024]
N_PARTICLES = [int(1e6)]
N_ITERATIONS = 3
F_GROWTH = 0.8
BIAS = 2.0
R_SMOOTH = 15.0
PADDING = 100.0          # Mpc/h added around the catalogue extent (absolute).
THRESHOLD_RANDOMS = 0.7
SEED_DATA = 1
SEED_RANDOM = 2

# Save sets to sweep: label -> list written into the YAML `output.save` block.
SAVE_SETS = {
    "cat": ["catalogs"],
    "grids": ["catalogs", "grid_potential", "grid_displacement"],
}

# Per-stage breakdown emitted by every worker, in execution order.
PIPELINE_STAGES = ("load", "to_xyz", "reconstruct", "convert_back", "save")


# ---------------------------------------------------------------------------
# Worker side (runs inside a per-configuration subprocess)
# ---------------------------------------------------------------------------
def _write_mock_catalog(path: str, n: int, seed: int) -> None:
    """Write a synthetic RA/DEC/z + WEIGHT FITS catalogue to ``path``."""
    from astropy.table import Table

    ra, dec, z = bc.gen_radec_z(n, seed=seed)
    table = Table()
    table["RA"] = ra
    table["DEC"] = dec
    table["REDSHIFT"] = z
    table["WEIGHT"] = np.ones(n, dtype=np.float32)
    table.write(path, overwrite=True)


def _write_config(cfg_path, data_path, random_path, out_folder,
                  nmesh, solver, device, save_list) -> None:
    """Emit a pipeline YAML config wired to the temporary mock catalogues."""
    config = {
        "catalog": {
            "data_path": data_path,
            "random_path": random_path,
            "data_hdu": 1,
            "random_hdu": 1,
        },
        "columns": {
            "coordinates": {"ra": "RA", "dec": "DEC", "redshift": "REDSHIFT"},
            "weights": {"data": "WEIGHT", "random": "WEIGHT"},
            "ids": {"data": None, "random": None},
            "keep_cols": ["RA", "DEC", "REDSHIFT", "WEIGHT"],
        },
        "coordinate_system": {
            "input": "ra_dec_z",
            "frame": "icrs",
            "ra_dec_unit": "deg",
            "distance_unit": "Mpc/h",
        },
        "cosmology": {"H0": 67.11, "Om0": 0.3175, "Ob0": 0.049, "Tcmb0": 2.7255},
        "reconstruction": {
            "dtype": "float32",
            "RSDspace": "RedshiftSpace",
            "nmesh": int(nmesh),
            "boxsize": None,
            "boxcentre": None,
            "padding": PADDING,
            "los": None,
            "R_sm": R_SMOOTH,
            "pbc": False,
            "rectype": "rec-sym",
            "f": F_GROWTH,
            "bias": BIAS,
            "MAS": "CIC",
            "threshold_randoms": THRESHOLD_RANDOMS,
            "solver_type": solver,
            "n_iterations": N_ITERATIONS,
            "device": device,
        },
        "output": {
            "folder": out_folder,
            "naming_pattern": "rec_{name}_{solver}_n{nmesh}",
            "save_metadata": True,
            "save": list(save_list),
        },
        "masking": {"apply_mask": False},
        "catalog_name": "bench",
    }
    with open(cfg_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def _worker_baorecon(n, nmesh, device, repeats, solver, save_label, mode):
    from baorecon.pipeline import ReconstructionPipeline

    backend = f"baorecon_{device}_{save_label}"
    save_list = SAVE_SETS[save_label]

    workdir = tempfile.mkdtemp(prefix="baorecon_bench_")
    try:
        data_path = os.path.join(workdir, "data.fits")
        random_path = os.path.join(workdir, "random.fits")
        cfg_path = os.path.join(workdir, "config.yaml")
        out_folder = os.path.join(workdir, "out")

        _write_mock_catalog(data_path, n, seed=SEED_DATA)
        _write_mock_catalog(random_path, n, seed=SEED_RANDOM)
        _write_config(cfg_path, data_path, random_path, out_folder,
                      nmesh, solver, device, save_list)

        bc.reset_memory_baseline()
        rows = []

        # MODIFICA ARCHITETTURALE: Esecuzione condizionale in base al 'mode' del sottoprocesso
        if mode == "stages":
            pipeline = ReconstructionPipeline(config_file=cfg_path)

            stage_fns = {
                "load": pipeline.load_catalogs,
                "to_xyz": pipeline.convert_to_xyz,
                "reconstruct": pipeline.reconstruct,
                "convert_back": pipeline.convert_back,
                "save": pipeline.save_outputs,
            }

            for stage in PIPELINE_STAGES:
                m = bc.safe_measure(stage_fns[stage],
                                    label=f"{backend} {stage} nmesh={nmesh}",
                                    repeats=repeats, device=device)
                if m is not None:
                    rows.append(m.as_row(backend, n, nmesh, stage))

        elif mode == "total":
            def total():
                return ReconstructionPipeline(config_file=cfg_path).run()

            m = bc.safe_measure(total, label=f"{backend} total nmesh={nmesh}",
                                repeats=repeats, device=device)
            if m is not None:
                rows.append(m.as_row(backend, n, nmesh, "total"))

        return rows
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def worker(spec):
    backend = spec["backend"]
    n = int(spec["n_particles"])
    nmesh = int(spec["nmesh"])
    repeats = int(spec["repeats"])
    solver = str(spec["solver"])
    save_label = str(spec["save_set"])
    mode = str(spec["mode"]) # Recupera il task assegnato al sottoprocesso

    device = "gpu" if backend.endswith("gpu") else "cpu"
    return _worker_baorecon(n, nmesh, device, repeats, solver, save_label, mode)


# ---------------------------------------------------------------------------
# Parent side (orchestration only -- no baorecon imports)
# ---------------------------------------------------------------------------
def run(n_particles, nmeshes, repeats, solver):
    info = bc.system_info()
    rows = []
    print("Pipeline-class (ReconstructionPipeline) benchmark")
    print(f"per-stage + total | repeats={repeats} | solver={solver}")
    print(f"GPU available: {bc.GPU_AVAILABLE}\n")

    devices = ["cpu"]
    if bc.GPU_AVAILABLE:
        devices.append("gpu")

    for n in n_particles:
        print(f"N={n:.0e}")
        for nmesh in nmeshes:
            print(f"nmesh={nmesh}")
            for device in devices:
                for save_label in SAVE_SETS:
                    backend = f"baorecon_{device}"
                    
                    # 1. Spawn del Sottoprocesso per gli Step Granulari
                    label_stages = f"{backend}_{save_label} N={n:.0e} nmesh={nmesh} (Stages)"
                    print(f"backend: {backend} save={save_label} -> Esecuzione Stage...")
                    spec_stages = {
                        "backend": backend, "n_particles": n, "nmesh": nmesh,
                        "repeats": repeats, "solver": solver, "save_set": save_label,
                        "mode": "stages"
                    }
                    rows.extend(bc.spawn_worker(__file__, spec_stages, label=label_stages))
                    
                    # 2. Spawn di un NUOVO Sottoprocesso Vergine solo per la fase 'Total'
                    label_total = f"{backend}_{save_label} N={n:.0e} nmesh={nmesh} (Total)"
                    print(f"backend: {backend} save={save_label} -> Esecuzione Total Isolato...")
                    spec_total = {
                        "backend": backend, "n_particles": n, "nmesh": nmesh,
                        "repeats": repeats, "solver": solver, "save_set": save_label,
                        "mode": "total"
                    }
                    rows.extend(bc.spawn_worker(__file__, spec_total, label=label_total))
                    
                    bc.print_table(rows, title="Intermediate data")

    bc.print_table(rows, title="Pipeline class (per-stage + total)")
    suffix = "_gpu" if bc.GPU_AVAILABLE else ""
    bc.save_csv(bc.RESULTS_DIR / f"pipeline_class_{solver}{suffix}.csv", rows, info)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                        help="small parameter set for a fast sanity run")
    parser.add_argument("--repeats", type=int, default=1,
                        help="number of timed repetitions (default 1)")
    parser.add_argument("--solver", type=str, choices=["multigrid", "ifft"],
                        default="ifft", help="either multigrid or ifft")
    parser.add_argument("--worker", type=str, help="Internal use by bench_common")
    args = parser.parse_args()

    if bc.is_worker():
        bc.run_worker(worker)
        return

    if args.quick:
        run([int(1e4), int(1e5)], [64], repeats=max(2, args.repeats), solver=args.solver)
    else:
        run(N_PARTICLES, NMESH, repeats=args.repeats, solver=args.solver)


if __name__ == "__main__":
    main()