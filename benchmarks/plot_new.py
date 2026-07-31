"""Generate figures from the benchmark CSVs in ``benchmarks/results/``.

Produces five PDFs in ``benchmarks/figures/``:

1. ``fig1_mass_assignment_time.pdf`` -- time vs N particles for mass assignment.
2. ``fig2_fft_solver_time.pdf``      -- time vs nmesh for the FFT solver.
3. ``fig3_pipeline_time.pdf``        -- end-to-end time vs N particles.
4. ``fig4_gpu_speedup.pdf``          -- GPU/CPU speedup vs N and vs nmesh.
5. ``fig5_memory.pdf``               -- peak RSS increase per step vs N particles.

Run::

    python benchmarks/plot_results.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "results"
FIGURES_DIR = Path(__file__).resolve().parent / "figures"

plt.rcParams.update({
    "font.size": 12,
    "axes.grid": True,
    "grid.alpha": 0.4,
    "legend.frameon": True,
    "figure.autolayout": True,
})
# Consistent per-backend styling across all figures.
STYLE = {
    "baorecon_cpu": {"color": "#1f77b4", "label": "baorecon (CPU)"},
    "baorecon_gpu": {"color": "#ff7f0e", "label": "baorecon (GPU)"},
    "pyrecon": {"color": "#2ca02c", "label": "pyrecon"},
}
BACKEND_ORDER = ["baorecon_cpu", "baorecon_gpu", "pyrecon"]

# Per-method styling for the reconstructor figures, which compare solver variants
# (one series per method) rather than backends. pyrecon is split by the solver it
# was run with: its ifft and multigrid timings come from different algorithms and
# must not collapse into a single "PyRecon" bar.
METHOD_STYLE = {
    "IFFT (PyFFTW)": "#1f77b4",
    "IFFT (SciPy)": "#ff7f0e",
    "Multigrid (Jacobi)": "#d62728",
    "Multigrid (MCGS)": "#9467bd",
    "PyRecon (IFFT)": "#2ca02c",
    "PyRecon (Multigrid)": "#8c564b",
}
METHOD_ORDER = list(METHOD_STYLE)


def _solver_label(stem: str) -> tuple[str, str]:
    """Map a ``bao_reconstructor_*.csv`` stem to (solver, variant) labels.

    The variant is the FFT backend for ifft and the smoother for multigrid -- the
    knob that ``bench_bao_reconstructor.py`` encodes in the filename because it is
    constant per run.
    """
    base = stem.lower()
    if "ifft" in base:
        variant = "PyFFTW" if "pyfftw" in base else "SciPy" if "scipy" in base else ""
        return "IFFT", variant
    variant = "Jacobi" if "jacobi" in base else "MCGS" if "mcgs" in base else ""
    return "Multigrid", variant


def _load_reconstructor_methods(figure_tag: str) -> Optional[pd.DataFrame]:
    """Concatenate the bao_reconstructor CSVs, tagging each row with its method."""
    files = sorted(RESULTS_DIR.glob("bao_reconstructor_*.csv"))
    if not files:
        print(f"[skip] {figure_tag}: no solver files found")
        return None

    dfs = []
    for f in files:
        df = _load(f.name)
        if df is None:
            continue
        solver, variant = _solver_label(f.stem)
        full = f"{solver} ({variant})" if variant else solver
        # baorecon rows carry the solver variant; pyrecon only ever ran the bare
        # solver, so it is labelled by that alone.
        df["method"] = df["backend"].map(
            lambda b: f"PyRecon ({solver})" if b == "pyrecon" else full)
        # The GPU backend has its own figure (fig4); keep these CPU-only.
        df = df[df["backend"] != "baorecon_gpu"]
        dfs.append(df)

    if not dfs:
        return None
    return pd.concat(dfs, ignore_index=True)


def _method_pivot(df: pd.DataFrame, value: str) -> tuple[pd.DataFrame, list, list]:
    """Pivot to nmesh x method, returning the frame plus its column order/colours."""
    pivot = df.pivot_table(index="nmesh", columns="method", values=value,
                           aggfunc="mean")
    methods = [m for m in METHOD_ORDER if m in pivot.columns]
    # Anything unrecognised still gets drawn, just after the known methods.
    methods += [m for m in pivot.columns if m not in METHOD_ORDER]
    return pivot[methods], methods, [METHOD_STYLE.get(m, "#7f7f7f") for m in methods]


def _load(name: str) -> Optional[pd.DataFrame]:
    path = RESULTS_DIR / name
    if not path.exists():
        print(f"[skip] {name} not found")
        return None
    try:
        df = pd.read_csv(path, comment="#")
    except Exception as exc:  # noqa: BLE001
        print(f"[skip] could not read {name}: {exc}")
        return None
    if df.empty:
        print(f"[skip] {name} is empty")
        return None
    return df


def _save(fig, name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] wrote {path}")


def _plot_backends(ax, df, xcol, ycol, logy=False):
    """Plot grouped bar charts (Pandas style). Returns number of backends drawn."""
    # Filtriamo solo i backend presenti nei dati, mantenendo l'ordine voluto
    available_backends = [b for b in BACKEND_ORDER if b in df["backend"].unique()]
    if not available_backends:
        return 0

    # Usiamo pivot_table per raggruppare i dati. 
    # aggfunc="mean" gestisce automaticamente eventuali righe duplicate in modo sicuro.
    df_pivot = df.pivot_table(index=xcol, columns="backend", values=ycol, aggfunc="mean")
    
    # Riordiniamo le colonne
    df_pivot = df_pivot[available_backends]

    # Estraiamo i colori e le label dal dizionario di stile
    colors = [STYLE[b]["color"] for b in available_backends]
    labels = [STYLE[b]["label"] for b in available_backends]

    # Plottiamo le barre tramite Pandas
    df_pivot.plot.bar(ax=ax, color=colors, logy=logy, rot=0, width=0.7, edgecolor='black', linewidth=0.5)

    # Aggiorniamo la legenda con i nomi puliti
    ax.legend(labels, loc='upper left')

    return len(available_backends)


# ---------------------------------------------------------------------------
# Figure 1: mass assignment time vs N
# ---------------------------------------------------------------------------
def figure_mass_assignment():
    df = _load("mass_assignment.csv")
    if df is None:
        return
    method = "CIC" if (df["step"] == "CIC").any() else df["step"].iloc[0]
    df = df[df["step"] == method]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    
    n = _plot_backends(ax, df, "n_particles", "time_mean", logy=False)
    if n == 0:
        plt.close(fig)
        print("[skip] fig1: no data")
        return
        
    ax.set_xlabel("Number of particles")
    ax.set_ylabel("Time [s]")
    ax.set_title(f"Mass assignment ({method}, nmesh={int(df['nmesh'].iloc[0])})")
    _save(fig, "fig1_mass_assignment_time.pdf")

# ---------------------------------------------------------------------------
# Figure 2: FFT solver time vs nmesh
# ---------------------------------------------------------------------------
def figure_fft_solver():
    df = _load("fft_solver.csv")
    if df is None:
        return
        
    # Sum the per-step times into a single total per configuration.
    df_total = df.groupby(["backend", "n_particles", "nmesh"], as_index=False)["time_mean"].sum()
    df_total["step"] = "total"

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    n = _plot_backends(ax, df_total, "nmesh", "time_mean", logy=False)
    
    if n == 0:
        plt.close(fig)
        print("[skip] fig2: no data")
        return
        
    ax.set_xlabel("nmesh")
    ax.set_ylabel("Solve Time [s]")
    
    if 'n_particles' in df.columns:
        n_part = df['n_particles'].iloc[0]
        ax.set_title(f"FFT Solver time vs nmesh (N={n_part:.0e})")
    else:
        ax.set_title("FFT Solver time vs nmesh")
        
    _save(fig, "fig2_fft_gpu_solver_time.pdf")

# ---------------------------------------------------------------------------
# Figure 3: end-to-end pipeline time vs N
# ---------------------------------------------------------------------------
def figure_reconstructor_time():
    df = _load_reconstructor_methods("fig3")
    if df is None:
        return

    times, methods, colors = _method_pivot(df, "time_mean")
    # time_std is np.std over --repeats samples; it is identically 0 when the
    # benchmark ran with repeats=1, in which case the error bars simply vanish.
    stds, _, _ = _method_pivot(df, "time_std")
    stds = stds.reindex(columns=methods)

    fig, ax = plt.subplots(figsize=(9, 5))
    times.plot.bar(ax=ax, color=colors, rot=0, width=0.8,
                   edgecolor="black", linewidth=0.5,
                   yerr=stds, capsize=2, error_kw={"elinewidth": 0.8})

    ax.set_xlabel("Mesh Size (nmesh)")
    ax.set_ylabel("Time [s] (log)")
    ax.set_yscale("log")
    ax.set_title("Full Pipeline Performance Comparison")

    ax.grid(axis="y", which="both", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(methods, title="Method", loc="upper left")

    _save(fig, "fig3_reconstructor_performance_comparison.pdf")

# ---------------------------------------------------------------------------
# Figure 4: GPU/CPU speedup vs N (mass assignment) and vs nmesh (scaling)
# ---------------------------------------------------------------------------
def _speedup(df, xcol, step_filter=None):
    if step_filter is not None:
        df = df[df["step"] == step_filter]
    
    df = df.copy()
    df[xcol] = df[xcol].astype(float).round(4)
    
    cpu = df[df["backend"] == "baorecon_cpu"][[xcol, "time_mean"]]
    gpu = df[df["backend"] == "baorecon_gpu"][[xcol, "time_mean"]]
    
    if cpu.empty or gpu.empty:
        return None
        
    merged = pd.merge(cpu, gpu, on=xcol, suffixes=("_cpu", "_gpu")).sort_values(xcol)
    
    if merged.empty:
        return None
        
    merged["speedup"] = merged["time_mean_cpu"] / merged["time_mean_gpu"]
    return merged

def figure_speedup():
    df = _load("scaling.csv") 
    if df is None:
        return
        
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax in axes:
        ax.set_axisbelow(True)
        ax.axhline(1.0, color="grey", ls="--", lw=1)

    # 1. Left: speedup vs nmesh (per Mass assignment)
    sp_ma = _speedup(df, "nmesh", step_filter="mass_assignment")
    if sp_ma is not None and not sp_ma.empty:
        # Usiamo Pandas per fare il plot a barre
        sp_ma.set_index("nmesh")["speedup"].plot.bar(
            ax=axes[0], color="#d62728", rot=0, width=0.5, edgecolor='black'
        )
        axes[0].set_title("Mass assignment speedup vs nmesh")
    else:
        axes[0].text(0.5, 0.5, "Data mismatch", ha="center", va="center", transform=axes[0].transAxes)

    # 2. Right: speedup vs nmesh (per FFT solver)
    sp_fft = _speedup(df, "nmesh", step_filter="fft_solver")
    if sp_fft is not None and not sp_fft.empty:
        sp_fft.set_index("nmesh")["speedup"].plot.bar(
            ax=axes[1], color="#9467bd", rot=0, width=0.5, edgecolor='black'
        )
        axes[1].set_title("FFT solver speedup vs nmesh")
    else:
        axes[1].text(0.5, 0.5, "Data mismatch", ha="center", va="center", transform=axes[1].transAxes)

    axes[0].set_xlabel("nmesh")
    axes[1].set_xlabel("nmesh")
    axes[0].set_ylabel("Speedup (CPU / GPU time)")
    
    _save(fig, "fig4_gpu_speedup.pdf")

# ---------------------------------------------------------------------------
# Figure 5: peak memory vs N
# ---------------------------------------------------------------------------
def figure_reconstructor_memory():
    df = _load_reconstructor_methods("fig5")
    if df is None:
        return

    df["memory_peak_gb"] = df["memory_peak_mb"] / 1024.0
    # No error bars here: memory_peak_mb comes from ru_maxrss, a monotonic
    # high-water mark over all repeats, so it has no spread to report.
    mem, methods, colors = _method_pivot(df, "memory_peak_gb")

    fig, ax = plt.subplots(figsize=(9, 5))
    mem.plot.bar(ax=ax, color=colors, rot=0, width=0.8,
                 edgecolor="black", linewidth=0.5)

    ax.set_xlabel("Mesh Size (nmesh)")
    ax.set_ylabel("Peak Memory [GB]")
    ax.set_yscale("log")
    ax.set_title("Peak Memory Usage Comparison")
    ax.grid(axis="y", which="both", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(methods, title="Method", loc="upper left")

    _save(fig, "fig5_reconstructor_memory_comparison.pdf")

def main():
    global RESULTS_DIR, FIGURES_DIR

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_dir", type=str, default=None,
                        help="directory holding the benchmark CSVs; absolute, or "
                             "relative to benchmarks/ (default: results)")
    parser.add_argument("--figures_dir", type=str, default=None,
                        help="output directory for the PDFs; absolute, or relative "
                             "to benchmarks/ (default: figures)")
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    if args.results_dir is not None:
        RESULTS_DIR = Path(args.results_dir)
        if not RESULTS_DIR.is_absolute():
            RESULTS_DIR = here / RESULTS_DIR
    if args.figures_dir is not None:
        FIGURES_DIR = Path(args.figures_dir)
        if not FIGURES_DIR.is_absolute():
            FIGURES_DIR = here / FIGURES_DIR

    print(f"reading CSVs from {RESULTS_DIR}")
    print(f"writing figures to {FIGURES_DIR}")

    figure_mass_assignment()
    figure_fft_solver()
    figure_reconstructor_time()
    figure_speedup()
    figure_reconstructor_memory()


if __name__ == "__main__":
    main()