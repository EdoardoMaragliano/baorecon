"""Generate figures from the benchmark CSVs in ``benchmarks/results/``.

Produces five PDFs in ``benchmarks/figures/``:

1. ``fig1_mass_assignment_time.pdf`` -- time vs N particles for mass assignment.
2. ``fig2_fft_solver_time.pdf``      -- time vs nmesh for the FFT solver.
3. ``fig3_pipeline_time.pdf``        -- end-to-end time vs N particles.
4. ``fig4_gpu_speedup.pdf``          -- GPU/CPU speedup vs N and vs nmesh.
5. ``fig5_memory.pdf``               -- peak RSS increase per step vs N particles.

Each figure is drawn from whatever data is present; missing backends (e.g. no
GPU rows) are simply omitted, and figures with no usable data are skipped with a
message instead of crashing.

Run::

    python benchmarks/plot_results.py
"""

from __future__ import annotations

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
    "baorecon_cpu": {"color": "#1f77b4", "marker": "o", "label": "baorecon (CPU)"},
    "baorecon_gpu": {"color": "#ff7f0e", "marker": "s", "label": "baorecon (GPU)"},
    "pyrecon": {"color": "#2ca02c", "marker": "^", "label": "pyrecon"},
}
BACKEND_ORDER = ["baorecon_cpu", "baorecon_gpu", "pyrecon"]


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


def _plot_backends(ax, df, xcol, ycol, logx=True, logy=True):
    """Plot one curve per backend, sorted by x. Returns number of curves drawn."""
    drawn = 0
    for backend in BACKEND_ORDER:
        sub = df[df["backend"] == backend].sort_values(xcol)
        if sub.empty:
            continue
        style = STYLE[backend]
        ax.plot(sub[xcol], sub[ycol], color=style["color"], marker=style["marker"],
                label=style["label"], markersize=6, linewidth=1.8)
        drawn += 1
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    return drawn


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
    n = _plot_backends(ax, df, "n_particles", "time_mean", logx=False, logy=False)
    if n == 0:
        plt.close(fig)
        print("[skip] fig1: no data")
        return
    ax.set_xlabel("Number of particles")
    ax.set_ylabel("Time [s]")
    ax.set_title(f"Mass assignment ({method}, nmesh={int(df['nmesh'].iloc[0])})")
    ax.legend()
    ax.grid(True, which="both", alpha=0.4)
    _save(fig, "fig1_mass_assignment_time.pdf")

# ---------------------------------------------------------------------------
# Figure 2: FFT solver time vs nmesh
# ---------------------------------------------------------------------------
def figure_fft_solver():
    df = _load("fft_solver.csv")
    if df is None:
        return
        
    # Sum the per-step times into a single total per configuration.
    # (To plot only the "solve" phase, filter df[df["step"] == "solve"] instead.)
    df_total = df.groupby(["backend", "n_particles", "nmesh"], as_index=False)["time_mean"].sum()

    # Tag the summed rows as the total time.
    df_total["step"] = "total"

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    # X axis: nmesh, Y axis: time_mean
    n = _plot_backends(ax, df_total, "nmesh", "time_mean", logx=False, logy=False)
    
    if n == 0:
        plt.close(fig)
        print("[skip] fig2: no data")
        return
        
    ax.set_xlabel("nmesh")
    ax.set_ylabel("Solve Time [s]")
    
    if 'n_particles' in df.columns:
        n_part = df['n_particles'].iloc[0]
        # Format the particle count in scientific notation (e.g. 1e+06)
        ax.set_title(f"FFT Solver time vs nmesh (N={n_part:.0e})")
    else:
        ax.set_title("FFT Solver time vs nmesh")
        
    ax.legend()
    ax.grid(True, which="both", alpha=0.4)
    _save(fig, "fig2_fft_gpu_solver_time.pdf")

# ---------------------------------------------------------------------------
# Figure 3: end-to-end pipeline time vs N
# ---------------------------------------------------------------------------
solver = 'ifft'
def figure_pipeline():
    df = _load(f"pipeline_{solver}.csv")
    if df is None:
        return
        
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    
    # Maps nmesh on the X axis and time_mean on the Y axis
    n = _plot_backends(ax, df, "nmesh", "time_mean", logx=False, logy=False)
    
    if n == 0:
        plt.close(fig)
        print("[skip] fig3: no data")
        return
        
    ax.set_xlabel("nmesh")
    ax.set_ylabel("End-to-end time [s]")

    # nmesh varies on the X axis, so n_particles is the fixed quantity.
    if 'n_particles' in df.columns:
        ax.set_title(f"Full pipeline {solver} (n_particles={int(df['n_particles'].iloc[0])})")
    else:
        ax.set_title(f"Full pipeline {solver}")
        
    ax.legend()
    ax.grid(True, which="both", alpha=0.4)
    _save(fig, f"fig3_pipeline_time_{solver}.pdf")


# ---------------------------------------------------------------------------
# Figure 4: GPU/CPU speedup vs N (mass assignment) and vs nmesh (scaling)
# ---------------------------------------------------------------------------
def _speedup(df, xcol, step_filter=None):
    if step_filter is not None:
        df = df[df["step"] == step_filter]
    
    # Arrotondiamo la colonna xcol per evitare problemi di floating point nel merge
    df = df.copy()
    df[xcol] = df[xcol].astype(float).round(4)
    
    cpu = df[df["backend"] == "baorecon_cpu"][[xcol, "time_mean"]]
    gpu = df[df["backend"] == "baorecon_gpu"][[xcol, "time_mean"]]
    
    if cpu.empty or gpu.empty:
        return None
        
    # Merge "outer" per vedere se ci sono discrepanze, poi pulisci
    merged = pd.merge(cpu, gpu, on=xcol, suffixes=("_cpu", "_gpu")).sort_values(xcol)
    
    if merged.empty:
        return None
        
    merged["speedup"] = merged["time_mean_cpu"] / merged["time_mean_gpu"]
    return merged
def figure_speedup():
    # Assumendo che i dati siano in un unico DataFrame o caricati come prima
    # Se hai un unico file, carica quello. Qui ipotizzo tu abbia un dataframe 'df'
    df = _load("scaling.csv") 
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Helper per configurare scala log-log
    for ax in axes:
        ax.set_xscale("linear")
        ax.set_yscale("linear")
        ax.grid(True, which="both", alpha=0.4)
        ax.axhline(1.0, color="grey", ls="--", lw=1)

    # 1. Left: speedup vs nmesh (per Mass assignment)
    # MODIFICA: Usiamo "nmesh" invece di "n_particles"
    sp_ma = _speedup(df, "nmesh", step_filter="mass_assignment")
    if sp_ma is not None and not sp_ma.empty:
        axes[0].plot(sp_ma["nmesh"], sp_ma["speedup"], color="#d62728", marker="o", lw=2)
        axes[0].set_title("Mass assignment speedup vs nmesh")
    else:
        axes[0].text(0.5, 0.5, "Data mismatch", ha="center", va="center", transform=axes[0].transAxes)

    # 2. Right: speedup vs nmesh (per FFT solver)
    sp_fft = _speedup(df, "nmesh", step_filter="fft_solver")
    if sp_fft is not None and not sp_fft.empty:
        axes[1].plot(sp_fft["nmesh"], sp_fft["speedup"], color="#9467bd", marker="s", lw=2)
        axes[1].set_title("FFT solver speedup vs nmesh")
    else:
        axes[1].text(0.5, 0.5, "Data mismatch", ha="center", va="center", transform=axes[1].transAxes)

    axes[0].set_xlabel("nmesh")
    axes[1].set_xlabel("nmesh")
    axes[0].set_ylabel("Speedup [log]")
    
    _save(fig, "fig4_gpu_speedup.pdf")

# ---------------------------------------------------------------------------
# Figure 5: peak memory vs N
# ---------------------------------------------------------------------------

def figure_pipeline_memory():
    # Note: to show ifft and multigrid together, load the combined data
    # rather than just "pipeline_{solver}.csv".
    df = _load(f"pipeline_{solver}.csv")
    if df is None:
        return
        
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    
    # Plot "memory_peak_mb" instead of "time_mean"
    n = _plot_backends(ax, df, "nmesh", "memory_peak_mb", logx=False, logy=False)
    
    if n == 0:
        plt.close(fig)
        print("[skip] fig4: no data")
        return
        
    ax.set_xlabel("nmesh")
    ax.set_ylabel("Peak Memory [MB]")
    
    if 'n_particles' in df.columns:
        ax.set_title(f"Full pipeline {solver} (n_particles={int(df['n_particles'].iloc[0])})")
    else:
        ax.set_title(f"Full pipeline {solver}")
        
    ax.legend()
    ax.grid(True, which="both", alpha=0.4)
    _save(fig, f"fig4_pipeline_memory_{solver}.pdf")


def main():
    figure_mass_assignment()
    figure_fft_solver()
    figure_pipeline()
    figure_speedup()
    figure_pipeline_memory()


if __name__ == "__main__":
    main()
