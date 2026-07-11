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

# Stile originale per i backend
STYLE = {
    "baorecon_cpu": {"color": "#1f77b4", "label": "baorecon (CPU)"},
    "baorecon_gpu": {"color": "#ff7f0e", "label": "baorecon (GPU)"},
    "pyrecon": {"color": "#2ca02c", "label": "pyrecon"},
}
BACKEND_ORDER = ["baorecon_cpu", "baorecon_gpu", "pyrecon"]

# Nuovi colori specifici per i metodi nei plot della pipeline raggruppati per nmesh
METHOD_COLORS = {
    "IFFT (PyFFTW)": "#1f77b4",       # Blu
    "IFFT (SciPy)": "#ff7f0e",        # Arancione
    "Multigrid (Jacobi)": "#d62728",  # Rosso
    "Multigrid (MCGS)": "#9467bd",    # Viola
    "PyRecon": "#2ca02c"              # Verde
}
METHOD_ORDER = ["IFFT (PyFFTW)", "IFFT (SciPy)", "Multigrid (Jacobi)", "Multigrid (MCGS)", "PyRecon"]


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
    available_backends = [b for b in BACKEND_ORDER if b in df["backend"].unique()]
    if not available_backends:
        return 0

    df_pivot = df.pivot_table(index=xcol, columns="backend", values=ycol, aggfunc="mean")
    df_pivot = df_pivot[available_backends]

    colors = [STYLE[b]["color"] for b in available_backends]
    labels = [STYLE[b]["label"] for b in available_backends]

    df_pivot.plot.bar(ax=ax, color=colors, logy=logy, rot=0, width=0.7, edgecolor='black', linewidth=0.5)
    ax.legend(labels, loc='upper left')

    return len(available_backends)


def _extract_solver_label(fname: str) -> str:
    """Utility per mappare il nome file all'etichetta del solver desiderata."""
    fname_base = fname.replace("bao_reconstructor_", "").replace(".csv", "")
    parts = fname_base.split("_")
    
    solver_type = parts[0].upper() if parts[0] == "ifft" else parts[0].capitalize()
    
    if len(parts) > 1:
        sub = parts[1].lower()
        if sub == "pyfftw":
            sub_str = "PyFFTW"
        elif sub == "scipy":
            sub_str = "SciPy"
        elif sub == "mcgs":
            sub_str = "MCGS"
        else:
            sub_str = sub.capitalize()
        return f"{solver_type} ({sub_str})"
    
    return solver_type


# ---------------------------------------------------------------------------
# Figure 1 & 2: invariate
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
        return
        
    ax.set_xlabel("Number of particles")
    ax.set_ylabel("Time [s]")
    ax.set_title(f"Mass assignment ({method}, nmesh={int(df['nmesh'].iloc[0])})")
    _save(fig, "fig1_mass_assignment_time.pdf")

def figure_fft_solver():
    df = _load("fft_solver.csv")
    if df is None:
        return
        
    df_total = df.groupby(["backend", "n_particles", "nmesh"], as_index=False)["time_mean"].sum()
    df_total["step"] = "total"

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    n = _plot_backends(ax, df_total, "nmesh", "time_mean", logy=False)
    
    if n == 0:
        plt.close(fig)
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
# Figure 3: Raggruppata solo per nmesh con metodi colorati
# ---------------------------------------------------------------------------
def figure_reconstructor_time():
    files = list(RESULTS_DIR.glob("bao_reconstructor_*.csv"))
    if not files:
        print("[skip] fig3: no solver files found")
        return
        
    dfs = []
    for f in files:
        df = _load(f.name)
        if df is not None:
            # Escludiamo la GPU per questo plot
            df = df[df["backend"] != "baorecon_gpu"].copy()
            if df.empty:
                continue
                
            solver_label = _extract_solver_label(f.name)
            
            # Assegniamo la categoria unificata "method"
            df["method"] = df["backend"].apply(lambda b: "PyRecon" if b == "pyrecon" else solver_label)
            dfs.append(df)
            
    if not dfs:
        return
    df = pd.concat(dfs, ignore_index=True)
    
    # Pivot usando SOLO nmesh sull'asse x e i vari metodi come colonne
    df_pivot = df.pivot_table(
        index="nmesh", 
        columns="method", 
        values="time_mean", 
        aggfunc="mean" # media in caso di multiple letture di pyrecon
    )
    
    # Ordiniamo le colonne per averle nel giusto ordine (se presenti nei dati)
    plot_cols = [c for c in METHOD_ORDER if c in df_pivot.columns]
    colors = [METHOD_COLORS[c] for c in plot_cols]
    
    fig, ax = plt.subplots(figsize=(9, 5))
    
    df_pivot[plot_cols].plot.bar(
        ax=ax, 
        color=colors, 
        rot=0, # Manteniamo dritto l'asse x dato che è solo un numero adesso
        width=0.8, 
        edgecolor='black', 
        linewidth=0.5
    )
    
    ax.set_xlabel("Mesh Size (nmesh)")
    ax.set_ylabel("Time [s] (log)")
    ax.set_yscale("log")
    ax.set_title("Full Pipeline Performance Comparison")
    
    ax.grid(axis='y', which='both', alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(title="Method", loc='upper left')
    
    _save(fig, "fig3_reconstructor_performance_comparison.pdf")


# ---------------------------------------------------------------------------
# Figure 4: invariata
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

    sp_ma = _speedup(df, "nmesh", step_filter="mass_assignment")
    if sp_ma is not None and not sp_ma.empty:
        sp_ma.set_index("nmesh")["speedup"].plot.bar(
            ax=axes[0], color="#d62728", rot=0, width=0.5, edgecolor='black'
        )
        axes[0].set_title("Mass assignment speedup vs nmesh")
    else:
        axes[0].text(0.5, 0.5, "Data mismatch", ha="center", va="center", transform=axes[0].transAxes)

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
# Figure 5: Raggruppata solo per nmesh con metodi colorati
# ---------------------------------------------------------------------------
def figure_reconstructor_memory():
    files = list(RESULTS_DIR.glob("bao_reconstructor_*.csv"))
    if not files:
        print("[skip] fig5: no solver files found")
        return
        
    dfs = []
    for f in files:
        df = _load(f.name)
        if df is not None:
            df = df[df["backend"] != "baorecon_gpu"].copy()
            if df.empty:
                continue
                
            solver_label = _extract_solver_label(f.name)
            
            df["method"] = df["backend"].apply(lambda b: "PyRecon" if b == "pyrecon" else solver_label)
            df["memory_peak_gb"] = df["memory_peak_mb"] / 1024.0
            dfs.append(df)
            
    if not dfs:
        return
    df = pd.concat(dfs, ignore_index=True)
    
    # Stessa logica della fig 3: solo nmesh sull'indice
    df_pivot = df.pivot_table(
        index="nmesh", 
        columns="method", 
        values="memory_peak_gb", 
        aggfunc="mean"
    )

    plot_cols = [c for c in METHOD_ORDER if c in df_pivot.columns]
    colors = [METHOD_COLORS[c] for c in plot_cols]
    
    fig, ax = plt.subplots(figsize=(9, 5))
    
    df_pivot[plot_cols].plot.bar(
        ax=ax, 
        color=colors, 
        rot=0, 
        width=0.8, 
        edgecolor='black', 
        linewidth=0.5
    )
    
    ax.set_xlabel("Mesh Size (nmesh)")
    ax.set_ylabel("Peak Memory [GB]")
    ax.set_yscale("log")
    ax.set_title("Peak Memory Usage by Method")
    
    ax.grid(axis='y', which='major', alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(title="Method", loc='upper left')
    
    _save(fig, "fig5_reconstructor_memory_comparison.pdf")

def main():
    figure_mass_assignment()
    figure_fft_solver()
    figure_reconstructor_time()
    figure_speedup()
    figure_reconstructor_memory()


if __name__ == "__main__":
    main()
