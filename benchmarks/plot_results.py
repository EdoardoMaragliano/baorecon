"""Generate figures from the benchmark CSVs in ``benchmarks/results/``.

Produces four PDFs in ``benchmarks/figures/``:

1. ``fig1_mass_assignment_time.pdf`` -- time vs N particles for mass assignment.
2. ``fig2_fft_solver_time.pdf``      -- time vs nmesh for the FFT solver.
3. ``fig3_pipeline_time.pdf``        -- end-to-end time vs N particles.
4. ``fig5_memory.pdf``               -- peak RSS increase per step vs N particles.

Each figure is drawn from whatever data is present; missing backends are simply
omitted, and figures with no usable data are skipped with a message instead of
crashing.

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
    "pyrecon": {"color": "#2ca02c", "marker": "^", "label": "pyrecon"},
}
BACKEND_ORDER = ["baorecon_cpu", "pyrecon"]


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
    n = _plot_backends(ax, df, "n_particles", "time_mean")
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
# Figure 2: FFT solver time vs nmesh (total of setup+solve+readout)
# ---------------------------------------------------------------------------
def figure_fft_solver():
    df = _load("fft_solver.csv")
    if df is None:
        return
    total = (df.groupby(["backend", "nmesh"], as_index=False)["time_mean"].sum())
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    n = _plot_backends(ax, total, "nmesh", "time_mean")
    if n == 0:
        plt.close(fig)
        print("[skip] fig2: no data")
        return
    ax.set_xlabel("nmesh")
    ax.set_ylabel("Total time [s] (setup + solve + readout)")
    ax.set_title("Iterative FFT solver")
    ax.legend()
    ax.grid(True, which="both", alpha=0.4)
    _save(fig, "fig2_fft_solver_time.pdf")


# ---------------------------------------------------------------------------
# Figure 3: end-to-end pipeline time vs N
# ---------------------------------------------------------------------------
def figure_pipeline():
    df = _load("pipeline.csv")
    if df is None:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    n = _plot_backends(ax, df, "n_particles", "time_mean")
    if n == 0:
        plt.close(fig)
        print("[skip] fig3: no data")
        return
    ax.set_xlabel("Number of particles")
    ax.set_ylabel("End-to-end time [s]")
    ax.set_title(f"Full pipeline (nmesh={int(df['nmesh'].iloc[0])})")
    ax.legend()
    ax.grid(True, which="both", alpha=0.4)
    _save(fig, "fig3_pipeline_time.pdf")


# ---------------------------------------------------------------------------
# Figure 5: peak memory vs N
# ---------------------------------------------------------------------------
def figure_memory():
    df = _load("mass_assignment.csv")
    if df is None:
        return
    method = "CIC" if (df["step"] == "CIC").any() else df["step"].iloc[0]
    df = df[df["step"] == method]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    n = _plot_backends(ax, df, "n_particles", "memory_peak_mb", logy=True)
    if n == 0:
        plt.close(fig)
        print("[skip] fig5: no data")
        return
    ax.set_xlabel("Number of particles")
    ax.set_ylabel("Peak RSS increase per step [MiB]")
    ax.set_title(f"Memory (peak RSS) — {method} mass assignment")
    ax.legend()
    ax.grid(True, which="both", alpha=0.4)
    _save(fig, "fig5_memory.pdf")


def main():
    figure_mass_assignment()
    figure_fft_solver()
    figure_pipeline()
    figure_memory()


if __name__ == "__main__":
    main()
