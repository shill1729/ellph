#!/usr/bin/env python3
"""
Render ellipsoidal intersection benchmark results into paper-ready plots and LaTeX tables.

Expected CSV schema (from C++ benchmark):
    d,n,method,mean_ms,std_ms,num_trials
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ----------------------------------------------------------------------
# Configuration knobs
# ----------------------------------------------------------------------

CSV_PATH = Path("benchmark_results.csv")

# Which fixed d to use when plotting runtime vs n
FIXED_D_FOR_N_SWEEPS = [2, 4, 5]

# Which fixed n to use when plotting runtime vs d
FIXED_N_FOR_D_SWEEPS = [2, 3, 4, 5, 10, 20, 50]

# Methods order for consistent legend / tables
METHOD_ORDER = [
    "Raw-SLSQP",
    "Raw-PGD",
    "Raw-Cauchy",
    "LP-Seidel",
    "LP-Clarkson",
]

# Output directories
FIG_DIR = Path("figs")
TABLE_DIR = Path("tables")


# ----------------------------------------------------------------------
# Core utilities
# ----------------------------------------------------------------------

def load_results(csv_path: Path = CSV_PATH) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Ensure correct dtypes
    df["d"] = df["d"].astype(int)
    df["n"] = df["n"].astype(int)
    df["method"] = df["method"].astype(str)
    return df


def ensure_dirs():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------

def plot_runtime_vs_n(df: pd.DataFrame, fixed_d: int, use_logy: bool = True):
    """
    For a fixed dimension d, plot mean runtime vs n for all methods.
    Saves a single PDF figure.
    """
    subset = df[df["d"] == fixed_d].copy()
    if subset.empty:
        print(f"[WARN] No data for d={fixed_d}, skipping runtime_vs_n plot.")
        return

    # Ensure consistent ordering in n and method
    subset = subset.sort_values(["n", "method"])
    methods = [m for m in METHOD_ORDER if m in subset["method"].unique()]

    plt.figure()
    for m in methods:
        sub_m = subset[subset["method"] == m]
        # Sort by n to get proper lines
        sub_m = sub_m.sort_values("n")
        plt.plot(
            sub_m["n"],
            sub_m["mean_ms"],
            marker="o",
            label=m,
        )

    if use_logy:
        plt.yscale("log")

    plt.xlabel(r"$n$ (number of ellipsoids)")
    plt.ylabel("Mean runtime (ms)")
    plt.title(fr"Mean runtime vs $n$ at fixed $d={fixed_d}$")
    plt.grid(True, which="both", linestyle="--", alpha=0.3)
    plt.legend()

    out_path = FIG_DIR / f"runtime_vs_n_d{fixed_d}.pdf"
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[INFO] Saved {out_path}")


def plot_runtime_vs_d(df: pd.DataFrame, fixed_n: int, use_logy: bool = True):
    """
    For a fixed n, plot mean runtime vs d for all methods.
    Saves a single PDF figure.
    """
    subset = df[df["n"] == fixed_n].copy()
    if subset.empty:
        print(f"[WARN] No data for n={fixed_n}, skipping runtime_vs_d plot.")
        return

    subset = subset.sort_values(["d", "method"])
    methods = [m for m in METHOD_ORDER if m in subset["method"].unique()]

    plt.figure()
    for m in methods:
        sub_m = subset[subset["method"] == m]
        sub_m = sub_m.sort_values("d")
        plt.plot(
            sub_m["d"],
            sub_m["mean_ms"],
            marker="o",
            label=m,
        )

    if use_logy:
        plt.yscale("log")

    plt.xlabel(r"$d$ (dimension)")
    plt.ylabel("Mean runtime (ms)")
    plt.title(fr"Mean runtime vs $d$ at fixed $n={fixed_n}$")
    plt.grid(True, which="both", linestyle="--", alpha=0.3)
    plt.legend()

    out_path = FIG_DIR / f"runtime_vs_d_n{fixed_n}.pdf"
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[INFO] Saved {out_path}")


def plot_heatmap_for_method(df: pd.DataFrame, method: str):
    """
    Heatmap of log10 mean runtime over the (n,d) grid for a single method.
    Rows = d, columns = n.
    """
    sub = df[df["method"] == method].copy()
    if sub.empty:
        print(f"[WARN] No data for method={method}, skipping heatmap.")
        return

    # Pivot to a d x n matrix
    pivot = sub.pivot(index="d", columns="n", values="mean_ms")
    # Sort axes
    pivot = pivot.sort_index(axis=0).sort_index(axis=1)

    plt.figure()
    # Avoid log of zero
    data = pivot.to_numpy()
    # tiny epsilon to avoid warnings if something is extremely small
    eps = 1e-12
    log_data = np.log10(data + eps)

    im = plt.imshow(
        log_data,
        origin="lower",
        aspect="auto",
        extent=[
            pivot.columns.min() - 0.5,
            pivot.columns.max() + 0.5,
            pivot.index.min() - 0.5,
            pivot.index.max() + 0.5,
        ],
    )
    plt.colorbar(im, label=r"$\log_{10}$ mean runtime (ms)")

    plt.xlabel(r"$n$")
    plt.ylabel(r"$d$")
    plt.title(fr"Runtime heatmap for method {method}")
    plt.xticks(pivot.columns)
    plt.yticks(pivot.index)

    out_path = FIG_DIR / f"heatmap_{method}.pdf"
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[INFO] Saved {out_path}")


# ----------------------------------------------------------------------
# LaTeX table generation
# ----------------------------------------------------------------------

def format_cell(mean_ms: float, std_ms: float, precision: int = 3) -> str:
    """
    Format a mean ± std cell as a LaTeX string, e.g. 1.23 (0.45).
    """
    fmt = f"{{:.{precision}g}}"
    return f"{fmt.format(mean_ms)} ({fmt.format(std_ms)})"


def latex_table_fixed_d(df: pd.DataFrame, fixed_d: int, precision: int = 3):
    """
    For a fixed d, produce a LaTeX table:
    rows = methods, columns = n, entries = mean(std) in ms.
    """
    sub = df[df["d"] == fixed_d].copy()
    if sub.empty:
        print(f"[WARN] No data for d={fixed_d}, skipping LaTeX table.")
        return

    sub = sub.sort_values(["method", "n"])
    methods = [m for m in METHOD_ORDER if m in sub["method"].unique()]
    ns = sorted(sub["n"].unique())

    # Build a DataFrame of formatted strings
    table_data = {}
    for m in methods:
        row = []
        for n in ns:
            entry = sub[(sub["method"] == m) & (sub["n"] == n)]
            if entry.empty:
                cell = "-"
            else:
                mean_ms = entry["mean_ms"].iloc[0]
                std_ms = entry["std_ms"].iloc[0]
                cell = format_cell(mean_ms, std_ms, precision=precision)
            row.append(cell)
        table_data[m] = row

    table_df = pd.DataFrame(table_data, index=[str(n) for n in ns]).T
    table_df.index.name = "Method"
    table_df.columns = [fr"$n={n}$" for n in ns]

    caption = fr"Mean runtime (ms) and standard deviation for fixed $d={fixed_d}$."
    label = fr"tab:runtime_d{fixed_d}"

    latex_str = table_df.to_latex(
        escape=False,
        caption=caption,
        label=label,
        column_format="l" + "c" * len(ns),
    )

    out_path = TABLE_DIR / f"runtime_fixed_d{fixed_d}.tex"
    with open(out_path, "w") as f:
        f.write(latex_str)

    print(f"[INFO] Saved {out_path}")


def latex_table_fixed_n(df: pd.DataFrame, fixed_n: int, precision: int = 3):
    """
    For a fixed n, produce a LaTeX table:
    rows = methods, columns = d, entries = mean(std) in ms.
    """
    sub = df[df["n"] == fixed_n].copy()
    if sub.empty:
        print(f"[WARN] No data for n={fixed_n}, skipping LaTeX table.")
        return

    sub = sub.sort_values(["method", "d"])
    methods = [m for m in METHOD_ORDER if m in sub["method"].unique()]
    ds = sorted(sub["d"].unique())

    table_data = {}
    for m in methods:
        row = []
        for d in ds:
            entry = sub[(sub["method"] == m) & (sub["d"] == d)]
            if entry.empty:
                cell = "-"
            else:
                mean_ms = entry["mean_ms"].iloc[0]
                std_ms = entry["std_ms"].iloc[0]
                cell = format_cell(mean_ms, std_ms, precision=precision)
            row.append(cell)
        table_data[m] = row

    table_df = pd.DataFrame(table_data, index=[str(d) for d in ds]).T
    table_df.index.name = "Method"
    table_df.columns = [fr"$d={d}$" for d in ds]

    caption = fr"Mean runtime (ms) and standard deviation for fixed $n={fixed_n}$."
    label = fr"tab:runtime_n{fixed_n}"

    latex_str = table_df.to_latex(
        escape=False,
        caption=caption,
        label=label,
        column_format="l" + "c" * len(ds),
    )

    out_path = TABLE_DIR / f"runtime_fixed_n{fixed_n}.tex"
    with open(out_path, "w") as f:
        f.write(latex_str)

    print(f"[INFO] Saved {out_path}")


# ----------------------------------------------------------------------
# Main driver
# ----------------------------------------------------------------------

def main():
    ensure_dirs()
    df = load_results()

    # Basic sanity check: aggregate again to ensure we are consistent
    # (in case one ever switches to trial-level CSV).
    grouped = df.groupby(["d", "n", "method"], as_index=False).agg(
        mean_ms=("mean_ms", "mean"),
        std_ms=("std_ms", "mean"),
        num_trials=("num_trials", "mean"),  # should already be constant
    )
    # If df is already aggregated (one row per (d,n,method)), grouped == df
    df_ag = grouped

    # Plots: runtime vs n for each fixed d
    for d in FIXED_D_FOR_N_SWEEPS:
        plot_runtime_vs_n(df_ag, fixed_d=d, use_logy=True)

    # Plots: runtime vs d for each fixed n
    for n in FIXED_N_FOR_D_SWEEPS:
        plot_runtime_vs_d(df_ag, fixed_n=n, use_logy=True)

    # Heatmaps per method
    for m in METHOD_ORDER:
        plot_heatmap_for_method(df_ag, method=m)

    # LaTeX tables for fixed d
    for d in FIXED_D_FOR_N_SWEEPS:
        latex_table_fixed_d(df_ag, fixed_d=d, precision=3)

    # LaTeX tables for fixed n
    for n in FIXED_N_FOR_D_SWEEPS:
        latex_table_fixed_n(df_ag, fixed_n=n, precision=3)


if __name__ == "__main__":
    main()
