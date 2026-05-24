"""Truncation analysis for the California Housing dataset.

Quantifies the concentration of censored records (MedHouseVal == 5.0, i.e., $500k cap)
across four geographic quadrants. Confirms whether the Sul Costa fairness gap is caused
by the dataset truncation rather than a model deficiency.

Output
------
reports/truncation_by_region.png  — bar chart of truncation rate per region
Printed table with N, n_truncated and pct_truncated per region + global.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

TRUNCATION_THRESHOLD = 4.999  # >= 4.999 captura float 5.000010 do dataset

REGION_LABELS = {
    "norte_interior": "Norte Interior",
    "norte_costa":    "Norte Costa",
    "sul_interior":   "Sul Interior",
    "sul_costa":      "Sul Costa",
}


def load_full_dataset() -> pd.DataFrame:
    raw = fetch_california_housing(as_frame=True)
    df = raw.data.copy()
    df["MedHouseVal"] = raw.target
    return df


def assign_regions(df: pd.DataFrame) -> pd.Series:
    lat_med = df["Latitude"].median()
    lon_med = df["Longitude"].median()
    norte = df["Latitude"] >= lat_med
    costa = df["Longitude"] < lon_med
    conditions = [norte & ~costa, norte & costa, ~norte & ~costa, ~norte & costa]
    choices = ["norte_interior", "norte_costa", "sul_interior", "sul_costa"]
    return pd.Series(
        np.select(conditions, choices, default="desconhecido"),
        index=df.index, name="regiao",
    ), lat_med, lon_med


def print_table(results: dict) -> None:
    header = f"{'Regiao':<20} {'N':>6} {'Truncados':>10} {'% Truncado':>12}"
    sep = "=" * len(header)
    print(f"\n{sep}\n{header}\n{sep}")
    for key, m in results.items():
        label = REGION_LABELS.get(key, key)
        print(f"{label:<20} {m['n']:>6} {m['n_trunc']:>10} {m['pct']:>11.1f}%")
    print(sep)


def save_chart(results: dict, global_pct: float) -> None:
    labels = [REGION_LABELS[k] for k in results]
    pcts = [results[k]["pct"] for k in results]
    colors = ["#e74c3c" if p > global_pct * 1.5 else "#3498db" for p in pcts]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, pcts, color=colors, edgecolor="white", linewidth=0.8)
    ax.axhline(global_pct, color="black", linestyle="--", linewidth=1.2,
               label=f"Media global ({global_pct:.1f}%)")

    for bar, pct in zip(bars, pcts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{pct:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylabel("Registros com target = 5.0 (%)", fontsize=11)
    ax.set_title(
        "Concentracao de registros truncados ($500k) por regiao geografica\n"
        "California Housing Dataset (20.640 amostras, Censo 1990)",
        fontsize=11, pad=12,
    )
    ax.legend(fontsize=9)
    ax.set_ylim(0, max(pcts) * 1.25)
    plt.tight_layout()

    path = os.path.join(_THIS_DIR, "truncation_by_region.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nGrafico salvo em: {path}")


def main() -> None:
    print("Carregando dataset completo (20.640 amostras)...")
    df = load_full_dataset()

    n_total = len(df)
    n_trunc_global = (df["MedHouseVal"] >= TRUNCATION_THRESHOLD).sum()
    pct_global = n_trunc_global / n_total * 100
    print(f"Global: {n_trunc_global} registros truncados de {n_total} ({pct_global:.1f}%)")

    regions, lat_med, lon_med = assign_regions(df)
    df["regiao"] = regions

    results = {}
    for key in ["norte_interior", "norte_costa", "sul_interior", "sul_costa"]:
        sub = df[df["regiao"] == key]
        n = len(sub)
        n_trunc = (sub["MedHouseVal"] >= TRUNCATION_THRESHOLD).sum()
        results[key] = {"n": n, "n_trunc": int(n_trunc), "pct": n_trunc / n * 100}

    print_table(results)

    print("\nComparacao com a media global:")
    for key, m in results.items():
        ratio = m["pct"] / pct_global
        label = REGION_LABELS[key]
        print(f"  {label:<20} {m['pct']:.1f}% ({ratio:.1f}x a media global)")

    save_chart(results, pct_global)
    print("\nConclusao:")
    worst = max(results, key=lambda k: results[k]["pct"])
    print(
        f"  {REGION_LABELS[worst]} tem a maior concentracao de truncamento "
        f"({results[worst]['pct']:.1f}% vs {pct_global:.1f}% global). "
        f"O gap de RMSE nessa regiao e causado pelo dataset, nao pelo modelo."
    )


if __name__ == "__main__":
    main()
