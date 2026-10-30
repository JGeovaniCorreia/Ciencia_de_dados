"""Fairness analysis by geographic region for the California Housing XGBoost model.

Segments X_test into 5 geographic regions (3 metropolitan areas + 2 remaining
regions, see reports/regions.py), then reports RMSE, MAE, R2 and MAPE per region,
with 95% bootstrap confidence intervals for RMSE, MAPE and the RMSE gap. Raises a
fairness alert if the point RMSE gap between best and worst region exceeds 0.15
($15k) — the alert looks at RMSE only; R2 and MAPE are reported, not thresholded.

Outputs
-------
reports/fairness_geo.csv              — metrics table with CIs, one row per region
reports/fairness_geo_sensitivity.csv  — RMSE gap and sample sizes across metro radii
reports/fairness_geo_map.png          — scatter of Lat x Lon coloured by absolute error
"""

import os
import sys
from pathlib import Path as _Path

_PROJECT_ROOT = _Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
from src.transformers import WinsorizacaoTransformer, CaliforniaHousingTransformer  # noqa: E402
from reports.regions import (  # noqa: E402
    NORTH_SOUTH_LATITUDE, REGION_LABELS, REGION_ORDER, assign_regions, metro_areas,
)

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
FAIRNESS_RMSE_THRESHOLD = 0.15  # $15k in $100k units
N_BOOTSTRAP = 1000
CI_LEVEL = 0.95
# (metro radius km, San Diego radius km) — sensitivity grid around the 80/50 default
SENSITIVITY_RADII = [(60, 40), (70, 45), (80, 50), (90, 55), (100, 60)]

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.join(_THIS_DIR, "..")
ARTIFACTS_DIR = os.path.join(_ROOT_DIR, "artifacts")
REPORTS_DIR = _THIS_DIR

RENAME_MAP = {
    "MedInc": "RendaMediana",
    "HouseAge": "IdadeMediaResidencias",
    "AveRooms": "MediaComodos",
    "AveBedrms": "MediaQuartos",
    "Population": "Populacao",
    "AveOccup": "MediaOcupacao",
    "Latitude": "Latitude",
    "Longitude": "Longitude",
}


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_and_split() -> tuple[pd.DataFrame, pd.Series]:
    """Load California Housing and reproduce the exact notebook split.

    Returns
    -------
    X_test : pd.DataFrame
        Test features with renamed columns.
    y_test : pd.Series
        Test target in log1p scale.
    """
    raw = fetch_california_housing(as_frame=True)
    X = raw.data.rename(columns=RENAME_MAP)
    y = np.log1p(raw.target.rename("ValorMedioResidencias"))

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )
    # Keep the second split consistent with the notebook (used for calibration)
    _X_train, _X_cal, _y_train, _y_cal = train_test_split(
        X_trainval, y_trainval, test_size=0.25, random_state=RANDOM_STATE
    )

    return X_test, y_test


# ---------------------------------------------------------------------------
# Metrics helper
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true_log: np.ndarray,
    y_pred_log: np.ndarray,
) -> dict[str, float]:
    """Compute RMSE, MAE, R2 and MAPE in original $100k scale.

    Args:
        y_true_log: Ground-truth values in log1p scale.
        y_pred_log: Predicted values in log1p scale.

    Returns:
        Dictionary with keys: n, rmse, mae, r2, mape.
    """
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    # MAPE — avoid division by zero (target is always > 0 for this dataset)
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)

    return {"n": len(y_true), "rmse": rmse, "mae": mae, "r2": r2, "mape": mape}


def bootstrap_region_cis(
    y_true_log: np.ndarray,
    y_pred_log: np.ndarray,
    regions: np.ndarray,
) -> tuple[dict[str, dict[str, float]], tuple[float, float]]:
    """Percentile bootstrap CIs for per-region RMSE/MAPE and for the RMSE gap.

    Resamples the whole test set with replacement, so the gap CI accounts for the
    joint uncertainty of all regions.

    Args:
        y_true_log: Ground-truth values in log1p scale.
        y_pred_log: Predicted values in log1p scale.
        regions: Region key per sample, aligned to the targets.

    Returns:
        Tuple of (mapping region -> {rmse_lo, rmse_hi, mape_lo, mape_hi},
        (gap_lo, gap_hi)), all in $100k units except MAPE (%).
    """
    rng = np.random.default_rng(RANDOM_STATE)
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    sq_err = (y_true - y_pred) ** 2
    pct_err = np.abs((y_true - y_pred) / y_true) * 100

    n = len(y_true)
    rmse_boot = {k: np.empty(N_BOOTSTRAP) for k in REGION_ORDER}
    mape_boot = {k: np.empty(N_BOOTSTRAP) for k in REGION_ORDER}
    gap_boot = np.empty(N_BOOTSTRAP)
    for b in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, n)
        reg_b = regions[idx]
        for key in REGION_ORDER:
            m = reg_b == key
            rmse_boot[key][b] = np.sqrt(sq_err[idx][m].mean())
            mape_boot[key][b] = pct_err[idx][m].mean()
        rmses = [rmse_boot[k][b] for k in REGION_ORDER]
        gap_boot[b] = max(rmses) - min(rmses)

    alpha = (1 - CI_LEVEL) / 2 * 100
    q = (alpha, 100 - alpha)
    cis = {}
    for key in REGION_ORDER:
        rmse_lo, rmse_hi = np.percentile(rmse_boot[key], q)
        mape_lo, mape_hi = np.percentile(mape_boot[key], q)
        cis[key] = {"rmse_lo": float(rmse_lo), "rmse_hi": float(rmse_hi),
                    "mape_lo": float(mape_lo), "mape_hi": float(mape_hi)}
    gap_lo, gap_hi = np.percentile(gap_boot, q)
    return cis, (float(gap_lo), float(gap_hi))


def radii_sensitivity(
    X_test: pd.DataFrame,
    y_true_log: np.ndarray,
    y_pred_log: np.ndarray,
) -> pd.DataFrame:
    """RMSE gap and sample sizes for each pair of radii in SENSITIVITY_RADII.

    Args:
        X_test: Test DataFrame with 'Latitude' and 'Longitude' columns.
        y_true_log: Ground-truth values in log1p scale.
        y_pred_log: Predicted values in log1p scale.

    Returns:
        DataFrame with one row per radii pair.
    """
    rows = []
    for metro_km, sd_km in SENSITIVITY_RADII:
        regions = assign_regions(X_test, metro_km, sd_km).to_numpy()
        metrics = {k: compute_metrics(y_true_log[regions == k], y_pred_log[regions == k])
                   for k in REGION_ORDER}
        rmses = {k: m["rmse"] for k, m in metrics.items()}
        worst = max(rmses, key=rmses.get)
        best = min(rmses, key=rmses.get)
        rows.append({
            "metro_km": metro_km,
            "san_diego_km": sd_km,
            "gap_rmse_100k": round(rmses[worst] - rmses[best], 4),
            "pior_regiao": REGION_LABELS[worst],
            "rmse_pior_100k": round(rmses[worst], 4),
            "n_min": min(m["n"] for m in metrics.values()),
            "n_demais_sul": metrics["demais_sul"]["n"],
            "rmse_demais_sul_100k": round(rmses["demais_sul"], 4),
            "acima_limiar": rmses[worst] - rmses[best] > FAIRNESS_RMSE_THRESHOLD,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_metrics_table(results: dict[str, dict], cis: dict[str, dict]) -> None:
    """Print a formatted metrics table to stdout.

    Args:
        results: Mapping region_key -> metrics dict.
        cis: Mapping region_key -> bootstrap CI bounds.
    """
    header = (f"{'Região':<16} {'N':>5} {'RMSE':>7} {'IC95 RMSE':>15} {'MAE':>7} "
              f"{'R²':>6} {'MAPE%':>6} {'IC95 MAPE':>13}")
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for key, m in results.items():
        label = REGION_LABELS.get(key, key)
        ci = cis[key]
        print(
            f"{label:<16} {m['n']:>5} {m['rmse']:>7.4f} "
            f"[{ci['rmse_lo']:.3f}, {ci['rmse_hi']:.3f}] {m['mae']:>7.4f} "
            f"{m['r2']:>6.3f} {m['mape']:>6.2f} [{ci['mape_lo']:.1f}, {ci['mape_hi']:.1f}]"
        )
    print("=" * len(header))


def save_csv(results: dict[str, dict], cis: dict[str, dict], path: str) -> None:
    """Save metrics table to CSV.

    Args:
        results: Mapping region_key -> metrics dict.
        cis: Mapping region_key -> bootstrap CI bounds.
        path: Absolute path for the output CSV file.
    """
    rows = []
    for key, m in results.items():
        ci = cis[key]
        rows.append({
            "regiao_key": key,
            "regiao": REGION_LABELS.get(key, key),
            "n": m["n"],
            "rmse_100k": round(m["rmse"], 4),
            "rmse_ic95_lo": round(ci["rmse_lo"], 4),
            "rmse_ic95_hi": round(ci["rmse_hi"], 4),
            "mae_100k": round(m["mae"], 4),
            "r2": round(m["r2"], 4),
            "mape_pct": round(m["mape"], 2),
            "mape_ic95_lo": round(ci["mape_lo"], 2),
            "mape_ic95_hi": round(ci["mape_hi"], 2),
        })
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"\nTabela de métricas salva em: {path}")


def save_map(
    X_test: pd.DataFrame,
    abs_errors: np.ndarray,
    path: str,
) -> None:
    """Generate and save scatter plot of Lat x Lon coloured by absolute error.

    Draws the metro radii and the north/south latitude cut used by
    reports/regions.py.

    Args:
        X_test: Test DataFrame with 'Latitude' and 'Longitude' columns.
        abs_errors: Array of absolute errors ($100k scale), aligned to X_test.
        path: Absolute path for the output PNG file.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    sc = ax.scatter(
        X_test["Longitude"],
        X_test["Latitude"],
        c=abs_errors,
        cmap="RdYlGn_r",
        s=5,
        alpha=0.6,
        vmin=0,
        vmax=np.percentile(abs_errors, 95),  # clip colour scale at 95th pct for readability
    )

    # Metro radii — km converted to degrees (1° lat ~ 111 km; lon scaled by cos of the
    # centre latitude). Approximate ellipse, negligible error at these radii.
    theta = np.linspace(0, 2 * np.pi, 200)
    for key, (lat, lon), radius_km in metro_areas():
        dlat = radius_km / 111.0
        dlon = radius_km / (111.0 * np.cos(np.radians(lat)))
        ax.plot(lon + dlon * np.cos(theta), lat + dlat * np.sin(theta),
                color="steelblue", linestyle="--", linewidth=1.0, alpha=0.9)
        ax.text(lon + dlon, lat + dlat, f"{REGION_LABELS[key]} ({radius_km:.0f} km)",
                fontsize=8, color="black",
                bbox=dict(facecolor="white", alpha=0.55, edgecolor="none", pad=1.5))

    # North/south cut for samples outside the three metros
    ax.axhline(NORTH_SOUTH_LATITUDE, color="darkorange", linestyle="--", linewidth=1.0,
               alpha=0.8, label=f"Corte norte/sul fora das 3 metrópoles ({NORTH_SOUTH_LATITUDE}°)")
    x_max = X_test["Longitude"].max()
    for y_offset, key in [(0.3, "demais_norte"), (-0.5, "demais_sul")]:
        ax.text(x_max - 0.1, NORTH_SOUTH_LATITUDE + y_offset, REGION_LABELS[key],
                fontsize=8, color="black", ha="right",
                bbox=dict(facecolor="white", alpha=0.55, edgecolor="none", pad=1.5))

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Erro absoluto ($100k)", fontsize=10)

    ax.set_xlabel("Longitude", fontsize=11)
    ax.set_ylabel("Latitude", fontsize=11)
    ax.set_title(
        "Erro absoluto por localização geográfica\nXGBoost — California Housing (test set)",
        fontsize=12,
        pad=12,
    )
    ax.legend(fontsize=9, loc="lower right")

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Mapa salvo em: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # --- Load pipeline ---
    pipeline_path = os.path.join(ARTIFACTS_DIR, "competition_winner_xgboost.joblib")
    print(f"Carregando pipeline de: {pipeline_path}")
    pipeline = joblib.load(pipeline_path)

    # --- Load data and reproduce exact notebook split ---
    print("Carregando dados e reproduzindo split exato do notebook (60/20/20)...")
    X_test, y_test = load_and_split()
    print(f"X_test shape: {X_test.shape} | y_test shape: {y_test.shape}")

    # --- Predict ---
    print("Gerando predições...")
    y_pred_log = pipeline.predict(X_test)

    # --- Assign geographic regions ---
    print("Atribuindo regiões geográficas (regras fixas em reports/regions.py)...")
    regions = assign_regions(X_test)
    print(f"  Distribuição de amostras por região:\n{regions.value_counts().to_string()}\n")

    # --- Compute per-region metrics ---
    results = {}
    for key in REGION_ORDER:
        mask = regions == key
        assert mask.sum() > 0, f"Nenhuma amostra para a região '{key}' — revisar reports/regions.py"
        results[key] = compute_metrics(
            y_test[mask].values,
            y_pred_log[mask],
        )

    # --- Global metrics for reference ---
    global_metrics = compute_metrics(y_test.values, y_pred_log)
    print(f"\nMétricas globais (test set completo):")
    print(f"  N={global_metrics['n']}  RMSE={global_metrics['rmse']:.4f}  "
          f"MAE={global_metrics['mae']:.4f}  R²={global_metrics['r2']:.4f}  "
          f"MAPE={global_metrics['mape']:.2f}%")

    # --- Bootstrap CIs ---
    print(f"\nCalculando IC {CI_LEVEL:.0%} por bootstrap ({N_BOOTSTRAP} reamostras do test set)...")
    cis, (gap_lo, gap_hi) = bootstrap_region_cis(y_test.values, y_pred_log, regions.to_numpy())

    # --- Print table ---
    print_metrics_table(results, cis)

    # --- Fairness gap ---
    rmse_values = {k: v["rmse"] for k, v in results.items()}
    best_region = min(rmse_values, key=rmse_values.get)
    worst_region = max(rmse_values, key=rmse_values.get)
    gap = rmse_values[worst_region] - rmse_values[best_region]

    print(f"\nFairness Gap (RMSE):")
    print(f"  Melhor região : {REGION_LABELS[best_region]} — RMSE = {rmse_values[best_region]:.4f} ($100k)")
    print(f"  Pior região   : {REGION_LABELS[worst_region]} — RMSE = {rmse_values[worst_region]:.4f} ($100k)")
    print(f"  Gap           : {gap:.4f} ($100k) = ${gap * 100_000:,.0f}")
    print(f"  IC {CI_LEVEL:.0%} do gap : [{gap_lo:.4f}, {gap_hi:.4f}] ($100k)")

    if gap > FAIRNESS_RMSE_THRESHOLD:
        print(
            f"\n[ALERTA FAIRNESS] Gap de RMSE entre regiões ({gap:.4f}) excede o limiar "
            f"de {FAIRNESS_RMSE_THRESHOLD} ($15k). O modelo apresenta desempenho "
            f"desigual entre regiões geográficas — investigar antes de produção."
        )
    else:
        print(
            f"\n[OK] Gap de RMSE entre regiões ({gap:.4f}) está dentro do limiar "
            f"de {FAIRNESS_RMSE_THRESHOLD} ($15k)."
        )

    # --- Save CSV ---
    csv_path = os.path.join(REPORTS_DIR, "fairness_geo.csv")
    save_csv(results, cis, csv_path)

    # --- Sensitivity to the metro radii ---
    sensitivity = radii_sensitivity(X_test, y_test.values, y_pred_log)
    print("\nSensibilidade aos raios (metrópoles / San Diego):")
    print(sensitivity.to_string(index=False))
    sens_path = os.path.join(REPORTS_DIR, "fairness_geo_sensitivity.csv")
    sensitivity.to_csv(sens_path, index=False)
    print(f"Sensibilidade salva em: {sens_path}")

    # --- Save map ---
    abs_errors = np.abs(np.expm1(y_test.values) - np.expm1(y_pred_log))
    map_path = os.path.join(REPORTS_DIR, "fairness_geo_map.png")
    save_map(X_test, abs_errors, map_path)

    print("\nAnálise de fairness geográfica concluída.")


if __name__ == "__main__":
    main()
