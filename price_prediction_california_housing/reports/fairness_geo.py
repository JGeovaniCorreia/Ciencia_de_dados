"""Fairness analysis by geographic region for the California Housing XGBoost model.

Segments X_test into 4 geographic quadrants (Norte/Sul x Interior/Costa) using
median Latitude and Longitude as cut-points, then reports RMSE, MAE, R2 and MAPE
per region. Raises a fairness alert if the RMSE gap between best and worst region
exceeds 0.15 ($15k).

Outputs
-------
reports/fairness_geo.csv      — metrics table, one row per region
reports/fairness_geo_map.png  — scatter of Lat x Lon coloured by absolute error
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

import numpy as np
import pandas as pd
import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


# Classes do pipeline — necessárias para joblib.load reconstruir o pipeline serializado
class WinsorizacaoTransformer(BaseEstimator, TransformerMixin):
    """Winsorização com bounds IQR aprendidos no treino (sem data leakage)."""

    def __init__(self, colunas=None, k=3.0):
        self.colunas = colunas
        self.k = k

    def fit(self, X, y=None):
        df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        cols = self.colunas or df.columns.tolist()
        self.bounds_ = {}
        for col in cols:
            if col in df.columns:
                q1, q3 = df[col].quantile([0.25, 0.75])
                iqr = q3 - q1
                self.bounds_[col] = (q1 - self.k * iqr, q3 + self.k * iqr)
        return self

    def transform(self, X):
        check_is_fitted(self, attributes=["bounds_"])
        df = pd.DataFrame(X).copy() if not isinstance(X, pd.DataFrame) else X.copy()
        for col, (lo, hi) in self.bounds_.items():
            if col in df.columns:
                df[col] = df[col].clip(lo, hi)
        return df


class CaliforniaHousingTransformer(BaseEstimator, TransformerMixin):
    """Feature engineering para o California Housing Dataset."""

    _CITIES = {
        "sf": (37.7749, -122.4194),
        "la": (34.0522, -118.2437),
        "sd": (32.7157, -117.1611),
    }
    LOG1P_FEATURES = ["RendaMediana", "Populacao", "MediaOcupacao"]
    INPUT_COLS = [
        "RendaMediana", "IdadeMediaResidencias", "MediaComodos",
        "MediaQuartos", "Populacao", "MediaOcupacao", "Latitude", "Longitude",
    ]
    OUTPUT_COLS = INPUT_COLS + ["razao_quartos", "comodos_por_pessoa", "dist_sf", "dist_la", "dist_sd"]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X[self.INPUT_COLS].copy() if isinstance(X, pd.DataFrame) \
             else pd.DataFrame(X, columns=self.INPUT_COLS[:X.shape[1]])
        for col in self.LOG1P_FEATURES:
            df[col] = np.log1p(df[col])
        df["razao_quartos"]      = df["MediaQuartos"]  / (df["MediaComodos"]  + 1e-8)
        df["comodos_por_pessoa"] = df["MediaComodos"]  / (df["MediaOcupacao"] + 1e-8)
        for city, (lat, lon) in self._CITIES.items():
            df[f"dist_{city}"] = np.sqrt(
                (df["Latitude"]  - lat) ** 2 +
                (df["Longitude"] - lon) ** 2
            )
        return df[self.OUTPUT_COLS].values

    def get_feature_names_out(self, input_features=None):
        return np.array(self.OUTPUT_COLS)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
FAIRNESS_RMSE_THRESHOLD = 0.15  # $15k in $100k units

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

REGION_LABELS = {
    "norte_interior": "Norte Interior",
    "norte_costa": "Norte Costa",
    "sul_interior": "Sul Interior",
    "sul_costa": "Sul Costa",
}


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_and_split() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
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


# ---------------------------------------------------------------------------
# Region assignment
# ---------------------------------------------------------------------------

def assign_regions(X_test: pd.DataFrame) -> pd.Series:
    """Assign each test sample to one of four geographic quadrants.

    Quadrant definition uses median Latitude and Longitude of X_test:
      - Norte Interior : Latitude >= median AND Longitude >= median (east)
      - Norte Costa    : Latitude >= median AND Longitude <  median (west, SF area)
      - Sul Interior   : Latitude <  median AND Longitude >= median
      - Sul Costa      : Latitude <  median AND Longitude <  median (LA, SD area)

    Args:
        X_test: Test DataFrame with 'Latitude' and 'Longitude' columns.

    Returns:
        pd.Series of string region keys aligned to X_test index.
    """
    lat_med = X_test["Latitude"].median()
    lon_med = X_test["Longitude"].median()

    norte = X_test["Latitude"] >= lat_med
    costa = X_test["Longitude"] < lon_med  # longitude is negative; less negative = more west

    conditions = [
        norte & ~costa,   # Norte Interior
        norte & costa,    # Norte Costa
        ~norte & ~costa,  # Sul Interior
        ~norte & costa,   # Sul Costa
    ]
    choices = ["norte_interior", "norte_costa", "sul_interior", "sul_costa"]

    return pd.Series(
        np.select(conditions, choices, default="desconhecido"),
        index=X_test.index,
        name="regiao",
    ), lat_med, lon_med


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_metrics_table(results: dict[str, dict]) -> None:
    """Print a formatted metrics table to stdout.

    Args:
        results: Mapping region_key -> metrics dict.
    """
    header = f"{'Região':<20} {'N':>6} {'RMSE ($100k)':>13} {'MAE ($100k)':>12} {'R²':>7} {'MAPE (%)':>10}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for key, m in results.items():
        label = REGION_LABELS.get(key, key)
        print(
            f"{label:<20} {m['n']:>6} {m['rmse']:>13.4f} {m['mae']:>12.4f} "
            f"{m['r2']:>7.4f} {m['mape']:>10.2f}"
        )
    print("=" * len(header))


def save_csv(results: dict[str, dict], path: str) -> None:
    """Save metrics table to CSV.

    Args:
        results: Mapping region_key -> metrics dict.
        path: Absolute path for the output CSV file.
    """
    rows = []
    for key, m in results.items():
        rows.append({
            "regiao_key": key,
            "regiao": REGION_LABELS.get(key, key),
            "n": m["n"],
            "rmse_100k": round(m["rmse"], 4),
            "mae_100k": round(m["mae"], 4),
            "r2": round(m["r2"], 4),
            "mape_pct": round(m["mape"], 2),
        })
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"\nTabela de métricas salva em: {path}")


def save_map(
    X_test: pd.DataFrame,
    abs_errors: np.ndarray,
    lat_med: float,
    lon_med: float,
    path: str,
) -> None:
    """Generate and save scatter plot of Lat x Lon coloured by absolute error.

    Args:
        X_test: Test DataFrame with 'Latitude' and 'Longitude' columns.
        abs_errors: Array of absolute errors ($100k scale), aligned to X_test.
        lat_med: Latitude cut-point (median of X_test).
        lon_med: Longitude cut-point (median of X_test).
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

    # Quadrant dividers
    ax.axhline(lat_med, color="steelblue", linestyle="--", linewidth=1.0, alpha=0.8, label=f"Lat mediana ({lat_med:.2f}°)")
    ax.axvline(lon_med, color="darkorange", linestyle="--", linewidth=1.0, alpha=0.8, label=f"Lon mediana ({lon_med:.2f}°)")

    # Quadrant labels
    x_min, x_max = X_test["Longitude"].min(), X_test["Longitude"].max()
    y_min, y_max = X_test["Latitude"].min(), X_test["Latitude"].max()
    offset = 0.05

    quadrant_labels = [
        (lon_med + offset, lat_med + offset, "Norte Interior"),
        (x_min + offset, lat_med + offset, "Norte Costa"),
        (lon_med + offset, y_min + offset, "Sul Interior"),
        (x_min + offset, y_min + offset, "Sul Costa"),
    ]
    for lx, ly, txt in quadrant_labels:
        ax.text(lx, ly, txt, fontsize=8, color="black",
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
    print("Atribuindo regiões geográficas...")
    regions, lat_med, lon_med = assign_regions(X_test)
    print(f"  Mediana Latitude : {lat_med:.4f}°")
    print(f"  Mediana Longitude: {lon_med:.4f}°")
    print(f"  Distribuição de amostras por região:\n{regions.value_counts().to_string()}\n")

    # --- Compute per-region metrics ---
    results = {}
    for key in ["norte_interior", "norte_costa", "sul_interior", "sul_costa"]:
        mask = regions == key
        if mask.sum() == 0:
            print(f"AVISO: nenhuma amostra para região '{key}' — ignorando.")
            continue
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

    # --- Print table ---
    print_metrics_table(results)

    # --- Fairness gap ---
    rmse_values = {k: v["rmse"] for k, v in results.items()}
    best_region = min(rmse_values, key=rmse_values.get)
    worst_region = max(rmse_values, key=rmse_values.get)
    gap = rmse_values[worst_region] - rmse_values[best_region]

    print(f"\nFairness Gap (RMSE):")
    print(f"  Melhor região : {REGION_LABELS[best_region]} — RMSE = {rmse_values[best_region]:.4f} ($100k)")
    print(f"  Pior região   : {REGION_LABELS[worst_region]} — RMSE = {rmse_values[worst_region]:.4f} ($100k)")
    print(f"  Gap           : {gap:.4f} ($100k) = ${gap * 100_000:,.0f}")

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
    save_csv(results, csv_path)

    # --- Save map ---
    abs_errors = np.abs(np.expm1(y_test.values) - np.expm1(y_pred_log))
    map_path = os.path.join(REPORTS_DIR, "fairness_geo_map.png")
    save_map(X_test, abs_errors, lat_med, lon_med, map_path)

    print("\nAnálise de fairness geográfica concluída.")


if __name__ == "__main__":
    main()
