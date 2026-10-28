"""Ablation study: do dist_sf/dist_la/dist_sd add signal beyond Latitude/Longitude?

dist_* are deterministic functions of Latitude/Longitude. This script retrains the
approved XGBoost pipeline (same hyperparameters, same 60/20/20 split) in four
feature-set variants and compares them on the test set:

  a_completo     — all 13 features (reproduces the approved model)
  b_sem_dist     — without dist_sf, dist_la, dist_sd
  c1_sem_latlon  — without Latitude, Longitude (dist_* still carry location)
  c2_sem_geo     — without any geographic feature (floor: value of location)

Method
------
- Base pipeline: sklearn.clone of artifacts/competition_winner_xgboost.joblib, with
  a column-selection step inserted after CaliforniaHousingTransformer. Training
  runs on GPU when available (project rule); evaluation and SHAP run on CPU.
- Each variant is trained with N_SEEDS seeds to separate effect from the noise of
  subsample/colsample_bytree. Metrics are reported as mean ± std across seeds.
- Paired bootstrap (B resamples of the test set) on the seed-averaged predictions
  gives the 95% CI of each variant's delta vs a_completo.
- Metric definitions follow the notebook: R² in log1p scale; RMSE, MAE and MAPE
  in $100k after expm1.

Decision rule (fixed before running)
------------------------------------
Removing a feature group has a relevant effect when the 95% CI of ΔRMSE excludes
zero AND |ΔRMSE| >= 1% of the a_completo RMSE. Otherwise the group is redundant
given the remaining features.

Outputs
-------
reports/ablation_dist_features.csv        — metrics per variant
reports/ablation_dist_features.txt        — summary, deltas with CI and verdict
reports/ablation_dist_features_shap.png   — mean |SHAP| share per feature/variant
MLflow (mlflow.db, experiment california-housing-price-prediction): one parent
run per variant with one nested run per seed, tagged backlog=B04.
"""

import os
import sys
from datetime import datetime
from pathlib import Path as _Path

_PROJECT_ROOT = _Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
from src.transformers import CaliforniaHousingTransformer  # noqa: E402

import joblib  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
import torch  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.datasets import fetch_california_housing  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import FunctionTransformer  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RANDOM_STATE = 42
N_SEEDS = 5
N_BOOTSTRAP = 2_000
SHAP_SAMPLE_SIZE = 2_000
RELEVANCE_THRESHOLD = 0.01  # |ΔRMSE| as a fraction of the a_completo RMSE

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

FEATURES = list(CaliforniaHousingTransformer.OUTPUT_COLS)
GEO_FEATURES = ["Latitude", "Longitude", "dist_sf", "dist_la", "dist_sd"]
VARIANTS: dict[str, list[str]] = {
    "a_completo": [],
    "b_sem_dist": ["dist_sf", "dist_la", "dist_sd"],
    "c1_sem_latlon": ["Latitude", "Longitude"],
    "c2_sem_geo": GEO_FEATURES,
}
BASELINE = "a_completo"

REPORTS_DIR = _PROJECT_ROOT / "reports"
PIPELINE_PATH = _PROJECT_ROOT / "artifacts" / "competition_winner_xgboost.joblib"
MLFLOW_URI = f"sqlite:///{(_PROJECT_ROOT / 'mlflow.db').as_posix()}"
MLFLOW_EXPERIMENT = "california-housing-price-prediction"


# ---------------------------------------------------------------------------
# Data and pipeline
# ---------------------------------------------------------------------------

def load_and_split() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Load California Housing and reproduce the notebook split (60/20/20).

    Returns:
        X_train, y_train, X_test, y_test with targets in log1p scale. The
        calibration fold (20%) is not used here.
    """
    raw = fetch_california_housing(as_frame=True)
    X = raw.data.rename(columns=RENAME_MAP)
    y = np.log1p(raw.target.rename("ValorMedioResidencias"))
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )
    X_train, _X_cal, y_train, _y_cal = train_test_split(
        X_trainval, y_trainval, test_size=0.25, random_state=RANDOM_STATE
    )
    return X_train, y_train, X_test, y_test


def _select_columns(X: np.ndarray, idx: list[int]) -> np.ndarray:
    """Keep only the given column indices of the engineered feature matrix."""
    return X[:, idx]


def build_variant(base: Pipeline, dropped: list[str], seed: int, device: str) -> Pipeline:
    """Clone the approved pipeline and drop feature columns after engineering.

    Args:
        base: Approved fitted pipeline (only its configuration is reused).
        dropped: Engineered feature names to remove.
        seed: random_state for the XGBRegressor.
        device: XGBoost training device ("cuda" or "cpu").

    Returns:
        Unfitted pipeline: winsorizacao → transformer → selecao → scaler → modelo.
    """
    keep_idx = [i for i, name in enumerate(FEATURES) if name not in dropped]
    steps = clone(base).steps
    selector = FunctionTransformer(_select_columns, kw_args={"idx": keep_idx})
    steps.insert(2, ("selecao", selector))
    pipeline = Pipeline(steps)
    pipeline.named_steps["modelo"].set_params(random_state=seed, device=device)
    return pipeline


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> dict[str, float]:
    """Point metrics with the notebook definitions.

    Args:
        y_true_log: Target in log1p scale.
        y_pred_log: Prediction in log1p scale.

    Returns:
        Dict with r2 (log scale), rmse_100k, mae_100k and mape_pct.
    """
    y_true, y_pred = np.expm1(y_true_log), np.expm1(y_pred_log)
    return {
        "r2": r2_score(y_true_log, y_pred_log),
        "rmse_100k": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae_100k": mean_absolute_error(y_true, y_pred),
        "mape_pct": mean_absolute_percentage_error(y_true, y_pred) * 100,
    }


def paired_bootstrap_delta(
    y_true_log: np.ndarray, pred_variant: np.ndarray, pred_base: np.ndarray
) -> dict[str, tuple[float, float, float]]:
    """95% CI of (variant − baseline) for each metric via paired bootstrap.

    Args:
        y_true_log: Test target in log1p scale.
        pred_variant: Variant predictions (log1p), seed-averaged.
        pred_base: Baseline predictions (log1p), seed-averaged.

    Returns:
        Dict metric → (point delta, CI lower, CI upper).
    """
    rng = np.random.default_rng(RANDOM_STATE)
    n = len(y_true_log)
    point_v, point_b = compute_metrics(y_true_log, pred_variant), compute_metrics(y_true_log, pred_base)
    samples: dict[str, list[float]] = {k: [] for k in point_v}
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, n)
        m_v = compute_metrics(y_true_log[idx], pred_variant[idx])
        m_b = compute_metrics(y_true_log[idx], pred_base[idx])
        for k in samples:
            samples[k].append(m_v[k] - m_b[k])
    return {
        k: (point_v[k] - point_b[k], *np.percentile(samples[k], [2.5, 97.5]))
        for k in samples
    }


def shap_share(pipeline: Pipeline, X_sample: pd.DataFrame, kept: list[str]) -> pd.Series:
    """Mean |SHAP| per feature, normalized to sum 1 (computed on CPU).

    Args:
        pipeline: Fitted variant pipeline.
        X_sample: Raw test sample.
        kept: Feature names in the order the model sees them.

    Returns:
        Series indexed by feature name with each feature's share of total |SHAP|.
    """
    model = pipeline.named_steps["modelo"]
    model.set_params(device="cpu")
    X_model = pd.DataFrame(pipeline[:-1].transform(X_sample), columns=kept)
    values = shap.TreeExplainer(model).shap_values(X_model)
    mean_abs = pd.Series(np.abs(values).mean(axis=0), index=kept)
    return mean_abs / mean_abs.sum()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Treino em: {device}")
    base = joblib.load(PIPELINE_PATH)
    hyperparams = {
        k: v for k, v in base.named_steps["modelo"].get_params().items()
        if k in ("n_estimators", "learning_rate", "max_depth", "subsample", "colsample_bytree",
                 "min_child_weight", "reg_alpha", "reg_lambda", "gamma")
    }
    X_train, y_train, X_test, y_test = load_and_split()
    y_test_arr = y_test.to_numpy()
    seeds = [RANDOM_STATE + i for i in range(N_SEEDS)]
    rng = np.random.default_rng(RANDOM_STATE)
    shap_idx = rng.choice(len(X_test), size=min(SHAP_SAMPLE_SIZE, len(X_test)), replace=False)
    X_shap = X_test.iloc[shap_idx]

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    rows: list[dict] = []
    mean_preds: dict[str, np.ndarray] = {}
    shares: dict[str, pd.Series] = {}

    for variant, dropped in VARIANTS.items():
        kept = [f for f in FEATURES if f not in dropped]
        per_seed: list[dict[str, float]] = []
        preds: list[np.ndarray] = []
        with mlflow.start_run(run_name=f"ablation-b04-{variant}") as parent:
            mlflow.set_tags({"backlog": "B04", "study": "ablation_dist_features", "variant": variant})
            mlflow.log_params({
                "variant": variant,
                "dropped_features": ",".join(dropped) or "none",
                "n_features": len(kept),
                "seeds": ",".join(map(str, seeds)),
                "train_device": device,
                **hyperparams,
            })
            for seed in seeds:
                pipeline = build_variant(base, dropped, seed, device)
                pipeline.fit(X_train, y_train)
                pipeline.named_steps["modelo"].set_params(device="cpu")
                pred = pipeline.predict(X_test)
                metrics = compute_metrics(y_test_arr, pred)
                per_seed.append(metrics)
                preds.append(pred)
                with mlflow.start_run(run_name=f"{variant}-seed{seed}", nested=True):
                    mlflow.set_tags({"backlog": "B04", "variant": variant, "seed": str(seed)})
                    mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})
                if seed == RANDOM_STATE:
                    shares[variant] = shap_share(pipeline, X_shap, kept)
                print(f"  {variant:<14} seed {seed}: R²={metrics['r2']:.4f} RMSE={metrics['rmse_100k']:.4f}")

            mean_preds[variant] = np.mean(preds, axis=0)
            df_seed = pd.DataFrame(per_seed)
            row = {"variant": variant, "n_features": len(kept)}
            for k in df_seed.columns:
                row[f"{k}_mean"] = df_seed[k].mean()
                row[f"{k}_std"] = df_seed[k].std(ddof=1)
            row["geo_shap_share"] = float(shares[variant].reindex(GEO_FEATURES).fillna(0).sum())
            rows.append(row)
            mlflow.log_metrics({k: v for k, v in row.items() if k not in ("variant", "n_features")})

    results = pd.DataFrame(rows).set_index("variant")
    lines = [
        "Ablation B04 — features de distância geográfica (dist_sf, dist_la, dist_sd)",
        f"Gerado em {datetime.now():%Y-%m-%d %H:%M} | treino: {device} | seeds: {seeds} | "
        f"bootstrap: {N_BOOTSTRAP} | test set: {len(X_test)} linhas",
        "",
        "Métricas no test set (média ± desvio entre seeds; R² em log1p, RMSE/MAE em $100k):",
    ]
    for v, r in results.iterrows():
        lines.append(
            f"  {v:<14} n={int(r.n_features):>2}  R²={r.r2_mean:.4f}±{r.r2_std:.4f}  "
            f"RMSE={r.rmse_100k_mean:.4f}±{r.rmse_100k_std:.4f}  MAE={r.mae_100k_mean:.4f}  "
            f"MAPE={r.mape_pct_mean:.2f}%  SHAP geo={r.geo_shap_share:.1%}"
        )

    base_rmse = results.loc[BASELINE, "rmse_100k_mean"]
    lines += ["", f"Δ vs {BASELINE} (predições médias entre seeds; IC 95% bootstrap pareado):"]
    with mlflow.start_run(run_name="ablation-b04-summary"):
        mlflow.set_tags({"backlog": "B04", "study": "ablation_dist_features", "variant": "summary"})
        mlflow.log_params({"relevance_threshold": RELEVANCE_THRESHOLD, "n_bootstrap": N_BOOTSTRAP})
        for variant in VARIANTS:
            if variant == BASELINE:
                continue
            delta = paired_bootstrap_delta(y_test_arr, mean_preds[variant], mean_preds[BASELINE])
            d_rmse, lo, hi = delta["rmse_100k"]
            relevant = (lo > 0 or hi < 0) and abs(d_rmse) >= RELEVANCE_THRESHOLD * base_rmse
            verdict = "EFEITO RELEVANTE" if relevant else "sem efeito relevante (redundante)"
            lines.append(
                f"  {variant:<14} ΔRMSE={d_rmse:+.4f} [{lo:+.4f}, {hi:+.4f}] "
                f"({d_rmse / base_rmse:+.1%})  ΔR²={delta['r2'][0]:+.4f} "
                f"[{delta['r2'][1]:+.4f}, {delta['r2'][2]:+.4f}]  → {verdict}"
            )
            mlflow.log_metrics({
                f"{variant}_delta_rmse": d_rmse, f"{variant}_delta_rmse_ci_low": lo,
                f"{variant}_delta_rmse_ci_high": hi, f"{variant}_delta_r2": delta["r2"][0],
            })
        lines += [
            "",
            f"Critério: efeito relevante se o IC 95% de ΔRMSE exclui zero E |ΔRMSE| >= "
            f"{RELEVANCE_THRESHOLD:.0%} do RMSE de {BASELINE} ({base_rmse:.4f}).",
        ]
        summary = "\n".join(lines)
        print("\n" + summary)

        results.to_csv(REPORTS_DIR / "ablation_dist_features.csv", float_format="%.6f")
        (REPORTS_DIR / "ablation_dist_features.txt").write_text(summary + "\n", encoding="utf-8")

        share_df = pd.DataFrame(shares).reindex(FEATURES).fillna(0.0)
        fig, ax = plt.subplots(figsize=(11, 6))
        share_df.plot.barh(ax=ax, width=0.8)
        ax.invert_yaxis()
        ax.set_xlabel("Participação no |SHAP| médio")
        ax.set_title("Ablation B04 — redistribuição da importância entre variantes (seed 42)")
        ax.legend(title="Variante")
        fig.tight_layout()
        shap_path = REPORTS_DIR / "ablation_dist_features_shap.png"
        fig.savefig(shap_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        mlflow.log_artifact(str(REPORTS_DIR / "ablation_dist_features.txt"))
        mlflow.log_artifact(str(shap_path))
        print(f"\nSaídas: {REPORTS_DIR / 'ablation_dist_features.csv'}, .txt e {shap_path.name}")


if __name__ == "__main__":
    main()
