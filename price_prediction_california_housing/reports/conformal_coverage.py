"""Conditional coverage of the conformal prediction intervals (backlog B10).

The production interval uses one global q_hat per confidence level (split conformal
on absolute log-scale residuals), which guarantees marginal coverage only. This
script measures the empirical coverage (PICP) and mean width (MPIW) on the test set
by geographic region, by predicted price band and by true price band, with 95%
Wilson confidence intervals.

It also evaluates a Mondrian (group-conditional) alternative: one q_hat per
predicted price band, calibrated on the same calibration set. The band is known at
inference time (it comes from the prediction itself), so the per-band guarantee
holds for new samples.

Outcome: Mondrian was rejected for production — it lowers the coverage of the
non-metro regions to ~75% and halves the >= $450k interval, where the true value is
censored. Nothing here changes production: the artefact and competition_metadata.json are
only read.

Outputs
-------
reports/conformal_coverage.csv          — coverage/width per method, level, cut and group
reports/conformal_mondrian_qhats.csv    — Mondrian q_hat per level and predicted band
reports/conformal_coverage_by_band.png  — coverage by predicted band, global vs Mondrian
MLflow run "conformal-coverage-b10" (tag backlog=B10) with the files above
"""

import json
import sys
from pathlib import Path as _Path

_PROJECT_ROOT = _Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
from src.transformers import WinsorizacaoTransformer, CaliforniaHousingTransformer  # noqa: E402,F401
from reports.fairness_geo import RENAME_MAP  # noqa: E402
from reports.regions import REGION_LABELS, assign_regions  # noqa: E402

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
LEVELS = [0.8, 0.9, 0.95]
TRUNCATION_THRESHOLD = 4.999  # >= 4.999 captures the 5.000010 float of the dataset
Z_95 = 1.96

# Price bands in $100k — same edges for predicted and true values
BAND_EDGES = [0.0, 1.0, 2.0, 3.0, 4.5, np.inf]
BAND_LABELS = ["< $100k", "$100–200k", "$200–300k", "$300–450k", ">= $450k"]

ARTIFACTS_DIR = _PROJECT_ROOT / "artifacts"
REPORTS_DIR = _PROJECT_ROOT / "reports"
PIPELINE_PATH = ARTIFACTS_DIR / "competition_winner_xgboost.joblib"
METADATA_PATH = ARTIFACTS_DIR / "competition_metadata.json"

MLFLOW_URI = f"sqlite:///{(_PROJECT_ROOT / 'mlflow.db').as_posix()}"
MLFLOW_EXPERIMENT = "california-housing-price-prediction"


# ---------------------------------------------------------------------------
# Data and calibration
# ---------------------------------------------------------------------------

def load_cal_test() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Reproduce the notebook 60/20/20 split and return calibration and test sets.

    Returns:
        X_cal, y_cal, X_test, y_test — targets in log1p scale.
    """
    raw = fetch_california_housing(as_frame=True)
    X = raw.data.rename(columns=RENAME_MAP)
    y = np.log1p(raw.target.rename("ValorMedioResidencias"))

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )
    _X_train, X_cal, _y_train, y_cal = train_test_split(
        X_trainval, y_trainval, test_size=0.25, random_state=RANDOM_STATE
    )
    return X_cal, y_cal, X_test, y_test


def conformal_q_hat(scores: np.ndarray, level: float) -> float:
    """Split-conformal quantile with the finite-sample correction used in the notebook.

    Args:
        scores: Non-conformity scores (absolute log-scale residuals).
        level: Target coverage, e.g. 0.8.

    Returns:
        q_hat in log scale.
    """
    q_level = min(1.0, level * (1 + 1 / len(scores)))
    return float(np.quantile(scores, q_level))


def price_band(values_100k: np.ndarray) -> np.ndarray:
    """Map values in $100k to BAND_LABELS.

    Args:
        values_100k: Values in $100k units.

    Returns:
        Array of band labels.
    """
    idx = np.digitize(values_100k, BAND_EDGES[1:-1], right=False)
    return np.asarray(BAND_LABELS, dtype=object)[idx]


def wilson_ci(successes: int, n: int) -> tuple[float, float]:
    """95% Wilson score interval for a proportion.

    Args:
        successes: Number of covered samples.
        n: Number of samples.

    Returns:
        (lower, upper) bounds.
    """
    p = successes / n
    denom = 1 + Z_95 ** 2 / n
    centre = (p + Z_95 ** 2 / (2 * n)) / denom
    half = Z_95 * np.sqrt(p * (1 - p) / n + Z_95 ** 2 / (4 * n ** 2)) / denom
    return float(centre - half), float(centre + half)


# ---------------------------------------------------------------------------
# Coverage table
# ---------------------------------------------------------------------------

def coverage_rows(
    method: str,
    level: float,
    covered: np.ndarray,
    width_100k: np.ndarray,
    cuts: dict[str, np.ndarray],
) -> list[dict]:
    """Coverage and width rows for every group of every cut.

    Args:
        method: "global" or "mondrian".
        level: Nominal coverage level.
        covered: Boolean array, True when the true value is inside the interval.
        width_100k: Interval width in $100k (original scale).
        cuts: Mapping cut name -> group label per sample.

    Returns:
        List of row dicts.
    """
    rows = []
    for cut, groups in cuts.items():
        for group in pd.unique(groups):
            m = groups == group
            n = int(m.sum())
            k = int(covered[m].sum())
            lo, hi = wilson_ci(k, n)
            rows.append({
                "metodo": method,
                "nivel": level,
                "recorte": cut,
                "grupo": group,
                "n": n,
                "cobertura": round(k / n, 4),
                "ic95_lo": round(lo, 4),
                "ic95_hi": round(hi, 4),
                "abaixo_nominal": hi < level,
                "mpiw_100k": round(float(width_100k[m].mean()), 4),
            })
    return rows


def save_band_plot(df: pd.DataFrame, path: _Path) -> None:
    """Plot coverage by predicted band for 80% and 95%, global vs Mondrian.

    Args:
        df: Coverage table from coverage_rows().
        path: Output PNG path.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    x = np.arange(len(BAND_LABELS))
    offsets = {"global": -0.17, "mondrian": 0.17}
    colours = {"global": "#3498db", "mondrian": "#e67e22"}
    names = {"global": "q_hat global (produção)", "mondrian": "q_hat por faixa (Mondrian)"}

    for ax, level in zip(axes, [0.8, 0.95]):
        for method in ["global", "mondrian"]:
            sub = (df[(df.metodo == method) & (df.nivel == level) & (df.recorte == "faixa_prevista")]
                   .set_index("grupo").reindex(BAND_LABELS))
            yerr = np.vstack([sub.cobertura - sub.ic95_lo, sub.ic95_hi - sub.cobertura])
            ax.errorbar(x + offsets[method], sub.cobertura, yerr=yerr, fmt="o", capsize=4,
                        color=colours[method], label=names[method])
        ax.axhline(level, color="black", linestyle="--", linewidth=1.0,
                   label=f"Nominal ({level:.0%})")
        ax.set_xticks(x)
        ax.set_xticklabels(BAND_LABELS, fontsize=9)
        ax.set_xlabel("Faixa da predição", fontsize=10)
        ax.set_ylabel("Cobertura empírica (test set)", fontsize=10)
        ax.set_title(f"Intervalo de {level:.0%}", fontsize=11)
        ax.legend(fontsize=8, loc="lower left")

    fig.suptitle("Cobertura do intervalo conformal por faixa de preço prevista (IC 95% de Wilson)",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Gráfico salvo em: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    pipeline = joblib.load(PIPELINE_PATH)
    with open(METADATA_PATH, encoding="utf-8") as f:
        meta_q_hats = json.load(f)["conformal_q_hats"]

    X_cal, y_cal, X_test, y_test = load_cal_test()
    print(f"Calibração: {len(X_cal)} | Teste: {len(X_test)}")

    pred_cal = pipeline.predict(X_cal)
    pred_test = pipeline.predict(X_test)
    scores_cal = np.abs(y_cal.to_numpy() - pred_cal)
    res_test = np.abs(y_test.to_numpy() - pred_test)

    band_cal = price_band(np.expm1(pred_cal))
    true_100k = np.expm1(y_test.to_numpy())
    cuts = {
        "global": np.full(len(X_test), "todos", dtype=object),
        "regiao": np.asarray([REGION_LABELS[k] for k in assign_regions(X_test)], dtype=object),
        "faixa_prevista": price_band(np.expm1(pred_test)),
        "faixa_real": price_band(true_100k),
        "truncado": np.where(true_100k >= TRUNCATION_THRESHOLD, "truncado ($500k)", "não truncado"),
    }

    rows: list[dict] = []
    mondrian_rows: list[dict] = []
    for level in LEVELS:
        # Global q_hat — must match the production metadata
        q_global = conformal_q_hat(scores_cal, level)
        q_meta = float(meta_q_hats[str(level)])
        assert abs(q_global - q_meta) < 1e-5, (
            f"q_hat recalculado ({q_global:.6f}) diverge do metadata ({q_meta:.6f}) no nível {level}"
        )
        lower = np.maximum(np.expm1(pred_test - q_meta), 0.0)
        upper = np.expm1(pred_test + q_meta)
        rows += coverage_rows("global", level, res_test <= q_meta, upper - lower, cuts)

        # Mondrian — one q_hat per predicted band, calibrated on the same set
        q_band = {}
        for band in BAND_LABELS:
            m = band_cal == band
            q_band[band] = conformal_q_hat(scores_cal[m], level)
            mondrian_rows.append({"nivel": level, "faixa_prevista": band, "n_cal": int(m.sum()),
                                  "q_hat": round(q_band[band], 6), "q_hat_global": round(q_meta, 6)})
        q_test = np.array([q_band[b] for b in cuts["faixa_prevista"]])
        lower_m = np.maximum(np.expm1(pred_test - q_test), 0.0)
        upper_m = np.expm1(pred_test + q_test)
        rows += coverage_rows("mondrian", level, res_test <= q_test, upper_m - lower_m, cuts)

    df = pd.DataFrame(rows)
    df_q = pd.DataFrame(mondrian_rows)

    pd.set_option("display.width", 160)
    print("\nq_hat por faixa prevista (Mondrian):")
    print(df_q.to_string(index=False))
    for level in LEVELS:
        print(f"\n=== Nível {level:.0%} ===")
        view = df[df.nivel == level].pivot_table(
            index=["recorte", "grupo"], columns="metodo",
            values=["cobertura", "mpiw_100k"], aggfunc="first",
        )
        print(view.round(3).to_string())
    flagged = df[df.abaixo_nominal]
    print("\nGrupos com IC 95% inteiro abaixo do nominal:")
    print(flagged[["metodo", "nivel", "recorte", "grupo", "n", "cobertura", "ic95_hi"]].to_string(index=False)
          if len(flagged) else "  nenhum")

    csv_path = REPORTS_DIR / "conformal_coverage.csv"
    q_path = REPORTS_DIR / "conformal_mondrian_qhats.csv"
    png_path = REPORTS_DIR / "conformal_coverage_by_band.png"
    df.to_csv(csv_path, index=False)
    df_q.to_csv(q_path, index=False)
    save_band_plot(df, png_path)
    print(f"Tabelas salvas em: {csv_path} e {q_path}")

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name="conformal-coverage-b10"):
        mlflow.set_tags({"backlog": "B10", "study": "conformal_coverage"})
        mlflow.log_params({"levels": LEVELS, "band_edges_100k": BAND_EDGES[1:-1],
                           "n_cal": len(X_cal), "n_test": len(X_test)})
        for level in LEVELS:
            tag = int(level * 100)
            for method in ["global", "mondrian"]:
                sub = df[(df.metodo == method) & (df.nivel == level)]
                bands = sub[sub.recorte == "faixa_prevista"]
                mlflow.log_metrics({
                    f"{method}_picp_{tag}": float(sub[sub.recorte == "global"].cobertura.iloc[0]),
                    f"{method}_mpiw_{tag}": float(sub[sub.recorte == "global"].mpiw_100k.iloc[0]),
                    f"{method}_picp_min_faixa_{tag}": float(bands.cobertura.min()),
                    f"{method}_picp_max_faixa_{tag}": float(bands.cobertura.max()),
                })
        for path in (csv_path, q_path, png_path):
            mlflow.log_artifact(str(path))
    print("Run MLflow registrado: conformal-coverage-b10")


if __name__ == "__main__":
    main()
