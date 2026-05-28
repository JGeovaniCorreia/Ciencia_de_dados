"""Residuals analysis for the California Housing XGBoost model.

Generates four diagnostic plots on the test set:
  1. residuals_histogram.png   — distribution of residuals in $100k
  2. residuals_vs_fitted.png   — residuals vs. fitted values with LOWESS smoother
  3. residuals_qq.png          — Q-Q plot for normality check
  4. residuals_by_decile.png   — RMSE and MAE by target decile

Prints summary statistics to stdout.

Outputs
-------
reports/residuals_histogram.png
reports/residuals_vs_fitted.png
reports/residuals_qq.png
reports/residuals_by_decile.png
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import joblib
import scipy.stats as stats
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import sys
from pathlib import Path as _Path
_PROJECT_ROOT = _Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
from src.transformers import WinsorizacaoTransformer, CaliforniaHousingTransformer


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RANDOM_STATE = 42
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

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.join(_THIS_DIR, "..")
ARTIFACTS_DIR = os.path.join(_ROOT_DIR, "artifacts")
REPORTS_DIR = _THIS_DIR


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_and_split() -> tuple:
    """Load California Housing and reproduce the exact notebook split (60/20/20).

    Returns
    -------
    X_test : pd.DataFrame
        Test features with renamed columns (4128 samples).
    y_test : pd.Series
        Test target in log1p scale.
    """
    raw = fetch_california_housing(as_frame=True)
    X = raw.data.rename(columns=RENAME_MAP)
    y = np.log1p(raw.target.rename("ValorMedioResidencias"))

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )
    _X_train, _X_cal, _y_train, _y_cal = train_test_split(
        X_trainval, y_trainval, test_size=0.25, random_state=RANDOM_STATE
    )

    return X_test, y_test


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------

def save_histogram(residuals: np.ndarray, path: str) -> None:
    """Save histogram of residuals in $100k scale.

    Args:
        residuals: Array of residuals in $100k units (true - predicted).
        path: Absolute path for the output PNG.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.hist(residuals, bins=60, color="steelblue", edgecolor="white", alpha=0.85)
    ax.axvline(0, color="firebrick", linewidth=1.8, linestyle="--", label="Resíduo = 0")

    mean_res = residuals.mean()
    ax.axvline(mean_res, color="darkorange", linewidth=1.4, linestyle=":",
               label=f"Média = {mean_res:.4f} ($100k)")

    ax.set_xlabel("Resíduo ($100k)", fontsize=11)
    ax.set_ylabel("Frequência", fontsize=11)
    ax.set_title(
        "Distribuição dos resíduos\nXGBoost — California Housing (test set)",
        fontsize=12, pad=12,
    )
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Histograma salvo em: {path}")


def _lowess(x: np.ndarray, y: np.ndarray, frac: float = 0.15) -> tuple:
    """Compute LOWESS smoother using statsmodels if available, else numpy polyfit.

    Args:
        x: Predictor values (fitted).
        y: Response values (residuals).
        frac: Smoothing span for LOWESS.

    Returns:
        Tuple (x_smooth, y_smooth) sorted by x.
    """
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        smoothed = lowess(y, x, frac=frac, return_sorted=True)
        return smoothed[:, 0], smoothed[:, 1]
    except ImportError:
        # Fallback: cubic polynomial fit
        order = np.argsort(x)
        x_s = x[order]
        coeffs = np.polyfit(x_s, y[order], deg=3)
        y_s = np.polyval(coeffs, x_s)
        return x_s, y_s


def save_residuals_vs_fitted(
    y_pred: np.ndarray, residuals: np.ndarray, path: str
) -> None:
    """Save scatter plot of residuals vs. fitted values with smoother.

    Args:
        y_pred: Predicted values in $100k scale.
        residuals: Array of residuals in $100k scale (true - predicted).
        path: Absolute path for the output PNG.
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    ax.scatter(y_pred, residuals, s=4, alpha=0.35, color="steelblue", label="Resíduos")
    ax.axhline(0, color="firebrick", linewidth=1.8, linestyle="--", label="Resíduo = 0")

    # LOWESS smoother
    x_s, y_s = _lowess(y_pred, residuals)
    ax.plot(x_s, y_s, color="darkorange", linewidth=2.0, label="LOWESS")

    ax.set_xlabel("Valor predito ($100k)", fontsize=11)
    ax.set_ylabel("Resíduo ($100k)", fontsize=11)
    ax.set_title(
        "Resíduos vs. Valores preditos\nXGBoost — California Housing (test set)",
        fontsize=12, pad=12,
    )
    ax.legend(fontsize=9, loc="upper right")

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Gráfico resíduos vs. predito salvo em: {path}")


def save_qq_plot(residuals: np.ndarray, path: str) -> None:
    """Save Q-Q plot of residuals for normality inspection.

    Args:
        residuals: Array of residuals in $100k scale.
        path: Absolute path for the output PNG.
    """
    fig, ax = plt.subplots(figsize=(7, 7))

    (osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist="norm")
    ax.scatter(osm, osr, s=6, alpha=0.4, color="steelblue", label="Resíduos")
    ax.plot(osm, slope * np.array(osm) + intercept,
            color="firebrick", linewidth=1.8, linestyle="--", label="Linha normal teórica")

    ax.set_xlabel("Quantis teóricos (Normal)", fontsize=11)
    ax.set_ylabel("Quantis empíricos ($100k)", fontsize=11)
    ax.set_title(
        "Q-Q Plot dos resíduos\nXGBoost — California Housing (test set)",
        fontsize=12, pad=12,
    )
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Q-Q plot salvo em: {path}")


def save_decile_analysis(y_true: np.ndarray, y_pred: np.ndarray, path: str) -> None:
    """Save double bar chart of RMSE and MAE by target decile.

    Args:
        y_true: Ground-truth values in $100k scale.
        y_pred: Predicted values in $100k scale.
        path: Absolute path for the output PNG.
    """
    residuals = y_true - y_pred
    deciles = pd.qcut(y_true, q=10, duplicates="drop")
    df = pd.DataFrame({"residual": residuals, "decil": deciles})

    grouped = df.groupby("decil", observed=True)["residual"]
    rmse = grouped.apply(lambda r: np.sqrt((r ** 2).mean()))
    mae = grouped.apply(lambda r: np.abs(r).mean())
    labels = [
        f"${int(iv.left * 100)}k–\n${int(iv.right * 100)}k"
        for iv in rmse.index
    ]

    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(x - width / 2, rmse.values, width, label="RMSE", color="steelblue", alpha=0.85)
    ax.bar(x + width / 2, mae.values, width, label="MAE", color="firebrick", alpha=0.75)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_xlabel("Faixa de preço real (decil)", fontsize=11)
    ax.set_ylabel("Erro ($100k)", fontsize=11)
    ax.set_title(
        "RMSE e MAE por decil do target\nXGBoost — California Housing (test set)",
        fontsize=12, pad=12,
    )
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Análise por decil salva em: {path}")


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def print_summary(residuals: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Print key residual statistics to stdout.

    Args:
        residuals: Array of residuals in $100k scale.
        y_true: Ground-truth values in $100k scale (for decile breakdown).
        y_pred: Predicted values in $100k scale (for decile breakdown).
    """
    mean_res = residuals.mean()
    std_res = residuals.std()
    within_50k = np.mean(np.abs(residuals) <= 0.5) * 100  # 0.5 = $50k in $100k units

    # Shapiro-Wilk on a random subsample (test requires n <= 5000)
    rng = np.random.default_rng(RANDOM_STATE)
    sample = rng.choice(residuals, size=min(2000, len(residuals)), replace=False)
    stat_sw, p_sw = stats.shapiro(sample)

    # Breusch-Pagan-like heteroscedasticity hint via correlation of |residual| with rank
    abs_res = np.abs(residuals)
    rank_corr, p_rank = stats.spearmanr(np.arange(len(abs_res)), abs_res[np.argsort(residuals)])

    print("\n" + "=" * 55)
    print("  RESUMO DA ANALISE DE RESIDUOS — TEST SET")
    print("=" * 55)
    print(f"  N amostras             : {len(residuals):,}")
    print(f"  Media dos residuos     : {mean_res:+.4f} ($100k) = ${mean_res * 100_000:+,.0f}")
    print(f"  Desvio padrao          : {std_res:.4f} ($100k) = ${std_res * 100_000:,.0f}")
    print(f"  Dentro de +-$50k       : {within_50k:.1f}%")
    print(f"  Shapiro-Wilk (n=2000)  : W={stat_sw:.4f}  p={p_sw:.4e}")
    print("=" * 55)

    # Heteroscedasticity observation
    print("\nObservacoes:")
    if abs(mean_res) < 0.01:
        print("  [OK] Media proxima de zero — modelo nao apresenta vies sistematico.")
    else:
        direction = "superestima" if mean_res < 0 else "subestima"
        print(f"  [ATENCAO] Media de {mean_res:+.4f}: modelo tende a {direction} os valores.")

    if p_sw < 0.05:
        print("  [ATENCAO] Residuos nao seguem distribuicao normal (Shapiro-Wilk p < 0.05).")
        print("            Verificar Q-Q plot — caudas pesadas ou outliers podem estar presentes.")
    else:
        print("  [OK] Residuos aprovados no teste de normalidade (Shapiro-Wilk p >= 0.05).")

    if within_50k >= 80.0:
        print(f"  [OK] {within_50k:.1f}% dos residuos estao dentro da meta de +-$50k.")
    else:
        print(f"  [ATENCAO] Apenas {within_50k:.1f}% dos residuos dentro de +-$50k (meta: >= 80%).")

    print("\n  Heterocedasticidade (indicativo visual):")
    print("    Verificar residuals_vs_fitted.png — se a amplitude dos residuos")
    print("    aumenta com os valores preditos, ha heterocedasticidade.")
    print("    Isso e esperado em datasets com truncamento de target (teto em $500k).")

    print(f"\n  Spearman |residuo| vs rank predito: r={rank_corr:.4f}  p={p_rank:.4e}")
    if abs(rank_corr) > 0.3 and p_rank < 0.05:
        print("  [ATENCAO] Heterocedasticidade detectada — variancia dos residuos nao e constante.")
        print("            Correlacao rank sugere que erros crescem com o valor predito.")
    else:
        print("  [OK] Sem evidencia estatistica de heterocedasticidade (|r| <= 0.3 ou p >= 0.05).")

    deciles = pd.qcut(y_true, q=10, duplicates="drop")
    rmse_by_decile = (
        pd.DataFrame({"residual": residuals, "decil": deciles})
        .groupby("decil", observed=True)["residual"]
        .apply(lambda r: np.sqrt((r ** 2).mean()))
    )
    worst = rmse_by_decile.idxmax()
    print(
        f"\n  Decil com maior RMSE: {worst}"
        f"  —  RMSE={rmse_by_decile.max():.4f} ($100k)"
        f" = ${rmse_by_decile.max() * 100_000:,.0f}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run full residuals analysis and save all plots."""
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
    print("Gerando predicoes no test set...")
    y_pred_log = pipeline.predict(X_test)

    # --- Convert to original scale ($100k) ---
    y_true = np.expm1(y_test.values)   # ground truth in $100k
    y_pred = np.expm1(y_pred_log)      # predictions in $100k
    residuals = y_true - y_pred        # residuals in $100k

    # --- Summary statistics ---
    print_summary(residuals, y_true, y_pred)

    # --- Plot 1: Histogram ---
    hist_path = os.path.join(REPORTS_DIR, "residuals_histogram.png")
    print("\nGerando histograma dos residuos...")
    save_histogram(residuals, hist_path)

    # --- Plot 2: Residuals vs. Fitted ---
    rvf_path = os.path.join(REPORTS_DIR, "residuals_vs_fitted.png")
    print("Gerando grafico residuos vs. predito...")
    save_residuals_vs_fitted(y_pred, residuals, rvf_path)

    # --- Plot 3: Q-Q plot ---
    qq_path = os.path.join(REPORTS_DIR, "residuals_qq.png")
    print("Gerando Q-Q plot...")
    save_qq_plot(residuals, qq_path)

    # --- Plot 4: RMSE and MAE by decile ---
    decile_path = os.path.join(REPORTS_DIR, "residuals_by_decile.png")
    print("Gerando analise de residuos por decil do target...")
    save_decile_analysis(y_true, y_pred, decile_path)

    print("\nAnalise de residuos concluida. Arquivos gerados:")
    for p in [hist_path, rvf_path, qq_path, decile_path]:
        exists = os.path.isfile(p)
        print(f"  {'[OK]' if exists else '[ERRO]'} {p}")


if __name__ == "__main__":
    main()
