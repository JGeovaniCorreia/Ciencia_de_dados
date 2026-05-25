"""Statistical significance test: XGBoost vs LightGBM (California Housing).

Answers: is the 0.46 pp R² difference (0.9017 vs 0.8971 in MLflow) statistically
significant, or could it be noise from the CV split?

Method: 10-fold CV on the training set, Wilcoxon signed-rank test on paired RMSE
scores (non-parametric, appropriate for small n).

Note: XGBoost uses Optuna-tuned params (500 trials). LightGBM uses defaults — no
Optuna tuning was run for it. The comparison is conservative for XGBoost: with
equivalent tuning, the gap could be smaller.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
from src.transformers import WinsorizacaoTransformer, CaliforniaHousingTransformer  # noqa: E402

SEED = 42
N_FOLDS = 10

RENAME_MAP = {
    "MedInc": "RendaMediana",
    "HouseAge": "IdadeMediaResidencias",
    "AveRooms": "MediaComodos",
    "AveBedrms": "MediaQuartos",
    "Population": "Populacao",
    "AveOccup": "MediaOcupacao",
}

# XGBoost params from Optuna trial (logged in MLflow run xgboost-winner-v1)
XGB_PARAMS = {
    "n_estimators": 573,
    "max_depth": 9,
    "learning_rate": 0.02537407815079108,
    "subsample": 0.8972672148504683,
    "colsample_bytree": 0.500998607593687,
    "min_child_weight": 1,
    "gamma": 1.4214841513542551e-07,
    "reg_alpha": 3.4158867038635544e-08,
    "reg_lambda": 0.05805605402091764,
    "random_state": SEED,
    "device": "cpu",
}


def build_pipeline(model) -> Pipeline:
    return Pipeline([
        ("winsorize", WinsorizacaoTransformer()),
        ("features", CaliforniaHousingTransformer()),
        ("scaler", StandardScaler()),
        ("model", model),
    ])


def run_cv(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    rmse_scores = []
    for train_idx, val_idx in kf.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        pipeline.fit(X_tr, np.log1p(y_tr))
        pred = np.expm1(pipeline.predict(X_val))
        rmse = np.sqrt(np.mean((pred - y_val.values) ** 2))
        rmse_scores.append(rmse)
    return np.array(rmse_scores)


def main() -> None:
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame.rename(columns=RENAME_MAP)
    X = df.drop(columns=["MedHouseVal"])
    y = df["MedHouseVal"]

    # Same 80/20 split as model_competition.ipynb (seed=42)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=SEED)

    print(f"Dataset treino: {X_train.shape[0]} amostras")
    print(f"CV: {N_FOLDS} folds, seed={SEED}\n")

    print("Rodando CV — XGBoost (params Optuna)...")
    xgb_rmse = run_cv(build_pipeline(XGBRegressor(**XGB_PARAMS)), X_train, y_train)

    print("Rodando CV — LightGBM (params default)...")
    lgbm_rmse = run_cv(
        build_pipeline(LGBMRegressor(random_state=SEED, verbose=-1)),
        X_train, y_train,
    )

    stat, p_value = stats.wilcoxon(xgb_rmse, lgbm_rmse, alternative="two-sided")
    # Rank-biserial correlation as effect size
    n = N_FOLDS
    r_effect = 1 - (2 * stat) / (n * (n + 1) / 2)

    lines = [
        "=" * 62,
        "Teste de significância: XGBoost vs LightGBM",
        "=" * 62,
        "",
        f"RMSE por fold ($100k):",
        f"  XGBoost  : {xgb_rmse.round(4).tolist()}",
        f"  LightGBM : {lgbm_rmse.round(4).tolist()}",
        "",
        f"Média RMSE XGBoost  : {xgb_rmse.mean():.4f} ± {xgb_rmse.std():.4f}",
        f"Média RMSE LightGBM : {lgbm_rmse.mean():.4f} ± {lgbm_rmse.std():.4f}",
        f"Diferença média     : {(lgbm_rmse - xgb_rmse).mean():+.4f} (+ = LightGBM pior)",
        "",
        "Wilcoxon signed-rank (pareado, bilateral, n=10 folds):",
        f"  Estatística W       : {stat:.4f}",
        f"  p-value             : {p_value:.4f}",
        f"  Tamanho do efeito r : {r_effect:.4f}",
        "",
    ]

    if p_value < 0.05:
        conclusion = (
            f"CONCLUSAO: diferenca estatisticamente significativa (p={p_value:.4f} < 0.05). "
            f"XGBoost e superior ao LightGBM neste dataset e com estes parametros."
        )
    else:
        conclusion = (
            f"CONCLUSAO: diferenca NAO e estatisticamente significativa (p={p_value:.4f} >= 0.05). "
            f"XGBoost e LightGBM sao equivalentes neste dataset. A escolha do XGBoost "
            f"se sustenta pelo tuning Optuna (500 trials) — sem tuning equivalente "
            f"para o LightGBM, a superioridade nao e demonstravel estatisticamente."
        )

    lines.append(conclusion)
    lines += [
        "",
        "NOTA METODOLOGICA:",
        "  XGBoost usou params do Optuna (500 trials, cv_r2=0.9017 no MLflow).",
        "  LightGBM usou params default (cv_r2=0.8971 no MLflow — sem tuning).",
        "  Comparacao e conservadora para o XGBoost. Com tuning equivalente,",
        "  a diferenca poderia ser menor ou reverter.",
    ]

    output = "\n".join(lines)
    print("\n" + output)

    out_path = _ROOT / "reports" / "model_significance_test.txt"
    out_path.write_text(output, encoding="utf-8")
    print(f"\nSalvo em: {out_path}")


if __name__ == "__main__":
    main()
