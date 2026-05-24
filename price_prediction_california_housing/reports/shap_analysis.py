"""SHAP analysis for the California Housing XGBoost winner model."""

import os
import sys
from pathlib import Path as _Path

_PROJECT_ROOT = _Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
from src.transformers import WinsorizacaoTransformer, CaliforniaHousingTransformer  # noqa: E402

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
SAMPLE_SIZE = 2_000
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")
REPORTS_DIR = os.path.dirname(__file__)

FEATURE_NAMES = [
    "RendaMediana", "IdadeMediaResidencias", "MediaComodos", "MediaQuartos",
    "Populacao", "MediaOcupacao", "Latitude", "Longitude",
    "razao_quartos", "comodos_por_pessoa", "dist_sf", "dist_la", "dist_sd",
]


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    raw = fetch_california_housing(as_frame=True)
    X = raw.data.rename(columns={
        "MedInc": "RendaMediana",
        "HouseAge": "IdadeMediaResidencias",
        "AveRooms": "MediaComodos",
        "AveBedrms": "MediaQuartos",
        "Population": "Populacao",
        "AveOccup": "MediaOcupacao",
        "Latitude": "Latitude",
        "Longitude": "Longitude",
    })
    y = np.log1p(raw.target.rename("ValorMedioResidencias"))
    return X, y


def reproduce_train_split(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Reproduz o split exato do notebook: 60/20/20."""
    X_trainval, _, y_trainval, _ = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )
    X_train, _, _, _ = train_test_split(
        X_trainval, y_trainval, test_size=0.25, random_state=RANDOM_STATE
    )
    return X_train


def main() -> None:
    pipeline_path = os.path.join(ARTIFACTS_DIR, "competition_winner_xgboost.joblib")
    print(f"Carregando pipeline de {pipeline_path}...")
    pipeline = joblib.load(pipeline_path)

    print("Carregando dados e reproduzindo split do notebook...")
    X, y = load_data()
    X_train = reproduce_train_split(X, y)

    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(X_train), size=min(SAMPLE_SIZE, len(X_train)), replace=False)
    X_sample = X_train.iloc[idx].reset_index(drop=True)
    print(f"Amostra de {len(X_sample)} linhas do conjunto de treino selecionada.")

    print("Transformando features via pipeline (exceto o modelo final)...")
    X_transformed = pipeline[:-1].transform(X_sample)
    X_transformed_df = pd.DataFrame(X_transformed, columns=FEATURE_NAMES)

    model = pipeline[-1]
    print("Calculando SHAP values com TreeExplainer...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_transformed_df)

    npy_path = os.path.join(REPORTS_DIR, "shap_values.npy")
    np.save(npy_path, shap_values)
    print(f"SHAP values salvos em {npy_path}")

    print("Gerando shap_summary_plot.png (beeswarm)...")
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X_transformed_df, feature_names=FEATURE_NAMES, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "shap_summary_plot.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print("Gerando shap_bar_plot.png (importância média absoluta)...")
    plt.figure(figsize=(9, 6))
    shap.summary_plot(
        shap_values, X_transformed_df, feature_names=FEATURE_NAMES,
        plot_type="bar", show=False
    )
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "shap_bar_plot.png"), dpi=150, bbox_inches="tight")
    plt.close()

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    ranking = sorted(zip(FEATURE_NAMES, mean_abs_shap), key=lambda x: x[1], reverse=True)

    print("\n--- Top-5 features por importância SHAP (|mean SHAP|) ---")
    for i, (feat, val) in enumerate(ranking[:5], 1):
        print(f"  {i}. {feat:<25} {val:.4f}")

    print("\nAnálise SHAP concluída. Arquivos salvos em reports/:")
    print("  - shap_summary_plot.png")
    print("  - shap_bar_plot.png")
    print("  - shap_values.npy")


if __name__ == "__main__":
    main()
