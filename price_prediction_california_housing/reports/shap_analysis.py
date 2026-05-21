"""SHAP analysis for the California Housing XGBoost winner model."""

import os
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
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
