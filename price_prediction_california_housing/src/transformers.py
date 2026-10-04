"""Shared pipeline transformers for the California Housing XGBoost model.

This module is the single source of truth for WinsorizacaoTransformer and
CaliforniaHousingTransformer. Import from here in all scripts so that joblib
can deserialise the pipeline regardless of the caller's working directory.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class WinsorizacaoTransformer(BaseEstimator, TransformerMixin):
    """Winsorization with IQR bounds learned at fit time."""

    def __init__(self, colunas: list[str] | None = None, k: float = 3.0):
        self.colunas = colunas
        self.k = k

    def fit(self, X: pd.DataFrame, y=None) -> "WinsorizacaoTransformer":
        df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        cols = self.colunas or df.columns.tolist()
        self.bounds_: dict = {}
        for col in cols:
            if col in df.columns:
                q1, q3 = df[col].quantile([0.25, 0.75])
                iqr = q3 - q1
                self.bounds_[col] = (q1 - self.k * iqr, q3 + self.k * iqr)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, attributes=["bounds_"])
        df = pd.DataFrame(X).copy() if not isinstance(X, pd.DataFrame) else X.copy()
        for col, (lo, hi) in self.bounds_.items():
            if col in df.columns:
                df[col] = df[col].clip(lo, hi)
        return df


class CaliforniaHousingTransformer(BaseEstimator, TransformerMixin):
    """Feature engineering for the California Housing dataset."""

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

    def fit(self, X: pd.DataFrame, y=None) -> "CaliforniaHousingTransformer":
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                f"X deve ser um DataFrame pandas com colunas nomeadas. "
                f"Recebido: {type(X).__name__}"
            )
        df = X[self.INPUT_COLS].copy()
        for col in self.LOG1P_FEATURES:
            df[col] = np.log1p(df[col])
        df["razao_quartos"] = df["MediaQuartos"] / (df["MediaComodos"] + 1e-8)
        df["comodos_por_pessoa"] = df["MediaComodos"] / (df["MediaOcupacao"] + 1e-8)
        for city, (lat, lon) in self._CITIES.items():
            df[f"dist_{city}"] = np.sqrt(
                (df["Latitude"] - lat) ** 2 + (df["Longitude"] - lon) ** 2
            )
        return df[self.OUTPUT_COLS].values

    def get_feature_names_out(self, input_features=None):
        return np.array(self.OUTPUT_COLS)
