"""Production inference script for the California Housing XGBoost model.

Usage
-----
Single prediction from a dict:
    from src.predict import prever
    result = prever({"MedInc": 5.0, "HouseAge": 30, "AveRooms": 5.5,
                     "AveBedrms": 1.1, "Population": 1200, "AveOccup": 3.0,
                     "Latitude": 34.05, "Longitude": -118.24})

Batch prediction from a DataFrame (original sklearn column names):
    result = prever(X_df)

Returns a dict with keys:
    predicao_100k   — predicted median house value in $100k units
    predicao_usd    — same in dollars
    intervalo_80pct — list of (lower, upper) tuples in $100k (conformal, 80%)
    alerta_teto     — True if prediction is in the $450k+ risk zone
"""

import functools
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from src.transformers import WinsorizacaoTransformer, CaliforniaHousingTransformer  # noqa: E402

import joblib
import numpy as np
import pandas as pd

_ARTIFACTS = _ROOT / "artifacts"
_METADATA_PATH = _ARTIFACTS / "competition_metadata.json"
_PIPELINE_PATH = _ARTIFACTS / "competition_winner_xgboost.joblib"

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

TETO_ALERTA_100K = 4.5
TETO_MODELO_100K = 5.0


def _load_pipeline():
    return joblib.load(_PIPELINE_PATH)


@functools.lru_cache(maxsize=None)
def _load_q_hat(nivel: float) -> float:
    with open(_METADATA_PATH) as f:
        meta = json.load(f)
    q_hats = meta["conformal_q_hats"]
    key = str(nivel)
    if key not in q_hats:
        raise ValueError(f"nivel={nivel} nao disponivel. Opcoes: {list(q_hats.keys())}")
    return float(q_hats[key])


def _normalize_input(X) -> pd.DataFrame:
    """Accept dict, list-of-dicts, or DataFrame; return DataFrame with PT column names."""
    if isinstance(X, dict):
        X = pd.DataFrame([X])
    elif isinstance(X, list):
        X = pd.DataFrame(X)

    if not isinstance(X, pd.DataFrame):
        raise TypeError(f"X deve ser dict, list-of-dicts ou DataFrame. Recebido: {type(X)}")

    if set(RENAME_MAP.keys()).issubset(X.columns):
        X = X.rename(columns=RENAME_MAP)

    required = list(RENAME_MAP.values())
    missing = [c for c in required if c not in X.columns]
    if missing:
        raise ValueError(f"Colunas faltando: {missing}")

    return X[required]


def prever(
    X: dict | list[dict] | pd.DataFrame,
    nivel_confianca: float = 0.8,
    pipeline=None,
) -> dict:
    """Predict median house value with conformal interval and ceiling alert.

    Args:
        X: Input features as dict (single), list-of-dicts, or DataFrame.
           Accepts original sklearn names (MedInc, HouseAge, ...) or
           Portuguese names (RendaMediana, IdadeMediaResidencias, ...).
        nivel_confianca: Confidence level for the conformal interval (0.8, 0.9, or 0.95).
        pipeline: Pre-loaded pipeline (optional — avoids reloading on every call).

    Returns:
        Dict with keys:
            predicao_100k   (np.ndarray) — predicted value in $100k
            predicao_usd    (np.ndarray) — predicted value in USD
            intervalo_lower (np.ndarray) — lower bound in $100k
            intervalo_upper (np.ndarray) — upper bound in $100k
            alerta_teto     (list[bool]) — True if prediction >= $450k risk zone
            mensagem_alerta (list[str])  — human-readable warning (empty str if OK)
    """
    if pipeline is None:
        pipeline = _load_pipeline()

    q_hat = _load_q_hat(nivel_confianca)
    X_df = _normalize_input(X)

    pred_log = pipeline.predict(X_df)
    pred_100k = np.expm1(pred_log)
    pred_usd = pred_100k * 100_000

    lower = np.expm1(pred_log - q_hat)
    upper = np.expm1(pred_log + q_hat)

    alertas = pred_100k >= TETO_ALERTA_100K
    mensagens = [
        (
            f"AVISO: predicao proxima ao limite do modelo (${TETO_MODELO_100K * 100_000:,.0f}). "
            "O modelo pode subestimar imoveis de alto valor — "
            "consulte comparativos de mercado diretamente."
        )
        if a else ""
        for a in alertas
    ]

    return {
        "predicao_100k": pred_100k,
        "predicao_usd": pred_usd,
        "intervalo_lower": lower,
        "intervalo_upper": upper,
        "alerta_teto": alertas.tolist(),
        "mensagem_alerta": mensagens,
    }


if __name__ == "__main__":
    print("Testando script de inferencia com 3 amostras...")

    exemplos = [
        {
            "MedInc": 8.3, "HouseAge": 25, "AveRooms": 6.2, "AveBedrms": 1.0,
            "Population": 1800, "AveOccup": 3.1, "Latitude": 37.77, "Longitude": -122.42,
        },
        {
            "MedInc": 2.5, "HouseAge": 40, "AveRooms": 4.0, "AveBedrms": 1.2,
            "Population": 900, "AveOccup": 2.8, "Latitude": 34.05, "Longitude": -118.24,
        },
        {
            "MedInc": 15.0, "HouseAge": 10, "AveRooms": 8.0, "AveBedrms": 1.0,
            "Population": 500, "AveOccup": 2.5, "Latitude": 34.02, "Longitude": -118.50,
        },
    ]

    resultado = prever(exemplos, nivel_confianca=0.8)

    for i, (pred, lower, upper, alerta, msg) in enumerate(zip(
        resultado["predicao_100k"],
        resultado["intervalo_lower"],
        resultado["intervalo_upper"],
        resultado["alerta_teto"],
        resultado["mensagem_alerta"],
    )):
        print(f"\nAmostra {i+1}:")
        print(f"  Predicao : ${pred * 100_000:,.0f} ({pred:.3f} x $100k)")
        print(f"  IC 80%   : [${lower * 100_000:,.0f}, ${upper * 100_000:,.0f}]")
        print(f"  Alerta   : {'SIM' if alerta else 'nao'}")
        if msg:
            print(f"  Mensagem : {msg}")
