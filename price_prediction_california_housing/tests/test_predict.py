"""Unit tests for src/predict.py — uses real artifacts, no mocks."""

import pytest
import pandas as pd

from src.predict import prever

EXPECTED_KEYS = {
    "predicao_100k",
    "predicao_usd",
    "intervalo_lower",
    "intervalo_upper",
    "alerta_teto",
    "mensagem_alerta",
    "alerta_teto_ic",
    "mensagem_alerta_ic",
}


@pytest.fixture
def sample_input() -> dict:
    return {
        "MedInc": 5.0,
        "HouseAge": 30,
        "AveRooms": 5.5,
        "AveBedrms": 1.1,
        "Population": 1200,
        "AveOccup": 3.0,
        "Latitude": 34.05,
        "Longitude": -118.24,
    }


def test_prever_dict_en(sample_input):
    result = prever(sample_input)
    assert set(result.keys()) == EXPECTED_KEYS
    assert result["predicao_usd"][0] > 0


def test_prever_list_of_dicts(sample_input):
    result = prever([sample_input, sample_input])
    assert len(result["predicao_100k"]) == 2
    assert len(result["intervalo_lower"]) == 2
    assert len(result["alerta_teto"]) == 2


def test_prever_dataframe_pt():
    df = pd.DataFrame([{
        "RendaMediana": 5.0,
        "IdadeMediaResidencias": 30,
        "MediaComodos": 5.5,
        "MediaQuartos": 1.1,
        "Populacao": 1200,
        "MediaOcupacao": 3.0,
        "Latitude": 34.05,
        "Longitude": -118.24,
    }])
    result = prever(df)
    assert result["predicao_usd"][0] > 0


def test_alerta_teto_ativado():
    result = prever({
        "MedInc": 15.0,
        "HouseAge": 10,
        "AveRooms": 8.0,
        "AveBedrms": 1.0,
        "Population": 500,
        "AveOccup": 2.5,
        "Latitude": 34.02,
        "Longitude": -118.50,
    })
    assert result["alerta_teto"][0] is True


def test_alerta_teto_desativado():
    result = prever({
        "MedInc": 1.0,
        "HouseAge": 40,
        "AveRooms": 4.0,
        "AveBedrms": 1.2,
        "Population": 900,
        "AveOccup": 2.8,
        "Latitude": 40.0,
        "Longitude": -120.0,
    })
    assert result["alerta_teto"][0] is False


def test_intervalo_contem_ponto(sample_input):
    result = prever(sample_input)
    lower = result["intervalo_lower"][0]
    pred = result["predicao_100k"][0]
    upper = result["intervalo_upper"][0]
    assert lower <= pred <= upper


def test_colunas_faltando_raises():
    with pytest.raises(ValueError):
        prever({
            "HouseAge": 30,
            "AveRooms": 5.5,
            "AveBedrms": 1.1,
            "Population": 1200,
            "AveOccup": 3.0,
            "Latitude": 34.05,
            "Longitude": -118.24,
        })


def test_tipo_invalido_raises():
    with pytest.raises(TypeError):
        prever("isso nao e um input valido")


def test_nivel_confianca_invalido(sample_input):
    with pytest.raises(ValueError):
        prever(sample_input, nivel_confianca=0.75)
