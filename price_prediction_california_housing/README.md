# California Housing — Previsão de Preço de Imóveis

Modelo de regressão para prever o **valor mediano de imóveis por block group** na Califórnia,
treinado sobre o dataset do Censo Americano de 1990 (StatLib / scikit-learn).

> **Contexto de uso:** apoiar corretores de imóveis e analistas de mercado na precificação de
> imóveis em propostas e avaliações. O modelo prevê o valor mediano do *block group* — a menor
> unidade geográfica publicada pelo censo (600 a 3.000 habitantes) — não o preço de uma
> casa individual.

---

## Resultados

| Métrica | Meta | Resultado |
|---------|------|-----------|
| R² | > 0.80 | **0.8737** |
| RMSE | < $50k | **$43.3k** |
| MAE | < $35k | **$27.0k** |
| MAPE | — | 14.71% |
| Cobertura IC 80% (conformal) | ≥ 80% | **80.55%** |
| MACE (calibração) | — | **0.0076** (excelente) |

**Baseline** (predição pela média global): RMSE ≈ $115k → redução de **62%** com o modelo.

---

## Modelo

**XGBoost** — vencedor de uma competição entre 5 algoritmos, todos otimizados com Optuna
(500 trials por modelo, TPESampler multivariado):

| Modelo | R² (CV) |
|--------|---------|
| **XGBoost** | **0.9017** |
| LightGBM | 0.8971 |
| CatBoost | 0.8527 |
| TabNet | 0.5846 |
| Ridge | 0.20 |

Intervalos de confiança via **Conformal Prediction** — split 60/20/20 (treino / calibração / teste).

---

## Explicabilidade (SHAP)

Top features por importância média absoluta (SHAP TreeExplainer):

1. **RendaMediana** (MedInc) — driver principal
2. **Latitude** — localização geográfica
3. **Longitude** — localização geográfica

"Localização, localização, localização" — confirmado estatisticamente.

---

## Como usar

```python
from src.predict import prever

# Predição única com nomes originais do sklearn
resultado = prever({
    "MedInc": 5.0, "HouseAge": 30, "AveRooms": 5.5,
    "AveBedrms": 1.1, "Population": 1200, "AveOccup": 3.0,
    "Latitude": 37.77, "Longitude": -122.42,
})

print(f"Previsão : ${resultado['predicao_usd'][0]:,.0f}")
print(f"IC 80%   : ${resultado['intervalo_lower'][0] * 100_000:,.0f} – "
      f"${resultado['intervalo_upper'][0] * 100_000:,.0f}")

if resultado['alerta_teto'][0]:
    print(resultado['mensagem_alerta'][0])
```

Também aceita `list-of-dicts` (batch) e `pd.DataFrame` com nomes em português ou inglês.

Para testar rapidamente:

```bash
.venv\Scripts\python src\predict.py
```

---

## Setup

```bash
# 1. Criar ambiente virtual
python -m venv .venv

# 2. Instalar dependências (workaround SSL necessário neste ambiente)
.venv\Scripts\pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt

# 3. PyTorch com CUDA 12.8 (necessário apenas para retreinar — inferência roda em CPU)
#    Ver comentário no requirements.txt para a URL específica
```

**Versões críticas fixadas:** `xgboost==2.1.4`, `scikit-learn==1.7.2`, `optuna==4.8.0`.
Não altere `xgboost` sem regenerar o artefato — incompatibilidade entre versões produz
predições erradas silenciosamente (ver histórico de decisões).

---

## Estrutura do projeto

```
├── artifacts/
│   ├── competition_winner_xgboost.joblib  # Pipeline serializado
│   └── competition_metadata.json          # Hiperparâmetros, métricas e q_hats
├── configs/
│   └── california_housing_optuna.db       # Banco de trials Optuna (não deletar)
├── notebooks/
│   ├── model_competition.ipynb            # EDA + competição de modelos (fases 2–4)
│   └── model_competition_notes.md         # Decisões de design documentadas
├── reports/
│   ├── shap_analysis.py                   # Gera plots SHAP
│   ├── fairness_geo.py                    # Gera análise de fairness geográfica
│   ├── residuals.py                       # Gera análise de resíduos
│   ├── shap_summary_plot.png
│   ├── shap_bar_plot.png
│   ├── fairness_geo.csv
│   ├── fairness_geo_map.png
│   └── residuals_*.png
├── src/
│   ├── predict.py                         # Inferência em produção (com IC conformal)
│   └── transformers.py                    # Transformadores customizados do pipeline
├── mlflow.db                              # Registro de experimentos MLflow
└── requirements.txt
```

---

## Reprodução completa

```bash
# Competição de modelos (fases 2–4 do CRISP-DM)
jupyter nbconvert --to notebook --execute notebooks/model_competition.ipynb

# Análise SHAP
.venv\Scripts\python reports\shap_analysis.py

# Análise de fairness geográfica
.venv\Scripts\python reports\fairness_geo.py

# Análise de resíduos
.venv\Scripts\python reports\residuals.py

# Teste de inferência
.venv\Scripts\python src\predict.py
```

---

## Limitações conhecidas

### Truncamento do target

O dataset do Censo de 1990 **capeou todos os imóveis acima de $500k** em exatamente 5.0
(escala $100k). O modelo aprende esse teto como se fosse um valor real — para imóveis
genuinamente acima de $500k, a predição **sempre subestima** de forma sistemática.

O truncamento não está distribuído igualmente pelo estado. Regiões de alto valor
imobiliário concentram muito mais registros com o teto ativado:

| Região          | Amostras   | % truncados |
|---------------- |----------  |-------------|
| Norte Interior  | 492        | 0.2%        |
| Norte Costa     | 9.835      | 4.0%        |
| Sul Interior    | 9.830      | 5.3%        |
| **Sul Costa**   | **483**    | **15.7% ⚠️**|
| **Global**      | **20.640** | **4.8%**    |

Sul Costa (LA e San Diego) concentra **15.7% de registros truncados — 3.3× a média
global**. Somado à sub-representação da região (483 amostras = 2.3% do dataset), isso
explica o RMSE elevado de $59.3k: 1 em cada 6 registros tem o valor verdadeiro
desconhecido, e o modelo teve poucos exemplos para aprender os padrões locais.
Esta é uma limitação estrutural do dataset, não um defeito do modelo.

Um alerta automático é emitido para predições acima de **$450k** (onde o IC conformal
superior já extrapola $500k com q_hat=0.133). Execute `reports/truncation_analysis.py`
para reproduzir a análise de truncamento por região.

### Fairness geográfica

| Região | N (test) | RMSE | R² |
|--------|----------|------|----|
| Norte Interior | 120 | $36.7k | **0.69 ⚠️** |
| Norte Costa | 1.945 | $41.7k | 0.88 |
| Sul Interior | 1.952 | $44.2k | 0.83 |
| Sul Costa | 111 | **$59.3k ⚠️** | 0.75 |

Dois alertas ativos:
- **Sul Costa**: RMSE acima da meta de $50k — causado pela concentração de truncamento e
  sub-representação (ver seção acima).
- **Norte Interior**: R²=0.69 abaixo do threshold de 0.80 — sub-representação estrutural
  do dataset de 1990 (apenas 120 amostras no test set).

### Dado histórico
O dataset é do Censo de 1990 — não reflete o mercado imobiliário atual da Califórnia.

---

## Metodologia

Ciclo **CRISP-DM** completo (6 fases) com:

| Ferramenta | Uso |
|------------|-----|
| **MLflow** | Tracking de experimentos (backend SQLite) |
| **Optuna** | Otimização de hiperparâmetros (trials persistidos em SQLite) |
| **SHAP** | Explicabilidade (TreeExplainer) |
| **Conformal Prediction** | Intervalos de confiança calibrados (split 3-way) |
| **Fairlearn** | Análise de fairness geográfica (manual) |

---

*Portfólio público — demonstração do ciclo completo de ML com boas práticas de engenharia.*
