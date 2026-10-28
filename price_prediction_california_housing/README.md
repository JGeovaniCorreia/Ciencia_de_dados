# California Housing — Previsão de Preço de Imóveis

Modelo de regressão para prever o **valor mediano de imóveis por block group** na Califórnia,
treinado sobre o dataset do Censo Americano de 1990 (StatLib / scikit-learn).

> **Contexto de uso:** apoiar corretores de imóveis e analistas de mercado na precificação de
> imóveis em propostas e avaliações. O modelo prevê o valor mediano do *block group* — a menor
> unidade geográfica publicada pelo censo (600 a 3.000 habitantes) — não o preço de uma
> casa individual.
> **Nota:** os dados são do Censo Americano de 1990 — o modelo não reflete valores de mercado
> atuais e destina-se a fins educacionais e de portfólio.

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

> ¹ CV 5-fold em X_train (60% dos dados). R² no test set: 0.8737 (gap de 2.8pp — overfitting leve, dentro do limiar aceitável).

Intervalos de confiança via **Conformal Prediction** — split 60/20/20 (treino / calibração / teste).

---

## Explicabilidade (SHAP)

Top features por importância média absoluta (SHAP TreeExplainer):

1. **RendaMediana** (MedInc) — driver principal
2. **Latitude** — localização geográfica
3. **Longitude** — localização geográfica

"Localização, localização, localização" — confirmado estatisticamente.

**Leitura por grupo.** As cinco features geográficas (`Latitude`, `Longitude`, `dist_sf`,
`dist_la`, `dist_sd`) são colineares e o SHAP divide o crédito entre elas. Somadas, respondem
por ~56% do |SHAP| médio — cerca de 3× a `RendaMediana` (~18%). Por feature isolada a renda
lidera; como grupo, a localização domina.

### Ablation das distâncias geográficas

As `dist_*` são funções determinísticas de Latitude/Longitude, então foi testado se acrescentam
sinal ou só redistribuem importância. O XGBoost aprovado foi re-treinado com os mesmos
hiperparâmetros e o mesmo split, em 4 variantes × 5 seeds, com IC 95% por bootstrap pareado
(`reports/ablation_dist_features.py`):

| Variante | Features | RMSE ($100k) | Δ RMSE vs completo [IC 95%] | R² |
|---|---|---|---|---|
| Completo (modelo aprovado) | 13 | 0.434 | — | 0.873 |
| Sem `dist_*` | 10 | 0.465 | +0.031 [+0.026, +0.036] (+7.2%) | 0.854 |
| Sem Latitude/Longitude | 11 | 0.494 | +0.060 [+0.052, +0.067] (+13.8%) | 0.835 |
| Sem nenhuma feature geográfica | 8 | 0.658 | +0.224 [+0.210, +0.238] (+51.5%) | 0.672 |

As distâncias **não são redundantes**: removê-las piora o RMSE em ~$3.1k, fora da margem de
ruído. Árvores fazem cortes alinhados aos eixos e precisariam de muitos splits em lat/lon para
aproximar uma distância radial até um polo; as `dist_*` entregam essa forma pronta. As
coordenadas cruas também carregam sinal próprio. O modelo aprovado é mantido como está.

---

## Dicionário de variáveis

Todas as variáveis do projeto estão documentadas em **[`docs/data_dictionary.md`](docs/data_dictionary.md)**:

- **Target** — `MedHouseVal` / `ValorMedioResidencias`, truncamento em $500k e `log1p`
- **8 features originais** — nome sklearn ↔ nome em português, unidade, faixa e transformação (winsorização, `log1p`)
- **5 features derivadas** — `razao_quartos`, `comodos_por_pessoa`, `dist_sf`, `dist_la`, `dist_sd` com fórmula e justificativa
- **Variáveis auxiliares** — `regiao` (fairness), resíduos e máscara de truncamento
- **Saídas de `prever()`** — predição, intervalo conformal e alertas de teto

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
├── docs/
│   └── data_dictionary.md                 # Dicionário de todas as variáveis
├── artifacts/
│   ├── competition_winner_xgboost.joblib  # Pipeline serializado
│   └── competition_metadata.json          # Hiperparâmetros, métricas e q_hats
├── notebooks/
│   ├── model_competition.ipynb            # EDA + competição de modelos (fases 2–4)
│   └── model_competition_notes.md         # Decisões de design documentadas
├── reports/
│   ├── shap_analysis.py                   # Gera plots SHAP
│   ├── fairness_geo.py                    # Gera análise de fairness geográfica
│   ├── residuals.py                       # Gera análise de resíduos
│   ├── ablation_dist_features.py          # Ablation das features de distância (+ .csv/.txt/_shap.png)
│   ├── shap_summary_plot.png
│   ├── shap_bar_plot.png
│   ├── fairness_geo.csv
│   ├── fairness_geo_map.png
│   └── residuals_*.png
├── src/
│   ├── predict.py                         # Inferência em produção (com IC conformal)
│   └── transformers.py                    # Transformadores customizados do pipeline
├── tests/
│   ├── test_predict.py                    # 10 testes unitários para src/predict.py
│   └── conftest.py                        # sys.path do projeto e relatório dos testes
├── california_housing_optuna.db           # Banco de trials Optuna (não deletar)
├── mlflow.db                              # Registro de experimentos MLflow
└── requirements.txt
```

> **Nota sobre versionamento:** `mlflow.db` e `california_housing_optuna.db` são versionados
> propositalmente. Este é um projeto de portfólio — clonar o repositório deve reproduzir o
> histórico de experimentos (MLflow) e a busca de hiperparâmetros (Optuna) sem precisar
> re-executar os treinos.

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

# Ablation das features de distância geográfica (re-treina 4 variantes × 5 seeds)
.venv\Scripts\python reports\ablation_dist_features.py

# Teste de inferência
.venv\Scripts\python src\predict.py
```

---

## Limitações conhecidas

### Dado histórico

O dataset é do Censo Americano de 1990 — reflete o mercado imobiliário da Califórnia de mais
de 35 anos atrás. Os valores nominais estão em dólares de 1990 e a estrutura socioeconômica
das regiões mudou significativamente desde então. **Não use este modelo para precificação em
mercado real atual.** Destina-se exclusivamente a fins educacionais, portfólio e demonstração
de metodologia de ML.

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

O intervalo de confiança conformal (80%) foi calibrado sobre um conjunto que inclui os
~4.8% de registros com target truncado em $500k. Para esses registros, o score de
não-conformidade é artificialmente pequeno — o modelo prevê ~$500k e o target também é
$500k, não porque acertou, mas porque o valor real foi censurado. Por isso, o IC pode
ser **ligeiramente estreito para imóveis próximos ao teto**: a cobertura de 80.55% é
válida para a distribuição do dataset, mas pode ser menor para imóveis genuinamente
acima de $500k.

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
