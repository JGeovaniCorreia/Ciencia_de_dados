# Dicionário de Variáveis — California Housing

Referência única de todas as variáveis do projeto: as originais do dataset, o target,
as derivadas pelo pipeline de feature engineering, as auxiliares usadas nas análises
e as saídas da API de inferência.

**Fonte de verdade no código:** `src/transformers.py` (transformações e features derivadas)
e `src/predict.py` (`RENAME_MAP`, saídas). Se este documento e o código divergirem,
o código prevalece — abra um PR corrigindo aqui.

**Dataset:** `sklearn.datasets.fetch_california_housing()` — Censo Americano de 1990
(StatLib), 20.640 block groups, 8 features numéricas, sem valores ausentes.
Um *block group* é a menor unidade geográfica publicada pelo censo (600–3.000 habitantes).
Todas as variáveis são agregados por block group — nenhuma descreve uma casa individual.

---

## 1. Target

| Nome (sklearn) | Nome (projeto) | Descrição | Unidade | Faixa | Transformação no treino |
|---|---|---|---|---|---|
| `MedHouseVal` | `ValorMedioResidencias` | Valor mediano dos imóveis do block group | $100.000 USD (1990) | 0.15 – 5.0 | `log1p` no treino; predições revertidas com `expm1` |

**Cuidados:**

- **Truncamento em 5.0:** todo block group com valor mediano acima de $500k foi registrado
  como exatamente 5.0 (992 amostras, 4.8%). O modelo não consegue prever acima desse teto.
  Ver `reports/truncation_analysis.py` e a seção *Limitações conhecidas* do README.
- **Por que `log1p`:** distribuição assimétrica à direita (cauda longa de imóveis caros).
  O log aproxima de uma normal, melhora modelos lineares e reduz o peso de outliers em
  modelos de árvore. Decisão documentada em `notebooks/model_competition_notes.md`.

---

## 2. Features originais (entrada do modelo)

O pipeline aceita os nomes em inglês (sklearn) ou em português — `src/predict.py`
renomeia via `RENAME_MAP`. Internamente, todo o pipeline usa os nomes em português.

| Nome (sklearn) | Nome (projeto) | Descrição | Unidade | Faixa observada | Winsorização (IQR k=3) | `log1p` |
|---|---|---|---|---|---|---|
| `MedInc` | `RendaMediana` | Renda mediana dos domicílios do block group | $10.000 USD/ano | 0.50 – 15.00 | — | ✔ |
| `HouseAge` | `IdadeMediaResidencias` | Idade mediana das casas do block group | anos | 1 – 52 | — | — |
| `AveRooms` | `MediaComodos` | Média de cômodos por domicílio | cômodos/domicílio | 0.85 – 141.9 | ✔ | — |
| `AveBedrms` | `MediaQuartos` | Média de quartos por domicílio | quartos/domicílio | 0.33 – 34.1 | ✔ | — |
| `Population` | `Populacao` | População total do block group | pessoas | 3 – 35.682 | ✔ | ✔ |
| `AveOccup` | `MediaOcupacao` | Média de moradores por domicílio | pessoas/domicílio | 0.69 – 1.243 | ✔ | ✔ |
| `Latitude` | `Latitude` | Latitude do centróide do block group | graus decimais | 32.54 – 41.95 | — | — |
| `Longitude` | `Longitude` | Longitude do centróide do block group | graus decimais | −124.35 – −114.31 | — | — |

**Notas sobre as transformações:**

- **Winsorização:** limites `[Q1 − 3·IQR, Q3 + 3·IQR]` aprendidos **apenas no treino**
  (`WinsorizacaoTransformer.fit`) e aplicados por *clipping* em todos os conjuntos —
  sem data leakage. `k=3.0` é mais conservador que o `k=1.5` do boxplot clássico.
  Motivação: `AveRooms`/`AveBedrms` atingem valores absurdos em block groups com
  poucos domicílios e muitas casas vazias (ex: resorts de férias); `Population` e
  `AveOccup` têm caudas extremas pelo mesmo motivo.
- **`log1p`:** aplicado em `RendaMediana`, `Populacao` e `MediaOcupacao` por assimetria
  positiva severa identificada na EDA (`notebooks/model_competition.ipynb`, seção 1.6).
- **`MedInc`** tem a maior correlação linear com o target (Pearson ≈ 0.69) e é o driver
  principal confirmado pelo SHAP (`reports/shap_bar_plot.png`).
- **`Latitude`/`Longitude`** são proxies de localização (bairro, proximidade ao mar) e a
  **variável sensível** do projeto para análise de fairness — ver seção 4.

---

## 3. Features derivadas (criadas pelo pipeline)

Geradas por `CaliforniaHousingTransformer` em `src/transformers.py`, **após** a
winsorização e o `log1p`. O modelo final recebe 13 features: as 8 originais
(transformadas) + as 5 abaixo. `ε = 1e-8` evita divisão por zero.

| Nome | Fórmula | Descrição | Justificativa |
|---|---|---|---|
| `razao_quartos` | `MediaQuartos / (MediaComodos + ε)` | Proporção de quartos em relação ao total de cômodos | Distingue o **tipo** de imóvel: razão alta indica unidades pequenas/compactas (apartamentos), razão baixa indica casas grandes com salas, escritórios etc. Captura informação que os valores absolutos não separam. |
| `comodos_por_pessoa` | `MediaComodos / (MediaOcupacao + ε)` | Cômodos disponíveis por morador | Proxy de **densidade habitacional / conforto**: mais cômodos por pessoa tende a correlacionar com renda e valor do imóvel. Note que `MediaOcupacao` já está em `log1p` neste ponto. |
| `dist_sf` | `√((Lat − 37.7749)² + (Lon + 122.4194)²)` | Distância euclidiana até San Francisco | Captura o gradiente de preço ao redor do maior polo econômico do norte. |
| `dist_la` | `√((Lat − 34.0522)² + (Lon + 118.2437)²)` | Distância euclidiana até Los Angeles | Idem para o maior polo do sul. |
| `dist_sd` | `√((Lat − 32.7157)² + (Lon + 117.1611)²)` | Distância euclidiana até San Diego | Idem para o terceiro polo, na fronteira sul. |

**Notas sobre as distâncias:**

- São euclidianas **em graus de lat/lon**, não geodésicas (km). Como a Califórnia ocupa
  uma faixa estreita de latitudes, a distorção é pequena e suficiente para o modelo capturar
  o padrão espacial — e evita dependência de bibliotecas geográficas em produção.
- As três distâncias são colineares entre si e com `Latitude`/`Longitude`. Isso é aceitável
  para XGBoost (robusto a colinearidade) mas foi apontado como gap pelo Chief Data
  Scientist: um *ablation study* ainda não foi feito.

**Ordem final das colunas que chegam ao modelo** (`CaliforniaHousingTransformer.OUTPUT_COLS`):

```
RendaMediana, IdadeMediaResidencias, MediaComodos, MediaQuartos, Populacao,
MediaOcupacao, Latitude, Longitude, razao_quartos, comodos_por_pessoa,
dist_sf, dist_la, dist_sd
```

Após o transformer, um `StandardScaler` padroniza todas as 13 colunas antes do XGBoost.

---

## 4. Variáveis auxiliares de análise (não entram no modelo)

Criadas apenas nos scripts de `reports/` para avaliação, fairness e diagnóstico.

| Nome | Onde é criada | Definição | Uso |
|---|---|---|---|
| `regiao` | `reports/fairness_geo.py`, `reports/truncation_analysis.py` | Quadrante geográfico pelo corte nas **medianas globais** de Latitude (34.26) e Longitude (−118.49): `norte_interior`, `norte_costa`, `sul_interior`, `sul_costa` | Grupo sensível para fairness — RMSE, MAE, R² e MAPE segmentados por região (`reports/fairness_geo.csv`) |
| `residuals` | `reports/residuals.py` | `y_true − y_pred` na escala original ($100k) | Diagnóstico de heterocedasticidade, normalidade (Q-Q) e erro por decil |
| máscara de truncamento | `reports/truncation_analysis.py` | `MedHouseVal >= 4.999` (`TRUNCATION_THRESHOLD`) — o limiar 4.999 captura o float `5.000010` do dataset, que `== 5.0` não pega | Concentração do teto por região — explica o gap de fairness em Sul Costa |

---

## 5. Saídas da inferência (`src/predict.prever`)

Retorno de `prever()` — dicionário com arrays alinhados às linhas da entrada.

| Chave | Tipo | Unidade | Descrição |
|---|---|---|---|
| `predicao_100k` | `np.ndarray` | $100.000 USD | Predição pontual na escala original do dataset |
| `predicao_usd` | `np.ndarray` | USD | Predição pontual em dólares (`predicao_100k × 100.000`) |
| `intervalo_lower` | `np.ndarray` | $100.000 USD | Limite inferior do intervalo conformal (`expm1(pred_log − q_hat)`) |
| `intervalo_upper` | `np.ndarray` | $100.000 USD | Limite superior do intervalo conformal |
| `alerta_teto` | `list[bool]` | — | `True` se predição ≥ $450k — zona de risco do truncamento |
| `mensagem_alerta` | `list[str]` | — | Texto do alerta de teto (string vazia se OK) |
| `alerta_teto_ic` | `list[bool]` | — | `True` se `intervalo_upper` ≥ $500k — IC ultrapassa o teto do modelo |
| `mensagem_alerta_ic` | `list[str]` | — | Texto do alerta de IC (string vazia se OK) |

**Intervalos:** Conformal Prediction (split 60/20/20). O `q_hat` de cada nível de confiança
fica em `artifacts/competition_metadata.json` (`conformal_q_hats`: 0.80, 0.90, 0.95).
Alerta em $450k e não $500k porque o IC 80% (`q_hat = 0.133`) já cruza o teto a partir desse valor.

---

## Referências

- `src/transformers.py` — `WinsorizacaoTransformer`, `CaliforniaHousingTransformer`
- `src/predict.py` — `RENAME_MAP`, `prever()`
- `artifacts/competition_metadata.json` — `input_features`, `engineered_features`, `target_transform`
- `notebooks/model_competition.ipynb` — EDA (seção 1) e decisões de feature engineering
- `notebooks/model_competition_notes.md` — justificativas de design do pipeline
- [Documentação do dataset no scikit-learn](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset)
