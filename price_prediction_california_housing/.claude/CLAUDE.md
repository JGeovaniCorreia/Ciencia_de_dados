# Projeto: prediction-ca-housing

---

## Objetivo de negócio

Prever o valor mediano de imóveis por bloco censitário (block group) na Califórnia,
para apoiar precificação de imóveis em propostas imobiliárias.

Um block group é a menor unidade geográfica publicada pelo censo americano,
tipicamente com 600 a 3.000 habitantes. O modelo prevê o valor mediano daquele
agrupamento — não o preço de uma casa individual.

Usuário final: corretores de imóveis e analistas do mercado imobiliário.
Decisão apoiada: precificação de imóveis em propostas e avaliações de mercado.

---

## Dataset

- **Fonte**: scikit-learn — fetch_california_housing()
- **Origem**: Censo americano de 1990, StatLib repository
- **Volume**: 20.640 amostras, 8 features numéricas + 1 target
- **Target**: MedHouseVal — valor mediano do imóvel por block group, em $100.000
- **Missing values**: nenhum (dataset limpo)
- **Atenção**: target tem efeito de teto — valores acima de $500.000 foram truncados em 5.0

### Features disponíveis

| Feature | Descrição |
|---------|-----------|
| MedInc | Renda mediana do block group |
| HouseAge | Idade mediana das casas do block group |
| AveRooms | Média de cômodos por domicílio |
| AveBedrms | Média de quartos por domicílio |
| Population | População do block group |
| AveOccup | Média de moradores por domicílio |
| Latitude | Latitude do block group |
| Longitude | Longitude do block group |

### Cuidados conhecidos do dataset

- AveRooms e AveBedrms podem ter valores muito altos em block groups com poucos
  domicílios e muitas casas vazias (ex: resorts de férias) — tratar como outliers
- Latitude e Longitude são features geográficas importantes — considerar como proxy
  de localização (bairro, proximidade ao mar, etc)
- O truncamento do target em 5.0 pode afetar a performance em imóveis de alto valor

---

## Fase CRISP-DM atual

- [x] 1. Entendimento do negócio
- [x] 2. Entendimento dos dados
- [x] 3. Preparação dos dados
- [x] 4. Modelagem
- [x] 5. Avaliação
- [x] 6. Implantação

**Fase ativa**: concluído — ciclo CRISP-DM encerrado. Veredicto final do Chief Data
Scientist: **APROVADO** (2026-09-11). Deploy real ainda não realizado — aguardando
revisão do dono do projeto para definir o momento.

**Nota (2026-05-18):** Fases 2–4 concluídas no `model_competition.ipynb`.
XGBoost venceu a competição de 5 modelos (R²=0.87, RMSE=0.433, MAE=0.270).

**Nota (2026-05-20):** Fases 5–6 concluídas.
- SHAP values: RendaMediana é o driver principal, seguido de Latitude/Longitude.
- Fairness: gap de RMSE $22.6k entre Sul Costa e Norte Interior — alerta ativo.
- MLflow: experimento registrado em `mlflow.db`, modelo `california-housing-xgboost` v1.
- Inferência: `src/predict.py` com IC conformal 80% e alerta de teto ($450k).
- Artefato XGBoost regenerado (incompatibilidade joblib entre versões detectada e corrigida).

**Nota (2026-05-24):** Chief Data Scientist avalia o ciclo: **Aprovado com ressalvas**
(5 pendências — README, diagnóstico Sul Costa, código duplicado, artefato de resíduos,
EDA formal).

**Nota (2026-05-28):** Backlog de 16 itens (revisão CRISP-DM de 2026-05-25) aplicado
por completo, cobrindo as 5 ressalvas do CDS.

**Nota (2026-09-11):** Segunda passada do Chief Data Scientist confirma as 5 ressalvas
resolvidas (evidência verificada arquivo a arquivo) → veredicto **Aprovado**. Restam
2 gaps cosméticos não-bloqueantes: comentário obsoleto em `requirements.txt` e ausência
de ablation study para colinearidade das features de distância geográfica. Detalhe
completo no banco MCP.

---

## Métricas de sucesso

### Métricas primárias (desempenho)

| Métrica | Descrição | Meta |
|---------|-----------|------|
| RMSE | Erro quadrático médio — penaliza erros grandes, mesma unidade do target | < 0.50 ($50k) |
| MAE | Erro absoluto médio — robusto a outliers, fácil de interpretar | < 0.35 ($35k) |
| R² | Variância explicada pelo modelo | > 0.80 |

### Métricas secundárias (qualidade e confiabilidade)

| Métrica | Descrição |
|---------|-----------|
| MAPE | Erro percentual médio — complementa RMSE/MAE com visão relativa |
| Adjusted R² | R² ajustado pelo número de features — para comparar modelos de complexidades diferentes |
| Análise de resíduos | Distribuição dos erros — verificar heterocedasticidade e padrões |
| Curva de calibração | Verificar se intervalos de predição são confiáveis |

### Fairness

Variável sensível identificada: localização geográfica (Latitude/Longitude).
Verificar se o modelo performa consistentemente entre regiões da Califórnia —
erros sistemáticos em determinadas regiões podem indicar viés geográfico.

### Baseline

Baseline simples: predição pela média global do target.
RMSE esperado do baseline: ~1.15 ($115k).
Qualquer modelo deve superar esse valor com folga.

---

## Restrições técnicas

- Preferência por modelos interpretáveis (ex: regressão linear, árvore de decisão,
  regressão ridge/lasso) — desde que a diferença de performance vs modelos caixa
  preta seja inferior a 10% no RMSE
- Se modelo caixa preta for necessário, SHAP values obrigatórios para explicabilidade
- Inferência deve rodar em CPU — sem dependência de GPU em produção
- GPU (RTX 2050) disponível apenas para treino

---

## Histórico de decisões arquiteturais

Registro das decisões importantes do projeto — metodologia, arquitetura e negócio.
Decisões técnicas detalhadas e aprendizados ficam no banco MCP (search_knowledge).

Regra: uma decisão entra aqui quando impacta a arquitetura ou metodologia do projeto
e deve ser visível para qualquer pessoa que visitar o repositório.

| Data | Decisão | Justificativa | Agente |
|------|---------|---------------|--------|
| 2026-05-13 | Métricas: RMSE + MAE + R² como primárias | Padrão de mercado para regressão imobiliária | negocio |
| 2026-05-13 | Fairness por região geográfica | Latitude/Longitude são proxies de localização | negocio |
| 2026-05-13 | Preferência por modelos interpretáveis | Corretores precisam entender o modelo | negocio |
| 2026-05-20 | MLflow com backend SQLite (`mlflow.db`) | MLflow 3.12+ deprecou filesystem store — SQLite é o mínimo recomendado | mlops |
| 2026-05-20 | Alerta de teto em $450k (não $500k) | IC conformal 80% com q_hat=0.133 já ultrapassa $500k para predições a partir de $450k | mlops |
| 2026-05-20 | Pipeline XGBoost salvo via joblib — versão do XGBoost deve ser fixada | Incompatibilidade entre versões corrompeu predições silenciosamente (R²=-1.58) | orquestrador |
| 2026-05-24 | Gap RMSE Sul Costa confirmado como falha de dataset, não de modelo | Sul Costa tem 15.7% de truncamento (3.3× média global) e apenas 483 amostras (2.3%) — dupla causa estrutural; `reports/truncation_analysis.py` reproduz a análise | avaliacao |
| 2026-08-05 | `mlflow.db` e `california_housing_optuna.db` versionados no git | Projeto de portfólio: clonar o repo deve reproduzir o histórico completo de experimentos MLflow e trials Optuna, sem exigir re-treino do zero | mlops |
| 2026-09-11 | Ciclo CRISP-DM formalmente encerrado — veredicto Aprovado | Segunda passada do CDS confirmou as 5 ressalvas de 2026-05-24 resolvidas pelo backlog de 2026-05-28; deploy fica pendente de avaliação própria do dono do projeto | chief_data_scientist |

---

## Destinação

Portfólio público no GitHub — projeto de vitrine para demonstrar domínio do ciclo
completo de ML com boas práticas (CRISP-DM, MLflow, Optuna, SHAP, fairness).