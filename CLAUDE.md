# Contexto Data Science — compartilhado entre projetos

Aplicado em todos os subprojetos dentro de Ciencia_de_dados-1\.
Geovani é cientista de dados focado em ML e IA generativa.

---

## Ambiente e hardware

- **SO**: Windows nativo, sem Docker
- **GPU**: NVIDIA GeForce RTX 2050 — 4GB VRAM, CUDA 12.8
- **Regra de GPU**: sempre verificar e usar GPU quando disponível para treino.
  Antes de qualquer treino com PyTorch ou XGBoost, verificar:
  - PyTorch: torch.cuda.is_available()
  - XGBoost/LightGBM: device="cuda"
- **Limitação de VRAM**: 4GB — evitar modelos com mais de 100M parâmetros.
  Para IA Gen, preferir modelos quantizados ou APIs externas

---

## Stack padrão

### Linguagem
- Python 3.11+

### ML supervisionado
- **Regressão**: LinearRegression, Ridge, Lasso, ElasticNet (scikit-learn)
- **Boosting**: XGBoost, LightGBM
- **Outros**: scikit-learn (SVM, Random Forest, Decision Tree, KNN)

### ML não supervisionado
- **Clustering**: KMeans, DBSCAN, AgglomerativeClustering (scikit-learn)
- **Redução de dimensionalidade**: PCA, UMAP, t-SNE
- **Detecção de anomalia**: IsolationForest, LocalOutlierFactor

### Semi-supervisionado
- LabelPropagation, LabelSpreading (scikit-learn)

### Aprendizado por reforço
- Stable-Baselines3 + Gymnasium
- Rodar em CPU por padrão — GPU só se o ambiente justificar

### Redes neurais
- **Framework**: PyTorch (CUDA 12.8)
- **Arquitetura padrão para tabular**: MLP com no máximo 3-4 camadas
- Sempre mover modelo e dados para GPU antes de treinar:
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### IA Generativa
- LangChain, LlamaIndex, Anthropic SDK
- Modelos locais: preferir quantizados (GGUF, 4bit) dado limite de 4GB VRAM
- Modelos grandes: usar APIs externas (Anthropic, OpenAI)

### Dados e infra
- pandas, polars, DuckDB
- MLflow (tracking + model registry)
- Hydra + OmegaConf (configurações)
- Optuna + SQLite (otimização de hiperparâmetros)

---

## Padrões de projeto ML

- Framework de processo: CRISP-DM (sempre seguir as fases)
- Tracking de experimentos: MLflow em todo treino
- Estrutura obrigatória de pastas: data/ models/ src/ configs/ notebooks/ reports/
- Notebooks: apenas para exploração, nunca para código de produção
- Todo código de treino deve ser reproduzível (seed fixo, logging de params)

---

## Padrões de modelagem

### Validação e métricas
- Sempre stratified k-fold para classificação
- Nunca reportar só accuracy — sempre F1, AUC, PR curve quando relevante
- Para regressão: RMSE, MAE e R² juntos
- Sempre verificar consistência entre treino, validação e teste

### Feature engineering
- Documentar cada feature criada e sua justificativa
- Sempre começar com baseline simples antes de modelos complexos
- Registrar TODOS os experimentos no MLflow, não só os bons

### Otimização de hiperparâmetros
- Sempre usar Optuna para busca de hiperparâmetros
- Persistir trials em banco SQLite separado do MCP:
  study = optuna.create_study(storage="sqlite:///configs/optuna_trials.db")
- Nunca deletar o banco de trials — permite retomar de onde parou
- Nomear estudos com versão: "modelo_v1", "modelo_v2"

### Calibração e confiabilidade
- Sempre verificar calibração de modelos probabilísticos (curva de calibração)
- Usar CalibratedClassifierCV quando necessário (isotonic ou sigmoid)
- Reportar intervalos de confiança nas métricas principais

### Explicabilidade
- Sempre gerar SHAP values para modelos em produção
- Para modelos tabulares: shap.TreeExplainer (XGBoost, LightGBM, RF)
- Para redes neurais: shap.DeepExplainer ou LIME como alternativa
- Documentar as top-5 features mais importantes em reports/

### Justiça do modelo (Fairness)
- Verificar performance segmentada por grupos sensíveis quando aplicável
- Variáveis sensíveis comuns: gênero, raça, idade, região geográfica
- Reportar métricas de fairness: equalized odds, demographic parity
- Usar biblioteca Fairlearn quando análise de fairness for necessária

---

## Agentes disponíveis

Este projeto usa um sistema de agentes especializados em .claude/agents/:

| Agente | Papel | Quando atua |
|--------|-------|-------------|
| orquestrador | Gerencia o fluxo e decide quais agentes acionar | O tempo todo |
| negocio | Valida o problema e as métricas de sucesso | Início e avaliação |
| dados | EDA, qualidade de dados, feature engineering | Fase de dados |
| modelagem | Algoritmos, experimentos, validação | Fase de modelagem |
| avaliacao | Avaliação crítica técnica do modelo | Pré-deploy |
| mlops | Pipeline, MLflow, monitoramento em produção | Implantação |
| mediador | Decide em debates entre agentes | Quando há conflito |
| chief_data_scientist | Avalia o projeto completo ao final | Última etapa |

O chief_data_scientist é o único agente que não participa do processo —
ele só é acionado no final, pelo orquestrador, para avaliar o projeto inteiro.
Veredicto em três saídas: Aprovado, Aprovado com ressalvas ou Reprovado.

---

## Memória e contexto

- Ao tomar decisões importantes, salve no banco com save_knowledge
- Ao iniciar uma tarefa nova, consulte o banco com search_knowledge
- Ao finalizar uma fase CRISP-DM, registre os aprendizados
- Banco de trials do Optuna (optuna_trials.db) é separado do banco MCP
