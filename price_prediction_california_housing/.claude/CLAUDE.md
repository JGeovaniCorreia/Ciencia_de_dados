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
- [ ] 2. Entendimento dos dados
- [ ] 3. Preparação dos dados
- [ ] 4. Modelagem
- [ ] 5. Avaliação
- [ ] 6. Implantação

**Fase ativa**: Entendimento dos dados

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

---

## Destinação

Portfólio público no GitHub — projeto de vitrine para demonstrar domínio do ciclo
completo de ML com boas práticas (CRISP-DM, MLflow, Optuna, SHAP, fairness).