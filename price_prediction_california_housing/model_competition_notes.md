# Notas Técnicas — Competição de Modelos (California Housing)

> Este arquivo documenta o raciocínio técnico por trás de cada seção de `model_competition.ipynb`.
> A estrutura de seções espelha o notebook — use os números de seção para navegar entre os dois.

---

## 1. Configuração do Ambiente

### 1.1 Imports e Dependências

O notebook é auto-suficiente: todas as classes (transformadores, wrapper TabNet) são definidas internamente, sem depender de módulos externos ao arquivo. Isso evita problemas de importação e facilita portabilidade.

**Pacotes não-padrão necessários:**

| Pacote | Papel |
|--------|-------|
| `xgboost` | Gradient Boosting level-wise |
| `lightgbm` | Gradient Boosting leaf-wise (mais rápido em datasets grandes) |
| `catboost` | Gradient Boosting com Ordered Boosting |
| `pytorch-tabnet` | Rede neural tabular baseada em Transformer |
| `torch` | Backend do TabNet — instalado com CUDA 12.8 para GPU |
| `optuna` | Otimização bayesiana de hiperparâmetros |
| `optuna-dashboard` | Dashboard interativo dos estudos |
| `joblib` | Serialização eficiente de pipelines sklearn |

A detecção de GPU é feita no momento dos imports:

```python
CUDA_AVAILABLE = torch.cuda.is_available()
DEVICE = 'cuda' if CUDA_AVAILABLE else 'cpu'
```

Todos os modelos com suporte a GPU (`XGBoost`, `CatBoost`, `TabNet`) usam `CUDA_AVAILABLE` para decidir o device em tempo de execução.

---

### 1.2 Métricas de Avaliação

#### Métricas de Ponto — Round 1

Métricas clássicas de regressão que avaliam o **valor central** previsto pelo modelo:

- **R²** — proporção da variância explicada. Interpretação: R² = 0.85 significa que o modelo explica 85% da variação dos preços.
- **RMSE** — raiz do erro quadrático médio, na mesma unidade do target ($100k USD). Penaliza erros grandes de forma quadrática.
- **MAE** — erro absoluto médio. Mais robusto a outliers que o RMSE.
- **MAPE** — erro percentual médio. Útil para comunicação com stakeholders sem background técnico.

#### Métricas de Confiabilidade — Round 2

Para um modelo de produção, não basta acertar o ponto central — os **intervalos de predição** precisam ser calibrados. Um modelo que diz "80% de confiança" deve cobrir ~80% dos pontos reais.

**Interval Score (IS)** — a métrica principal do Round 2. Análogo ao Brier Score para regressão:

```
IS(α) = MPIW + (2/α) * (sub-coverage penalty)
```

Penaliza simultaneamente:
- Intervalos **largos** (mesmo com boa cobertura)
- Pontos **fora do intervalo** (mesmo com intervalos estreitos)

Um modelo que "chuta" intervalos enormes para garantir cobertura é penalizado tanto quanto um que tem intervalos estreitos mas deixa muitos pontos de fora.

| Métrica | Fórmula simplificada | Interpretação |
|---------|---------------------|---------------|
| **PICP** | `mean(y ∈ [ŷ_low, ŷ_high])` | Deve ser ≥ nível nominal. Garantido pelo Conformal. |
| **MPIW** | `mean(ŷ_high - ŷ_low)` | Menor = intervalos mais estreitos = modelo mais preciso |
| **MACE** | `mean(|PICP_emp - PICP_nom|)` | Calibração em múltiplos níveis. Ideal = 0. |
| **IS** | Ver fórmula acima | Menor = melhor (única métrica que resume tudo) |

> **Por que MPIW não entra no scoreboard final?** O IS já captura a largura dos intervalos. Incluir MPIW separadamente seria dupla-contagem.

---

## 2. Preparação do Pipeline

### 2.1 Transformadores Customizados

O pipeline sklearn exige que todos os estágios implementem `fit(X, y)` e `transform(X)`. As classes abaixo são definidas no notebook para manter auto-suficiência.

#### WinsorizacaoTransformer

Aplica winsorização usando limites IQR calculados **apenas nos dados de treino** (método `fit`). No `transform`, qualquer valor acima/abaixo dos limites é clipado.

**Por que IQR e não percentil fixo?**  
O IQR é robusto: ele usa os quartis Q1 e Q3, que são pouco afetados por outliers. O parâmetro `k` controla a sensibilidade: `k=3.0` (padrão) é mais conservador que o `k=1.5` do boxplot clássico.

**Data leakage:** os bounds são aprendidos apenas no treino e depois aplicados ao teste — nunca o contrário. Isso é garantido pela interface `fit/transform` do sklearn.

#### CaliforniaHousingTransformer

Feature engineering específico para o dataset California Housing:

| Feature criada | Cálculo | Motivação |
|----------------|---------|-----------|
| `log1p(RendaMediana)` | `log1p(x)` | Distribuição assimétrica à direita |
| `log1p(Populacao)` | `log1p(x)` | Idem |
| `log1p(MediaOcupacao)` | `log1p(x)` | Idem |
| `razao_quartos` | `MediaQuartos / (MediaComodos + ε)` | Proporção de quartos por cômodo |
| `comodos_por_pessoa` | `MediaComodos / (MediaOcupacao + ε)` | Densidade de cômodos |
| `dist_sf` | distância euclidiana para San Francisco | Localização relativa |
| `dist_la` | distância euclidiana para Los Angeles | Localização relativa |
| `dist_sd` | distância euclidiana para San Diego | Localização relativa |

O `+ε` (1e-8) evita divisão por zero. As distâncias são euclidiana em graus de lat/lon — não é distância geodésica real, mas é suficiente para capturar o padrão espacial.

---

### 2.2 Wrapper TabNet

O `TabNetRegressor` original (biblioteca `pytorch-tabnet`) não segue a interface sklearn:
- Exige que o `fit` receba arrays numpy (não DataFrames)
- Parâmetros de treinamento (`max_epochs`, `patience`, etc.) são passados no `fit`, não no `__init__`
- Isso é incompatível com `sklearn.Pipeline`

O `TabNetRegressorWrapper` resolve isso armazenando todos os parâmetros no `__init__` e internamente:
1. Converte `X` para `float32` numpy
2. Separa uma validação interna (`val_ratio=0.15`) para o early stopping do TabNet
3. Instancia e treina o `TabNetRegressor` dentro do `fit`

**Por que TabNet em um dataset pequeno?**  
California Housing tem ~12.000 amostras e 11 features. Redes neurais geralmente precisam de muito mais dados para superar boosting em tabular. Incluir TabNet aqui tem dois propósitos:
1. **Benchmarking completo**: mostrar onde redes neurais ficam em relação a boosting
2. **Educacional**: TabNet oferece explicabilidade via máscaras de atenção por step — algo que XGBoost/LightGBM não têm nativamente

---

## 3. Dados

### 3.1 Carregamento e Divisão

**Por que 3 partições em vez de 2 (treino/teste)?**

A Conformal Prediction exige um conjunto de calibração separado para calcular os quantis de não-conformidade (`q_hat`). Se o mesmo conjunto de treino fosse usado para calibração, haveria data leakage nos intervalos — o modelo teria "visto" esses pontos, e os intervalos seriam otimisticamente estreitos.

| Conjunto | Proporção | Papel |
|----------|-----------|-------|
| `X_train` (60%) | Fit do modelo + validação cruzada no Optuna |
| `X_cal` (20%) | Calibrar `q_hat` para os intervalos conformal |
| `X_test` (20%) | Avaliação final — **intocável até a última célula** |

A divisão é `train_test_split(test_size=0.20)` → `train_test_split(test_size=0.25)`, que resulta em 60/20/20.

**Por que `log1p` no target?**  
`ValorMedioResidencias` tem distribuição assimétrica à direita (cauda longa de imóveis caros). O `log1p` a aproxima de uma normal, o que melhora o ajuste de modelos lineares (Ridge) e reduz o efeito de outliers nos modelos tree-based. As predições são revertidas com `np.expm1` na avaliação final.

---

### 3.2 Factory de Pipelines

A função `criar_pipeline(modelo)` encapsula os 4 estágios em sequência:

```
X_raw → WinsorizacaoTransformer → CaliforniaHousingTransformer → StandardScaler → Modelo → ŷ
```

**Por que StandardScaler depois do feature engineering?**  
O TabNet e o Ridge são sensíveis à escala. Para XGBoost, LightGBM e CatBoost, o scaler é irrelevante (árvores são invariantes a monotransformações), mas mantê-lo no pipeline garante que o mesmo código funcione para todos os modelos — sem precisar de pipelines diferentes por tipo de modelo.

---

## 4. Competidores

### 4.1 Descrição dos Modelos

#### Ridge Regression
- **Família:** Linear com regularização L2
- **GPU:** Não aplicável (CPU, muito rápido)
- **Papel no projeto:** Baseline linear. Estabelece o piso de desempenho — qualquer modelo mais complexo deve superá-lo para justificar sua adoção.
- **Limitação fundamental:** Não captura interações não-lineares entre features. R² esperado neste dataset: ~0.55–0.65.

#### XGBoost
- **Família:** Gradient Boosting level-wise (cresce a árvore nível por nível)
- **GPU:** `device='cuda'` quando disponível — transfere matrizes de histograma para a GPU
- **Regularização:** L1 (`reg_alpha`) + L2 (`reg_lambda`) + poda por `gamma` e `min_child_weight`
- **R² esperado neste dataset:** ~0.82–0.85 com bom tuning

#### LightGBM
- **Família:** Gradient Boosting leaf-wise (cresce pela folha de maior ganho)
- **GPU:** Requer compilação especial — não disponível no pip padrão. CPU apenas.
- **Vantagem sobre XGBoost:** Até 10× mais rápido em datasets grandes; `num_leaves` permite árvores mais expressivas
- **Risco:** Leaf-wise pode overfit mais facilmente — `min_child_samples` é crítico para regularizar
- **R² esperado:** ~0.84–0.86

#### CatBoost
- **Família:** Gradient Boosting com symmetric trees (árvores binárias balanceadas)
- **GPU:** `task_type='GPU'` — suporte mais maduro e estável que XGBoost/LightGBM
- **Diferencial — Ordered Boosting:** Técnica que reduz target leakage durante o treino ao calcular os resíduos em uma ordem aleatória, impedindo que o modelo "veja" o target da própria amostra durante o boosting.
- **Vantagem prática:** Excelente resultado out-of-the-box, menos sensível a hiperparâmetros que os concorrentes
- **R² esperado:** ~0.83–0.86

#### TabNet
- **Família:** Deep Learning — Transformer adaptado para tabular (Google Research, 2019)
- **GPU:** `device_name='cuda'` via PyTorch — GPU sempre ajuda, independente do tamanho do dataset
- **Mecanismo de atenção:** A cada "step" do TabNet, uma máscara de atenção seleciona um subconjunto de features a considerar — isso cria explicabilidade nativa por feature e por amostra
- **Expectativa realista neste dataset:** R² ~0.75–0.82 — competitivo mas provavelmente abaixo dos boosting
- **Valor do exercício:** Benchmarking + familiaridade com redes neurais tabulares

---

## 5. Round 1 — Acurácia de Ponto

### 5.1 Baseline: CV com Parâmetros Padrão

Antes de qualquer tuning, todos os modelos são avaliados com hiperparâmetros padrão em 5-Fold CV. Isso é importante por dois motivos:

1. **Referência de comparação:** permite medir o ganho real que o Optuna trouxe (baseline sem tuning vs. melhor tunado)
2. **Sanity check:** se um modelo com parâmetros padrão já supera outros modelos tunados, algo está errado no processo de tuning

---

### 5.2 Otimização de Hiperparâmetros com Optuna

#### 5.2.1 Estratégia de Busca

**Por que Optuna e não GridSearchCV ou RandomizedSearchCV?**

| Critério | GridSearchCV | RandomizedSearchCV | Optuna |
|----------|--------------|--------------------|--------|
| Estratégia | Exaustiva | Aleatória | Bayesiana (TPE) |
| Correlação entre hiperparâmetros | Ignora | Ignora | Modela explicitamente |
| Pruning de trials ruins | Não | Não | Sim (MedianPruner) |
| Escalabilidade | Exponencial no nº de params | Linear | Sublinear |

**TPE (Tree-structured Parzen Estimator):** o sampler aprende quais regiões do espaço de hiperparâmetros produzem bons resultados e foca nelas — em vez de explorar uniformemente.

**`multivariate=True`:** modela correlações entre hiperparâmetros. Por exemplo: `learning_rate` alto requer `n_estimators` alto para manter desempenho — o TPE univariado ignora essa dependência, o multivariado não.

**MedianPruner para boosting:** a cada fold da CV, o score acumulado é reportado ao Optuna. Se após o fold 1 o R² médio estiver abaixo da mediana dos trials já completos, o trial é cancelado — economizando até 4 dos 5 folds de um trial que claramente não vai ser bom.

**NopPruner para Ridge e TabNet:**
- Ridge: a CV é uma operação única (sem steps intermediários para reportar)
- TabNet: cada fold é caro (treinamento de rede neural) — implementar pruning tornaria o código mais complexo sem ganho prático significativo dado o número baixo de trials (40 vs 100)

#### 5.2.2 Persistência dos Estudos

Cada modelo tem seu próprio estudo Optuna com nome único (`CalHousing_Ridge`, `CalHousing_XGBoost`, etc.) persistido em `california_housing_optuna.db` (SQLite).

**`load_if_exists=True`:** ao rodar a célula novamente, o Optuna carrega o estudo existente e **acumula** novos trials sobre o histórico anterior. Isso permite uma exploração incremental:

```python
# Primeira execução: N_TRIALS_GB = 10 → estuda rápido para validar
# Segunda execução: N_TRIALS_GB = 50 → adiciona 50 trials ao estudo
# Terceira execução: N_TRIALS_GB = 100 → adiciona mais 100, acumulando 160 no total
```

O TPE usará todos os trials históricos para guiar os próximos — quanto mais trials acumulados, melhor a sugestão.

**Por que um `_mk_sampler()` por estudo?** Para evitar estado compartilhado entre estudos. Se o mesmo objeto sampler fosse reutilizado em 5 `create_study`, o estado interno do TPE de um estudo poderia influenciar o outro.

#### 5.2.3 Dashboard Optuna

O `optuna-dashboard` é iniciado em background via `subprocess.Popen` com `CREATE_NEW_PROCESS_GROUP` — isso desacopla o processo do kernel Jupyter, então o servidor continua rodando mesmo após reiniciar o kernel.

A função `launch_optuna_dashboard()` tenta abrir o dashboard no **VS Code Simple Browser** (aba interna, sem janela externa) via `code --open-url`. Se falhar, abre no navegador padrão do sistema. O servidor roda em `http://localhost:8080`.

---

### 5.3 Resultados do Round 1

#### 5.3.1 Métricas de Cross-Validation

As métricas do Round 1 são calculadas em 5-Fold CV sobre `X_train`. Importante: **CV não é avaliação no conjunto de teste** — é uma estimativa da capacidade de generalização do modelo.

#### 5.3.2 Diagnóstico de Overfitting

Compara métricas calculadas em `X_train` (após retreinamento full) vs. `X_test`. O **Gap R²** (Teste − Treino) deve ser próximo de zero:

| Gap R² | Semáforo | Interpretação |
|--------|----------|---------------|
| > −2 % | Verde | Generalização saudável |
| −2 % a −5 % | Laranja | Overfitting leve — monitorar |
| < −5 % | Vermelho | Overfitting — revisar regularização |

Um gap negativo grande indica que o modelo memorizou os dados de treino sem generalizar. Modelos tree-based com profundidade alta (XGBoost, CatBoost) são candidatos mais frequentes a overfitting do que Ridge ou LightGBM com `min_child_samples` alto.

---

## 6. Round 2 — Confiabilidade das Predições

### 6.1 Split Conformal Prediction

A **Conformal Prediction** é uma framework de cobertura garantida: dado um nível nominal de confiança (80%), o método garante que pelo menos 80% dos intervalos conterão o valor real — sem assumir distribuição dos erros.

**Metodologia Split Conformal:**

1. Treinar o pipeline em `X_train`
2. Computar scores de não-conformidade em `X_cal`: `s_i = |y_i - ŷ_i|`
3. Calcular o quantil `q_hat = quantile(s, (1−α)(1 + 1/n_cal))`
4. Para novos pontos: `intervalo = [ŷ - q_hat, ŷ + q_hat]`

O `(1 + 1/n_cal)` é uma correção finita que garante cobertura marginal exata (não apenas assintótica).

**Limitação:** os intervalos têm largura constante (simétrica em torno de ŷ). Modelos mais precisos geram `q_hat` menor → intervalos mais estreitos → vantagem no IS e MPIW.

### 6.2 a 6.5 — Avaliações

Após calcular os intervalos, o Round 2 avalia:

- **6.2** — IS, PICP, MPIW, MACE para cada modelo a 80% de confiança
- **6.3** — Curvas de calibração (10% a 90%): um modelo bem calibrado segue a diagonal y=x. Desvios sistemáticos acima indicam super-cobertura (intervalos largos demais); abaixo indicam sub-cobertura.
- **6.4** — Visualização dos intervalos sobre 100 amostras ordenadas por valor real. Útil para ver se intervalos são mais largos onde os preços são mais altos (heteroscedasticidade).
- **6.5** — Painel resumo das 4 métricas para os 5 modelos.

---

## 7. Scoreboard Final

### 7.1 Ranking Ponderado

O scoreboard normaliza cada métrica para [0, 1] e calcula uma nota ponderada:

| Dimensão | Métrica | Peso | Direção |
|----------|---------|------|---------|
| Ponto | R² | 30 % | Maior = melhor |
| Ponto | RMSE | 15 % | Menor = melhor |
| Ponto | MAE | 10 % | Menor = melhor |
| Ponto | MAPE | 5 % | Menor = melhor |
| **Subtotal Round 1** | | **60 %** | |
| Confiabilidade | Interval Score | 20 % | Menor = melhor |
| Confiabilidade | PICP Error | 10 % | Menor = melhor |
| Confiabilidade | MACE | 10 % | Menor = melhor |
| **Subtotal Round 2** | | **40 %** | |

**Decisão de design — 60/40 Round1/Round2:** reflete que acurácia de ponto ainda é a métrica primária em regressão, mas confiabilidade tem peso relevante para uso em produção (onde o usuário precisa saber o intervalo de incerteza, não só o ponto).

**Por que PICP não entra diretamente?** O Conformal garante PICP ≈ nominal para todos os modelos — seria uma métrica sem discriminação. Entra apenas via "PICP Error" = distância do PICP ao nível nominal, que penaliza quem fica muito acima (intervalos super-conservadores).

---

## 8. Experimento Final — Modelo Vencedor

### 8.1 Protocolo

Após a eleição do vencedor pelo scoreboard:

1. **Retreinar em `X_trainval`** (treino + calibração = 80% dos dados) — maximiza os dados de ajuste
2. **Avaliar em `X_test`** — avaliação final única, nunca usada antes
3. **Construir intervalos conformal** — usando `X_cal` para calibração (mantido separado)
4. **Serializar** — pipeline + metadados + `q_hats`

**Por que não retreinar em 100% dos dados para produção?**  
Para manter `X_test` como estimativa não-enviesada do desempenho em produção. Retreinar em 100% não tem avaliação independente — o desempenho reportado seria o do treino.

### 8.4 Serialização e Artefatos

Dois arquivos são gerados em `artifacts/`:

**`competition_winner_<modelo>.joblib`** — pipeline sklearn completo com todos os estágios treinados. Basta chamar `.predict(X)` para obter previsões sem re-aplicar transformações.

**`competition_metadata.json`** — contém:
- Qual modelo venceu e todos os concorrentes avaliados
- Configuração de tuning (n_trials, tipo de sampler, GPU)
- Versões de bibliotecas e data de treinamento
- Features de entrada e features engenhadas
- Métricas no conjunto de teste
- **`conformal_q_hats`** para 80/90/95% — valores críticos para geração de intervalos na inferência sem re-rodar calibração

O `q_hat` é o elemento mais importante do metadata: sem ele, a função de inferência não consegue gerar intervalos calibrados sem acesso ao conjunto de calibração.

---

## 9. Inferência

### 9.1 Função de Inferência

A função `carregar_e_prever()` implementa um pipeline de inferência completo:

1. Carrega o pipeline do `.joblib`
2. Lê os metadados do `.json` (para obter `q_hat`, features esperadas, transformação do target)
3. Valida que as features fornecidas correspondem ao schema de treinamento
4. Aplica o pipeline (inclui todas as transformações automaticamente)
5. Reverte o `log1p` com `np.expm1`
6. Adiciona intervalos conformal usando o `q_hat` salvo

**Stateless:** a inferência não depende de dados de treino/calibração — só do `.joblib` e do `.json`.

---

## 10. Conclusão

### Resultados Esperados

| Métrica | Ridge | XGBoost | LightGBM | CatBoost | TabNet |
|---------|-------|---------|----------|----------|--------|
| R² (CV) | ~0.60 | ~0.83 | ~0.85 | ~0.84 | ~0.78 |
| MAPE (%) | ~30 % | ~18 % | ~17 % | ~18 % | ~22 % |
| IS (80 %) | alto | médio | baixo | baixo-médio | médio |
| PICP (80 %) | ≈ 0.80 | ≈ 0.80 | ≈ 0.80 | ≈ 0.80 | ≈ 0.80 |

> **PICP garantido pelo Conformal:** todos os modelos têm PICP ≈ nível nominal. O diferencial no Round 2 é a **largura** (MPIW) — modelos mais precisos geram `q_hat` menor → intervalos mais estreitos → melhor IS.

### GPU: quando vale a pena por modelo e tamanho de dataset

| Dataset | XGBoost GPU | LightGBM GPU | CatBoost GPU | TabNet GPU |
|---------|-------------|--------------|--------------|------------|
| ~12 k (este) | CPU ≈ GPU | CPU apenas | CPU ≈ GPU | **GPU melhor** |
| ~100 k | GPU ligeiramente melhor | GPU ligeiramente melhor | **GPU melhor** | **GPU melhor** |
| ~1 M+ | **GPU muito melhor** | **GPU muito melhor** | **GPU muito melhor** | **GPU melhor** |

Para boosting em tabular pequeno, a transferência de dados CPU→GPU e o overhead de kernel CUDA supera o ganho computacional. GPU compensa para boosting a partir de ~100 k amostras. Redes neurais (TabNet) se beneficiam de GPU independente do tamanho.

### LightGBM sem GPU

LightGBM no pip padrão é compilado sem suporte a GPU. Ativar GPU requereria compilar a partir do código-fonte com `-DUSE_GPU=1` ou usar o pacote `lightgbm-gpu` (menos estável). Para este projeto, a diferença de velocidade em ~12 k amostras é negligenciável.

---

*Notebook: `model_competition.ipynb` | Fase 1 — Descoberta de Modelos | 5 competidores | Conformal Prediction + Optuna*
