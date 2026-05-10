---
description: Generate a full model evaluation report from artifacts/competition_metadata.json — interprets metrics in business terms, assesses overfitting, evaluates conformal calibration quality, and gives a go/no-go for Phase 2.
---

## Usage

`/model-report`

Reads `artifacts/competition_metadata.json` and produces a structured, business-readable evaluation report for the winning model.

## Report sections

### 1. Model Identity
```
Model:          <competition_winner>
Training date:  <training_date>
Data split:     <n_train> train / <n_cal> calibration / <n_test> test
                (~60% / ~20% / ~20%)
GPU used:       <gpu_used> (<device>)
```

### 2. Business Interpretation of Accuracy
> Raw metric values are available in full via `/scoreboard`. Here, interpret them in business terms.

Read `metrics_test` and write 3–4 sentences in plain language:
- From R²: "The model explains X% of the variance in California housing prices."
- From RMSE: "On average, predictions deviate by ±$X (RMSE_100k × 100,000)."
- From MAE: "Half of predictions are within $X of the true value."
- From MAPE: "Relative error is X% — the model tends to be off by roughly 1 in Y dollars."

Then apply a single verdict label:
- R² ≥ 0.90 → **Excellent** — strong predictive power, production-ready
- R² 0.80–0.89 → **Good** — suitable for production use
- R² 0.70–0.79 → **Fair** — acceptable for low-stakes decisions
- R² < 0.70 → **Poor** — revisit feature engineering or model choice

### 3. Overfitting Assessment
Compute gap: `train_R2 - test_R2` (use `best_cv_r2` as proxy for train performance).

| Gap | Status |
|---|---|
| < 0.02 | 🟢 No overfitting |
| 0.02–0.05 | 🟡 Mild overfitting — monitor |
| > 0.05 | 🔴 Significant overfitting — consider regularization |

Print: `CV R² = <best_cv_r2> vs Test R² = <R2> → gap = <gap>`

### 4. Conformal Prediction Quality
Read `conformal_q_hats` and `metrics_test`:

| Coverage | q_hat | PICP (actual) | Calibration |
|---|---|---|---|
| 80% | X.XXX | X.XXXX | 🟢/🟡/🔴 |

Calibration verdict:
- If |PICP_80 − 0.80| ≤ 0.03: 🟢 Well-calibrated
- If |PICP_80 − 0.80| ≤ 0.07: 🟡 Acceptable
- If |PICP_80 − 0.80| > 0.07: 🔴 Poor calibration — q_hats need recalibration

Also check MACE (Mean Absolute Coverage Error): < 0.05 is good.

Compute interval width in USD: `q_hat_80 * 100_000` = ±$X,XXX (log-scale approximation).

### 5. All Competitors Scoreboard
> Use `/scoreboard` to see the full competitor table. Here, print only a one-line reference:
> `CatBoost 0.9526 > XGBoost 0.9019 > … (full table via /scoreboard)`

### 6. Phase 2 Readiness
Based on sections 2–4, give one of:
- ✅ **Ready for Phase 2** — metrics are solid, winner is clear, calibration is good
- ⚠️ **Proceed with caution** — list specific concerns
- ❌ **Revisit Phase 1** — state what needs to be fixed

### 7. Key hyperparameters
Print `best_hyperparams` in a table with a one-line note on what each controls:
- `iterations` — number of boosting rounds (more = potentially better, slower)
- `learning_rate` — step size (lower = needs more iterations)
- `depth` — tree depth (higher = more complex, overfitting risk)
- `l2_leaf_reg` — L2 regularization on leaves
- etc.
