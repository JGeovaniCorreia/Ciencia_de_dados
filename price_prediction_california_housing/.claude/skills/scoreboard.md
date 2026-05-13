---
description: Display Phase 1 model competition results from artifacts/competition_metadata.json — scoreboard, winner, metrics, conformal q_hats, and training config.
---

Read `artifacts/competition_metadata.json` and render a structured summary in this exact order:

## 1. Winner
Print: `🏆 Winner: <competition_winner> (scoreboard: <all_scores[winner]>)`
Print training date and whether GPU was used.

## 2. Full Scoreboard
Print a markdown table with columns: `Model | Score | R² | RMSE | MAE | MAPE%`
Use `all_scores` for Score and `metrics_test` for the remaining columns.
Sort descending by Score.

## 3. Test Metrics (winner only)
Print a table with all keys from `metrics_test`: R2, RMSE_100k, MAE_100k, MAPE_pct, IS_80pct, PICP_80pct, MACE.
Add a note that RMSE and MAE are in $100k USD units.

## 4. Conformal Prediction Intervals
Print a table: `Confidence | q_hat | Interpretation`
For each level in `conformal_q_hats`, interpret the q_hat as: ±`round(q_hat * 100, 1)`% of the log-scale prediction.

## 5. Best Hyperparameters
Print a code block with the `best_hyperparams` dict (key: value, one per line).
Also print `best_cv_r2`.

## 6. Training Config
Print: n_train_samples, n_cal_samples, n_test_samples, split ratio (compute from n values), sklearn/optuna/python versions.

If `artifacts/competition_metadata.json` does not exist, print:
> Phase 1 has not been run yet. Execute `/run-phase1` first.
