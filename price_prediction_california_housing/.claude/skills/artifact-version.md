---
description: Register a model artifact with a versioned name and update artifacts/REGISTRY.json. Used by the mlops-specialist agent to maintain an artifact registry with traceability from model to metrics.
---

## Usage

`/artifact-version [joblib_path]`

Default: reads `artifacts/competition_metadata.json` to find the current winner pipeline path.

## What this skill does

1. Reads `artifacts/competition_metadata.json` to extract: winner name, training date, R²
2. Creates a versioned copy following the naming convention:
   `artifacts/<YYYY-MM-DD>_<model_lower>_R<r2_nodot>.joblib`
   Example: `artifacts/2026-05-08_catboost_R087.joblib`
   where `r2_nodot = f"R{r2:.2f}".replace(".", "")`
3. Appends an entry to `artifacts/REGISTRY.json` (creates the file if absent)

## REGISTRY.json entry structure

Each entry has: `artifact` (filename), `model`, `training_date`, `metrics` (R2, RMSE_100k, MAE_100k, MAPE_pct), `best_hyperparams`, `registered_at` (ISO timestamp).

Never overwrite an existing versioned artifact — skip with a warning if the destination already exists.

Write the Python code to perform these steps using `shutil.copy2` and `json` from the standard library, then run it with `.venv\Scripts\python`.

## Display after running

Show a formatted summary:

```
ARTIFACT REGISTERED
===================
Source:    artifacts/competition_winner_catboost.joblib
Versioned: artifacts/2026-05-08_catboost_R087.joblib
Model:     CatBoost | R²=0.8654 | RMSE=$44,450 | MAPE=15.37%
Registry:  artifacts/REGISTRY.json (N total versions)
```

If `REGISTRY.json` has more than 1 entry, also show the full registry as a table:

| Version | Model | Date | R² | RMSE ($100k) | MAPE% |
|---|---|---|---|---|---|

Sorted by date descending (newest first).
