---
description: Execute california_housing_crisp_dm.ipynb end-to-end using nbconvert, validate artifacts, and report final model metrics.
---

## Pre-flight checks

1. Verify `artifacts/competition_metadata.json` exists. If not:
   > Phase 1 has not been run yet. Run `/run-phase1` first.
   Stop.

2. Read `artifacts/competition_metadata.json` and extract the `competition_winner`.

3. Read `california_housing_crisp_dm.ipynb` and find the cell that defines the estimator used for tuning (look for `CatBoostRegressor`, `XGBRegressor`, `LGBMRegressor`, or similar).

4. Compare the notebook's estimator with the Phase 1 winner. If they differ:
   > ⚠️ Mismatch detected: Phase 1 winner is **<WINNER>** but Phase 2 notebook uses **<NOTEBOOK_MODEL>**.
   > Update the estimator in the tuning cell before proceeding.
   Ask the user whether to continue anyway or abort.

## Execute

Run:

```powershell
$jupyter = ".venv\Scripts\jupyter.exe"
& $jupyter nbconvert --to notebook --execute california_housing_crisp_dm.ipynb --output california_housing_crisp_dm.ipynb --ExecutePreprocessor.timeout=3600 2>&1
```

Stream output to the user. If the exit code is non-zero, stop and show the last 30 lines of stderr.

## Post-run validation

Check for:

| Artifact | Status |
|---|---|
| `artifacts/california_housing_pipeline.joblib` | ✅/❌ |
| `artifacts/metadata.json` | ✅/❌ |

If `artifacts/metadata.json` exists, read it and print the final model metrics (R², RMSE, MAE, MAPE).

## Summary

Print:
```
Phase 2 complete — CRISP-DM pipeline saved
Model: <winner>
Test R²: <value> | RMSE: <value> | MAPE: <value>%
Pipeline: artifacts/california_housing_pipeline.joblib
Next: use /pipeline-inspect to verify the saved pipeline, or /validate-input to test inference.
```
