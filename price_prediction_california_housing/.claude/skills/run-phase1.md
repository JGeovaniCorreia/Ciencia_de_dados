---
description: Execute model_competition.ipynb end-to-end using nbconvert, then validate that all expected artifacts were created and report a pass/fail summary.
---

## Pre-flight checks (do before running)

1. Verify `.venv\Scripts\python` exists. If not, instruct user to create the venv first.
2. Check that `california_housing_optuna.db` exists and warn the user:
   > Optuna DB found with existing trials. Re-running will **accumulate** more trials (not restart).
   > To start fresh, delete `california_housing_optuna.db` first.
   Ask the user to confirm they want to proceed.

## Execute

Run (using PowerShell, with the project venv active):

```powershell
$jupyter = ".venv\Scripts\jupyter.exe"
& $jupyter nbconvert --to notebook --execute model_competition.ipynb --output model_competition.ipynb --ExecutePreprocessor.timeout=3600 2>&1
```

Stream output to the user. If the exit code is non-zero, stop and show the last 30 lines of stderr.

## Post-run validation

After successful execution, read `artifacts/competition_metadata.json` to determine the winner, then check:

| Artifact | Expected | Status |
|---|---|---|
| `artifacts/competition_winner_<winner>.joblib` | winner from metadata | ✅/❌ |
| `artifacts/competition_metadata.json` | required | ✅/❌ |
| `california_housing_optuna.db` | required | ✅/❌ |

Where `<winner>` = `competition_metadata.json["competition_winner"].lower()` (e.g. `catboost`).

From the metadata, print:
- Winner model and scoreboard score
- Test R² and MAPE
- Whether GPU was detected (`gpu_used`)

## Summary

Print a final block:
```
Phase 1 complete — <WINNER> elected (score: <SCORE>)
Artifacts saved in artifacts/
Next step: run /run-phase2 to execute the CRISP-DM notebook
```

If the elected winner differs from the model currently set in `california_housing_crisp_dm.ipynb`, warn:
> Phase 2 notebook may still reference the old winner. Review the estimator in the tuning cell before running `/run-phase2`.
