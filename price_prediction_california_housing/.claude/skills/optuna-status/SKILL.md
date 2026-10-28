---
name: optuna-status
description: Query california_housing_optuna.db and show a summary of all Optuna studies — trial counts, best values, best hyperparams, and whether more trials are worth running.
---

The Optuna SQLite database is at `california_housing_optuna.db`.
Study names follow the convention: `CalHousing_Ridge`, `CalHousing_XGBoost`, `CalHousing_LightGBM`, `CalHousing_CatBoost`, `CalHousing_TabNet`.

Run the following Python snippet using the project venv (`.venv\Scripts\python`) to query it:

```python
import optuna
import os

db_path = "california_housing_optuna.db"
storage = f"sqlite:///{db_path}"

studies = optuna.study.get_all_study_names(storage=storage)
for name in studies:
    study = optuna.load_study(study_name=name, storage=storage)
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned   = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    failed   = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
    best = study.best_trial if completed else None
    print(f"\n{'='*50}")
    print(f"Study: {name}")
    print(f"  Completed: {len(completed)} | Pruned: {len(pruned)} | Failed: {len(failed)}")
    if best:
        print(f"  Best value (CV R²): {best.value:.4f}")
        print(f"  Best params: {best.params}")
    else:
        print("  No completed trials yet.")
```

After running, display the output formatted as a markdown table with columns:
`Study | Completed | Pruned | Failed | Best CV R² | Best params (top 3)`

Then print a management tip:
> Never delete `california_housing_optuna.db` or its studies — it is the versioned trial history.
> For a new search, create a new versioned study name instead of reusing an existing one.
> For convergence analysis and tuning recommendations, use the `cientista_senior` agent.

If the database file does not exist, print:
> No Optuna database found. Run `notebooks/model_competition.ipynb` first.
