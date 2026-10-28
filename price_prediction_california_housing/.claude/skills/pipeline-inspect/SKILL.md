---
name: pipeline-inspect
description: Load a saved .joblib pipeline and display its stages, parameters, and a data-flow diagram. Accepts an optional path argument; defaults to the competition winner recorded in artifacts/competition_metadata.json.
---

## Usage

`/pipeline-inspect [path/to/pipeline.joblib]`

Default path (when no argument given): `artifacts/competition_winner_<winner>.joblib`, with `<winner>` read from `artifacts/competition_metadata.json`.

## Execution

Run this Python snippet with `.venv\Scripts\python`. When no argument is given, the default path is resolved dynamically from `competition_metadata.json`:

```python
import joblib, sys, json
from pathlib import Path

if len(sys.argv) > 1:
    path = sys.argv[1]
else:
    meta = Path("artifacts/competition_metadata.json")
    if meta.exists():
        winner = json.loads(meta.read_text())["competition_winner"].lower()
        path = f"artifacts/competition_winner_{winner}.joblib"
    else:
        raise SystemExit("artifacts/competition_metadata.json not found — pass the pipeline path")

pipeline = joblib.load(path)

print(f"Pipeline loaded from: {path}\n")
print("=== Stages ===")
for i, (name, step) in enumerate(pipeline.steps):
    cls = type(step).__name__
    print(f"  [{i+1}] {name}: {cls}")
    # Print key params (skip defaults and None)
    try:
        params = {k: v for k, v in step.get_params().items() if v is not None and k != "estimator"}
        for k, v in list(params.items())[:8]:
            print(f"       {k} = {v}")
    except Exception:
        pass

estimator = pipeline.steps[-1][1]
print(f"\n=== Final Estimator: {type(estimator).__name__} ===")
try:
    print(f"  n_features_in_: {estimator.n_features_in_}")
except AttributeError:
    pass
try:
    print(f"  device: {estimator.get_params().get('device')}")
except Exception:
    pass
```

## Display

After running, show:
1. A data-flow diagram using ASCII:
   ```
   X_raw (8 features)
     → WinsorizacaoTransformer     [clips MediaComodos, MediaQuartos, Populacao, MediaOcupacao]
     → CaliforniaHousingTransformer [log1p 3 cols + 5 engineered = 13 features out]
     → StandardScaler               [standardize all 13]
     → <FinalEstimator>             [predicts log1p(ValorMedioResidencias)]
   ```

2. A validation checklist:
   - [ ] All 4 stages present
   - [ ] WinsorizacaoTransformer is first
   - [ ] StandardScaler is second-to-last
   - [ ] Final estimator is a regressor (not a classifier)
   - [ ] Pipeline has been fitted (check `pipeline.steps[-1][1]` has `n_features_in_` or equivalent)
   - [ ] Estimator `device` is `cpu` or unset — inference must run on CPU (flag `cuda` as a risk)

3. Inference reminder:
   > Input must use Portuguese column names. Predictions are in log-scale — apply `np.expm1()` and multiply by 100_000 for USD.

If the file does not exist, list all `.joblib` files found under `artifacts/` and prompt the user to specify which to inspect.
