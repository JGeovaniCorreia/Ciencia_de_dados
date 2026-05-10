---
description: Detect distribution drift between two datasets using PSI (Population Stability Index) and summary statistics. Used by the data-engineer agent to decide whether retraining or recalibration is needed.
---

## Usage

`/check-drift <reference.csv|parquet> <new.csv|parquet>`

- `reference`: baseline dataset (e.g., training data)
- `new`: incoming data to compare against the baseline

If only one path is given, compare it against the California Housing training distribution (loaded from sklearn).

## Execution

Run with `.venv\Scripts\python`. The rename mapping below matches the canonical schema in **CLAUDE.md > Data Schema** — update both if column names ever change.

```python
import sys
import numpy as np
import pandas as pd

def load(path):
    if path == "__sklearn__":
        from sklearn.datasets import fetch_california_housing
        return fetch_california_housing(as_frame=True).frame.rename(columns={
            'MedInc': 'RendaMediana', 'HouseAge': 'IdadeMediaResidencias',
            'AveRooms': 'MediaComodos', 'AveBedrms': 'MediaQuartos',
            'Population': 'Populacao', 'AveOccup': 'MediaOcupacao',
            'MedHouseVal': 'ValorMedioResidencias'
        })
    return pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)

def psi(ref, new, bins=10):
    ref = ref.dropna()
    new = new.dropna()
    breakpoints = np.percentile(ref, np.linspace(0, 100, bins + 1))
    breakpoints = np.unique(breakpoints)
    ref_counts, _ = np.histogram(ref, bins=breakpoints)
    new_counts, _ = np.histogram(new, bins=breakpoints)
    ref_pct = np.where(ref_counts == 0, 1e-6, ref_counts / len(ref))
    new_pct = np.where(new_counts == 0, 1e-6, new_counts / len(new))
    return np.sum((new_pct - ref_pct) * np.log(new_pct / ref_pct))

args = sys.argv[1:]
ref_path = args[0] if len(args) >= 1 else "__sklearn__"
new_path = args[1] if len(args) >= 2 else args[0] if len(args) == 1 else None

if new_path is None:
    print("ERROR: provide at least one file path to compare against training baseline")
    sys.exit(1)

ref_df = load(ref_path)
new_df = load(new_path)

common_cols = [c for c in ref_df.columns if c in new_df.columns and ref_df[c].dtype.kind in 'fi']

print(f"Reference: {ref_path} ({len(ref_df):,} rows)")
print(f"New data:  {new_path} ({len(new_df):,} rows)")
print(f"\n{'Column':<28} {'PSI':>7} {'Status':>12} {'Mean shift':>12}")
print("-" * 65)

results = []
for col in common_cols:
    psi_val = psi(ref_df[col], new_df[col])
    mean_shift = new_df[col].mean() - ref_df[col].mean()
    mean_shift_pct = 100 * mean_shift / (ref_df[col].mean() + 1e-9)
    if psi_val < 0.1:
        status = "✅ Stable"
    elif psi_val < 0.2:
        status = "🟡 Monitor"
    else:
        status = "🔴 DRIFT"
    results.append((col, psi_val, status, mean_shift_pct))
    print(f"{col:<28} {psi_val:>7.4f} {status:>12} {mean_shift_pct:>+11.1f}%")

drifted = [r for r in results if "DRIFT" in r[2]]
monitored = [r for r in results if "Monitor" in r[2]]
print(f"\nSummary: {len(drifted)} drifted | {len(monitored)} to monitor | {len(results)-len(drifted)-len(monitored)} stable")
```

## Report format

After running, display:

**1. Drift Summary Table** — PSI per feature with status icons

**2. PSI Interpretation guide**
| PSI | Meaning | Action |
|---|---|---|
| < 0.10 | 🟢 Stable | None needed |
| 0.10–0.20 | 🟡 Monitor | Log and watch |
| > 0.20 | 🔴 Significant drift | Investigate + possibly retrain |

**3. High-impact drift alert**
If any of these key features drift (PSI > 0.20), flag with elevated priority:
- `RendaMediana` — highest predictor importance
- `Latitude` / `Longitude` — geographic distribution shift
- `MediaOcupacao` — density changes

**4. Recommended action**
Based on which features drifted:

- Only `Populacao` or `IdadeMediaResidencias` drifted → 🟡 Monitor; model likely still valid
- `RendaMediana` or geographic features drifted → ⚠️ Recalibrate Conformal q_hats with new calibration set
- More than 3 features with PSI > 0.20 → 🔴 Full retraining recommended (run `/run-phase1`)
- Target column drifted → 🔴 Data pipeline issue — investigate source before retraining

**5. Next steps checklist** (only for detected drift)
- [ ] Investigate source of distribution change (seasonal? data quality? scope change?)
- [ ] Recollect calibration set from new distribution if only recalibrating q_hats
- [ ] If retraining: delete `california_housing_optuna.db` or use a new study name for fresh tuning
