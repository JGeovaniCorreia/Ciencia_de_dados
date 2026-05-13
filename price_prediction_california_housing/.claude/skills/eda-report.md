---
description: Generate a diagnostic EDA report on the California Housing training dataset (loaded from sklearn) — distributions, correlations with target, outliers, skewness, and log1p justification. Run this BEFORE modeling decisions or feature engineering changes. For external/new data files use /profile-data instead.
---

## Usage

`/eda-report`

Loads the California Housing dataset, applies the Portuguese rename mapping, and produces a structured EDA report. No file path needed — data comes from sklearn.

## Execution

Run with `.venv\Scripts\python`. The rename mapping below matches the canonical schema in **CLAUDE.md > Data Schema** — update both if column names ever change.

```python
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing

data = fetch_california_housing(as_frame=True)
df = data.frame.rename(columns={
    'MedInc': 'RendaMediana', 'HouseAge': 'IdadeMediaResidencias',
    'AveRooms': 'MediaComodos', 'AveBedrms': 'MediaQuartos',
    'Population': 'Populacao', 'AveOccup': 'MediaOcupacao',
    'MedHouseVal': 'ValorMedioResidencias'
})

print(f"Shape: {df.shape}")
print("\n--- Basic Stats ---")
print(df.describe().round(2).to_string())

print("\n--- Missing Values ---")
print(df.isnull().sum())

print("\n--- Skewness ---")
print(df.skew().round(3))

print("\n--- Outliers (IQR k=1.5) ---")
for col in df.columns:
    q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    iqr = q3 - q1
    n_out = ((df[col] < q1 - 1.5*iqr) | (df[col] > q3 + 1.5*iqr)).sum()
    pct = 100 * n_out / len(df)
    print(f"  {col}: {n_out} outliers ({pct:.1f}%)")

print("\n--- Correlation with target ---")
corr = df.corr()['ValorMedioResidencias'].drop('ValorMedioResidencias').sort_values(ascending=False)
print(corr.round(3))

print("\n--- Target distribution ---")
target = df['ValorMedioResidencias']
print(f"  min={target.min():.2f}  max={target.max():.2f}  mean={target.mean():.2f}  median={target.median():.2f}")
print(f"  skewness={target.skew():.3f}")
log_target = np.log1p(target)
print(f"  log1p skewness={log_target.skew():.3f}  (target: close to 0)")
```

## Report sections to display

**1. Dataset Overview** — shape, feature count, missing values (expected: none in California Housing)

**2. Feature Distributions** — for each of the 8 features, show: mean ± std, median, skewness, and outlier count. Flag features with |skewness| > 1.5 as "high skew".

**3. Outlier Summary** — table sorted by outlier % descending. Note: `MediaComodos`, `MediaQuartos`, `Populacao`, `MediaOcupacao` are the ones Winsorization targets (k=3.0 in the pipeline).

**4. Target Analysis**
- Raw `ValorMedioResidencias`: distribution stats + skewness
- `log1p(ValorMedioResidencias)`: skewness after transform (should be much closer to 0)
- Conclude: "log1p transformation is justified / not needed" based on skewness reduction

**5. Correlation Ranking** — table sorted by |correlation| with target. Flag top 3 predictors.

**6. Feature Engineering Hints** — based on correlations:
- If Latitude/Longitude correlate > 0.4 with target: geographic features are valuable ✅
- If RendaMediana correlates > 0.6: income dominates ✅ (expected)
- If any feature has near-zero correlation (<0.05): flag as potentially droppable
