---
description: Generate a data quality profile for an external CSV or Parquet file — missing values, dtypes, distributions, outliers, skewness, and cardinality. Use this when validating NEW or EXTERNAL data before inference (e.g., a file from a client, a new data dump). Do NOT use for the standard California Housing training dataset — use /eda-report for that.
---

## Usage

`/profile-data <path/to/file.csv or file.parquet>`

If no path given, profiles the California Housing dataset directly from sklearn.

## Execution

Run with `.venv\Scripts\python`. The rename mapping below matches the canonical schema in **CLAUDE.md > Data Schema** — update both if column names ever change.

```python
import sys, numpy as np, pandas as pd

path = sys.argv[1] if len(sys.argv) > 1 else None

if path is None:
    from sklearn.datasets import fetch_california_housing
    df = fetch_california_housing(as_frame=True).frame.rename(columns={
        'MedInc': 'RendaMediana', 'HouseAge': 'IdadeMediaResidencias',
        'AveRooms': 'MediaComodos', 'AveBedrms': 'MediaQuartos',
        'Population': 'Populacao', 'AveOccup': 'MediaOcupacao',
        'MedHouseVal': 'ValorMedioResidencias'
    })
    print("Source: sklearn California Housing (renamed to Portuguese)")
elif path.endswith(".parquet"):
    df = pd.read_parquet(path)
else:
    df = pd.read_csv(path)

print(f"\nShape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"Memory: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")

print("\n=== MISSING VALUES ===")
miss = df.isnull().sum()
miss_pct = 100 * miss / len(df)
for col, n in miss.items():
    status = "⚠️" if n > 0 else "✅"
    print(f"  {status} {col}: {n} ({miss_pct[col]:.1f}%)")

print("\n=== DTYPES ===")
for col, dtype in df.dtypes.items():
    print(f"  {col}: {dtype}")

print("\n=== DISTRIBUTIONS (numeric) ===")
num_cols = df.select_dtypes(include='number').columns
for col in num_cols:
    s = df[col]
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    n_out = ((s < q1 - 1.5*iqr) | (s > q3 + 1.5*iqr)).sum()
    skew_flag = " ⚠️ high skew" if abs(s.skew()) > 1.5 else ""
    print(f"  {col}:")
    print(f"    mean={s.mean():.3f}  std={s.std():.3f}  min={s.min():.3f}  max={s.max():.3f}")
    print(f"    median={s.median():.3f}  skew={s.skew():.3f}{skew_flag}")
    print(f"    outliers (IQR 1.5x): {n_out} ({100*n_out/len(df):.1f}%)")

cat_cols = df.select_dtypes(exclude='number').columns
if len(cat_cols) > 0:
    print("\n=== CATEGORICAL COLUMNS ===")
    for col in cat_cols:
        n_unique = df[col].nunique()
        top3 = df[col].value_counts().head(3).to_dict()
        print(f"  {col}: {n_unique} unique | top3: {top3}")
```

## Report format

After running the script, display the output organized as:

**1. Dataset Overview** — shape, memory, source

**2. Data Quality Issues** (only if any ⚠️ found):
- Missing values by column
- Unexpected dtypes (e.g., string column where numeric expected)
- Extreme outliers (> 5% of rows outside IQR 1.5x)

**3. Distribution Summary Table**

| Column | Mean | Std | Median | Skew | Outliers% | Flag |
|---|---|---|---|---|---|---|

**4. Recommendations**
- Columns with high skew (>1.5): suggest `log1p` transform or Winsorization
- Columns with many outliers (>5%): suggest Winsorization with k=3.0
- Missing values: suggest imputation strategy
- Unexpected cardinality: suggest investigation

**5. Comparison to training baseline** (if California Housing dataset)
Compare each column's current mean/std to the expected values (from `artifacts/competition_metadata.json` if it exists). Flag columns where mean differs > 1 std — potential drift indicator.
