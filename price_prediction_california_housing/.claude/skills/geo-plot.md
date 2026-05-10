---
description: Generate geographic scatter plots for the California Housing dataset — price distribution map and model prediction error map by Latitude/Longitude. Saves PNG files to artifacts/. Use after Phase 1 to visualize geographic price patterns and where the model underperforms.
---

## Usage

`/geo-plot`

No arguments needed. Loads the dataset and the fitted pipeline from `artifacts/`. Saves two PNGs:
- `artifacts/geo_price_map.png` — all data points colored by actual home value
- `artifacts/geo_error_map.png` — test split colored by absolute prediction error

## Execution

Run with `.venv\Scripts\python`. The rename mapping and column names match **CLAUDE.md > Data Schema**.

```python
import json, joblib, numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# --- Load data ---
data = fetch_california_housing(as_frame=True)
df = data.frame.rename(columns={
    'MedInc': 'RendaMediana', 'HouseAge': 'IdadeMediaResidencias',
    'AveRooms': 'MediaComodos', 'AveBedrms': 'MediaQuartos',
    'Population': 'Populacao', 'AveOccup': 'MediaOcupacao',
    'MedHouseVal': 'ValorMedioResidencias',
})
X = df.drop(columns=['ValorMedioResidencias'])
y = df['ValorMedioResidencias']

# --- Load pipeline ---
meta_path = Path('artifacts/competition_metadata.json')
if not meta_path.exists():
    raise FileNotFoundError("Run /run-phase1 first — artifacts/competition_metadata.json missing")
winner = json.loads(meta_path.read_text())['competition_winner'].lower()
pipeline = joblib.load(f'artifacts/competition_winner_{winner}.joblib')

# --- Test split (80/20 for visualization) ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_pred_log = pipeline.predict(X_test)
y_pred = np.expm1(y_pred_log)
abs_err = np.abs(y_pred - y_test.values)

Path('artifacts').mkdir(exist_ok=True)

# --- Plot 1: price distribution map ---
fig, ax = plt.subplots(figsize=(10, 7))
sc = ax.scatter(
    df['Longitude'], df['Latitude'],
    c=df['ValorMedioResidencias'], cmap='YlOrRd',
    s=2, alpha=0.4
)
plt.colorbar(sc, ax=ax, label='Valor Médio ($100k)')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title('California Housing — Distribuição Geográfica de Preços')
# Reference cities
cities = {'San Francisco': (-122.42, 37.77), 'Los Angeles': (-118.24, 34.05), 'San Diego': (-117.16, 32.72)}
for name, (lon, lat) in cities.items():
    ax.annotate(name, (lon, lat), fontsize=8, color='navy',
                xytext=(5, 5), textcoords='offset points')
    ax.scatter(lon, lat, color='navy', s=40, zorder=5)
plt.tight_layout()
plt.savefig('artifacts/geo_price_map.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: artifacts/geo_price_map.png")

# --- Plot 2: prediction error map ---
fig, ax = plt.subplots(figsize=(10, 7))
sc = ax.scatter(
    X_test['Longitude'], X_test['Latitude'],
    c=abs_err, cmap='RdYlGn_r',
    s=4, alpha=0.6, vmin=0, vmax=np.percentile(abs_err, 95)
)
plt.colorbar(sc, ax=ax, label='Erro Absoluto ($100k)')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title(f'Erros de Previsão por Região — {winner.upper()} (test 20%)')
for name, (lon, lat) in cities.items():
    ax.scatter(lon, lat, color='navy', s=40, zorder=5)
    ax.annotate(name, (lon, lat), fontsize=8, color='navy',
                xytext=(5, 5), textcoords='offset points')
plt.tight_layout()
plt.savefig('artifacts/geo_error_map.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: artifacts/geo_error_map.png")

# --- Text summary ---
print(f"\nModel: {winner.upper()}")
print(f"Test samples: {len(X_test):,}")
print(f"\nMean absolute error by latitude band:")
lat_bins = pd.cut(X_test['Latitude'], bins=[32, 34.5, 36.5, 38.5, 42], labels=['Sul', 'Centro-Sul', 'Centro-Norte', 'Norte'])
err_series = pd.Series(abs_err, index=X_test.index)
for region, mean_err in err_series.groupby(lat_bins).mean().items():
    print(f"  {region}: ${mean_err*100_000:,.0f} avg error")
```

## Display after running

Show the two saved file paths and the text summary table. Then add this analysis:

**Geographic patterns to look for:**

- **High errors in San Francisco Bay Area** (Lat 37–38, Lon -122 to -121): Urban density and extreme prices create high variance — errors expected.
- **High errors in Coastal LA** (Lat 33–34): Tourism/luxury properties above the $500k census cap.
- **Low errors in Central Valley** (Lat 36–37, Lon -119 to -121): More homogeneous pricing, model performs well.

If error map shows systematic patches of high error (not random scatter), flag to `data-scientist` agent — this suggests a missing geographic feature or interaction term.
