---
description: Visualize the effect of each pipeline transformation step on feature distributions — raw data → after WinsorizacaoTransformer → after CaliforniaHousingTransformer. Shows histograms for the 4 Winsorized columns and the 5 engineered features. Saves artifacts/feature_engineering_plots.png. Use when reviewing transformer behavior or preparing CRISP-DM documentation.
---

## Usage

`/feature-plots`

No arguments. Applies each transformer step from the fitted pipeline to the training data and plots before/after distributions. Saves a multi-panel PNG.

## What is shown

**Panel group 1 — Winsorization effect** (4 columns: `MediaComodos`, `MediaQuartos`, `Populacao`, `MediaOcupacao`):
- Each column: raw distribution (blue) vs after k=3.0 IQR clipping (orange)
- Vertical lines for IQR clip bounds

**Panel group 2 — Log1p transforms** (`RendaMediana`, `Populacao`, `MediaOcupacao`):
- Raw distribution vs log1p-transformed
- Skewness before and after

**Panel group 3 — Engineered features** (5 new features):
- Distribution of each engineered feature: `razao_quartos`, `comodos_por_pessoa`, `dist_sf`, `dist_la`, `dist_sd`
- With summary statistics

## Execution

Run with `.venv\Scripts\python`. Transformations are reimplementadas manualmente usando as fórmulas canônicas do CLAUDE.md — isso evita dependência dos nomes dos steps do pipeline.

```python
import json, numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

data = fetch_california_housing(as_frame=True)
df = data.frame.rename(columns={
    'MedInc': 'RendaMediana', 'HouseAge': 'IdadeMediaResidencias',
    'AveRooms': 'MediaComodos', 'AveBedrms': 'MediaQuartos',
    'Population': 'Populacao', 'AveOccup': 'MediaOcupacao',
    'MedHouseVal': 'ValorMedioResidencias',
})
X = df.drop(columns=['ValorMedioResidencias'])
X_train, _ = train_test_split(X, test_size=0.2, random_state=42)
X_raw = X_train.copy()

# Step 1: WinsorizacaoTransformer — IQR k=3.0, fit on train only
winsor_cols = ['MediaComodos', 'MediaQuartos', 'Populacao', 'MediaOcupacao']
X_win = X_raw.copy()
clip_pct = {}
for col in winsor_cols:
    q1, q3 = X_raw[col].quantile(0.25), X_raw[col].quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 3.0 * iqr, q3 + 3.0 * iqr
    X_win[col] = X_raw[col].clip(lo, hi)
    clip_pct[col] = 100 * (X_raw[col] != X_win[col]).mean()

# Step 2: CaliforniaHousingTransformer — log1p + 5 engineered features
X_eng = X_win.copy()
X_eng['RendaMediana']  = np.log1p(X_win['RendaMediana'])
X_eng['Populacao']     = np.log1p(X_win['Populacao'])
X_eng['MediaOcupacao'] = np.log1p(X_win['MediaOcupacao'])
X_eng['razao_quartos']      = X_win['MediaQuartos'] / (X_win['MediaComodos']  + 1e-8)
X_eng['comodos_por_pessoa'] = X_win['MediaComodos']  / (X_win['MediaOcupacao'] + 1e-8)
# Distância Euclidiana em graus — mesma fórmula usada no CaliforniaHousingTransformer
for city, (lat, lon) in [('sf', (37.7749, -122.4194)), ('la', (34.0522, -118.2437)), ('sd', (32.7157, -117.1611))]:
    X_eng[f'dist_{city}'] = np.sqrt((X_win['Latitude'] - lat)**2 + (X_win['Longitude'] - lon)**2)

Path('artifacts').mkdir(exist_ok=True)

# === Figure 1: Winsorization effect ===
winsor_cols = ['MediaComodos', 'MediaQuartos', 'Populacao', 'MediaOcupacao']
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()
for ax, col in zip(axes, winsor_cols):
    raw_vals = X_raw[col].values
    win_vals = X_win[col].values
    ax.hist(raw_vals, bins=60, alpha=0.55, color='steelblue', label='Original', density=True)
    ax.hist(win_vals, bins=60, alpha=0.55, color='darkorange', label='Após Winsor.', density=True)
    ax.set_title(f'{col}\nskew: {pd.Series(raw_vals).skew():.2f} → {pd.Series(win_vals).skew():.2f}')
    ax.set_xlabel('Valor')
    ax.legend(fontsize=8)
fig.suptitle('Efeito da WinsorizacaoTransformer (k=3.0)', fontsize=12)
plt.tight_layout()
plt.savefig('artifacts/feature_plots_winsorization.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: artifacts/feature_plots_winsorization.png")

# === Figure 2: Log1p transforms ===
log_cols = ['RendaMediana', 'Populacao', 'MediaOcupacao']
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, col in zip(axes, log_cols):
    raw_vals = X_win[col].values
    log_vals = np.log1p(raw_vals)
    ax.hist(raw_vals, bins=50, alpha=0.55, color='steelblue', label='Antes log1p', density=True)
    ax_twin = ax.twinx()
    ax_twin.hist(log_vals, bins=50, alpha=0.55, color='green', label='Após log1p', density=True)
    ax.set_title(f'{col}\nskew: {pd.Series(raw_vals).skew():.2f} → {pd.Series(log_vals).skew():.2f}')
    ax.set_xlabel('Valor')
    ax.legend(loc='upper right', fontsize=7)
    ax_twin.legend(loc='upper left', fontsize=7)
fig.suptitle('Efeito do log1p em CaliforniaHousingTransformer', fontsize=12)
plt.tight_layout()
plt.savefig('artifacts/feature_plots_log1p.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: artifacts/feature_plots_log1p.png")

# === Figure 3: Engineered features ===
eng_cols = ['razao_quartos', 'comodos_por_pessoa', 'dist_sf', 'dist_la', 'dist_sd']
fig, axes = plt.subplots(1, 5, figsize=(18, 4))
for ax, col in zip(axes, eng_cols):
    vals = X_eng[col].values
    ax.hist(vals, bins=50, color='#9b59b6', edgecolor='white', alpha=0.8)
    ax.set_title(f'{col}\nmean={np.mean(vals):.2f}\nstd={np.std(vals):.2f}', fontsize=9)
    ax.set_xlabel('Valor')
fig.suptitle('Distribuição das Features Engenheiradas', fontsize=12)
plt.tight_layout()
plt.savefig('artifacts/feature_plots_engineered.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: artifacts/feature_plots_engineered.png")

# Text summary
print("\nPipeline transformation summary:")
print(f"  Input: {X_raw.shape[1]} features → Output: {len(eng_cols) + 8} features")
print("\nWinsorization clipping (% of rows clipped per column):")
for col in winsor_cols:
    pct = 100 * (X_raw[col] != X_win[col]).mean()
    print(f"  {col}: {pct:.2f}% clipped")
```

## Display after running

Show the three saved file paths. Then add this interpretation guide:

**What to verify:**

- **Winsorization (Figure 1)**: the orange distribution should have visibly trimmed tails but preserve the bulk of the distribution. If > 10% of rows are clipped for any column, consider tightening k from 3.0 to 2.5 or investigate why outliers are so extreme.
- **Log1p (Figure 2)**: skewness should drop significantly (target: |skew| < 1.0 after transform). If it doesn't, the log1p may be insufficient and a Box-Cox might be worth testing.
- **Engineered features (Figure 3)**: `dist_sf`, `dist_la`, `dist_sd` should have bimodal-ish distributions (California is geographically split). `razao_quartos` and `comodos_por_pessoa` should be right-skewed — flag extreme values as potential transformer issues.

Use `/feature-importance` after this skill to confirm that the engineered features actually contribute to model performance.
