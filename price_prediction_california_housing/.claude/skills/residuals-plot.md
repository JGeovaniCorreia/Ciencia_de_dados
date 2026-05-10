---
description: Generate residual analysis plots for the competition winner — residuals vs predicted, error distribution, QQ plot, and error breakdown by California price tier (low/medium/high/capped). Saves artifacts/residuals_analysis.png. Use after Phase 1 to identify systematic model biases before proceeding to Phase 2.
---

## Usage

`/residuals-plot`

No arguments. Loads the fitted pipeline and computes residuals on a 20% hold-out. Saves a 4-panel figure.

## Execution

Run with `.venv\Scripts\python`. Target is in log-scale; residuals and business metrics are in the original $100k scale.

```python
import json, joblib, numpy as np, pandas as pd, matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.stats as stats
from pathlib import Path
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# --- Load ---
meta_path = Path('artifacts/competition_metadata.json')
if not meta_path.exists():
    raise FileNotFoundError("Run /run-phase1 first")
meta = json.loads(meta_path.read_text())
winner = meta['competition_winner'].lower()
pipeline = joblib.load(f'artifacts/competition_winner_{winner}.joblib')

data = fetch_california_housing(as_frame=True)
df = data.frame.rename(columns={
    'MedInc': 'RendaMediana', 'HouseAge': 'IdadeMediaResidencias',
    'AveRooms': 'MediaComodos', 'AveBedrms': 'MediaQuartos',
    'Population': 'Populacao', 'AveOccup': 'MediaOcupacao',
    'MedHouseVal': 'ValorMedioResidencias',
})
X = df.drop(columns=['ValorMedioResidencias'])
y = df['ValorMedioResidencias']  # in $100k units
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Predictions in $100k (reversed from log-scale)
y_pred_log = pipeline.predict(X_test)
y_pred = np.expm1(y_pred_log)
residuals = y_test.values - y_pred          # raw residual in $100k
pct_error = 100 * residuals / y_test.values # percent error

print(f"Model: {winner.upper()}")
print(f"Test samples: {len(y_test):,}")
print(f"Residuals (raw $100k): mean={residuals.mean():.4f}, std={residuals.std():.4f}")
print(f"% Error: mean={pct_error.mean():.2f}%, std={pct_error.std():.2f}%")

# Price tiers (California Housing has a hard cap at $500k / 5.0 in $100k)
price_tiers = pd.cut(
    y_test,
    bins=[0, 1.0, 2.0, 3.5, 5.0, 999],
    labels=['<$100k', '$100k–$200k', '$200k–$350k', '$350k–$500k', '>$500k (cap)']
)

print("\nMAE by price tier:")
for tier, grp in pd.Series(np.abs(residuals), index=y_test.index).groupby(price_tiers):
    print(f"  {tier}: MAE=${grp.mean()*100_000:,.0f}  n={len(grp)}")

# --- Plot ---
Path('artifacts').mkdir(exist_ok=True)
fig = plt.figure(figsize=(14, 10))
gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

# Panel 1: Residuals vs Predicted
ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(y_pred, residuals, alpha=0.3, s=3, color='steelblue')
ax1.axhline(0, color='red', linewidth=1.2, linestyle='--')
ax1.set_xlabel('Valor Previsto ($100k)')
ax1.set_ylabel('Resíduo ($100k)')
ax1.set_title('Resíduos vs Previsto')
# Annotate the $500k cap
ax1.axvline(5.0, color='orange', linewidth=1, linestyle=':', label='Cap $500k')
ax1.legend(fontsize=8)

# Panel 2: Residuals distribution
ax2 = fig.add_subplot(gs[0, 1])
ax2.hist(residuals, bins=60, color='steelblue', edgecolor='white', alpha=0.8)
ax2.axvline(0, color='red', linewidth=1.5, linestyle='--', label='Zero')
ax2.axvline(residuals.mean(), color='orange', linewidth=1.5, linestyle='--',
            label=f'Média={residuals.mean():.3f}')
ax2.set_xlabel('Resíduo ($100k)')
ax2.set_ylabel('Frequência')
ax2.set_title('Distribuição dos Resíduos')
ax2.legend(fontsize=8)

# Panel 3: QQ plot
ax3 = fig.add_subplot(gs[1, 0])
(osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist='norm', fit=True)
ax3.plot(osm, osr, 'o', alpha=0.3, markersize=2, color='steelblue')
ax3.plot(osm, slope * np.array(osm) + intercept, 'r-', linewidth=1.5)
ax3.set_xlabel('Quantis Teóricos (Normal)')
ax3.set_ylabel('Quantis Observados')
ax3.set_title(f'QQ Plot (r={r:.3f})')

# Panel 4: MAE by price tier
ax4 = fig.add_subplot(gs[1, 1])
tier_mae = pd.Series(np.abs(residuals), index=y_test.index).groupby(price_tiers).mean() * 100_000
colors = ['#2ecc71', '#f39c12', '#e67e22', '#e74c3c', '#8e44ad']
bars = ax4.bar(range(len(tier_mae)), tier_mae.values, color=colors, edgecolor='white')
ax4.set_xticks(range(len(tier_mae)))
ax4.set_xticklabels(tier_mae.index, fontsize=8, rotation=15)
ax4.set_ylabel('MAE Médio (USD)')
ax4.set_title('MAE por Faixa de Preço')
ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
for bar, val in zip(bars, tier_mae.values):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
             f'${val:,.0f}', ha='center', va='bottom', fontsize=7)

fig.suptitle(f'Análise de Resíduos — {winner.upper()} (test 20%)', fontsize=13)
plt.savefig('artifacts/residuals_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: artifacts/residuals_analysis.png")
```

## Display after running

Show the saved file path, the text summary table, and this interpretation guide:

**What to flag:**

- **Panel 1 (Residuals vs Predicted)**: a cone shape (variance grows with price) indicates heteroscedasticity — the log1p transform should mitigate this but doesn't eliminate it for luxury properties.
- **Panel 3 (QQ plot)**: heavy tails (points diverging from the line at extremes) are expected due to the $500k census cap artificially compressing high-value predictions.
- **Panel 4 (MAE by tier)**: if `>$500k (cap)` has dramatically higher error than other tiers, this is the census cap effect — not a modeling failure. Note this in the CRISP-DM documentation.
- **Systematic bias (Panel 1, mean residual far from 0)**: if `mean residual > 0.1` ($10k), the model has a positive bias — it systematically under-predicts. Flag to `data-scientist`.
