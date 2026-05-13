---
description: Compute and plot feature importance for the fitted competition pipeline — native model importance (CatBoost/XGBoost/LightGBM) plus permutation importance on the test set. Shows all 13 features (8 original + 5 engineered) with Portuguese names. Saves chart to artifacts/feature_importance.png.
---

## Usage

`/feature-importance`

No arguments. Loads `artifacts/competition_winner_<model>.joblib`, computes importances, saves PNG and prints a ranked table.

## Feature names (13 total after CaliforniaHousingTransformer)

Original 8 (Portuguese): `RendaMediana`, `IdadeMediaResidencias`, `MediaComodos`, `MediaQuartos`, `Populacao`, `MediaOcupacao`, `Latitude`, `Longitude`

Engineered 5: `razao_quartos`, `comodos_por_pessoa`, `dist_sf`, `dist_la`, `dist_sd`

## Execution

Run with `.venv\Scripts\python`.

```python
import json, joblib, numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

# --- Load ---
meta_path = Path('artifacts/competition_metadata.json')
if not meta_path.exists():
    raise FileNotFoundError("Run /run-phase1 first — artifacts/competition_metadata.json missing")
winner = json.loads(meta_path.read_text())['competition_winner'].lower()
pipeline = joblib.load(f'artifacts/competition_winner_{winner}.joblib')

data = fetch_california_housing(as_frame=True)
df = data.frame.rename(columns={
    'MedInc': 'RendaMediana', 'HouseAge': 'IdadeMediaResidencias',
    'AveRooms': 'MediaComodos', 'AveBedrms': 'MediaQuartos',
    'Population': 'Populacao', 'AveOccup': 'MediaOcupacao',
    'MedHouseVal': 'ValorMedioResidencias',
})
X = df.drop(columns=['ValorMedioResidencias'])
y_log = np.log1p(df['ValorMedioResidencias'])
X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.2, random_state=42)

# Feature names after CaliforniaHousingTransformer (same order as transformer output)
feature_names = [
    'RendaMediana', 'IdadeMediaResidencias', 'MediaComodos', 'MediaQuartos',
    'Populacao', 'MediaOcupacao', 'Latitude', 'Longitude',
    'razao_quartos', 'comodos_por_pessoa', 'dist_sf', 'dist_la', 'dist_sd'
]
engineered = {'razao_quartos', 'comodos_por_pessoa', 'dist_sf', 'dist_la', 'dist_sd'}

# --- Native importance (tree models) ---
estimator = pipeline.steps[-1][1]
native_imp = None
model_name = winner.upper()

if hasattr(estimator, 'feature_importances_'):
    native_imp = estimator.feature_importances_
    print(f"\nNative importance from {model_name}:")
    for name, imp in sorted(zip(feature_names, native_imp), key=lambda x: -x[1]):
        tag = " [eng]" if name in engineered else ""
        print(f"  {name:<28}{tag}: {imp:.4f}")
else:
    print(f"\n{model_name} does not expose feature_importances_ — skipping native importance")

# --- Permutation importance ---
print("\nComputing permutation importance on test set (n_repeats=10)...")
result = permutation_importance(pipeline, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
perm_mean = result.importances_mean

print(f"\nPermutation importance (mean R² drop when feature is shuffled):")
ranked = sorted(zip(feature_names, perm_mean), key=lambda x: -x[1])
for name, imp in ranked:
    tag = " [eng]" if name in engineered else ""
    bar = '█' * max(0, int(imp * 200))
    print(f"  {name:<28}{tag}: {imp:+.4f}  {bar}")

# --- Plot ---
Path('artifacts').mkdir(exist_ok=True)
fig, axes = plt.subplots(1, 2 if native_imp is not None else 1,
                          figsize=(16 if native_imp is not None else 8, 7))

if native_imp is None:
    axes = [axes]

# Permutation importance chart
ax = axes[-1]
sorted_idx = np.argsort(perm_mean)
colors = ['#e67e22' if feature_names[i] in engineered else '#3498db' for i in sorted_idx]
ax.barh(range(len(feature_names)), perm_mean[sorted_idx], color=colors)
ax.set_yticks(range(len(feature_names)))
ax.set_yticklabels([feature_names[i] for i in sorted_idx], fontsize=9)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_xlabel('Importância por Permutação (queda no R²)')
ax.set_title(f'Importância por Permutação — {model_name}')

from matplotlib.patches import Patch
legend = [Patch(color='#3498db', label='Original'), Patch(color='#e67e22', label='Engenheirada')]
ax.legend(handles=legend, loc='lower right', fontsize=8)

# Native importance chart (if available)
if native_imp is not None:
    ax2 = axes[0]
    sorted_idx2 = np.argsort(native_imp)
    colors2 = ['#e67e22' if feature_names[i] in engineered else '#3498db' for i in sorted_idx2]
    ax2.barh(range(len(feature_names)), native_imp[sorted_idx2], color=colors2)
    ax2.set_yticks(range(len(feature_names)))
    ax2.set_yticklabels([feature_names[i] for i in sorted_idx2], fontsize=9)
    ax2.set_xlabel('Importância Nativa (gain)')
    ax2.set_title(f'Importância Nativa — {model_name}')
    ax2.legend(handles=legend, loc='lower right', fontsize=8)

plt.suptitle('Importância das Features — California Housing Pipeline', fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig('artifacts/feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: artifacts/feature_importance.png")
```

## Display after running

Show the saved file path and the ranked permutation table. Then add this interpretation guide:

**What to look for:**

- **`RendaMediana` dominates**: expected — income is the strongest predictor of housing price. If it doesn't lead, investigate the transformer.
- **Engineered geographic features (`dist_sf`, `dist_la`, `dist_sd`) rank high**: validates the feature engineering decision. If all three rank low, reconsider the Euclidean-in-degrees approach or add interaction terms with income.
- **`razao_quartos` or `comodos_por_pessoa` has negative permutation importance**: the feature is hurting the model — flag to `data-scientist` for removal consideration.
- **Native vs permutation mismatch**: if a feature ranks top in native importance but near-zero in permutation, it may be a splitting artifact rather than a true predictor.
