# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

```powershell
# Activate virtual environment (Windows)
.venv\Scripts\Activate.ps1

# Install dependencies (no requirements.txt — install manually as needed)
pip install xgboost lightgbm catboost pytorch-tabnet torch optuna optuna-dashboard joblib scikit-learn pandas numpy matplotlib seaborn scipy
```

## Running Notebooks

```powershell
# Run a notebook end-to-end (re-executes all cells, saves output)
jupyter nbconvert --to notebook --execute california_housing_crisp_dm.ipynb --output california_housing_crisp_dm.ipynb

# Launch Jupyter for interactive editing
jupyter notebook
```

## Two-Notebook Workflow

This project uses a two-phase workflow where notebooks are tightly coupled:

**Phase 1 — `model_competition.ipynb`**
- Competes Ridge, XGBoost, LightGBM, CatBoost, TabNet
- Uses a **60/20/20 train/calibration/test split** (not 80/20) — the calibration set is required for Split Conformal Prediction interval calibration
- Evaluates models on point accuracy (R², RMSE, MAE, MAPE) and reliability (Interval Score, PICP, MACE)
- Elects a winner via a weighted scoreboard (60% point metrics, 40% reliability metrics)
- Saves the winner pipeline as `artifacts/competition_winner_<model>.joblib` and `artifacts/competition_metadata.json`
- Optuna studies persist in `california_housing_optuna.db` (SQLite) with `load_if_exists=True` — re-running cells **accumulates** trials rather than restarting from zero

**Phase 2 — `california_housing_crisp_dm.ipynb`**
- Full CRISP-DM documentation + official modeling of the elected winner
- Defaults to XGBoost; if Phase 1 elects a different winner, swap the estimator in the tuning cell
- Uses **80/20 train/test split** (no calibration set needed here)
- Saves pipeline as `artifacts/california_housing_pipeline.joblib` and `artifacts/metadata.json`

As of the last run (2026-05-08), **CatBoost won Phase 1** (scoreboard 0.9526 vs XGBoost 0.9019).

## Data Schema

### Input features — pipeline requires Portuguese names

| Column (PT) | Column (EN/sklearn) | Type | Valid range |
|---|---|---|---|
| RendaMediana | MedInc | float64 | 0.5–15.0 |
| IdadeMediaResidencias | HouseAge | float64 | 1–52 |
| MediaComodos | AveRooms | float64 | 1.0–50.0 |
| MediaQuartos | AveBedrms | float64 | 0.5–10.0 |
| Populacao | Population | float64 | 3–35682 |
| MediaOcupacao | AveOccup | float64 | 0.5–20.0 |
| Latitude | Latitude | float64 | 32.5–42.0 |
| Longitude | Longitude | float64 | -124.4–-114.3 |

```python
rename_map = {
    'MedInc': 'RendaMediana', 'HouseAge': 'IdadeMediaResidencias',
    'AveRooms': 'MediaComodos', 'AveBedrms': 'MediaQuartos',
    'Population': 'Populacao', 'AveOccup': 'MediaOcupacao',
}
df = df.rename(columns=rename_map)
```

### Engineered features (13 total after pipeline transforms)

Original 8 + `razao_quartos`, `comodos_por_pessoa`, `dist_sf`, `dist_la`, `dist_sd`

### Target

`ValorMedioResidencias` → `log1p(target)` for training → reverse with `np.expm1()` → multiply by `100_000` for USD.

## Pipeline Architecture

All preprocessing is inside a single `sklearn.Pipeline` — no transformations happen outside it:

```
X_raw → WinsorizacaoTransformer → CaliforniaHousingTransformer → StandardScaler → Model → ŷ_log
```

**WinsorizacaoTransformer:** clips outliers using IQR bounds learned on train only (`k=3.0`, more conservative than the classic `k=1.5` boxplot). Applied only to `['MediaComodos', 'MediaQuartos', 'Populacao', 'MediaOcupacao']`.

**CaliforniaHousingTransformer:** applies `log1p` to `RendaMediana`, `Populacao`, `MediaOcupacao`, then creates 5 engineered features: `razao_quartos`, `comodos_por_pessoa`, `dist_sf`, `dist_la`, `dist_sd`. Geographic distances use **Euclidean distance in degrees** (`sqrt(Δlat² + Δlon²)`), not Haversine — consistent between training and inference. Input is 8 features; output is 13.

**StandardScaler** is included for all model types (even tree-based) to keep a single pipeline factory usable for all competitors.

## Inference

```python
import joblib
import numpy as np
import pandas as pd

# Load once (e.g., at API startup)
pipeline = joblib.load('artifacts/competition_winner_catboost.joblib')

# Predict — input must use the Portuguese-renamed column names
X = pd.DataFrame([{
    'RendaMediana': 5.0, 'IdadeMediaResidencias': 20.0,
    'MediaComodos': 5.5, 'MediaQuartos': 1.1,
    'Populacao': 1200.0, 'MediaOcupacao': 2.8,
    'Latitude': 34.05, 'Longitude': -118.24,
}])
pred_100k = np.expm1(pipeline.predict(X))
# Multiply by 100_000 to get USD
```

For **Conformal Prediction intervals**, load `q_hats` from `artifacts/competition_metadata.json` (`conformal_q_hats` key) — they are pre-computed and do not require the calibration set at inference time.

## Key Design Decisions

- **Column names are Portuguese**: raw sklearn dataset columns are renamed at load time. See the Data Schema table above for the full mapping.
- **Optuna study persistence**: `california_housing_optuna.db` accumulates across runs. Delete or use a new study name to start fresh. Study names: `CalHousing_Ridge`, `CalHousing_XGBoost`, `CalHousing_LightGBM`, `CalHousing_CatBoost`, `CalHousing_TabNet`.
- **GPU detection**: XGBoost, CatBoost, and TabNet auto-detect CUDA. LightGBM uses CPU only (pip build lacks GPU support).
- **Data split difference between phases**: Phase 1 uses 60/20/20; Phase 2 uses 80/20. This is intentional — Conformal Prediction requires a held-out calibration set.
- **Artifacts directory**: `artifacts/` is created automatically by the notebooks (`Path('artifacts/').mkdir(exist_ok=True)`).
- **Scoreboard weights**: Phase 1 elects the winner via a weighted score — point metrics 60% (R² 30%, RMSE 15%, MAE 10%, MAPE 5%) and reliability metrics 40% (Interval Score 20%, PICP Error 10%, MACE 10%). Winner is stored in `artifacts/competition_metadata.json` under `competition_winner`.

## Session Guidance

- **Context limit warning**: When a session is approaching ~100k tokens (long notebook audits, multi-agent workflows, iterative edits), warn explicitly before continuing:
  > ⚠️ Esta sessão está se aproximando de 100k tokens. Considere `/compact` (comprime o histórico mantendo contexto) ou `/clear` (reinicia — use quando a tarefa atual terminou) antes de continuar.
