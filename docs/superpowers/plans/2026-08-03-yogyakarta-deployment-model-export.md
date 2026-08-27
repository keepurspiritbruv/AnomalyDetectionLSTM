# Yogyakarta Deployment Model Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build notebook `06_yogyakarta_export_deployment_models.ipynb` to retrain and export deployable `rain_7d` and `wind_30d` LSTM Autoencoder artifacts.

**Architecture:** Notebook 06 consumes the processed feature table from notebook 01, trains only the two selected hazard-specific deployment models, and writes each model's Keras file, scaler, thresholds, feature config, and training report under `artifacts/deployment/`. Notebook 05 remains a sensitivity-analysis report notebook.

**Tech Stack:** Jupyter Notebook, Python, pandas, numpy, scikit-learn, TensorFlow/Keras, joblib.

## Global Constraints

- Build the deliverable as an `.ipynb` notebook.
- Consume `data/processed/yogyakarta_weather_features.csv`.
- Export `rain_7d` for curah hujan tinggi.
- Export `wind_30d` for angin kencang.
- Fit scalers only on the training split for each deployment model.
- Use chronological train, validation, and test splitting.
- Build station-safe consecutive sequences and reject date-gap windows.
- Use old-Python-compatible typing: `list`, `dict`, and `tuple`, not `list[str]`.
- Do not modify notebook 05 behavior.
- Do not overwrite baseline notebook 02 artifacts.

---

### Task 1: Create Notebook 06

**Files:**
- Create: `notebooks/06_yogyakarta_export_deployment_models.ipynb`

**Interfaces:**
- Consumes: `data/processed/yogyakarta_weather_features.csv`
- Produces: deployment artifact folders under `artifacts/deployment/`

- [ ] **Step 1: Add notebook sections**

Create sections:

```markdown
# 06 - Export Yogyakarta Deployment Models
## 1. Setup
## 2. Load Processed Features
## 3. Deployment Model Configurations
## 4. Helper Functions
## 5. Train and Export Deployment Models
## 6. Verification
```

- [ ] **Step 2: Add setup cell**

Define paths:

```python
PROJECT_ROOT = Path("..").resolve()
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "yogyakarta_weather_features.csv"
DEPLOYMENT_DIR = PROJECT_ROOT / "artifacts" / "deployment"
REPORT_DIR = PROJECT_ROOT / "reports"
```

### Task 2: Add Training and Export Logic

**Files:**
- Modify: `notebooks/06_yogyakarta_export_deployment_models.ipynb`

**Interfaces:**
- Consumes: selected deployment configs
- Produces: trained model artifacts

- [ ] **Step 1: Define deployment configs**

Use:

```python
DEPLOYMENT_MODELS = [
    {"experiment_id": "rain_7d", "sequence_length": 7, "feature_group": "rain_only", "hazard": "curah_hujan_tinggi"},
    {"experiment_id": "wind_30d", "sequence_length": 30, "feature_group": "wind_only", "hazard": "angin_kencang"},
]
```

- [ ] **Step 2: Add helper functions**

Implement:

```python
resolve_feature_groups(data)
chronological_split(data)
fit_transform_splits(train_df, validation_df, test_df, feature_names)
build_sequences(feature_df, metadata_df, feature_names, sequence_length)
build_compact_lstm_autoencoder(sequence_length, n_features, learning_rate=0.001)
weight_vector(feature_names, feature_group)
weighted_reconstruction_error(x_true, x_pred, weights)
derive_thresholds(scores)
```

- [ ] **Step 3: Export files per model**

For each deployment config, save:

```text
artifacts/deployment/<experiment_id>/model.keras
artifacts/deployment/<experiment_id>/scaler.pkl
artifacts/deployment/<experiment_id>/threshold.json
artifacts/deployment/<experiment_id>/feature_config.json
artifacts/deployment/<experiment_id>/training_report.json
```

### Task 3: Update Notebook README

**Files:**
- Modify: `notebooks/README.md`

**Interfaces:**
- Consumes: notebook run order
- Produces: README entry for notebook 06

- [ ] **Step 1: Add notebook 06 to run order**

Add:

```markdown
6. `06_yogyakarta_export_deployment_models.ipynb`
```

- [ ] **Step 2: Add expected deployment outputs**

Add:

```markdown
Use `06_yogyakarta_export_deployment_models.ipynb` to export selected deployment models.
- `../artifacts/deployment/rain_7d/model.keras`
- `../artifacts/deployment/rain_7d/scaler.pkl`
- `../artifacts/deployment/rain_7d/threshold.json`
- `../artifacts/deployment/rain_7d/feature_config.json`
- `../artifacts/deployment/wind_30d/model.keras`
- `../artifacts/deployment/wind_30d/scaler.pkl`
- `../artifacts/deployment/wind_30d/threshold.json`
- `../artifacts/deployment/wind_30d/feature_config.json`
```

### Task 4: Verify Notebook Integrity

**Files:**
- Verify: `notebooks/06_yogyakarta_export_deployment_models.ipynb`
- Verify: `notebooks/README.md`

**Interfaces:**
- Consumes: created notebook and README
- Produces: JSON and syntax validation

- [ ] **Step 1: Validate notebook JSON and compile code cells**

Run:

```bash
rtk python3 -c "import json; nb=json.load(open('notebooks/06_yogyakarta_export_deployment_models.ipynb')); source='\n\n'.join(''.join(c.get('source', [])) for c in nb['cells'] if c.get('cell_type')=='code'); compile(source, 'notebooks/06_yogyakarta_export_deployment_models.ipynb', 'exec'); print('notebook json valid and code cells compile')"
```

Expected: command exits 0 and prints `notebook json valid and code cells compile`.

- [ ] **Step 2: Confirm old-Python-compatible typing**

Run:

```bash
rtk rg -n "list\\[|dict\\[|tuple\\[|set\\[" notebooks/06_yogyakarta_export_deployment_models.ipynb
```

Expected: no matches.
