# Yogyakarta LSTM Autoencoder Sensitivity Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build notebook `05_yogyakarta_sensitivity_analysis_experiments.ipynb` to train and compare six compact LSTM Autoencoder sensitivity experiments for DI Yogyakarta weather anomaly detection.

**Architecture:** The notebook is an experiment layer that consumes the processed feature table from notebook 01, trains independent compact models per window/feature configuration, and exports comparison reports without overwriting the baseline model or baseline reports. Reusable helper cells keep split, scaling, sequence building, scoring, and alert interpretation consistent with notebooks 02 and 03.

**Tech Stack:** Jupyter Notebook, Python, pandas, numpy, scikit-learn, TensorFlow/Keras, matplotlib, seaborn.

## Global Constraints

- Build the deliverable as an `.ipynb` notebook.
- Consume `data/processed/yogyakarta_weather_features.csv`.
- Compare sequence window lengths of 7 days and 30 days.
- Compare rainfall-only, wind-only, and all-weather feature groups.
- Train six independent compact LSTM Autoencoder models.
- Use chronological train, validation, and test splitting.
- Fit scalers only on the training split for each experiment.
- Build station-safe consecutive sequences and reject windows with date gaps.
- Do not use flood labels.
- Do not overwrite baseline artifacts or baseline reports.
- Export sensitivity reports under `reports/`.
- Optionally write experiment artifacts under `artifacts/sensitivity/`.

---

### Task 1: Build Notebook 05 Structure

**Files:**
- Create: `notebooks/05_yogyakarta_sensitivity_analysis_experiments.ipynb`
- Modify: `notebooks/README.md`

**Interfaces:**
- Consumes: existing notebook naming and run order
- Produces: notebook 05 markdown/code cell skeleton and README run-order entry

- [ ] **Step 1: Create notebook with these sections**

Create an `.ipynb` with cells titled:

```markdown
# 05 - Yogyakarta LSTM Autoencoder Sensitivity Analysis
## 1. Setup
## 2. Load Processed Features
## 3. Experiment Configurations
## 4. Helper Functions
## 5. Run Sensitivity Experiments
## 6. Export Reports
## 7. Visual Comparison
## 8. Lecturer-Facing Interpretation
## 9. Verification
```

- [ ] **Step 2: Add notebook path setup**

Add code that defines:

```python
PROJECT_ROOT = Path("..").resolve()
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "yogyakarta_weather_features.csv"
REPORT_DIR = PROJECT_ROOT / "reports"
SENSITIVITY_ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "sensitivity"
BASELINE_ALERTS_PATH = REPORT_DIR / "alerts.csv"
```

Expected: notebook can run from `notebooks/` as the kernel working directory.

- [ ] **Step 3: Update README**

Add `05_yogyakarta_sensitivity_analysis_experiments.ipynb` after notebook 04 and list the five sensitivity report outputs.

### Task 2: Implement Experiment Helpers

**Files:**
- Modify: `notebooks/05_yogyakarta_sensitivity_analysis_experiments.ipynb`

**Interfaces:**
- Consumes: processed feature DataFrame with `date`, `station_id`, metadata, base weather columns, engineered weather columns
- Produces: helper functions used by the experiment runner

- [ ] **Step 1: Add imports and reproducibility setup**

Use:

```python
from pathlib import Path
import json
import random
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM, RepeatVector, TimeDistributed
from tensorflow.keras.optimizers import Adam

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
sns.set_theme(style="whitegrid")
warnings.filterwarnings("ignore")
```

- [ ] **Step 2: Add experiment configurations**

Define:

```python
EXPERIMENTS = [
    {"experiment_id": "rain_7d", "sequence_length": 7, "feature_group": "rain_only"},
    {"experiment_id": "wind_7d", "sequence_length": 7, "feature_group": "wind_only"},
    {"experiment_id": "all_7d", "sequence_length": 7, "feature_group": "all_features"},
    {"experiment_id": "rain_30d", "sequence_length": 30, "feature_group": "rain_only"},
    {"experiment_id": "wind_30d", "sequence_length": 30, "feature_group": "wind_only"},
    {"experiment_id": "all_30d", "sequence_length": 30, "feature_group": "all_features"},
]
```

- [ ] **Step 3: Add helper functions**

Implement helpers with these names:

```python
resolve_feature_groups(data: pd.DataFrame) -> dict
validate_required_columns(data: pd.DataFrame, feature_names: list, experiment_id: str) -> None
chronological_split(data: pd.DataFrame) -> tuple
fit_transform_splits(train_df, validation_df, test_df, feature_names) -> tuple
build_sequences(feature_df, metadata_df, feature_names, sequence_length) -> tuple
build_compact_lstm_autoencoder(sequence_length, n_features, learning_rate=0.001) -> Model
weight_vector(feature_names: list, feature_group: str) -> np.ndarray
weighted_reconstruction_error(x_true, x_pred, weights) -> np.ndarray
status_from_score(score, thresholds) -> str
derive_physical_thresholds(training_df: pd.DataFrame) -> dict
classify_alert(row: pd.Series, physical_thresholds: dict) -> tuple
```

Expected: helpers support old Python kernels, so use `list`, `dict`, and `tuple` instead of `list[str]` or `dict[str, float]`.

### Task 3: Implement Experiment Runner and Reports

**Files:**
- Modify: `notebooks/05_yogyakarta_sensitivity_analysis_experiments.ipynb`

**Interfaces:**
- Consumes: helper functions from Task 2
- Produces: sensitivity summary, alert counts, anomaly rows, and optional per-experiment artifacts

- [ ] **Step 1: Load and validate processed data**

Read `PROCESSED_PATH` with:

```python
data = pd.read_csv(PROCESSED_PATH, parse_dates=["date"], dtype={"station_id": "string"})
data = data.sort_values(["station_id", "date"]).reset_index(drop=True)
```

If the file is missing, raise:

```python
FileNotFoundError("Run notebook 01 first: data/processed/yogyakarta_weather_features.csv was not found.")
```

- [ ] **Step 2: Run all six experiments**

For each experiment:

```python
feature_names = feature_groups[config["feature_group"]]
validate_required_columns(data, feature_names, config["experiment_id"])
train_df, validation_df, test_df = chronological_split(data)
x_train, x_val, x_test, train_meta, val_meta, test_meta, scaler = fit_transform_splits(...)
model = build_compact_lstm_autoencoder(...)
history = model.fit(...)
validation_scores = weighted_reconstruction_error(x_val, model.predict(x_val), weights)
thresholds = {"p95": ..., "p99": ..., "p995": ...}
evaluation_scores = weighted_reconstruction_error(x_eval, model.predict(x_eval), weights)
```

Expected: each experiment appends rows to summary, alert count, and top-anomaly collections.

- [ ] **Step 3: Export reports**

Write:

```python
summary_df.to_csv(REPORT_DIR / "sensitivity_experiment_summary.csv", index=False)
alert_counts_df.to_csv(REPORT_DIR / "sensitivity_alert_counts.csv", index=False)
top_anomalies_df.to_csv(REPORT_DIR / "sensitivity_top_anomalies.csv", index=False)
```

Expected: CSVs are non-empty after a successful run.

### Task 4: Add Visual Comparisons and Lecturer Summary

**Files:**
- Modify: `notebooks/05_yogyakarta_sensitivity_analysis_experiments.ipynb`

**Interfaces:**
- Consumes: sensitivity result DataFrames from Task 3
- Produces: comparison charts and Indonesian explanation cells

- [ ] **Step 1: Add status comparison plot**

Create a grouped bar chart of status counts by experiment and save:

```python
plt.savefig(REPORT_DIR / "sensitivity_status_comparison.png", dpi=160, bbox_inches="tight")
```

- [ ] **Step 2: Add score distribution plot**

Create boxplots or violin plots of anomaly score distributions by experiment and save:

```python
plt.savefig(REPORT_DIR / "sensitivity_score_distribution.png", dpi=160, bbox_inches="tight")
```

- [ ] **Step 3: Add baseline overlap table**

If `reports/alerts.csv` exists, compare non-normal alert dates from baseline with non-normal sensitivity dates by experiment. If it does not exist, display a clear note and continue.

- [ ] **Step 4: Add final Indonesian summary cell**

Display a concise summary explaining:

```text
Model 7 hari cenderung lebih responsif terhadap perubahan singkat.
Model 30 hari cenderung lebih stabil karena membaca pola cuaca yang lebih panjang.
Rain-only dan wind-only berguna sebagai pembanding, tetapi all-features tetap kandidat utama karena membaca konteks cuaca paling lengkap.
```

### Task 5: Verify Notebook File Integrity

**Files:**
- Verify: `notebooks/05_yogyakarta_sensitivity_analysis_experiments.ipynb`
- Verify: `notebooks/README.md`

**Interfaces:**
- Consumes: completed notebook and README
- Produces: syntax/JSON integrity confidence before user runs notebook in Jupyter

- [ ] **Step 1: Validate notebook JSON**

Run:

```bash
rtk python3 -m json.tool notebooks/05_yogyakarta_sensitivity_analysis_experiments.ipynb
```

Expected: command exits with status 0.

- [ ] **Step 2: Inspect README update**

Run:

```bash
rtk sed -n '1,220p' notebooks/README.md
```

Expected: notebook 05 is listed in run order and outputs.
