# Yogyakarta LSTM Autoencoder Weather Warning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Jupyter notebooks that train and evaluate one Yogyakarta-only multivariate LSTM Autoencoder for daily extreme rainfall and strong-wind anomaly warning.

**Architecture:** The primary deliverable is an `.ipynb` workflow intended for a lecturer-managed Jupyter server. The notebooks are split into data audit/preprocessing, LSTM Autoencoder training, and evaluation/alert interpretation so the dissertation workflow is transparent, rerunnable, and easy to present.

**Tech Stack:** Jupyter Notebook, Python 3.11+, pandas, numpy, scikit-learn, TensorFlow/Keras, joblib, matplotlib, seaborn.

## Global Constraints

- Build the implementation in `.ipynb` notebooks as the primary runnable artifacts.
- Train only on DI Yogyakarta climate rows from `Dataset/datasetIDNkaggle/climate_data.csv`.
- Use station metadata from `Dataset/datasetIDNkaggle/station_detail.csv` and `Dataset/datasetIDNkaggle/province_detail.csv`.
- First baseline trains on available Yogyakarta station rows for `96851` and `96855`; keep `96859` visible in metadata notes if data appears later.
- Do not use the `flood` column from any dataset as target, feature, or evaluation label.
- Do not merge supplementary Excel files into the hazard model unless they expose rainfall and wind columns matching the model schema.
- Use one multivariate LSTM Autoencoder for both hazards.
- Use a rule-based alert engine for `CURAH_HUJAN_EKSTREM`, `ANGIN_KENCANG`, and `CURAH_HUJAN_EKSTREM_DAN_ANGIN_KENCANG`.
- Use chronological train/validation/test splitting; never random splitting.
- Fit scalers only on the training split.
- Build LSTM windows per station; no sequence may cross station boundaries.
- The first model is daily historical anomaly detection, not minute-level IoT nowcasting.
- This workspace is currently not a git repository, so commit steps should be run only if a repository is initialized later.

---

## File Structure

- Create: `notebooks/01_yogyakarta_data_audit_preprocessing.ipynb`  
  Loads datasets, filters DI Yogyakarta stations, audits missing values/outliers, cleans data, engineers features, and exports `data/processed/yogyakarta_weather_features.csv`.
- Create: `notebooks/02_yogyakarta_lstm_autoencoder_training.ipynb`  
  Loads processed features, builds station-safe sequences, trains the LSTM Autoencoder, calculates validation thresholds, and exports model artifacts.
- Create: `notebooks/03_yogyakarta_anomaly_evaluation_alerts.ipynb`  
  Loads artifacts, scores validation/test windows, applies alert rules, creates dissertation-ready tables/plots, and exports reports.
- Create: `notebooks/README.md`  
  Explains notebook run order and expected outputs.
- Create: `artifacts/.gitkeep`, `reports/.gitkeep`, `data/processed/.gitkeep`  
  Output directories for model artifacts, evaluation reports, and processed data.
- Create: `requirements-notebook.txt`  
  Notebook/server dependencies.

## Task 1: Notebook Workspace and Dependency Manifest

**Files:**
- Create: `requirements-notebook.txt`
- Create: `notebooks/README.md`
- Create: `artifacts/.gitkeep`
- Create: `reports/.gitkeep`
- Create: `data/processed/.gitkeep`

**Interfaces:**
- Produces notebook run contract:
  - Run `01_yogyakarta_data_audit_preprocessing.ipynb` first.
  - Run `02_yogyakarta_lstm_autoencoder_training.ipynb` second.
  - Run `03_yogyakarta_anomaly_evaluation_alerts.ipynb` third.

- [ ] **Step 1: Create notebook dependency file**

Create `requirements-notebook.txt`:

```text
jupyter>=1.0
ipykernel>=6.29
pandas>=2.2
numpy>=1.26
scikit-learn>=1.4
tensorflow>=2.16
joblib>=1.4
matplotlib>=3.8
seaborn>=0.13
openpyxl>=3.1
```

- [ ] **Step 2: Create output directories**

Run:

```bash
rtk mkdir -p notebooks artifacts reports data/processed
rtk touch artifacts/.gitkeep reports/.gitkeep data/processed/.gitkeep
```

Expected: directories exist for notebooks and generated outputs.

- [ ] **Step 3: Create notebook README**

Create `notebooks/README.md`:

```markdown
# Yogyakarta LSTM Autoencoder Notebook Workflow

Run the notebooks in this order:

1. `01_yogyakarta_data_audit_preprocessing.ipynb`
2. `02_yogyakarta_lstm_autoencoder_training.ipynb`
3. `03_yogyakarta_anomaly_evaluation_alerts.ipynb`

The workflow trains one multivariate LSTM Autoencoder for DI Yogyakarta daily climate anomaly detection. It focuses on curah hujan ekstrem and angin kencang.

Primary data:

- `../Dataset/datasetIDNkaggle/climate_data.csv`
- `../Dataset/datasetIDNkaggle/station_detail.csv`
- `../Dataset/datasetIDNkaggle/province_detail.csv`

Important scope:

- Flood labels are not used.
- The baseline trains on Yogyakarta station rows available in `climate_data.csv`.
- Daily historical data is not the same as 5-minute or 10-minute IoT nowcasting data.

Expected outputs:

- `../data/processed/yogyakarta_weather_features.csv`
- `../artifacts/model_lstm_autoencoder.keras`
- `../artifacts/scaler.pkl`
- `../artifacts/threshold.json`
- `../artifacts/feature_config.json`
- `../artifacts/training_report.json`
- `../reports/anomaly_scores.csv`
- `../reports/alerts.csv`
- `../reports/top_anomalies.csv`
- `../reports/station_anomaly_counts.csv`
```

- [ ] **Step 4: Commit if a Git repository exists**

Run:

```bash
rtk git rev-parse --show-toplevel
rtk git add requirements-notebook.txt notebooks/README.md artifacts/.gitkeep reports/.gitkeep data/processed/.gitkeep
rtk git commit -m "chore: set up notebook weather anomaly workspace"
```

Expected in current workspace: skip commit if not a Git repository.

## Task 2: Data Audit and Preprocessing Notebook

**Files:**
- Create: `notebooks/01_yogyakarta_data_audit_preprocessing.ipynb`
- Output: `data/processed/yogyakarta_weather_features.csv`

**Interfaces:**
- Consumes:
  - `Dataset/datasetIDNkaggle/climate_data.csv`
  - `Dataset/datasetIDNkaggle/station_detail.csv`
  - `Dataset/datasetIDNkaggle/province_detail.csv`
- Produces processed CSV with columns:
  - `date`, `station_id`, `station_name`, `region_name`, `province_name`
  - base climate features
  - engineered model features
  - missingness flags

- [ ] **Step 1: Create notebook markdown introduction cell**

Notebook cell content:

```markdown
# 01 - Yogyakarta Data Audit and Preprocessing

This notebook prepares the daily DI Yogyakarta climate dataset for LSTM Autoencoder training.

Scope:

- Use `climate_data.csv` as the primary source.
- Filter to DI Yogyakarta station rows.
- Do not use flood labels.
- Engineer rainfall, wind, seasonal, and station features.
- Export a clean feature table for model training.
```

- [ ] **Step 2: Create imports and path setup cell**

Notebook code cell:

```python
from pathlib import Path
import json
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

PROJECT_ROOT = Path("..").resolve()
DATA_DIR = PROJECT_ROOT / "Dataset" / "datasetIDNkaggle"
CLIMATE_CSV = DATA_DIR / "climate_data.csv"
STATION_CSV = DATA_DIR / "station_detail.csv"
PROVINCE_CSV = DATA_DIR / "province_detail.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

YOGYAKARTA_STATION_IDS = ["96851", "96855", "96859"]
BASELINE_TRAINING_STATION_IDS = ["96851", "96855"]

print(CLIMATE_CSV)
print(STATION_CSV)
print(PROVINCE_CSV)
```

- [ ] **Step 3: Create data loading cell**

Notebook code cell:

```python
climate = pd.read_csv(CLIMATE_CSV, dtype={"station_id": "string"})
stations = pd.read_csv(STATION_CSV, dtype={"station_id": "string", "province_id": "string"})
provinces = pd.read_csv(PROVINCE_CSV, dtype={"province_id": "string"})

climate["date"] = pd.to_datetime(climate["date"], format="%d-%m-%Y", errors="coerce")
climate["station_id"] = climate["station_id"].astype(str)
stations["station_id"] = stations["station_id"].astype(str)

numeric_columns = ["Tn", "Tx", "Tavg", "RH_avg", "RR", "ss", "ff_x", "ddd_x", "ff_avg"]
for column in numeric_columns:
    climate[column] = pd.to_numeric(climate[column], errors="coerce")

station_metadata = stations.merge(provinces, on="province_id", how="left")

print("Climate shape:", climate.shape)
print("Station metadata shape:", station_metadata.shape)
display(climate.head())
display(station_metadata[station_metadata["station_id"].isin(YOGYAKARTA_STATION_IDS)])
```

- [ ] **Step 4: Create Yogyakarta filtering and coverage audit cell**

Notebook code cell:

```python
yogya = climate[climate["station_id"].isin(YOGYAKARTA_STATION_IDS)].copy()
yogya = yogya.merge(
    station_metadata[
        ["station_id", "station_name", "region_name", "latitude", "longitude", "province_name"]
    ],
    on="station_id",
    how="left",
)
yogya = yogya.sort_values(["station_id", "date"]).drop_duplicates(["station_id", "date"]).reset_index(drop=True)

coverage = (
    yogya.groupby(["station_id", "station_name", "region_name"])
    .agg(
        rows=("date", "size"),
        start_date=("date", "min"),
        end_date=("date", "max"),
        unique_days=("date", "nunique"),
    )
    .reset_index()
)
coverage["start_date"] = coverage["start_date"].dt.date
coverage["end_date"] = coverage["end_date"].dt.date

display(coverage)
print("Training stations with available rows:", sorted(yogya["station_id"].unique()))
```

- [ ] **Step 5: Create missingness and outlier audit cell**

Notebook code cell:

```python
missing_summary = (
    yogya[["date", "station_id"] + numeric_columns]
    .isna()
    .mean()
    .mul(100)
    .round(2)
    .rename("missing_percent")
    .reset_index()
    .rename(columns={"index": "column"})
)
display(missing_summary)

physical_ranges = {
    "Tn": (10.0, 40.0),
    "Tx": (15.0, 45.0),
    "Tavg": (10.0, 40.0),
    "RH_avg": (0.0, 100.0),
    "RR": (0.0, 500.0),
    "ss": (0.0, 15.0),
    "ff_x": (0.0, 60.0),
    "ff_avg": (0.0, 60.0),
    "ddd_x": (0.0, 360.0),
}

outlier_rows = []
for column, (lower, upper) in physical_ranges.items():
    mask = yogya[column].notna() & ((yogya[column] < lower) | (yogya[column] > upper))
    outlier_rows.append({"column": column, "outlier_count": int(mask.sum())})

outlier_summary = pd.DataFrame(outlier_rows)
display(outlier_summary)
```

- [ ] **Step 6: Create cleaning functions cell**

Notebook code cell:

```python
IMPORTANT_MISSING_COLUMNS = ["RR", "ff_x", "ff_avg", "RH_avg"]
IMPUTE_COLUMNS = ["Tn", "Tx", "Tavg", "RH_avg", "RR", "ss", "ff_x", "ddd_x", "ff_avg"]


def clean_physical_ranges(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    for column, (lower, upper) in physical_ranges.items():
        mask = cleaned[column].notna() & ((cleaned[column] < lower) | (cleaned[column] > upper))
        cleaned.loc[mask, column] = np.nan
    return cleaned


def add_missing_flags(df: pd.DataFrame) -> pd.DataFrame:
    flagged = df.copy()
    for column in IMPORTANT_MISSING_COLUMNS:
        flagged[f"missing_{column}"] = flagged[column].isna().astype(int)
    return flagged


def impute_by_station(df: pd.DataFrame, max_gap_days: int = 3) -> pd.DataFrame:
    frames = []
    for station_id, group in df.sort_values(["station_id", "date"]).groupby("station_id", sort=False):
        g = group.copy().set_index("date")
        for column in IMPUTE_COLUMNS:
            g[column] = g[column].interpolate(method="time", limit=max_gap_days, limit_direction="both")
        frames.append(g.reset_index())
    return pd.concat(frames, ignore_index=True).sort_values(["station_id", "date"]).reset_index(drop=True)
```

- [ ] **Step 7: Create feature engineering cell**

Notebook code cell:

```python
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    engineered = df.sort_values(["station_id", "date"]).copy()
    engineered["temp_range"] = engineered["Tx"] - engineered["Tn"]

    direction_rad = np.deg2rad(engineered["ddd_x"].fillna(0.0))
    engineered["ddd_x_sin"] = np.sin(direction_rad)
    engineered["ddd_x_cos"] = np.cos(direction_rad)

    day_angle = 2 * np.pi * engineered["date"].dt.dayofyear / 365.25
    engineered["day_of_year_sin"] = np.sin(day_angle)
    engineered["day_of_year_cos"] = np.cos(day_angle)

    engineered["station_96855"] = (engineered["station_id"].astype(str) == "96855").astype(int)

    frames = []
    for station_id, group in engineered.groupby("station_id", sort=False):
        g = group.sort_values("date").copy()
        g["rain_3d"] = g["RR"].rolling(window=3, min_periods=1).sum()
        g["rain_7d"] = g["RR"].rolling(window=7, min_periods=1).sum()
        g["rain_change_1d"] = g["RR"].diff().fillna(0.0)
        g["wind_change_1d"] = g["ff_x"].diff().fillna(0.0)
        frames.append(g)

    return pd.concat(frames, ignore_index=True).sort_values(["station_id", "date"]).reset_index(drop=True)


base_features = ["Tn", "Tx", "Tavg", "RH_avg", "RR", "ss", "ff_x", "ff_avg"]
model_features = [
    "Tn",
    "Tx",
    "Tavg",
    "RH_avg",
    "RR",
    "ss",
    "ff_x",
    "ff_avg",
    "temp_range",
    "rain_3d",
    "rain_7d",
    "rain_change_1d",
    "wind_change_1d",
    "ddd_x_sin",
    "ddd_x_cos",
    "day_of_year_sin",
    "day_of_year_cos",
    "station_96855",
    "missing_RR",
    "missing_ff_x",
    "missing_ff_avg",
    "missing_RH_avg",
]
```

- [ ] **Step 8: Create preprocessing execution and export cell**

Notebook code cell:

```python
prepared = yogya[yogya["station_id"].isin(BASELINE_TRAINING_STATION_IDS)].copy()
prepared = clean_physical_ranges(prepared)
prepared = add_missing_flags(prepared)
prepared = impute_by_station(prepared, max_gap_days=3)
prepared = engineer_features(prepared)
prepared = prepared.dropna(subset=model_features).reset_index(drop=True)

output_columns = [
    "date",
    "station_id",
    "station_name",
    "region_name",
    "province_name",
] + model_features

processed_path = PROCESSED_DIR / "yogyakarta_weather_features.csv"
prepared[output_columns].to_csv(processed_path, index=False)

print("Prepared shape:", prepared.shape)
print("Saved:", processed_path)
display(prepared[output_columns].head())
display(prepared.groupby("station_id")["date"].agg(["min", "max", "count"]))
```

- [ ] **Step 9: Create preprocessing visual audit cell**

Notebook code cell:

```python
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

for station_id, group in prepared.groupby("station_id"):
    axes[0].plot(group["date"], group["RR"], label=station_id, alpha=0.8)
    axes[1].plot(group["date"], group["ff_x"], label=station_id, alpha=0.8)
    axes[2].plot(group["date"], group["Tavg"], label=station_id, alpha=0.8)

axes[0].set_title("Daily Rainfall RR - DI Yogyakarta")
axes[1].set_title("Maximum Wind Speed ff_x - DI Yogyakarta")
axes[2].set_title("Average Temperature Tavg - DI Yogyakarta")
for ax in axes:
    ax.legend()
    ax.set_xlabel("Date")

plt.tight_layout()
plt.show()
```

- [ ] **Step 10: Execute notebook top to bottom**

Run inside Jupyter: `Kernel -> Restart Kernel and Run All Cells`.  
Expected: `data/processed/yogyakarta_weather_features.csv` is created, and no cell raises an exception.

## Task 3: LSTM Autoencoder Training Notebook

**Files:**
- Create: `notebooks/02_yogyakarta_lstm_autoencoder_training.ipynb`
- Output:
  - `artifacts/model_lstm_autoencoder.keras`
  - `artifacts/scaler.pkl`
  - `artifacts/threshold.json`
  - `artifacts/feature_config.json`
  - `artifacts/training_report.json`

**Interfaces:**
- Consumes: `data/processed/yogyakarta_weather_features.csv`.
- Produces saved model, scaler, feature config, threshold config, and training report.

- [ ] **Step 1: Create notebook markdown introduction cell**

Notebook cell content:

```markdown
# 02 - Yogyakarta LSTM Autoencoder Training

This notebook trains one multivariate LSTM Autoencoder on DI Yogyakarta daily climate sequences.

The model reconstructs 30-day weather windows. High reconstruction error indicates an unusual weather sequence. Warning type is interpreted later by physical rainfall and wind rules.
```

- [ ] **Step 2: Create imports, paths, and seed setup cell**

Notebook code cell:

```python
from pathlib import Path
import json
import random

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import RobustScaler
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, RepeatVector, TimeDistributed
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

PROJECT_ROOT = Path("..").resolve()
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "yogyakarta_weather_features.csv"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
REPORT_DIR = PROJECT_ROOT / "reports"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

SEQUENCE_LENGTH = 30
BATCH_SIZE = 32
MAX_EPOCHS = 200
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 15
VALIDATION_FRACTION = 0.15
TEST_FRACTION = 0.15
```

- [ ] **Step 3: Create feature configuration cell**

Notebook code cell:

```python
model_features = [
    "Tn",
    "Tx",
    "Tavg",
    "RH_avg",
    "RR",
    "ss",
    "ff_x",
    "ff_avg",
    "temp_range",
    "rain_3d",
    "rain_7d",
    "rain_change_1d",
    "wind_change_1d",
    "ddd_x_sin",
    "ddd_x_cos",
    "day_of_year_sin",
    "day_of_year_cos",
    "station_96855",
    "missing_RR",
    "missing_ff_x",
    "missing_ff_avg",
    "missing_RH_avg",
]

feature_weights = {
    "RR": 3.0,
    "rain_3d": 3.0,
    "rain_7d": 3.0,
    "ff_x": 2.5,
    "ff_avg": 2.5,
    "wind_change_1d": 2.5,
    "RH_avg": 1.5,
    "Tavg": 1.0,
    "Tn": 1.0,
    "Tx": 1.0,
    "temp_range": 1.0,
    "ss": 1.0,
    "ddd_x_sin": 0.8,
    "ddd_x_cos": 0.8,
    "day_of_year_sin": 0.5,
    "day_of_year_cos": 0.5,
    "station_96855": 0.5,
    "missing_RR": 0.5,
    "missing_ff_x": 0.5,
    "missing_ff_avg": 0.5,
    "missing_RH_avg": 0.5,
}
```

- [ ] **Step 4: Create load processed data cell**

Notebook code cell:

```python
data = pd.read_csv(PROCESSED_PATH, parse_dates=["date"], dtype={"station_id": "string"})
data = data.sort_values(["station_id", "date"]).reset_index(drop=True)

print("Processed shape:", data.shape)
display(data.head())
display(data.groupby("station_id")["date"].agg(["min", "max", "count"]))
```

- [ ] **Step 5: Create chronological split cell**

Notebook code cell:

```python
ordered_dates = np.array(sorted(data["date"].unique()))
n_dates = len(ordered_dates)
test_size = int(round(n_dates * TEST_FRACTION))
validation_size = int(round(n_dates * VALIDATION_FRACTION))
train_end = n_dates - validation_size - test_size
validation_end = n_dates - test_size

train_dates = ordered_dates[:train_end]
validation_dates = ordered_dates[train_end:validation_end]
test_dates = ordered_dates[validation_end:]

train_df = data[data["date"].isin(train_dates)].copy()
validation_df = data[data["date"].isin(validation_dates)].copy()
test_df = data[data["date"].isin(test_dates)].copy()

print("Train:", train_df["date"].min().date(), "to", train_df["date"].max().date(), train_df.shape)
print("Validation:", validation_df["date"].min().date(), "to", validation_df["date"].max().date(), validation_df.shape)
print("Test:", test_df["date"].min().date(), "to", test_df["date"].max().date(), test_df.shape)
```

- [ ] **Step 6: Create scaling cell**

Notebook code cell:

```python
scaler = RobustScaler()
scaler.fit(train_df[model_features])

def scale_frame(df: pd.DataFrame) -> pd.DataFrame:
    scaled = df.copy()
    scaled[model_features] = scaler.transform(scaled[model_features])
    return scaled

train_scaled = scale_frame(train_df)
validation_scaled = scale_frame(validation_df)
test_scaled = scale_frame(test_df)
```

- [ ] **Step 7: Create station-safe sequence builder cell**

Notebook code cell:

```python
def build_sequences(df: pd.DataFrame, feature_names: list[str], sequence_length: int):
    x_values = []
    metadata_rows = []

    for station_id, group in df.sort_values(["station_id", "date"]).groupby("station_id", sort=False):
        g = group.sort_values("date").reset_index(drop=True)
        values = g[feature_names].to_numpy(dtype=np.float32)

        for end_idx in range(sequence_length - 1, len(g)):
            start_idx = end_idx - sequence_length + 1
            x_values.append(values[start_idx:end_idx + 1])
            metadata_rows.append(
                {
                    "date": g.loc[end_idx, "date"],
                    "station_id": str(station_id),
                    "station_name": g.loc[end_idx, "station_name"],
                    "region_name": g.loc[end_idx, "region_name"],
                }
            )

    if not x_values:
        return np.empty((0, sequence_length, len(feature_names)), dtype=np.float32), pd.DataFrame(metadata_rows)

    return np.stack(x_values).astype(np.float32), pd.DataFrame(metadata_rows)


x_train, train_meta = build_sequences(train_scaled, model_features, SEQUENCE_LENGTH)
x_validation, validation_meta = build_sequences(validation_scaled, model_features, SEQUENCE_LENGTH)
x_test, test_meta = build_sequences(test_scaled, model_features, SEQUENCE_LENGTH)

print("x_train:", x_train.shape)
print("x_validation:", x_validation.shape)
print("x_test:", x_test.shape)
display(train_meta.head())
```

- [ ] **Step 8: Create model architecture cell**

Notebook code cell:

```python
def build_lstm_autoencoder(sequence_length: int, n_features: int, learning_rate: float) -> Model:
    inputs = Input(shape=(sequence_length, n_features))
    encoded = LSTM(64, activation="tanh", return_sequences=True)(inputs)
    encoded = Dropout(0.2)(encoded)
    encoded = LSTM(32, activation="tanh", return_sequences=False)(encoded)

    decoded = RepeatVector(sequence_length)(encoded)
    decoded = LSTM(32, activation="tanh", return_sequences=True)(decoded)
    decoded = Dropout(0.2)(decoded)
    decoded = LSTM(64, activation="tanh", return_sequences=True)(decoded)
    outputs = TimeDistributed(Dense(n_features))(decoded)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse")
    return model


model = build_lstm_autoencoder(SEQUENCE_LENGTH, len(model_features), LEARNING_RATE)
model.summary()
```

- [ ] **Step 9: Create training cell**

Notebook code cell:

```python
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=EARLY_STOPPING_PATIENCE,
    restore_best_weights=True,
)

history = model.fit(
    x_train,
    x_train,
    validation_data=(x_validation, x_validation),
    epochs=MAX_EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stopping],
    shuffle=False,
    verbose=1,
)
```

- [ ] **Step 10: Create training curve cell**

Notebook code cell:

```python
plt.figure(figsize=(10, 5))
plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.title("LSTM Autoencoder Training Curve")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.tight_layout()
plt.show()
```

- [ ] **Step 11: Create weighted reconstruction error and threshold cell**

Notebook code cell:

```python
def weight_vector(feature_names: list[str]) -> np.ndarray:
    weights = np.array([feature_weights.get(name, 1.0) for name in feature_names], dtype=np.float32)
    return weights / weights.mean()


def weighted_reconstruction_error(x_true: np.ndarray, x_pred: np.ndarray, feature_names: list[str]) -> np.ndarray:
    weights = weight_vector(feature_names).reshape(1, 1, -1)
    squared_error = np.square(x_true - x_pred)
    return np.mean(squared_error * weights, axis=(1, 2))


validation_pred = model.predict(x_validation, verbose=0)
validation_scores = weighted_reconstruction_error(x_validation, validation_pred, model_features)

thresholds = {
    "p95": float(np.percentile(validation_scores, 95)),
    "p99": float(np.percentile(validation_scores, 99)),
    "p995": float(np.percentile(validation_scores, 99.5)),
}

print(thresholds)
```

- [ ] **Step 12: Create artifact export cell**

Notebook code cell:

```python
model.save(ARTIFACT_DIR / "model_lstm_autoencoder.keras")
joblib.dump(scaler, ARTIFACT_DIR / "scaler.pkl")

feature_config = {
    "sequence_length": SEQUENCE_LENGTH,
    "model_features": model_features,
    "feature_weights": feature_weights,
    "training_stations": ["96851", "96855"],
}

training_report = {
    "train_sequences": int(len(x_train)),
    "validation_sequences": int(len(x_validation)),
    "test_sequences": int(len(x_test)),
    "thresholds": thresholds,
    "loss_history": {key: [float(v) for v in values] for key, values in history.history.items()},
}

(ARTIFACT_DIR / "threshold.json").write_text(json.dumps(thresholds, indent=2), encoding="utf-8")
(ARTIFACT_DIR / "feature_config.json").write_text(json.dumps(feature_config, indent=2), encoding="utf-8")
(ARTIFACT_DIR / "training_report.json").write_text(json.dumps(training_report, indent=2), encoding="utf-8")

print("Saved model and artifacts to:", ARTIFACT_DIR)
```

- [ ] **Step 13: Execute notebook top to bottom**

Run inside Jupyter: `Kernel -> Restart Kernel and Run All Cells`.  
Expected: all five model artifacts are created under `artifacts/`.

## Task 4: Evaluation and Alert Notebook

**Files:**
- Create: `notebooks/03_yogyakarta_anomaly_evaluation_alerts.ipynb`
- Output:
  - `reports/anomaly_scores.csv`
  - `reports/alerts.csv`
  - `reports/top_anomalies.csv`
  - `reports/station_anomaly_counts.csv`
  - `reports/threshold_sensitivity.csv`
  - `reports/reconstruction_error_distribution.png`

**Interfaces:**
- Consumes:
  - `data/processed/yogyakarta_weather_features.csv`
  - `artifacts/model_lstm_autoencoder.keras`
  - `artifacts/scaler.pkl`
  - `artifacts/threshold.json`
  - `artifacts/feature_config.json`
- Produces dissertation-ready evaluation tables and plots.

- [ ] **Step 1: Create notebook markdown introduction cell**

Notebook cell content:

```markdown
# 03 - Yogyakarta Anomaly Evaluation and Alerts

This notebook scores the trained LSTM Autoencoder, applies physical rainfall and wind rules, and exports dissertation-ready anomaly tables.
```

- [ ] **Step 2: Create imports and artifact loading cell**

Notebook code cell:

```python
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

sns.set_theme(style="whitegrid")

PROJECT_ROOT = Path("..").resolve()
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "yogyakarta_weather_features.csv"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
REPORT_DIR = PROJECT_ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

model = tf.keras.models.load_model(ARTIFACT_DIR / "model_lstm_autoencoder.keras")
scaler = joblib.load(ARTIFACT_DIR / "scaler.pkl")
thresholds = json.loads((ARTIFACT_DIR / "threshold.json").read_text(encoding="utf-8"))
feature_config = json.loads((ARTIFACT_DIR / "feature_config.json").read_text(encoding="utf-8"))

model_features = feature_config["model_features"]
feature_weights = feature_config["feature_weights"]
SEQUENCE_LENGTH = int(feature_config["sequence_length"])
```

- [ ] **Step 3: Create data split and sequence reconstruction cell**

Notebook code cell:

```python
data = pd.read_csv(PROCESSED_PATH, parse_dates=["date"], dtype={"station_id": "string"})
data = data.sort_values(["station_id", "date"]).reset_index(drop=True)

ordered_dates = np.array(sorted(data["date"].unique()))
n_dates = len(ordered_dates)
test_size = int(round(n_dates * 0.15))
validation_size = int(round(n_dates * 0.15))
train_end = n_dates - validation_size - test_size
validation_end = n_dates - test_size

evaluation_dates = ordered_dates[train_end:]
evaluation_df = data[data["date"].isin(evaluation_dates)].copy()
evaluation_scaled = evaluation_df.copy()
evaluation_scaled[model_features] = scaler.transform(evaluation_scaled[model_features])


def build_sequences(df: pd.DataFrame, feature_names: list[str], sequence_length: int):
    x_values = []
    metadata_rows = []

    for station_id, group in df.sort_values(["station_id", "date"]).groupby("station_id", sort=False):
        g = group.sort_values("date").reset_index(drop=True)
        values = g[feature_names].to_numpy(dtype=np.float32)
        for end_idx in range(sequence_length - 1, len(g)):
            start_idx = end_idx - sequence_length + 1
            x_values.append(values[start_idx:end_idx + 1])
            metadata_rows.append(g.iloc[end_idx].to_dict())

    return np.stack(x_values).astype(np.float32), pd.DataFrame(metadata_rows)


x_eval, eval_meta = build_sequences(evaluation_scaled, model_features, SEQUENCE_LENGTH)
eval_pred = model.predict(x_eval, verbose=0)

print("Evaluation sequences:", x_eval.shape)
display(eval_meta.head())
```

- [ ] **Step 4: Create scoring functions cell**

Notebook code cell:

```python
def weight_vector(feature_names: list[str]) -> np.ndarray:
    weights = np.array([feature_weights.get(name, 1.0) for name in feature_names], dtype=np.float32)
    return weights / weights.mean()


def weighted_reconstruction_error(x_true: np.ndarray, x_pred: np.ndarray, feature_names: list[str]) -> np.ndarray:
    weights = weight_vector(feature_names).reshape(1, 1, -1)
    squared_error = np.square(x_true - x_pred)
    return np.mean(squared_error * weights, axis=(1, 2))


def anomaly_level(score: float, thresholds: dict[str, float]) -> str:
    if score > thresholds["p995"]:
        return "AWAS"
    if score > thresholds["p99"]:
        return "SIAGA"
    if score > thresholds["p95"]:
        return "WASPADA"
    return "NORMAL"


scores = weighted_reconstruction_error(x_eval, eval_pred, model_features)
eval_meta["anomaly_score"] = scores
eval_meta["status"] = [anomaly_level(float(score), thresholds) for score in scores]

anomaly_scores_path = REPORT_DIR / "anomaly_scores.csv"
eval_meta.to_csv(anomaly_scores_path, index=False)

display(eval_meta[["date", "station_id", "region_name", "anomaly_score", "status"]].head())
print("Saved:", anomaly_scores_path)
```

- [ ] **Step 5: Create alert rule cell**

Notebook code cell:

```python
rain_daily_threshold = float(data["RR"].quantile(0.95))
rain_3d_threshold = float(data["rain_3d"].quantile(0.95))
wind_max_threshold = float(data["ff_x"].quantile(0.95))
wind_avg_threshold = float(data["ff_avg"].quantile(0.95))

print("Rain daily threshold:", rain_daily_threshold)
print("Rain 3-day threshold:", rain_3d_threshold)
print("Wind max threshold:", wind_max_threshold)
print("Wind avg threshold:", wind_avg_threshold)


def build_alert(row: pd.Series) -> dict:
    rain_alert = bool(row["RR"] >= rain_daily_threshold or row["rain_3d"] >= rain_3d_threshold)
    wind_alert = bool(row["ff_x"] >= wind_max_threshold or row["ff_avg"] >= wind_avg_threshold)

    if row["status"] == "NORMAL":
        alert_type = "NORMAL"
    elif rain_alert and wind_alert:
        alert_type = "CURAH_HUJAN_EKSTREM_DAN_ANGIN_KENCANG"
    elif rain_alert:
        alert_type = "CURAH_HUJAN_EKSTREM"
    elif wind_alert:
        alert_type = "ANGIN_KENCANG"
    else:
        alert_type = "ANOMALI_CUACA"

    triggers = []
    if row["status"] != "NORMAL":
        triggers.append(f"Anomaly score above {row['status']} threshold")
    if rain_alert:
        triggers.append("Rainfall above local percentile threshold")
    if wind_alert:
        triggers.append("Wind above local percentile threshold")

    return {
        "date": pd.Timestamp(row["date"]).date().isoformat(),
        "station_id": str(row["station_id"]),
        "region_name": row["region_name"],
        "status": row["status"],
        "alert_type": alert_type,
        "source": "LSTM_AUTOENCODER",
        "anomaly_score": float(row["anomaly_score"]),
        "threshold_p95": float(thresholds["p95"]),
        "threshold_p99": float(thresholds["p99"]),
        "threshold_p995": float(thresholds["p995"]),
        "rr_mm": float(row["RR"]),
        "rain_3d_mm": float(row["rain_3d"]),
        "rain_7d_mm": float(row["rain_7d"]),
        "ff_x": float(row["ff_x"]),
        "ff_avg": float(row["ff_avg"]),
        "triggers": "; ".join(triggers),
    }


alerts = pd.DataFrame([build_alert(row) for _, row in eval_meta.iterrows()])
alerts_path = REPORT_DIR / "alerts.csv"
alerts.to_csv(alerts_path, index=False)
display(alerts.head())
print("Saved:", alerts_path)
```

- [ ] **Step 6: Create top anomalies and station count cell**

Notebook code cell:

```python
top_anomalies = alerts.sort_values("anomaly_score", ascending=False).head(50).reset_index(drop=True)
station_counts = alerts.groupby(["station_id", "status", "alert_type"]).size().reset_index(name="count")

top_anomalies.to_csv(REPORT_DIR / "top_anomalies.csv", index=False)
station_counts.to_csv(REPORT_DIR / "station_anomaly_counts.csv", index=False)

display(top_anomalies.head(20))
display(station_counts)
```

- [ ] **Step 7: Create threshold sensitivity cell**

Notebook code cell:

```python
threshold_sensitivity = (
    alerts["status"]
    .value_counts()
    .reindex(["NORMAL", "WASPADA", "SIAGA", "AWAS"], fill_value=0)
    .rename_axis("status")
    .reset_index(name="count")
)
threshold_sensitivity.to_csv(REPORT_DIR / "threshold_sensitivity.csv", index=False)
display(threshold_sensitivity)
```

- [ ] **Step 8: Create reconstruction error distribution plot cell**

Notebook code cell:

```python
plt.figure(figsize=(12, 6))
sns.histplot(alerts["anomaly_score"], bins=50, kde=True)
plt.axvline(thresholds["p95"], color="orange", linestyle="--", label="P95")
plt.axvline(thresholds["p99"], color="red", linestyle="--", label="P99")
plt.axvline(thresholds["p995"], color="purple", linestyle="--", label="P99.5")
plt.title("Weighted Reconstruction Error Distribution")
plt.xlabel("Anomaly Score")
plt.ylabel("Window Count")
plt.legend()
plt.tight_layout()
plt.savefig(REPORT_DIR / "reconstruction_error_distribution.png", dpi=160)
plt.show()
```

- [ ] **Step 9: Create alert timeline plot cell**

Notebook code cell:

```python
plot_data = alerts.copy()
plot_data["date"] = pd.to_datetime(plot_data["date"])

plt.figure(figsize=(14, 6))
for station_id, group in plot_data.groupby("station_id"):
    plt.plot(group["date"], group["anomaly_score"], label=station_id, alpha=0.8)

plt.axhline(thresholds["p95"], color="orange", linestyle="--", label="P95")
plt.axhline(thresholds["p99"], color="red", linestyle="--", label="P99")
plt.axhline(thresholds["p995"], color="purple", linestyle="--", label="P99.5")
plt.title("Yogyakarta LSTM Autoencoder Anomaly Timeline")
plt.xlabel("Date")
plt.ylabel("Anomaly Score")
plt.legend()
plt.tight_layout()
plt.show()
```

- [ ] **Step 10: Execute notebook top to bottom**

Run inside Jupyter: `Kernel -> Restart Kernel and Run All Cells`.  
Expected: all report CSV/PNG files are created under `reports/`.

## Task 5: Final Notebook Verification

**Files:**
- Verify: `notebooks/01_yogyakarta_data_audit_preprocessing.ipynb`
- Verify: `notebooks/02_yogyakarta_lstm_autoencoder_training.ipynb`
- Verify: `notebooks/03_yogyakarta_anomaly_evaluation_alerts.ipynb`
- Verify: `notebooks/README.md`

**Interfaces:**
- Consumes all notebooks and outputs from prior tasks.
- Produces a rerunnable notebook workflow ready for the lecturer’s Jupyter server.

- [ ] **Step 1: Run notebooks in order on the Jupyter server**

Run:

```text
01_yogyakarta_data_audit_preprocessing.ipynb
02_yogyakarta_lstm_autoencoder_training.ipynb
03_yogyakarta_anomaly_evaluation_alerts.ipynb
```

Expected: each notebook completes from a fresh kernel without manual variable injection.

- [ ] **Step 2: Verify processed data output**

Run in a new notebook cell or terminal:

```python
from pathlib import Path
import pandas as pd

path = Path("data/processed/yogyakarta_weather_features.csv")
df = pd.read_csv(path)
print(df.shape)
print(df.columns.tolist())
print(df[["date", "station_id", "RR", "ff_x", "ff_avg"]].head())
```

Expected: file exists, contains Yogyakarta station rows, and includes rainfall and wind features.

- [ ] **Step 3: Verify artifacts**

Run in a new notebook cell or terminal:

```python
from pathlib import Path
import json

expected = [
    "artifacts/model_lstm_autoencoder.keras",
    "artifacts/scaler.pkl",
    "artifacts/threshold.json",
    "artifacts/feature_config.json",
    "artifacts/training_report.json",
]

for item in expected:
    path = Path(item)
    print(item, path.exists(), path.stat().st_size if path.exists() else 0)

print(json.loads(Path("artifacts/threshold.json").read_text()))
```

Expected: every artifact exists and `threshold.json` contains `p95`, `p99`, and `p995`.

- [ ] **Step 4: Verify reports**

Run in a new notebook cell or terminal:

```python
from pathlib import Path
import pandas as pd

for item in [
    "reports/anomaly_scores.csv",
    "reports/alerts.csv",
    "reports/top_anomalies.csv",
    "reports/station_anomaly_counts.csv",
    "reports/threshold_sensitivity.csv",
    "reports/reconstruction_error_distribution.png",
]:
    path = Path(item)
    print(item, path.exists(), path.stat().st_size if path.exists() else 0)

alerts = pd.read_csv("reports/alerts.csv")
print(alerts["alert_type"].value_counts())
print(alerts.sort_values("anomaly_score", ascending=False).head(10))
```

Expected: report files exist and alert counts are printed.

- [ ] **Step 5: Commit if a Git repository exists**

Run:

```bash
rtk git rev-parse --show-toplevel
rtk git add notebooks requirements-notebook.txt artifacts/.gitkeep reports/.gitkeep data/processed/.gitkeep docs/superpowers/specs/2026-07-21-yogyakarta-lstm-autoencoder-weather-warning-design.md docs/superpowers/plans/2026-07-21-yogyakarta-lstm-autoencoder-weather-warning.md
rtk git commit -m "feat: add yogyakarta lstm autoencoder notebooks"
```

Expected in current workspace: skip commit if not a Git repository.

## Self-Review Notes

- Spec coverage: the plan covers Yogyakarta station filtering, no flood usage, one multivariate LSTM Autoencoder, weighted reconstruction error, local thresholds, rule-based alert types, chronological splitting, station-safe windows, daily-data limitation, and dissertation-ready evaluation outputs.
- Notebook requirement: the implementation is now explicitly `.ipynb`-first, matching the lecturer’s Jupyter server environment.
- Placeholder scan: no `TBD`, `TODO`, or unspecified implementation steps should remain in this plan.
- Type consistency: notebook variables are introduced before later notebooks consume them through exported CSV/artifact files.
- Known dependency limitation: this local environment does not currently have pandas/openpyxl/TensorFlow installed; the lecturer’s Jupyter server should install or already provide the packages in `requirements-notebook.txt`.
- Known repository limitation: the current workspace is not a Git repository, so commit steps are conditional.
