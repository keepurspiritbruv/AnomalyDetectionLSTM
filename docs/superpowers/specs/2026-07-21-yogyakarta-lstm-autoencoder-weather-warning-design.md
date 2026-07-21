# Yogyakarta LSTM Autoencoder Weather Warning Design

Date: 2026-07-21

## Goal

Build a research-quality machine learning pipeline for detecting anomalous weather patterns in DI Yogyakarta, focused only on:

- Curah hujan ekstrem
- Angin kencang
- Kombinasi curah hujan ekstrem dan angin kencang

The first model will use historical daily climate data, especially `Dataset/datasetIDNkaggle/climate_data.csv`, filtered to DI Yogyakarta stations. The `flood` label from other datasets is out of scope and must not be used as a target, feature, or evaluation label.

## Recommended Model Strategy

Use one multivariate LSTM Autoencoder as the main anomaly detector, then use a rule-based alert engine to interpret the anomaly type.

This means the architecture is:

```text
Historical Yogyakarta climate data
        -> preprocessing
        -> feature engineering
        -> multivariate LSTM Autoencoder
        -> weighted reconstruction error
        -> Yogyakarta-specific anomaly threshold
        -> rule-based alert type classification
```

The model does not directly classify disasters. It learns normal Yogyakarta weather sequences. A high reconstruction error means the recent sequence is unusual. The rule engine then explains whether the anomaly is mainly rainfall, wind, or both.

This is preferred over separate models for rainfall and wind because rainfall and wind share atmospheric context. Humidity, temperature, sunshine duration, rainfall, wind speed, wind direction, and seasonality all interact. A shared anomaly model is simpler, more stable for the available data, and easier to defend academically.

## Data Sources

Primary training source:

- `Dataset/datasetIDNkaggle/climate_data.csv`

Supporting metadata:

- `Dataset/datasetIDNkaggle/station_detail.csv`
- `Dataset/datasetIDNkaggle/province_detail.csv`

Supplementary Yogyakarta files:

- `Dataset/Data_Cuaca_Jogja_Final_2021-Mei2026.xlsx`
- `Dataset/laporan_iklim_harian-jogja1.xlsx`

The supplementary Excel files can be used for context or future extension only if their available columns match the model feature schema. Based on initial inspection, the visible sheets mainly expose average temperature and humidity, so they should not be merged into the main hazard training table unless rainfall and wind fields are also available.

## Region Scope

The model should be trained only on DI Yogyakarta stations from the broader Indonesian climate dataset.

Known DI Yogyakarta station IDs from metadata:

| Station ID | Station Name | Region | Initial `climate_data.csv` Rows |
|---|---|---|---:|
| 96851 | Stasiun Klimatologi DI Yogyakarta | Kab. Sleman | 1195 |
| 96855 | Stasiun Geofisika Sleman | Kab. Sleman | 3750 |
| 96859 | Stasiun Meteorologi Yogyakarta | Kab. Kulon Progo | 0 found in initial scan |

The training pipeline should include DI Yogyakarta stations that have usable climate rows. Based on initial inspection, the first baseline will train on `96851` and `96855`. Station `96859` should remain in metadata handling so it can be included automatically if matching rows are added later.

The first production/research model should not train on all Indonesian stations. All-Indonesia pretraining can be evaluated later as a comparison experiment, but it is not the baseline recommendation because anomaly thresholds should reflect Yogyakarta local climate.

## Data Cleaning

The pipeline should:

1. Parse dates consistently.
2. Join climate rows with station metadata.
3. Filter to DI Yogyakarta stations.
4. Sort by `station_id` and date.
5. Remove duplicate station-date rows if any appear.
6. Convert impossible physical values to missing values.
7. Impute short missing gaps within each station using time-aware interpolation.
8. Keep missingness flags for important fields.
9. Drop rows or windows that still lack required hazard fields after cleaning.

Suggested physical sanity ranges:

| Feature | Valid Range |
|---|---:|
| `Tn` | 10 to 40 C |
| `Tx` | 15 to 45 C |
| `Tavg` | 10 to 40 C |
| `RH_avg` | 0 to 100 percent |
| `RR` | 0 to 500 mm/day |
| `ss` | 0 to 15 hours |
| `ff_x` | 0 to 60 m/s or dataset unit equivalent |
| `ff_avg` | 0 to 60 m/s or dataset unit equivalent |
| `ddd_x` | 0 to 360 degrees |

If BMKG unit documentation shows `ff_x` and `ff_avg` are knots instead of m/s, the pipeline should preserve the original unit in metadata and convert thresholds consistently.

## Feature Engineering

Base features:

- `Tn`
- `Tx`
- `Tavg`
- `RH_avg`
- `RR`
- `ss`
- `ff_x`
- `ff_avg`

Derived features:

- `temp_range = Tx - Tn`
- `rain_3d = rolling 3-day rainfall`
- `rain_7d = rolling 7-day rainfall`
- `rain_change_1d = RR[t] - RR[t-1]`
- `wind_change_1d = ff_x[t] - ff_x[t-1]`
- `ddd_x_sin = sin(ddd_x)`
- `ddd_x_cos = cos(ddd_x)`
- `day_of_year_sin`
- `day_of_year_cos`
- `station_id` encoding
- missingness flags for `RR`, `ff_x`, `ff_avg`, and `RH_avg`

The model should not use raw `ddd_car` as a numeric feature. If used at all, it should be treated as categorical metadata or replaced by the numeric direction transformation from `ddd_x`.

## Sequence Design

Because the historical dataset is daily, the LSTM window should also be daily.

Recommended first experiment:

```text
sequence_length = 30 days
prediction target = reconstruct the same 30-day multivariate sequence
```

Secondary experiments:

| Sequence Length | Purpose |
|---:|---|
| 14 days | More responsive to short abnormal periods |
| 30 days | Balanced default for rainfall and wind pattern shifts |
| 60 days | Captures longer seasonal transitions, but may smooth short hazards |

Windows must be built separately per station. A sequence must not cross from one station into another.

## Train, Validation, and Test Split

Use chronological splitting, never random splitting.

Recommended split:

```text
70 percent earliest data -> train
15 percent next data     -> validation
15 percent latest data   -> test
```

The scaler should be fitted only on the training split. Validation and test data must use the saved training scaler.

## Model Architecture

Baseline model:

```text
Input(shape = sequence_length x n_features)
LSTM(64, return_sequences=True)
Dropout(0.2)
LSTM(32, return_sequences=False)
RepeatVector(sequence_length)
LSTM(32, return_sequences=True)
Dropout(0.2)
LSTM(64, return_sequences=True)
TimeDistributed(Dense(n_features))
```

Training configuration:

| Parameter | Initial Value |
|---|---:|
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss | MSE or Huber |
| Batch size | 32 or 64 |
| Epochs | up to 200 |
| Early stopping | patience 10 to 20 |
| Validation metric | reconstruction error |

The first model should prioritize reliability and interpretability over architecture complexity.

## Anomaly Score

Use weighted reconstruction error.

```text
score = weighted mean squared reconstruction error across sequence and features
```

Suggested initial feature weights:

| Feature Group | Weight |
|---|---:|
| `RR`, `rain_3d`, `rain_7d` | 3.0 |
| `ff_x`, `ff_avg`, `wind_change_1d` | 2.5 |
| `RH_avg` | 1.5 |
| `Tavg`, `Tn`, `Tx`, `temp_range` | 1.0 |
| `ss` | 1.0 |
| wind direction sin/cos | 0.8 |
| seasonal features | 0.5 |
| missingness flags | 0.5 |

Thresholds should be derived from validation reconstruction error:

| Level | Threshold |
|---|---|
| Normal | score <= P95 |
| Waspada | score > P95 |
| Siaga | score > P99 |
| Awas | score > P99.5 |

The final threshold values must be saved in `threshold.json`.

## Alert Engine

The LSTM Autoencoder detects unusual sequences. The alert engine decides warning type using anomaly score plus physical weather rules.

Example decision logic:

```text
if anomaly_level is high and rainfall condition is high and wind condition is high:
    alert_type = CURAH_HUJAN_EKSTREM_DAN_ANGIN_KENCANG
elif anomaly_level is high and rainfall condition is high:
    alert_type = CURAH_HUJAN_EKSTREM
elif anomaly_level is high and wind condition is high:
    alert_type = ANGIN_KENCANG
elif anomaly_level is high:
    alert_type = ANOMALI_CUACA
else:
    alert_type = NORMAL
```

Rainfall conditions can use:

- Daily `RR`
- `rain_3d`
- `rain_7d`
- rainfall percentile by Yogyakarta station

Wind conditions can use:

- `ff_x`
- `ff_avg`
- `wind_change_1d`
- wind percentile by Yogyakarta station

Physical thresholds should be configurable and documented. If official BMKG warning thresholds are used later, they should replace percentile-only thresholds.

## Expected Outputs

Training artifacts:

- `artifacts/model_lstm_autoencoder.keras`
- `artifacts/scaler.pkl`
- `artifacts/threshold.json`
- `artifacts/feature_config.json`
- `artifacts/training_report.json`

Per-window inference output:

```json
{
  "date": "2020-12-14",
  "station_id": "96851",
  "region_name": "Kab. Sleman",
  "status": "SIAGA",
  "alert_type": "CURAH_HUJAN_EKSTREM",
  "source": "LSTM_AUTOENCODER",
  "anomaly_score": 1.42,
  "threshold": 1.20,
  "rr_mm": 230.5,
  "rain_3d_mm": 260.4,
  "ff_x": 8.0,
  "triggers": [
    "Anomaly score above P99 threshold",
    "Daily rainfall above local extreme threshold"
  ]
}
```

## Evaluation Plan

Because the model is unsupervised, evaluation should combine quantitative diagnostics and expert-readable case analysis.

Required evaluation:

1. Reconstruction error distribution on train, validation, and test splits.
2. Top anomaly days by station.
3. Extreme rainfall day inspection.
4. Strong wind day inspection.
5. Threshold sensitivity for P95, P99, and P99.5.
6. Comparison with rule-only baseline.
7. Station-by-station anomaly counts.
8. False alarm review for days where anomaly score is high but rainfall and wind are not high.

This gives the dissertation a defensible explanation of why the model flags certain days.

## Deployment Boundary

This first model is trained on daily historical climate data. It is appropriate for daily anomaly analysis and research validation.

It should not be presented as a minute-level realtime IoT nowcasting model unless future IoT data is collected at 5-minute or 10-minute frequency and the model is retrained on that frequency.

For website deployment later, the daily model can still produce warning records for dashboard display, but realtime IoT inference should use a model trained on data with the same sampling interval as the IoT stream.

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Supplementary Excel files have fewer columns | Do not merge them into the hazard model unless rainfall and wind columns are available |
| Missing rainfall values are ambiguous | Preserve missing flags; do not blindly convert blanks to zero |
| Broad Indonesian data weakens local sensitivity | Train the first model only on DI Yogyakarta stations |
| Extreme events are rare | Use percentile thresholds and manual inspection of top anomaly days |
| Model is hard to explain | Store feature-level reconstruction errors and rule triggers |
| Daily data differs from IoT frequency | State the limitation and train a separate IoT-frequency model later |

## Future Research Extensions

After the Yogyakarta-only baseline is complete, compare against:

- All-Indonesia pretraining followed by Yogyakarta calibration
- Separate rainfall and wind autoencoders
- Isolation Forest or One-Class SVM baseline
- Transformer autoencoder baseline
- IoT-frequency LSTM Autoencoder if realtime sensor history becomes available

These should be treated as comparison experiments, not the first baseline.
