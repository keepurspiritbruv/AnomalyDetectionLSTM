# Yogyakarta LSTM Autoencoder Sensitivity Analysis Design

Date: 2026-07-25

## Goal

Build notebook `05_yogyakarta_sensitivity_analysis_experiments.ipynb` to test how sensitive the DI Yogyakarta LSTM Autoencoder anomaly detection result is to two modeling choices:

- Sequence window length: 7 days versus 30 days
- Feature group: rainfall only, wind only, and all weather features

The notebook is an experiment layer. It does not replace the baseline notebooks. Its purpose is to support the dissertation workflow by showing whether the baseline design is stable, too sensitive, or missing important rainfall or wind behavior.

## Research Question

The notebook should help answer:

> Which LSTM Autoencoder configuration is most reasonable for detecting anomalous curah hujan ekstrem and angin kencang patterns in DI Yogyakarta?

The expected conclusion is not only "which score is highest." Because this is unsupervised anomaly detection, the notebook should compare warning behavior, threshold behavior, top anomaly dates, and interpretability.

## Scope

Included:

- Train separate compact LSTM Autoencoder models for each experiment configuration.
- Use the processed DI Yogyakarta feature table from notebook 01.
- Use chronological train, validation, and test splitting.
- Fit scalers only on the training split for each experiment.
- Build sequences separately per station, with no sequence crossing station boundaries.
- Reject windows with non-consecutive dates.
- Compute validation percentile thresholds for each experiment.
- Generate comparison tables and plots for lecturer review.

Excluded:

- Flood prediction or flood labels.
- Hyperparameter search.
- Minute-level IoT forecasting.
- Replacing the main baseline model artifacts from notebook 02.
- Claiming disaster ground-truth accuracy, because the current workflow has no verified event-label dataset.

## Experiment Matrix

The notebook will train six independent models:

| Experiment ID | Window Length | Feature Group |
|---|---:|---|
| `rain_7d` | 7 days | rainfall only |
| `wind_7d` | 7 days | wind only |
| `all_7d` | 7 days | all weather features |
| `rain_30d` | 30 days | rainfall only |
| `wind_30d` | 30 days | wind only |
| `all_30d` | 30 days | all weather features |

Each model has its own scaler, thresholds, training history, anomaly scores, and alert summary.

## Feature Groups

### Rainfall Only

Use rainfall behavior features:

- `RR`
- `rain_3d`
- `rain_7d`
- `rain_change_1d`
- `RR_missing_flag`

This model tests whether rainfall information alone is enough to detect extreme rainfall anomalies.

### Wind Only

Use wind behavior and wind direction features:

- `ff_x`
- `ff_avg`
- `wind_change_1d`
- `ddd_x_sin`
- `ddd_x_cos`
- `ff_x_missing_flag`
- `ff_avg_missing_flag`

This model tests whether wind information alone is enough to detect angin kencang anomalies.

### All Features

Use the same multivariate feature set as notebook 02:

- Temperature features
- Humidity
- Rainfall and rolling rainfall
- Wind speed and wind direction
- Sunshine duration
- Seasonal features
- Station encoding
- Missingness flags

This model tests whether full weather context gives more stable anomaly detection than hazard-specific subsets.

## Data Flow

```text
data/processed/yogyakarta_weather_features.csv
        -> choose experiment config
        -> chronological split
        -> fit scaler on train only
        -> transform train/validation/test
        -> build station-safe consecutive sequences
        -> train compact LSTM Autoencoder
        -> compute weighted reconstruction error
        -> derive validation thresholds
        -> score validation/test windows
        -> apply alert interpretation rules
        -> export sensitivity reports
```

The notebook should preserve the raw physical columns for interpretation, while scaled features are used only as model inputs.

## Model Design

Use a compact LSTM Autoencoder so the six experiments are practical on Jupyter:

```text
Input(sequence_length, n_features)
LSTM(32, return_sequences=True)
Dropout(0.15)
LSTM(16, return_sequences=False)
RepeatVector(sequence_length)
LSTM(16, return_sequences=True)
Dropout(0.15)
LSTM(32, return_sequences=True)
TimeDistributed(Dense(n_features))
```

Training configuration:

| Parameter | Value |
|---|---:|
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss | MSE |
| Batch size | 32 |
| Max epochs | 50 |
| Early stopping patience | 8 |
| Restore best weights | True |

The lower epoch limit is intentional. Notebook 05 is for sensitivity comparison, not final model training.

## Anomaly Score

Each experiment computes reconstruction error over its selected feature group:

```text
score = weighted mean squared reconstruction error across sequence and selected features
```

Feature weights should be simpler than the baseline:

- Rainfall features: higher weight in `rain_only`
- Wind features: higher weight in `wind_only`
- Existing baseline-like weights in `all_features`

Thresholds are computed from validation scores for each experiment:

| Level | Rule |
|---|---|
| `NORMAL` | score <= P95 |
| `WASPADA` | score > P95 |
| `SIAGA` | score > P99 |
| `AWAS` | score > P99.5 |

## Alert Interpretation

The alert status comes from the experiment-specific anomaly score threshold.

The alert type is interpreted using physical rainfall and wind context:

- `CURAH_HUJAN_EKSTREM`
- `ANGIN_KENCANG`
- `CURAH_HUJAN_EKSTREM_DAN_ANGIN_KENCANG`
- `ANOMALI_CUACA`
- `NORMAL`

For `rain_only`, the notebook should not pretend to evaluate wind anomaly detection as a direct feature result. It can still report whether rainfall anomalies coincide with high wind values in the raw metadata.

For `wind_only`, the notebook should not pretend to evaluate rainfall anomaly detection as a direct feature result. It can still report whether wind anomalies coincide with high rainfall values in the raw metadata.

## Outputs

Create these report files:

- `reports/sensitivity_experiment_summary.csv`
- `reports/sensitivity_alert_counts.csv`
- `reports/sensitivity_top_anomalies.csv`
- `reports/sensitivity_score_distribution.png`
- `reports/sensitivity_status_comparison.png`

Optional experiment artifacts may be written under:

- `artifacts/sensitivity/`

The notebook should not overwrite the baseline files:

- `artifacts/model_lstm_autoencoder.keras`
- `artifacts/scaler.pkl`
- `artifacts/threshold.json`
- `reports/alerts.csv`
- `reports/anomaly_scores.csv`

## Notebook Sections

1. Introduction and experiment purpose
2. Imports, paths, and reproducibility settings
3. Load processed Yogyakarta features
4. Define experiment configurations
5. Define reusable helpers:
   - chronological split
   - feature validation
   - scaling
   - station-safe sequence building
   - compact model builder
   - weighted reconstruction error
   - alert interpretation
6. Run six experiments
7. Export result tables
8. Plot comparison charts
9. Lecturer-facing interpretation summary

## Comparison Metrics

The notebook should compare:

- Number of train, validation, and test sequences
- Best validation loss
- Epoch stopped
- P95, P99, and P99.5 thresholds
- Count of `NORMAL`, `WASPADA`, `SIAGA`, and `AWAS`
- Count of each alert type
- Top anomaly dates per configuration
- Overlap with the baseline notebook 03 alert dates when `reports/alerts.csv` exists

The baseline overlap is supporting context only. It should not be treated as accuracy.

## Expected Interpretation

The notebook should end with an Indonesian explanation suitable for the lecturer:

- 7-day windows are expected to be more responsive to short changes, but may create more noisy alerts.
- 30-day windows are expected to be more stable because they include broader weather context.
- Rainfall-only models are useful for seeing rainfall behavior, but they miss wind context.
- Wind-only models are useful for seeing wind behavior, but they miss rainfall context.
- All-features models are expected to be the strongest main-candidate approach because they consider complete weather context.

The final recommendation should likely remain:

> Use the all-features LSTM Autoencoder as the main model, with rainfall-only and wind-only sensitivity experiments as supporting evidence.

Whether the final main window should remain 30 days or move to 7 days should be decided from the notebook 05 results.

## Error Handling

The notebook should fail clearly when:

- `data/processed/yogyakarta_weather_features.csv` does not exist.
- Required columns for a feature group are missing.
- A station has too few consecutive rows for a selected window length.
- TensorFlow is not installed in the active Jupyter kernel.

For missing baseline reports, the notebook should skip baseline-overlap comparison and continue the sensitivity analysis.

## Verification

A successful run should produce:

- Six completed experiment rows in `sensitivity_experiment_summary.csv`
- Non-empty alert count rows in `sensitivity_alert_counts.csv`
- Top anomaly examples for each experiment in `sensitivity_top_anomalies.csv`
- Two comparison PNG files in `reports/`

The notebook should also display a final summary table directly inside Jupyter so the lecturer does not need to open CSV files manually.
