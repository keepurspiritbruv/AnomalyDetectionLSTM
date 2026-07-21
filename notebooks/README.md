# Yogyakarta LSTM Autoencoder Notebook Workflow

Run the notebooks in this order:

1. `01_yogyakarta_data_audit_preprocessing.ipynb`
2. `02_yogyakarta_lstm_autoencoder_training.ipynb`
3. `03_yogyakarta_anomaly_evaluation_alerts.ipynb`

Start Jupyter with `notebooks/` as the kernel working directory. Each notebook
resolves the project root as `Path("..").resolve()`.

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
- `../reports/threshold_sensitivity.csv`
- `../reports/reconstruction_error_distribution.png`

## Verification on the Jupyter server

Run all three notebooks from a fresh kernel in the listed order. Then run the
processed-data, artifact, and report verification cells in Task 5 to confirm
that the expected files exist, `threshold.json` includes `p95`, `p99`, and
`p995`, and alert counts are available.
