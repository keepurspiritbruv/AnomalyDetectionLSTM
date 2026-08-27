# Yogyakarta LSTM Autoencoder Notebook Workflow

Run the notebooks in this order:

1. `01_yogyakarta_data_audit_preprocessing.ipynb`
2. `02_yogyakarta_lstm_autoencoder_training.ipynb`
3. `03_yogyakarta_anomaly_evaluation_alerts.ipynb`
4. `04_yogyakarta_report_dashboard.ipynb`
5. `05_yogyakarta_sensitivity_analysis_experiments.ipynb`
6. `06_yogyakarta_export_deployment_models.ipynb`

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

Use `04_yogyakarta_report_dashboard.ipynb` to inspect the generated reports as tables and charts without opening CSV files directly.
- `../reports/threshold_sensitivity.csv`
- `../reports/reconstruction_error_distribution.png`

Use `05_yogyakarta_sensitivity_analysis_experiments.ipynb` to compare 7-day versus 30-day windows and rainfall-only, wind-only, and all-features LSTM Autoencoder experiments.
- `../reports/sensitivity_experiment_summary.csv`
- `../reports/sensitivity_alert_counts.csv`
- `../reports/sensitivity_top_anomalies.csv`
- `../reports/sensitivity_status_comparison.png`
- `../reports/sensitivity_score_distribution.png`

Use `06_yogyakarta_export_deployment_models.ipynb` to export selected deployment models.
- `../artifacts/deployment/rain_7d/model.keras`
- `../artifacts/deployment/rain_7d/scaler.pkl`
- `../artifacts/deployment/rain_7d/threshold.json`
- `../artifacts/deployment/rain_7d/feature_config.json`
- `../artifacts/deployment/wind_30d/model.keras`
- `../artifacts/deployment/wind_30d/scaler.pkl`
- `../artifacts/deployment/wind_30d/threshold.json`
- `../artifacts/deployment/wind_30d/feature_config.json`
- `../reports/deployment_model_export_summary.csv`

## Verification on the Jupyter server

Run the notebooks from a fresh kernel in the listed order. Then run the
verification cells at the end of each notebook to confirm that the expected
processed data, artifacts, reports, and sensitivity outputs exist.
