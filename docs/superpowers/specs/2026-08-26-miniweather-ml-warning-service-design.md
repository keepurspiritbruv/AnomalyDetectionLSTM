# Miniweather ML Warning Service Design

## Goal

Build a separate Python worker service for deploying the Yogyakarta LSTM Autoencoder anomaly detection models. The service runs every 30 minutes, reads IoT history from ScyllaDB, evaluates rain and wind anomaly models, and optionally creates warnings in the existing Miniweather backend.

The first deployment mode is dry-run, so the service logs model results without creating public warnings.

## Deployment Shape

The service will live in a new repository named `miniweather-ml-warning-service`.

It will run as a Docker container beside the existing `miniweather-backend` container on the VPS. The backend currently exposes container port `3001` on host port `9000`. For container-to-container communication, the ML service should share a Docker network with `miniweather-backend` and call:

```text
http://miniweather-backend:3001
```

## Data Source

The backend `/v1/weather-data` endpoint is not suitable as the first data source because the minute endpoint returned empty data for the target device and the raw endpoint currently fails with `updated_at` instead of `_updated_at`.

The ML service will read ScyllaDB directly for the initial deployment.

Target Scylla configuration:

```text
keyspace: hyperbase
table: records_91e2e000cb17494c8a45c6182b2a89ac
collection_id: 91e2e000-cb17-494c-8a45-c6182b2a89ac
device name: DEPLOY TNTF UGM
```

Relevant columns:

```text
_id
_collection_id
_created_by
_updated_at
curah_hujan
kecepatan_angin
arah_angin
temperature
kelembapan
tekanan_udara
```

The table primary key is:

```sql
PRIMARY KEY ("_collection_id", "_id")
```

Because `_updated_at` is not part of the primary key, the first implementation may use `ALLOW FILTERING` for a single-device 30-day query. This is acceptable for an early deployment that runs every 30 minutes, but a future production improvement should add a dedicated time-series table or materialized query path.

## Model Artifacts

The service will include the exported deployment artifacts:

```text
artifacts/deployment/rain_7d/model.keras
artifacts/deployment/rain_7d/scaler.pkl
artifacts/deployment/rain_7d/threshold.json
artifacts/deployment/rain_7d/feature_config.json

artifacts/deployment/wind_30d/model.keras
artifacts/deployment/wind_30d/scaler.pkl
artifacts/deployment/wind_30d/threshold.json
artifacts/deployment/wind_30d/feature_config.json
```

Rain thresholds:

```text
p95: 3.7326470613479468
p99: 9.169132642745971
p995: 11.549763865470867
```

Wind thresholds:

```text
p95: 1.488192993402481
p99: 3.553187344074256
p995: 3.9175432944297786
```

## Feature Mapping

Rain model input is built from `curah_hujan`:

```text
RR = daily rainfall
rain_3d = rolling 3-day rainfall sum
rain_7d = rolling 7-day rainfall sum
rain_change_1d = daily rainfall difference from previous day
missing_RR = missing rainfall flag
```

Wind model input is built from `kecepatan_angin` and `arah_angin`:

```text
ff_x = daily max wind speed
ff_avg = daily average wind speed
wind_change_1d = daily average wind speed difference from previous day
ddd_x_sin = sin(wind direction degrees)
ddd_x_cos = cos(wind direction degrees)
missing_ff_x = missing max wind speed flag
missing_ff_avg = missing average wind speed flag
```

Rows are sorted by `_updated_at` in Python, then aggregated to daily data in `Asia/Jakarta`.

## Alert Logic

The service evaluates two hazards independently:

```text
curah_hujan_tinggi -> rain_7d
angin_kencang -> wind_30d
```

Default alert status:

```text
NORMAL: score < p95
WASPADA: score >= p95
SIAGA: score >= p99 and physical threshold is met
AWAS: score >= p995 and higher physical threshold is met
```

Default physical thresholds are configurable:

```text
RAIN_SIAGA_MM=50
RAIN_AWAS_MM=100
WIND_SIAGA_MS=10.8
WIND_AWAS_MS=17.2
```

These values are deployment defaults and should be calibrated with lecturer/admin feedback after dry-run logs are reviewed.

## Warning Posting

Warnings are created through the existing backend:

```http
POST /v1/warnings
Authorization: Bearer <admin-token>
Content-Type: application/json
```

Request body:

```json
{
  "message": "ML Warning: ...",
  "type": "weather",
  "is_active": true
}
```

The backend accepts warning types `general`, `weather`, `tsunami`, `earthquake`, `volcano`, and `flood`, so the ML service will use `weather`.

The backend access token expires quickly, so the service should log in using admin credentials from environment variables and refresh/re-login when needed. Secrets must be stored only in `.env` on the VPS and must not be committed.

## Runtime States

The service must handle incomplete data safely:

```text
WAITING_FOR_HISTORY: fewer than 7 daily rows for rain or fewer than 30 daily rows for wind
WAITING_FOR_RECENT_HISTORY: data exists but newest data is too old
NORMAL: model score below p95
WASPADA/SIAGA/AWAS: alert levels based on anomaly and physical rules
ERROR: connection/query/model/posting failure
```

When `DRY_RUN=true`, the service never posts warnings. It only logs status, score, physical values, timestamp range, and reason.

## Configuration

Expected `.env` values:

```text
DATA_SOURCE=scylla
SCYLLA_CONTACT_POINTS=10.42.28.70
SCYLLA_PORT=9042
SCYLLA_KEYSPACE=hyperbase
SCYLLA_USERNAME=hyperbase
SCYLLA_PASSWORD=
SCYLLA_TABLE=records_91e2e000cb17494c8a45c6182b2a89ac
SCYLLA_COLLECTION_ID=91e2e000-cb17-494c-8a45-c6182b2a89ac

MINIWEATHER_API_BASE_URL=http://miniweather-backend:3001
MINIWEATHER_AUTH_EMAIL=
MINIWEATHER_AUTH_PASSWORD=

RUN_INTERVAL_MINUTES=30
TIMEZONE=Asia/Jakarta
DRY_RUN=true

RAIN_SIAGA_MM=50
RAIN_AWAS_MM=100
WIND_SIAGA_MS=10.8
WIND_AWAS_MS=17.2
```

## Verification Plan

Before live warning creation:

1. Build the Docker image locally or on the VPS.
2. Run a one-shot dry-run command.
3. Confirm the service can connect to Scylla.
4. Confirm it reads rows from the target table.
5. Confirm daily aggregation produces expected rain and wind columns.
6. Confirm model artifacts load.
7. Confirm rain and wind status logs are emitted.
8. Confirm `DRY_RUN=true` does not call `/v1/warnings`.
9. Test login to backend with admin credentials.
10. Test one `[TEST ML]` warning only after admin approval.
11. Enable the 30-minute worker with `DRY_RUN=true`.
12. Switch to `DRY_RUN=false` only after dry-run results are reviewed.

## Out Of Scope

This deployment does not modify the Node.js backend, the frontend dashboard, or the Scylla schema. A later backend improvement may fix `/v1/weather-data` raw query and add a proper time-series query path.
