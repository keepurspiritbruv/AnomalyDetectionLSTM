# Miniweather ML Warning Service Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a new Dockerized Python worker repository named `miniweather-ml-warning-service` that reads ScyllaDB weather history, runs the exported Yogyakarta LSTM Autoencoder rain and wind models, and posts Miniweather warning records only when live mode is enabled.

**Architecture:** The service is a polling worker, not a web API. It reads ScyllaDB directly because the current backend weather-data API has a table mapping mismatch and a raw timestamp bug, then uses the existing backend only for auth and `POST /v1/warnings`.

**Tech Stack:** Python 3.10, TensorFlow/Keras, pandas, numpy, scikit-learn/joblib, cassandra-driver, requests, python-dotenv, pytest, Docker.

## Global Constraints

- New repository/folder: `miniweather-ml-warning-service`.
- Runtime interval: every 30 minutes by default.
- Initial mode: `DRY_RUN=true`.
- Data source: ScyllaDB keyspace `hyperbase`.
- Scylla table: `records_91e2e000cb17494c8a45c6182b2a89ac`.
- Scylla collection id: `91e2e000-cb17-494c-8a45-c6182b2a89ac`.
- Device name: `DEPLOY TNTF UGM`.
- Backend base URL inside Docker network: `http://miniweather-backend:3001`.
- Rain model: `artifacts/deployment/rain_7d`.
- Wind model: `artifacts/deployment/wind_30d`.
- Backend warning type: `weather`.
- Do not commit secrets. Real passwords and admin credentials must live only in `.env`.
- Do not modify the Node.js backend, frontend dashboard, or Scylla schema in this implementation.

---

## File Structure

- Create `miniweather-ml-warning-service/app/config.py`: parse and validate environment variables.
- Create `miniweather-ml-warning-service/app/scylla_client.py`: connect to Scylla and fetch rows for a time range.
- Create `miniweather-ml-warning-service/app/features.py`: convert raw rows to daily rain/wind features.
- Create `miniweather-ml-warning-service/app/model_runner.py`: load Keras models, scalers, configs, thresholds, and compute anomaly score.
- Create `miniweather-ml-warning-service/app/alert_rules.py`: convert model score and physical weather values into warning status.
- Create `miniweather-ml-warning-service/app/backend_client.py`: login/refresh with backend and create warnings.
- Create `miniweather-ml-warning-service/app/service.py`: orchestrate one full ML warning check.
- Create `miniweather-ml-warning-service/app/main.py`: CLI entrypoint for `once` and `worker` modes.
- Create `miniweather-ml-warning-service/tests/`: focused unit tests for config, feature building, alert rules, and orchestration dry-run behavior.
- Copy `artifacts/deployment/rain_7d` and `artifacts/deployment/wind_30d` into `miniweather-ml-warning-service/artifacts/deployment/`.
- Create `miniweather-ml-warning-service/requirements.txt`, `Dockerfile`, `.env.example`, `.gitignore`, and `README.md`.

---

### Task 1: Repository Scaffold And Configuration

**Files:**
- Create: `miniweather-ml-warning-service/app/__init__.py`
- Create: `miniweather-ml-warning-service/app/config.py`
- Create: `miniweather-ml-warning-service/tests/test_config.py`
- Create: `miniweather-ml-warning-service/requirements.txt`
- Create: `miniweather-ml-warning-service/.env.example`
- Create: `miniweather-ml-warning-service/.gitignore`

**Interfaces:**
- Produces: `Settings` dataclass and `load_settings(env: Mapping[str, str] | None = None) -> Settings`.
- Later tasks consume: `Settings.scylla_table`, `Settings.scylla_collection_id`, `Settings.dry_run`, `Settings.run_interval_minutes`, alert thresholds, backend credentials.

- [ ] **Step 1: Write config tests**

Create `miniweather-ml-warning-service/tests/test_config.py`:

```python
from app.config import load_settings


def test_load_settings_defaults():
    settings = load_settings(
        {
            "SCYLLA_PASSWORD": "secret",
            "MINIWEATHER_AUTH_EMAIL": "admin@example.com",
            "MINIWEATHER_AUTH_PASSWORD": "admin-secret",
        }
    )

    assert settings.data_source == "scylla"
    assert settings.scylla_contact_points == ["10.42.28.70"]
    assert settings.scylla_port == 9042
    assert settings.scylla_keyspace == "hyperbase"
    assert settings.scylla_username == "hyperbase"
    assert settings.scylla_password == "secret"
    assert settings.scylla_table == "records_91e2e000cb17494c8a45c6182b2a89ac"
    assert settings.scylla_collection_id == "91e2e000-cb17-494c-8a45-c6182b2a89ac"
    assert settings.backend_base_url == "http://miniweather-backend:3001"
    assert settings.run_interval_minutes == 30
    assert settings.timezone == "Asia/Jakarta"
    assert settings.dry_run is True
    assert settings.rain_siaga_mm == 50.0
    assert settings.rain_awas_mm == 100.0
    assert settings.wind_siaga_ms == 10.8
    assert settings.wind_awas_ms == 17.2


def test_load_settings_boolean_false():
    settings = load_settings(
        {
            "SCYLLA_PASSWORD": "secret",
            "MINIWEATHER_AUTH_EMAIL": "admin@example.com",
            "MINIWEATHER_AUTH_PASSWORD": "admin-secret",
            "DRY_RUN": "false",
        }
    )

    assert settings.dry_run is False


def test_load_settings_multiple_contact_points():
    settings = load_settings(
        {
            "SCYLLA_CONTACT_POINTS": "10.42.28.70,10.42.28.71",
            "SCYLLA_PASSWORD": "secret",
            "MINIWEATHER_AUTH_EMAIL": "admin@example.com",
            "MINIWEATHER_AUTH_PASSWORD": "admin-secret",
        }
    )

    assert settings.scylla_contact_points == ["10.42.28.70", "10.42.28.71"]
```

- [ ] **Step 2: Run the config tests and confirm they fail**

Run:

```bash
cd miniweather-ml-warning-service
pytest tests/test_config.py -q
```

Expected: fails because `app.config` does not exist yet.

- [ ] **Step 3: Implement config loader**

Create `miniweather-ml-warning-service/app/__init__.py` as an empty file.

Create `miniweather-ml-warning-service/app/config.py`:

```python
from dataclasses import dataclass
from os import environ
from typing import Mapping


def _get(env: Mapping[str, str], key: str, default: str) -> str:
    value = env.get(key, default)
    return value.strip() if isinstance(value, str) else default


def _get_required(env: Mapping[str, str], key: str) -> str:
    value = env.get(key, "").strip()
    if not value:
        raise ValueError(f"{key} is required")
    return value


def _get_bool(env: Mapping[str, str], key: str, default: bool) -> bool:
    raw = env.get(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _get_float(env: Mapping[str, str], key: str, default: float) -> float:
    return float(_get(env, key, str(default)))


def _get_int(env: Mapping[str, str], key: str, default: int) -> int:
    return int(_get(env, key, str(default)))


@dataclass(frozen=True)
class Settings:
    data_source: str
    scylla_contact_points: list[str]
    scylla_port: int
    scylla_keyspace: str
    scylla_username: str
    scylla_password: str
    scylla_table: str
    scylla_collection_id: str
    backend_base_url: str
    miniweather_auth_email: str
    miniweather_auth_password: str
    run_interval_minutes: int
    timezone: str
    dry_run: bool
    rain_siaga_mm: float
    rain_awas_mm: float
    wind_siaga_ms: float
    wind_awas_ms: float
    stale_after_hours: int
    query_limit: int


def load_settings(env: Mapping[str, str] | None = None) -> Settings:
    source = environ if env is None else env
    contact_points = [
        item.strip()
        for item in _get(source, "SCYLLA_CONTACT_POINTS", "10.42.28.70").split(",")
        if item.strip()
    ]

    return Settings(
        data_source=_get(source, "DATA_SOURCE", "scylla"),
        scylla_contact_points=contact_points,
        scylla_port=_get_int(source, "SCYLLA_PORT", 9042),
        scylla_keyspace=_get(source, "SCYLLA_KEYSPACE", "hyperbase"),
        scylla_username=_get(source, "SCYLLA_USERNAME", "hyperbase"),
        scylla_password=_get_required(source, "SCYLLA_PASSWORD"),
        scylla_table=_get(
            source,
            "SCYLLA_TABLE",
            "records_91e2e000cb17494c8a45c6182b2a89ac",
        ),
        scylla_collection_id=_get(
            source,
            "SCYLLA_COLLECTION_ID",
            "91e2e000-cb17-494c-8a45-c6182b2a89ac",
        ),
        backend_base_url=_get(
            source,
            "MINIWEATHER_API_BASE_URL",
            "http://miniweather-backend:3001",
        ).rstrip("/"),
        miniweather_auth_email=_get_required(source, "MINIWEATHER_AUTH_EMAIL"),
        miniweather_auth_password=_get_required(source, "MINIWEATHER_AUTH_PASSWORD"),
        run_interval_minutes=_get_int(source, "RUN_INTERVAL_MINUTES", 30),
        timezone=_get(source, "TIMEZONE", "Asia/Jakarta"),
        dry_run=_get_bool(source, "DRY_RUN", True),
        rain_siaga_mm=_get_float(source, "RAIN_SIAGA_MM", 50.0),
        rain_awas_mm=_get_float(source, "RAIN_AWAS_MM", 100.0),
        wind_siaga_ms=_get_float(source, "WIND_SIAGA_MS", 10.8),
        wind_awas_ms=_get_float(source, "WIND_AWAS_MS", 17.2),
        stale_after_hours=_get_int(source, "STALE_AFTER_HOURS", 48),
        query_limit=_get_int(source, "SCYLLA_QUERY_LIMIT", 250000),
    )
```

- [ ] **Step 4: Add dependency and env files**

Create `miniweather-ml-warning-service/requirements.txt`:

```text
cassandra-driver>=3.29,<4
joblib>=1.2,<1.4
numpy>=1.23,<1.25
pandas>=1.5,<2.1
python-dotenv>=1.0,<2
requests>=2.31,<3
scikit-learn>=1.1,<1.4
tensorflow>=2.13,<2.14
pytest>=7.4,<9
```

Create `miniweather-ml-warning-service/.env.example`:

```text
DATA_SOURCE=scylla
SCYLLA_CONTACT_POINTS=10.42.28.70
SCYLLA_PORT=9042
SCYLLA_KEYSPACE=hyperbase
SCYLLA_USERNAME=hyperbase
SCYLLA_PASSWORD=
SCYLLA_TABLE=records_91e2e000cb17494c8a45c6182b2a89ac
SCYLLA_COLLECTION_ID=91e2e000-cb17-494c-8a45-c6182b2a89ac
SCYLLA_QUERY_LIMIT=250000

MINIWEATHER_API_BASE_URL=http://miniweather-backend:3001
MINIWEATHER_AUTH_EMAIL=
MINIWEATHER_AUTH_PASSWORD=

RUN_INTERVAL_MINUTES=30
TIMEZONE=Asia/Jakarta
DRY_RUN=true
STALE_AFTER_HOURS=48

RAIN_SIAGA_MM=50
RAIN_AWAS_MM=100
WIND_SIAGA_MS=10.8
WIND_AWAS_MS=17.2
```

Create `miniweather-ml-warning-service/.gitignore`:

```text
.env
.venv/
__pycache__/
.pytest_cache/
*.pyc
```

- [ ] **Step 5: Run config tests**

Run:

```bash
cd miniweather-ml-warning-service
pytest tests/test_config.py -q
```

Expected: 3 passed.

---

### Task 2: Scylla Data Reader

**Files:**
- Create: `miniweather-ml-warning-service/app/scylla_client.py`
- Create: `miniweather-ml-warning-service/tests/test_scylla_client.py`

**Interfaces:**
- Consumes: `Settings`.
- Produces: `WeatherRow` dataclass and `build_weather_query(table_name: str) -> str`.
- Produces: `ScyllaWeatherClient.fetch_rows(start_time: datetime, end_time: datetime) -> list[WeatherRow]`.

- [ ] **Step 1: Write query construction tests**

Create `miniweather-ml-warning-service/tests/test_scylla_client.py`:

```python
import pytest

from app.scylla_client import build_weather_query


def test_build_weather_query_uses_quoted_timestamp_and_allow_filtering():
    query = build_weather_query("records_91e2e000cb17494c8a45c6182b2a89ac")

    assert 'SELECT "_id", "_collection_id", "_created_by", "_updated_at"' in query
    assert "curah_hujan" in query
    assert "kecepatan_angin" in query
    assert "arah_angin" in query
    assert 'WHERE "_updated_at" >= ? AND "_updated_at" <= ?' in query
    assert "ALLOW FILTERING" in query


def test_build_weather_query_rejects_invalid_table_name():
    with pytest.raises(ValueError):
        build_weather_query("DEPLOY TNTF UGM")
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cd miniweather-ml-warning-service
pytest tests/test_scylla_client.py -q
```

Expected: fails because `app.scylla_client` does not exist yet.

- [ ] **Step 3: Implement Scylla reader**

Create `miniweather-ml-warning-service/app/scylla_client.py`:

```python
from dataclasses import dataclass
from datetime import datetime
from typing import Any
import re

from cassandra.auth import PlainTextAuthProvider
from cassandra.cluster import Cluster

from app.config import Settings


TABLE_RE = re.compile(r"^records_[a-zA-Z0-9_]+$")


@dataclass(frozen=True)
class WeatherRow:
    row_id: str
    collection_id: str
    created_by: str | None
    updated_at: datetime
    curah_hujan: float | None
    kecepatan_angin: float | None
    arah_angin: float | None


def build_weather_query(table_name: str) -> str:
    if not TABLE_RE.match(table_name):
        raise ValueError(f"Invalid Scylla table name: {table_name}")

    return f"""
        SELECT "_id", "_collection_id", "_created_by", "_updated_at",
               curah_hujan, kecepatan_angin, arah_angin
        FROM {table_name}
        WHERE "_updated_at" >= ? AND "_updated_at" <= ?
        LIMIT ?
        ALLOW FILTERING
    """


def _to_str(value: Any) -> str:
    return str(value) if value is not None else ""


def _to_float(value: Any) -> float | None:
    return float(value) if value is not None else None


class ScyllaWeatherClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        auth_provider = PlainTextAuthProvider(
            username=settings.scylla_username,
            password=settings.scylla_password,
        )
        self.cluster = Cluster(
            contact_points=settings.scylla_contact_points,
            port=settings.scylla_port,
            auth_provider=auth_provider,
        )
        self.session = self.cluster.connect(settings.scylla_keyspace)
        self.query = build_weather_query(settings.scylla_table)

    def close(self) -> None:
        self.cluster.shutdown()

    def fetch_rows(self, start_time: datetime, end_time: datetime) -> list[WeatherRow]:
        result = self.session.execute(
            self.query,
            (start_time, end_time, self.settings.query_limit),
        )
        rows: list[WeatherRow] = []
        for row in result:
            rows.append(
                WeatherRow(
                    row_id=_to_str(row._id),
                    collection_id=_to_str(row._collection_id),
                    created_by=_to_str(row._created_by) or None,
                    updated_at=row._updated_at,
                    curah_hujan=_to_float(row.curah_hujan),
                    kecepatan_angin=_to_float(row.kecepatan_angin),
                    arah_angin=_to_float(row.arah_angin),
                )
            )
        return sorted(rows, key=lambda item: item.updated_at)
```

- [ ] **Step 4: Run Scylla tests**

Run:

```bash
cd miniweather-ml-warning-service
pytest tests/test_scylla_client.py -q
```

Expected: 2 passed.

---

### Task 3: Daily Feature Builder

**Files:**
- Create: `miniweather-ml-warning-service/app/features.py`
- Create: `miniweather-ml-warning-service/tests/test_features.py`

**Interfaces:**
- Consumes: `WeatherRow`.
- Produces: `build_daily_features(rows: Sequence[WeatherRow], timezone: str) -> pandas.DataFrame`.
- Later tasks require columns `RR`, `rain_3d`, `rain_7d`, `rain_change_1d`, `missing_RR`, `ff_x`, `ff_avg`, `wind_change_1d`, `ddd_x_sin`, `ddd_x_cos`, `missing_ff_x`, `missing_ff_avg`.

- [ ] **Step 1: Write daily feature tests**

Create `miniweather-ml-warning-service/tests/test_features.py`:

```python
from datetime import datetime, timezone

import numpy as np

from app.features import build_daily_features
from app.scylla_client import WeatherRow


def row(day: int, rain: float | None, wind: float | None, direction: float | None) -> WeatherRow:
    return WeatherRow(
        row_id=f"row-{day}",
        collection_id="91e2e000-cb17-494c-8a45-c6182b2a89ac",
        created_by="0194fd5b-1768-7533-ac8a-1200c9d6748c",
        updated_at=datetime(2026, 7, day, 12, 0, tzinfo=timezone.utc),
        curah_hujan=rain,
        kecepatan_angin=wind,
        arah_angin=direction,
    )


def test_build_daily_features_creates_rain_and_wind_columns():
    rows = [
        row(1, 1.0, 2.0, 0.0),
        row(2, 3.0, 4.0, 90.0),
        row(3, None, None, None),
    ]

    df = build_daily_features(rows, timezone_name="Asia/Jakarta")

    assert list(df["RR"]) == [1.0, 3.0, 0.0]
    assert list(df["rain_3d"]) == [1.0, 4.0, 4.0]
    assert list(df["rain_change_1d"]) == [0.0, 2.0, -3.0]
    assert list(df["missing_RR"]) == [0.0, 0.0, 1.0]
    assert list(df["ff_x"]) == [2.0, 4.0, 0.0]
    assert list(df["ff_avg"]) == [2.0, 4.0, 0.0]
    assert list(df["missing_ff_x"]) == [0.0, 0.0, 1.0]
    assert np.isclose(df.loc[0, "ddd_x_sin"], 0.0)
    assert np.isclose(df.loc[0, "ddd_x_cos"], 1.0)
    assert np.isclose(df.loc[1, "ddd_x_sin"], 1.0)
    assert np.isclose(df.loc[1, "ddd_x_cos"], 0.0, atol=1e-7)
```

- [ ] **Step 2: Run feature tests and confirm they fail**

Run:

```bash
cd miniweather-ml-warning-service
pytest tests/test_features.py -q
```

Expected: fails because `app.features` does not exist yet.

- [ ] **Step 3: Implement feature builder**

Create `miniweather-ml-warning-service/app/features.py`:

```python
from collections.abc import Sequence

import numpy as np
import pandas as pd

from app.scylla_client import WeatherRow


def build_daily_features(rows: Sequence[WeatherRow], timezone_name: str) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()

    raw = pd.DataFrame(
        {
            "timestamp": [row.updated_at for row in rows],
            "curah_hujan": [row.curah_hujan for row in rows],
            "kecepatan_angin": [row.kecepatan_angin for row in rows],
            "arah_angin": [row.arah_angin for row in rows],
        }
    )
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True).dt.tz_convert(timezone_name)
    raw["date"] = raw["timestamp"].dt.date

    grouped = raw.groupby("date", as_index=False).agg(
        RR=("curah_hujan", "sum"),
        rainfall_count=("curah_hujan", "count"),
        ff_x=("kecepatan_angin", "max"),
        ff_avg=("kecepatan_angin", "mean"),
        wind_count=("kecepatan_angin", "count"),
        ddd_x=("arah_angin", "mean"),
    )

    grouped["missing_RR"] = (grouped["rainfall_count"] == 0).astype(float)
    grouped["missing_ff_x"] = (grouped["wind_count"] == 0).astype(float)
    grouped["missing_ff_avg"] = (grouped["wind_count"] == 0).astype(float)

    grouped["RR"] = grouped["RR"].fillna(0.0)
    grouped["ff_x"] = grouped["ff_x"].fillna(0.0)
    grouped["ff_avg"] = grouped["ff_avg"].fillna(0.0)
    grouped["ddd_x"] = grouped["ddd_x"].fillna(0.0)

    grouped["rain_3d"] = grouped["RR"].rolling(window=3, min_periods=1).sum()
    grouped["rain_7d"] = grouped["RR"].rolling(window=7, min_periods=1).sum()
    grouped["rain_change_1d"] = grouped["RR"].diff().fillna(0.0)
    grouped["wind_change_1d"] = grouped["ff_avg"].diff().fillna(0.0)

    radians = np.deg2rad(grouped["ddd_x"])
    grouped["ddd_x_sin"] = np.sin(radians)
    grouped["ddd_x_cos"] = np.cos(radians)

    return grouped.drop(columns=["rainfall_count", "wind_count", "ddd_x"])
```

- [ ] **Step 4: Run feature tests**

Run:

```bash
cd miniweather-ml-warning-service
pytest tests/test_features.py -q
```

Expected: 1 passed.

---

### Task 4: Model Runner

**Files:**
- Create: `miniweather-ml-warning-service/app/model_runner.py`
- Create: `miniweather-ml-warning-service/tests/test_model_runner.py`
- Copy: `artifacts/deployment/rain_7d/*` to `miniweather-ml-warning-service/artifacts/deployment/rain_7d/`
- Copy: `artifacts/deployment/wind_30d/*` to `miniweather-ml-warning-service/artifacts/deployment/wind_30d/`

**Interfaces:**
- Consumes: daily feature `DataFrame`.
- Produces: `ModelResult` dataclass and `LSTMAutoencoderRunner.score_latest(daily_df: pd.DataFrame) -> ModelResult`.

- [ ] **Step 1: Write sequence validation tests**

Create `miniweather-ml-warning-service/tests/test_model_runner.py`:

```python
import pandas as pd

from app.model_runner import has_enough_history


def test_has_enough_history():
    df = pd.DataFrame({"RR": range(7)})

    assert has_enough_history(df, 7) is True
    assert has_enough_history(df, 8) is False
```

- [ ] **Step 2: Run model runner tests and confirm they fail**

Run:

```bash
cd miniweather-ml-warning-service
pytest tests/test_model_runner.py -q
```

Expected: fails because `app.model_runner` does not exist yet.

- [ ] **Step 3: Copy model artifacts**

Copy the existing model artifacts into the new repository:

```bash
mkdir -p miniweather-ml-warning-service/artifacts/deployment
cp -R artifacts/deployment/rain_7d miniweather-ml-warning-service/artifacts/deployment/rain_7d
cp -R artifacts/deployment/wind_30d miniweather-ml-warning-service/artifacts/deployment/wind_30d
```

- [ ] **Step 4: Implement model runner**

Create `miniweather-ml-warning-service/app/model_runner.py`:

```python
from dataclasses import dataclass
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model


@dataclass(frozen=True)
class ModelResult:
    experiment_id: str
    hazard: str
    status: str
    score: float | None
    thresholds: dict[str, float]
    reason: str


def has_enough_history(daily_df: pd.DataFrame, sequence_length: int) -> bool:
    return len(daily_df) >= sequence_length


class LSTMAutoencoderRunner:
    def __init__(self, artifact_dir: Path):
        self.artifact_dir = artifact_dir
        with (artifact_dir / "feature_config.json").open("r", encoding="utf-8") as file:
            self.feature_config = json.load(file)
        with (artifact_dir / "threshold.json").open("r", encoding="utf-8") as file:
            self.thresholds = {key: float(value) for key, value in json.load(file).items()}

        self.experiment_id = self.feature_config["experiment_id"]
        self.hazard = self.feature_config["hazard"]
        self.sequence_length = int(self.feature_config["sequence_length"])
        self.model_features = list(self.feature_config["model_features"])
        self.feature_weights = np.array(
            [float(self.feature_config["feature_weights"][name]) for name in self.model_features],
            dtype=np.float32,
        )
        self.scaler = joblib.load(artifact_dir / "scaler.pkl")
        self.model = load_model(artifact_dir / "model.keras")

    def score_latest(self, daily_df: pd.DataFrame) -> ModelResult:
        if not has_enough_history(daily_df, self.sequence_length):
            return ModelResult(
                experiment_id=self.experiment_id,
                hazard=self.hazard,
                status="WAITING_FOR_HISTORY",
                score=None,
                thresholds=self.thresholds,
                reason=f"Need {self.sequence_length} daily rows, got {len(daily_df)}",
            )

        missing_features = [name for name in self.model_features if name not in daily_df.columns]
        if missing_features:
            return ModelResult(
                experiment_id=self.experiment_id,
                hazard=self.hazard,
                status="ERROR",
                score=None,
                thresholds=self.thresholds,
                reason=f"Missing features: {', '.join(missing_features)}",
            )

        latest = daily_df.tail(self.sequence_length)
        values = latest[self.model_features].astype(float).to_numpy()
        scaled = self.scaler.transform(values)
        sequence = np.expand_dims(scaled, axis=0)
        reconstructed = self.model.predict(sequence, verbose=0)
        error = np.square(sequence - reconstructed)
        weighted_error = error * self.feature_weights.reshape(1, 1, -1)
        score = float(np.mean(weighted_error))

        return ModelResult(
            experiment_id=self.experiment_id,
            hazard=self.hazard,
            status="SCORED",
            score=score,
            thresholds=self.thresholds,
            reason="OK",
        )
```

- [ ] **Step 5: Run model runner tests**

Run:

```bash
cd miniweather-ml-warning-service
pytest tests/test_model_runner.py -q
```

Expected: 1 passed.

---

### Task 5: Alert Rules

**Files:**
- Create: `miniweather-ml-warning-service/app/alert_rules.py`
- Create: `miniweather-ml-warning-service/tests/test_alert_rules.py`

**Interfaces:**
- Consumes: `ModelResult`, `Settings`, latest physical values.
- Produces: `AlertDecision` dataclass and `decide_alert(result: ModelResult, latest_values: dict[str, float], settings: Settings) -> AlertDecision`.

- [ ] **Step 1: Write alert rule tests**

Create `miniweather-ml-warning-service/tests/test_alert_rules.py`:

```python
from app.alert_rules import decide_alert
from app.config import load_settings
from app.model_runner import ModelResult


def settings():
    return load_settings(
        {
            "SCYLLA_PASSWORD": "secret",
            "MINIWEATHER_AUTH_EMAIL": "admin@example.com",
            "MINIWEATHER_AUTH_PASSWORD": "admin-secret",
        }
    )


def result(hazard: str, score: float | None) -> ModelResult:
    return ModelResult(
        experiment_id="x",
        hazard=hazard,
        status="SCORED" if score is not None else "WAITING_FOR_HISTORY",
        score=score,
        thresholds={"p95": 1.0, "p99": 2.0, "p995": 3.0},
        reason="OK",
    )


def test_rain_waspada_uses_score_only():
    decision = decide_alert(result("curah_hujan_tinggi", 1.5), {"RR": 10.0}, settings())

    assert decision.status == "WASPADA"
    assert decision.should_post is True


def test_rain_siaga_requires_physical_threshold():
    decision = decide_alert(result("curah_hujan_tinggi", 2.5), {"RR": 10.0}, settings())

    assert decision.status == "WASPADA"

    decision = decide_alert(result("curah_hujan_tinggi", 2.5), {"RR": 60.0}, settings())

    assert decision.status == "SIAGA"


def test_wind_awas_requires_score_and_physical_threshold():
    decision = decide_alert(result("angin_kencang", 4.0), {"ff_x": 5.0}, settings())

    assert decision.status == "WASPADA"

    decision = decide_alert(result("angin_kencang", 4.0), {"ff_x": 20.0}, settings())

    assert decision.status == "AWAS"


def test_waiting_history_does_not_post():
    decision = decide_alert(result("angin_kencang", None), {"ff_x": 0.0}, settings())

    assert decision.status == "WAITING_FOR_HISTORY"
    assert decision.should_post is False
```

- [ ] **Step 2: Run alert tests and confirm they fail**

Run:

```bash
cd miniweather-ml-warning-service
pytest tests/test_alert_rules.py -q
```

Expected: fails because `app.alert_rules` does not exist yet.

- [ ] **Step 3: Implement alert rules**

Create `miniweather-ml-warning-service/app/alert_rules.py`:

```python
from dataclasses import dataclass

from app.config import Settings
from app.model_runner import ModelResult


@dataclass(frozen=True)
class AlertDecision:
    hazard: str
    status: str
    should_post: bool
    message: str
    score: float | None
    reason: str


def decide_alert(
    result: ModelResult,
    latest_values: dict[str, float],
    settings: Settings,
) -> AlertDecision:
    if result.status != "SCORED" or result.score is None:
        return AlertDecision(
            hazard=result.hazard,
            status=result.status,
            should_post=False,
            message=f"{result.hazard}: {result.status} - {result.reason}",
            score=result.score,
            reason=result.reason,
        )

    p95 = result.thresholds["p95"]
    p99 = result.thresholds["p99"]
    p995 = result.thresholds["p995"]

    status = "NORMAL"
    if result.score >= p95:
        status = "WASPADA"

    if result.hazard == "curah_hujan_tinggi":
        rainfall = float(latest_values.get("RR", 0.0))
        if result.score >= p995 and rainfall >= settings.rain_awas_mm:
            status = "AWAS"
        elif result.score >= p99 and rainfall >= settings.rain_siaga_mm:
            status = "SIAGA"
        message = (
            f"ML {status} curah hujan tinggi. "
            f"Anomaly score={result.score:.3f}, curah hujan harian={rainfall:.2f} mm."
        )
    elif result.hazard == "angin_kencang":
        wind_speed = float(latest_values.get("ff_x", latest_values.get("ff_avg", 0.0)))
        if result.score >= p995 and wind_speed >= settings.wind_awas_ms:
            status = "AWAS"
        elif result.score >= p99 and wind_speed >= settings.wind_siaga_ms:
            status = "SIAGA"
        message = (
            f"ML {status} angin kencang. "
            f"Anomaly score={result.score:.3f}, kecepatan angin maksimum={wind_speed:.2f} m/s."
        )
    else:
        message = f"ML {status} {result.hazard}. Anomaly score={result.score:.3f}."

    return AlertDecision(
        hazard=result.hazard,
        status=status,
        should_post=status != "NORMAL",
        message=message,
        score=result.score,
        reason="OK",
    )
```

- [ ] **Step 4: Run alert tests**

Run:

```bash
cd miniweather-ml-warning-service
pytest tests/test_alert_rules.py -q
```

Expected: 4 passed.

---

### Task 6: Backend Client

**Files:**
- Create: `miniweather-ml-warning-service/app/backend_client.py`
- Create: `miniweather-ml-warning-service/tests/test_backend_client.py`

**Interfaces:**
- Consumes: `Settings`.
- Produces: `MiniweatherBackendClient.login() -> None`, `create_warning(message: str, warning_type: str = "weather", is_active: bool = True) -> dict`.

- [ ] **Step 1: Write backend client tests with fake session**

Create `miniweather-ml-warning-service/tests/test_backend_client.py`:

```python
from app.backend_client import MiniweatherBackendClient
from app.config import load_settings


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self):
        self.posts = []

    def post(self, url, json=None, headers=None, timeout=None):
        self.posts.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        if url.endswith("/v1/auth/login"):
            return FakeResponse(
                {
                    "data": {
                        "accessToken": "access-token",
                        "refreshToken": "refresh-token",
                    }
                }
            )
        return FakeResponse({"data": {"warning": {"id": "warning-id"}}}, status_code=201)


def settings():
    return load_settings(
        {
            "SCYLLA_PASSWORD": "secret",
            "MINIWEATHER_AUTH_EMAIL": "admin@example.com",
            "MINIWEATHER_AUTH_PASSWORD": "admin-secret",
        }
    )


def test_login_and_create_warning():
    session = FakeSession()
    client = MiniweatherBackendClient(settings(), session=session)

    created = client.create_warning("test warning")

    assert created["data"]["warning"]["id"] == "warning-id"
    assert session.posts[0]["url"] == "http://miniweather-backend:3001/v1/auth/login"
    assert session.posts[0]["json"] == {
        "email": "admin@example.com",
        "password": "admin-secret",
    }
    assert session.posts[1]["url"] == "http://miniweather-backend:3001/v1/warnings"
    assert session.posts[1]["json"] == {
        "message": "test warning",
        "type": "weather",
        "is_active": True,
    }
    assert session.posts[1]["headers"] == {"Authorization": "Bearer access-token"}
```

- [ ] **Step 2: Run backend client tests and confirm they fail**

Run:

```bash
cd miniweather-ml-warning-service
pytest tests/test_backend_client.py -q
```

Expected: fails because `app.backend_client` does not exist yet.

- [ ] **Step 3: Implement backend client**

Create `miniweather-ml-warning-service/app/backend_client.py`:

```python
from typing import Any

import requests

from app.config import Settings


class MiniweatherBackendClient:
    def __init__(self, settings: Settings, session: requests.Session | None = None):
        self.settings = settings
        self.session = session or requests.Session()
        self.access_token: str | None = None
        self.refresh_token: str | None = None

    def login(self) -> None:
        response = self.session.post(
            f"{self.settings.backend_base_url}/v1/auth/login",
            json={
                "email": self.settings.miniweather_auth_email,
                "password": self.settings.miniweather_auth_password,
            },
            timeout=20,
        )
        response.raise_for_status()
        data = response.json()["data"]
        self.access_token = data["accessToken"]
        self.refresh_token = data["refreshToken"]

    def create_warning(
        self,
        message: str,
        warning_type: str = "weather",
        is_active: bool = True,
    ) -> dict[str, Any]:
        if not self.access_token:
            self.login()

        response = self.session.post(
            f"{self.settings.backend_base_url}/v1/warnings",
            json={
                "message": message,
                "type": warning_type,
                "is_active": is_active,
            },
            headers={"Authorization": f"Bearer {self.access_token}"},
            timeout=20,
        )

        if response.status_code == 401:
            self.login()
            response = self.session.post(
                f"{self.settings.backend_base_url}/v1/warnings",
                json={
                    "message": message,
                    "type": warning_type,
                    "is_active": is_active,
                },
                headers={"Authorization": f"Bearer {self.access_token}"},
                timeout=20,
            )

        response.raise_for_status()
        return response.json()
```

- [ ] **Step 4: Run backend client tests**

Run:

```bash
cd miniweather-ml-warning-service
pytest tests/test_backend_client.py -q
```

Expected: 1 passed.

---

### Task 7: Orchestration Service And CLI

**Files:**
- Create: `miniweather-ml-warning-service/app/service.py`
- Create: `miniweather-ml-warning-service/app/main.py`
- Create: `miniweather-ml-warning-service/tests/test_service.py`

**Interfaces:**
- Consumes: `Settings`, Scylla client, model runners, backend client.
- Produces: `run_once(settings: Settings, weather_client, backend_client, runners: list[LSTMAutoencoderRunner]) -> list[AlertDecision]`.
- Produces CLI: `python -m app.main once` and `python -m app.main worker`.

- [ ] **Step 1: Write dry-run orchestration test**

Create `miniweather-ml-warning-service/tests/test_service.py`:

```python
from datetime import datetime, timedelta, timezone

from app.alert_rules import AlertDecision
from app.config import load_settings
from app.model_runner import ModelResult
from app.scylla_client import WeatherRow
from app.service import run_once


class FakeWeatherClient:
    def fetch_rows(self, start_time, end_time):
        rows = []
        for i in range(30):
            rows.append(
                WeatherRow(
                    row_id=str(i),
                    collection_id="91e2e000-cb17-494c-8a45-c6182b2a89ac",
                    created_by=None,
                    updated_at=datetime.now(timezone.utc) - timedelta(days=29 - i),
                    curah_hujan=0.0,
                    kecepatan_angin=0.0,
                    arah_angin=2.0,
                )
            )
        return rows


class FakeRunner:
    def __init__(self, hazard):
        self.hazard = hazard

    def score_latest(self, daily_df):
        return ModelResult(
            experiment_id="fake",
            hazard=self.hazard,
            status="SCORED",
            score=1.5,
            thresholds={"p95": 1.0, "p99": 2.0, "p995": 3.0},
            reason="OK",
        )


class FakeBackendClient:
    def __init__(self):
        self.created = []

    def create_warning(self, message):
        self.created.append(message)
        return {"ok": True}


def test_run_once_dry_run_does_not_post_warning():
    settings = load_settings(
        {
            "SCYLLA_PASSWORD": "secret",
            "MINIWEATHER_AUTH_EMAIL": "admin@example.com",
            "MINIWEATHER_AUTH_PASSWORD": "admin-secret",
            "DRY_RUN": "true",
        }
    )
    backend = FakeBackendClient()

    decisions = run_once(
        settings=settings,
        weather_client=FakeWeatherClient(),
        backend_client=backend,
        runners=[FakeRunner("curah_hujan_tinggi")],
    )

    assert len(decisions) == 1
    assert isinstance(decisions[0], AlertDecision)
    assert decisions[0].status == "WASPADA"
    assert backend.created == []
```

- [ ] **Step 2: Run service test and confirm it fails**

Run:

```bash
cd miniweather-ml-warning-service
pytest tests/test_service.py -q
```

Expected: fails because `app.service` does not exist yet.

- [ ] **Step 3: Implement orchestration service**

Create `miniweather-ml-warning-service/app/service.py`:

```python
from datetime import datetime, timedelta, timezone
import logging

from app.alert_rules import AlertDecision, decide_alert
from app.config import Settings
from app.features import build_daily_features

LOGGER = logging.getLogger(__name__)


def _latest_values(daily_df, hazard: str) -> dict[str, float]:
    if daily_df.empty:
        return {}
    latest = daily_df.iloc[-1]
    if hazard == "curah_hujan_tinggi":
        return {"RR": float(latest.get("RR", 0.0))}
    if hazard == "angin_kencang":
        return {
            "ff_x": float(latest.get("ff_x", 0.0)),
            "ff_avg": float(latest.get("ff_avg", 0.0)),
        }
    return {}


def run_once(settings: Settings, weather_client, backend_client, runners) -> list[AlertDecision]:
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=35)
    rows = weather_client.fetch_rows(start_time=start_time, end_time=end_time)
    daily_df = build_daily_features(rows, timezone_name=settings.timezone)

    decisions: list[AlertDecision] = []
    for runner in runners:
        result = runner.score_latest(daily_df)
        decision = decide_alert(result, _latest_values(daily_df, result.hazard), settings)
        decisions.append(decision)

        LOGGER.info(
            "hazard=%s status=%s score=%s dry_run=%s reason=%s message=%s",
            decision.hazard,
            decision.status,
            decision.score,
            settings.dry_run,
            decision.reason,
            decision.message,
        )

        if decision.should_post and not settings.dry_run:
            backend_client.create_warning(decision.message)

    return decisions
```

- [ ] **Step 4: Implement CLI**

Create `miniweather-ml-warning-service/app/main.py`:

```python
from pathlib import Path
import argparse
import logging
import time

from dotenv import load_dotenv

from app.backend_client import MiniweatherBackendClient
from app.config import load_settings
from app.model_runner import LSTMAutoencoderRunner
from app.scylla_client import ScyllaWeatherClient
from app.service import run_once


def build_runners() -> list[LSTMAutoencoderRunner]:
    root = Path(__file__).resolve().parents[1]
    return [
        LSTMAutoencoderRunner(root / "artifacts" / "deployment" / "rain_7d"),
        LSTMAutoencoderRunner(root / "artifacts" / "deployment" / "wind_30d"),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["once", "worker"], help="Run once or loop forever")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    load_dotenv()
    settings = load_settings()
    weather_client = ScyllaWeatherClient(settings)
    backend_client = MiniweatherBackendClient(settings)
    runners = build_runners()

    try:
        if args.mode == "once":
            run_once(settings, weather_client, backend_client, runners)
            return

        while True:
            run_once(settings, weather_client, backend_client, runners)
            time.sleep(settings.run_interval_minutes * 60)
    finally:
        weather_client.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run service tests**

Run:

```bash
cd miniweather-ml-warning-service
pytest tests/test_service.py -q
```

Expected: 1 passed.

---

### Task 8: Docker And Deployment Docs

**Files:**
- Create: `miniweather-ml-warning-service/Dockerfile`
- Create: `miniweather-ml-warning-service/docker-compose.example.yml`
- Create: `miniweather-ml-warning-service/README.md`

**Interfaces:**
- Produces Docker command: `python -m app.main worker`.
- Produces one-shot test command: `python -m app.main once`.

- [ ] **Step 1: Create Dockerfile**

Create `miniweather-ml-warning-service/Dockerfile`:

```dockerfile
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY artifacts ./artifacts

CMD ["python", "-m", "app.main", "worker"]
```

- [ ] **Step 2: Create Docker compose example**

Create `miniweather-ml-warning-service/docker-compose.example.yml`:

```yaml
services:
  miniweather-ml-warning-service:
    build: .
    container_name: miniweather-ml-warning-service
    restart: always
    env_file:
      - .env
    networks:
      - miniweather-net

networks:
  miniweather-net:
    external: true
```

- [ ] **Step 3: Create README**

Create `miniweather-ml-warning-service/README.md`:

```markdown
# Miniweather ML Warning Service

Python worker service for Miniweather LSTM Autoencoder anomaly detection in Yogyakarta.

## Runtime Flow

```text
ScyllaDB records table
-> daily weather features
-> rain_7d and wind_30d LSTM Autoencoder models
-> alert rules
-> dry-run logs or POST /v1/warnings
```

## Setup On VPS

Create Docker network if it does not exist:

```bash
docker network create miniweather-net
```

Connect backend container:

```bash
docker network connect miniweather-net miniweather-backend
```

Prepare env:

```bash
cp .env.example .env
nano .env
```

Build:

```bash
docker build -t miniweather-ml-warning-service .
```

Run one dry-run check:

```bash
docker run --rm --network miniweather-net --env-file .env miniweather-ml-warning-service python -m app.main once
```

Run worker:

```bash
docker run -d \
  --name miniweather-ml-warning-service \
  --restart always \
  --network miniweather-net \
  --env-file .env \
  miniweather-ml-warning-service
```

Check logs:

```bash
docker logs -f miniweather-ml-warning-service
```

## Safety

Keep `DRY_RUN=true` until logs are reviewed by the lecturer/admin.

Never commit `.env`.
```

- [ ] **Step 4: Build Docker image**

Run:

```bash
cd miniweather-ml-warning-service
docker build -t miniweather-ml-warning-service .
```

Expected: image builds successfully.

---

### Task 9: Full Verification

**Files:**
- Modify only if failures reveal implementation mistakes in previous files.

**Interfaces:**
- Verifies the whole service can run unit tests.
- Verifies CLI imports without requiring Scylla by using unit tests.

- [ ] **Step 1: Run all unit tests**

Run:

```bash
cd miniweather-ml-warning-service
pytest -q
```

Expected: all tests pass.

- [ ] **Step 2: Run local import smoke test**

Run:

```bash
cd miniweather-ml-warning-service
python -m compileall app
```

Expected: all app files compile.

- [ ] **Step 3: Check secrets are not committed**

Run:

```bash
cd miniweather-ml-warning-service
git status --short
```

Expected: `.env` does not appear unless intentionally untracked and ignored; `.env.example` appears as a normal file.

- [ ] **Step 4: VPS dry-run verification after clone**

Run on VPS after filling `.env`:

```bash
docker run --rm --network miniweather-net --env-file .env miniweather-ml-warning-service python -m app.main once
```

Expected: logs show both hazards with one of `WAITING_FOR_HISTORY`, `WAITING_FOR_RECENT_HISTORY`, `NORMAL`, `WASPADA`, `SIAGA`, or `AWAS`, and no warning is posted while `DRY_RUN=true`.
