from pathlib import Path

import pandas as pd
import pytest

from airsense.config import load_data_config


@pytest.fixture
def rules():
    return load_data_config(Path(__file__).resolve().parents[1] / "params.yaml")[1]


@pytest.fixture
def training():
    dates = pd.date_range("2026-01-01", periods=30).strftime("%Y-%m-%d")
    return pd.DataFrame({
        "record_id": [f"r{index}" for index in range(len(dates))],
        "date": dates,
        "pm25_today": [100.0 + index for index in range(len(dates))],
        "wind_kmph": [10.0 + index for index in range(len(dates))],
        "humidity_pct": [40.0 + index for index in range(len(dates))],
        "temp_c": [10.0 + index for index in range(len(dates))],
        "traffic_index": [20.0 + index for index in range(len(dates))],
        "stubble_fires": [30.0 + index for index in range(len(dates))],
        "day_type": ["weekday"] * len(dates),
        "station_zone": ["Industrial"] * len(dates),
        "pm25_next_day": [100.0 + index for index in range(len(dates))],
    })
