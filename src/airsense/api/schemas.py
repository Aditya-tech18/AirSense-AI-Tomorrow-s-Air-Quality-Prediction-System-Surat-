"""MP252 inference contract: clean inputs, without training-data repairs."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


FEATURES = (
    "pm25_today", "wind_kmph", "humidity_pct", "temp_c", "traffic_index",
    "stubble_fires", "day_type", "station_zone",
)
NUMERIC_FEATURES = FEATURES[:6]


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    pm25_today: float = Field(..., ge=7.5)
    wind_kmph: float
    humidity_pct: float
    temp_c: float
    traffic_index: float
    stubble_fires: int = Field(..., ge=0, le=750)
    day_type: Literal["weekday", "weekend", "festival"]
    station_zone: Literal["Central", "Residential", "Industrial"]


class PredictionResponse(BaseModel):
    prediction: float = Field(..., allow_inf_nan=False)
