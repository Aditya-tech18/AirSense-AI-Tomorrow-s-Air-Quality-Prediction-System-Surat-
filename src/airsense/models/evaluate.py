"""Time-ordered validation utilities for next-day air quality forecasts."""

import math

import pandas as pd


def chronological_masks(
    dates: pd.Series, validation_fraction: float
) -> tuple[pd.Series, pd.Series]:
    """Split by whole observation dates and purge the immediately prior day.

    Today's observations carry tomorrow's target. Purging origins on the day
    before validation keeps those training labels outside the validation
    period. All stations observed on the same date stay together.
    """
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be strictly between zero and one")
    observations = pd.to_datetime(dates, errors="raise").dt.normalize()
    if observations.isna().any():
        raise ValueError("Chronological validation requires every observation date")
    days = pd.DatetimeIndex(observations.unique()).sort_values()
    if len(days) < 3:
        raise ValueError("Chronological validation requires at least three observation dates")
    validation_days = math.ceil(len(days) * validation_fraction)
    if validation_days >= len(days):
        raise ValueError("Validation consumes every observation date")
    validation_start = days[-validation_days]
    training = observations.lt(validation_start - pd.Timedelta(days=1))
    validation = observations.ge(validation_start)
    if not training.any():
        raise ValueError("No training observations remain after the one-day purge")
    return training, validation
