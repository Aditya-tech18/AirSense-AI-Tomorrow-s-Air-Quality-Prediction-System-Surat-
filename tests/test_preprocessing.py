import pandas as pd
import pytest

from airsense.data.preprocessing import prepare_split
from airsense.data.quality_checks import audit_training_data
from airsense.data.validation import validate_raw_schema


def test_repairs_are_deterministic_and_training_specific(training, rules):
    later = training.copy()
    holdout = later.drop(columns="pm25_next_day")
    training["pm25_today"] *= 1000
    training.loc[:1, "temp_c"] = 999
    training.loc[2, "temp_c"] = float("nan")
    source = training.copy()
    findings = audit_training_data(training, later, holdout, rules)

    first, labels, metadata = prepare_split(training, rules, split="train", findings=findings)
    second, second_labels, second_metadata = prepare_split(
        training, rules, split="train", findings=findings
    )
    future, _, _ = prepare_split(holdout, rules, split="holdout")
    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_series_equal(labels, second_labels)
    pd.testing.assert_frame_equal(metadata, second_metadata)
    pd.testing.assert_frame_equal(training, source)
    pd.testing.assert_series_equal(first["pm25_today"], future["pm25_today"])
    assert first["temp_c"].isna().sum() == 3
    assert future["temp_c"].notna().all()


def test_schema_flags_missing_features_and_holdout_target(training, rules):
    with pytest.raises(ValueError, match="missing columns.*wind_kmph"):
        validate_raw_schema(training.drop(columns="wind_kmph"), rules, split="train")
    with pytest.raises(ValueError, match="forbidden target"):
        validate_raw_schema(training, rules, split="holdout")


def test_unknown_category_fails_with_column_name(training, rules):
    training.loc[0, "station_zone"] = "Other"
    with pytest.raises(ValueError, match="invalid categories in station_zone"):
        prepare_split(training, rules, split="later")
