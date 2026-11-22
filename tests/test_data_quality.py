import pandas as pd

from airsense.data.quality_checks import audit_training_data


def test_detects_all_seven_issues(training, rules):
    later = training.copy()
    holdout = later.drop(columns="pm25_next_day")
    training.loc[:6, "humidity_pct"] = 61
    training.loc[:3, "temp_c"] = 999
    training.loc[0, "day_type"] = " Weekday"
    training["pm25_today"] *= 1000
    training.loc[29, "pm25_next_day"] = 1200
    training["after_close"] = training["pm25_next_day"]
    training = pd.concat([training, training.iloc[[0]]], ignore_index=True)

    result = audit_training_data(training, later, holdout, rules)

    assert all(issue["present"] for issue in result.values())
    assert result["duplicate_rows"]["count"] == 1
    assert result["leaking_column"]["column"] == "after_close"
    assert result["unit_mixup"]["factor"] == 1000
    assert result["sentinel_value"]["count"] == 5
    assert result["stuck_sensor"]["start_date"] == "2026-01-01"
    assert result["stuck_sensor"]["end_date"] == "2026-01-07"
    assert result["target_typos"]["count"] == 1
    assert result["spelling_variants"]["true_levels"] == 1


def test_clean_data_reports_no_issues(training, rules):
    holdout = training.drop(columns="pm25_next_day")
    assert not any(
        issue["present"] for issue in
        audit_training_data(training, training, holdout, rules).values()
    )
