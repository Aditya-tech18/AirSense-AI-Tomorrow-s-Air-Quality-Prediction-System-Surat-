import pandas as pd
import pytest

from airsense.models.evaluate import chronological_masks
from airsense.models.train import build_candidates, evaluate_candidate


def test_time_split_groups_dates_and_purges_next_day_label():
    dates = pd.Series(pd.to_datetime([
        "2026-01-05", "2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05",
    ]))
    train, validation = chronological_masks(dates, 0.2)
    assert train.tolist() == [False, True, True, True, False, False]
    assert validation.tolist() == [True, False, False, False, False, True]


def test_imputation_learns_only_from_training_rows(training, rules):
    features = training.loc[:, rules["features"]].copy()
    features.loc[25:, "temp_c"] = 10000.0
    features.loc[0, "temp_c"] = float("nan")
    original = features.copy()
    labels = training[rules["target"]]
    fit, validation = chronological_masks(training["date"], 0.2)
    settings = {"random_seed": 252, "random_forest": {}, "hist_gradient_boosting": {}}
    pipeline = build_candidates(rules, settings)["median_baseline"]

    trial = evaluate_candidate("median_baseline", pipeline, features, labels, fit, validation)

    numeric = [column for column in rules["features"] if column not in rules["categories"]]
    imputer = pipeline.named_steps["preprocessing"].named_transformers_["numeric"]
    assert imputer.statistics_[numeric.index("temp_c")] == features.loc[fit, "temp_c"].median()
    assert trial.validation_mae > 0
    pd.testing.assert_frame_equal(features, original)


def test_invalid_validation_fraction_is_rejected(training):
    with pytest.raises(ValueError, match="validation_fraction"):
        chronological_masks(training["date"], 1)


def test_champion_artifact_preserves_input_order(training, rules, tmp_path):
    import numpy as np

    from airsense.models.inference import ChampionPredictor
    from airsense.models.registry import export_pipeline

    features = training.loc[:, rules["features"]]
    labels = training[rules["target"]]
    settings = {
        "random_seed": 252,
        "random_forest": {"n_estimators": 10},
        "hist_gradient_boosting": {},
    }
    pipeline = build_candidates(rules, settings)["random_forest"].fit(features, labels)
    artifact = tmp_path / "champion.joblib"
    export_pipeline(pipeline, artifact)

    predictor = ChampionPredictor.load(artifact, tuple(rules["features"]))
    rows = features.iloc[::-1]
    shuffled_columns = rows.iloc[:, ::-1]
    np.testing.assert_allclose(predictor.predict(shuffled_columns), pipeline.predict(rows))

    with pytest.raises(ValueError, match="artifact has"):
        ChampionPredictor.load(artifact, tuple(reversed(rules["features"])))
    with pytest.raises(ValueError, match="missing=.*wind_kmph"):
        predictor.predict(rows.drop(columns="wind_kmph"))
    with pytest.raises(ValueError, match="extra=.*pm25_next_day"):
        predictor.predict(rows.assign(pm25_next_day=labels))
    with pytest.raises(FileNotFoundError, match="Champion artifact not found"):
        ChampionPredictor.load(tmp_path / "missing.joblib", tuple(rules["features"]))
