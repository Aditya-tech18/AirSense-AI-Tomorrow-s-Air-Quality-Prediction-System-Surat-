"""Construct and compare independent preprocessing and regression pipelines."""

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


@dataclass(frozen=True)
class Trial:
    """Validation result for one estimator configuration."""

    name: str
    pipeline: Pipeline
    validation_mae: float


def build_candidates(rules: dict, settings: dict) -> dict[str, Pipeline]:
    """Build one baseline and two non-linear candidates with isolated transforms."""
    features = rules["features"]
    categorical = [name for name in features if name in rules["categories"]]
    numeric = [name for name in features if name not in categorical]
    seed = settings["random_seed"]
    estimators: dict[str, BaseEstimator] = {
        "median_baseline": DummyRegressor(strategy="median"),
        "random_forest": RandomForestRegressor(
            **settings["random_forest"], random_state=seed, n_jobs=1
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            **settings["hist_gradient_boosting"], random_state=seed, early_stopping=False
        ),
    }
    candidates = {}
    for name, estimator in estimators.items():
        preprocessing = ColumnTransformer([
            ("numeric", SimpleImputer(strategy="median", keep_empty_features=True), numeric),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             categorical),
        ])
        candidates[name] = Pipeline([
            ("preprocessing", preprocessing),
            ("regressor", estimator),
        ])
    return candidates


def evaluate_candidate(
    name: str,
    candidate: Pipeline,
    features: pd.DataFrame,
    labels: pd.Series,
    training: pd.Series,
    validation: pd.Series,
) -> Trial:
    """Fit only on the training window and measure MAE on the later window."""
    candidate.fit(features.loc[training], labels.loc[training])
    predictions: Any = candidate.predict(features.loc[validation])
    mae = mean_absolute_error(labels.loc[validation], predictions)
    return Trial(name, candidate, float(mae))
