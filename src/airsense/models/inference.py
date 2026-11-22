"""Load the trusted champion artifact and enforce its prediction contract."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted


class ChampionPredictor:
    """Reuse one loaded, fitted pipeline across prediction requests."""

    def __init__(self, pipeline: Pipeline, features: tuple[str, ...]) -> None:
        if not isinstance(pipeline, Pipeline):
            raise ValueError("Champion artifact must contain a fitted sklearn Pipeline")
        if not features or len(set(features)) != len(features):
            raise ValueError("Expected feature names must be nonempty and unique")
        check_is_fitted(pipeline)
        trained_features = tuple(getattr(pipeline, "feature_names_in_", ()))
        if trained_features != features:
            raise ValueError(
                f"Champion feature contract mismatch: expected {features}, "
                f"artifact has {trained_features}"
            )
        self._pipeline = pipeline
        self._features = features

    @classmethod
    def load(cls, artifact: Path, features: tuple[str, ...]) -> "ChampionPredictor":
        """Load a release artifact from a trusted source such as the team repository."""
        if not artifact.is_file():
            raise FileNotFoundError(f"Champion artifact not found: {artifact}")
        return cls(joblib.load(artifact), features)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Predict one finite value per input row in the original row order."""
        if frame.empty or frame.columns.has_duplicates:
            raise ValueError("Prediction input needs rows and unique column names")
        expected = set(self._features)
        actual = set(frame.columns)
        if actual != expected:
            raise ValueError(
                f"Prediction feature mismatch: missing={sorted(expected - actual)}, "
                f"extra={sorted(actual - expected, key=str)}"
            )
        ordered = frame.loc[:, list(self._features)]
        predictions = np.asarray(self._pipeline.predict(ordered), dtype=float)
        if predictions.shape != (len(frame),) or not np.isfinite(predictions).all():
            raise ValueError("Champion must return one finite prediction per input row")
        return predictions
