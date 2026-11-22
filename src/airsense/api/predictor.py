"""Load a trusted, fitted pipeline and preserve its eight-column contract."""

from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from airsense.api.schemas import FEATURES, NUMERIC_FEATURES, PredictionRequest
from airsense.models.inference import ChampionPredictor


class Model(Protocol):
    def predict(self, frame: pd.DataFrame): ...


def check_contract(model: Model) -> None:
    if not callable(getattr(model, "predict", None)):
        raise ValueError("Artifact must expose predict(DataFrame)")
    if isinstance(model, BaseEstimator):
        check_is_fitted(model)
    columns = getattr(model, "feature_names_in_", None)
    if columns is not None and tuple(columns) != FEATURES:
        raise ValueError("Artifact feature names/order do not match MP252")
    count = getattr(model, "n_features_in_", len(FEATURES))
    if count != len(FEATURES):
        raise ValueError("Artifact must accept eight MP252 features")


def load_model(path: Path) -> Model:
    # Only load artifacts produced and approved by the team: joblib uses pickle.
    return ChampionPredictor.load(path, FEATURES)


def predict_one(model: Model, request: PredictionRequest) -> float:
    frame = pd.DataFrame([request.model_dump()], columns=list(FEATURES))
    # prepare_split in the training pipeline hands off six float64 columns.
    frame = frame.astype({column: "float64" for column in NUMERIC_FEATURES})
    values = np.asarray(model.predict(frame))
    if values.shape != (1,) or values.dtype.kind not in "fiu":
        raise ValueError("Expected one numeric scalar prediction")
    prediction = float(values[0])
    if not np.isfinite(prediction):
        raise ValueError("Prediction must be finite")
    return prediction
