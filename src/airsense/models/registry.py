"""Persist the serving estimator and register the same fitted model in MLflow."""

import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import joblib
import mlflow
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.pipeline import Pipeline


def export_pipeline(pipeline: Pipeline, destination: Path) -> None:
    """Replace the serving artifact atomically after serialization succeeds."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=destination.parent, suffix=".joblib", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        joblib.dump(pipeline, temporary, compress=3)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def register_champion(
    pipeline: Pipeline, example: pd.DataFrame, name: str, destination: Path
) -> str:
    """Register a fitted pipeline and assign the champion alias."""
    signature = infer_signature(example, pipeline.predict(example))
    info = mlflow.sklearn.log_model(
        pipeline, name="model", registered_model_name=name,
        serialization_format="cloudpickle", signature=signature, input_example=example,
    )
    version = str(info.registered_model_version)
    export_pipeline(pipeline, destination)
    MlflowClient().set_registered_model_alias(name, "champion", version)
    return version
