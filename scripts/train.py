"""Evaluate model candidates, track experiments and publish the fitted champion."""

import argparse
import hashlib
from pathlib import Path

import mlflow
import pandas as pd
import yaml
from sklearn.base import clone

from airsense.config import load_data_config, read_yaml_mapping
from airsense.data.preprocessing import prepare_split
from airsense.models.evaluate import chronological_masks
from airsense.models.registry import register_champion
from airsense.models.train import build_candidates, evaluate_candidate


def train(params_path: Path) -> None:
    """Run the training workflow from original data and verified cleaning rules."""
    root = params_path.resolve().parent
    paths, rules = load_data_config(params_path.resolve())
    params = read_yaml_mapping(params_path)
    settings = read_yaml_mapping(root / params["model"]["settings"])
    findings_path = paths["findings"]
    findings = read_yaml_mapping(findings_path)
    raw = pd.read_csv(paths["train"], dtype={"record_id": "string"})
    features, labels, metadata = prepare_split(
        raw, rules, split="train", findings=findings["data_issues"]
    )
    if labels is None:
        raise ValueError("The training dataset has no labels")
    training, validation = chronological_masks(metadata["date"], settings["validation_fraction"])
    fingerprint = hashlib.sha256(paths["train"].read_bytes()).hexdigest()

    mlflow.set_tracking_uri(f"sqlite:///{root / 'mlflow.db'}")
    mlflow.set_experiment(settings["experiment_name"])
    common = {
        "data_sha256": fingerprint,
        "training_rows": int(training.sum()),
        "validation_rows": int(validation.sum()),
        "training_last_date": metadata.loc[training, "date"].max().date().isoformat(),
        "validation_first_date": metadata.loc[validation, "date"].min().date().isoformat(),
        "random_seed": settings["random_seed"],
    }
    results = []
    for name, pipeline in build_candidates(rules, settings).items():
        with mlflow.start_run(run_name=name) as run:
            trial = evaluate_candidate(name, pipeline, features, labels, training, validation)
            mlflow.log_params({**common, "candidate": name, **settings.get(name, {})})
            mlflow.log_metric("validation_mae", trial.validation_mae)
            results.append((trial, run.info.run_id))
            print(f"{name}: validation MAE {trial.validation_mae:.4f}")

    winner, run_id = min(results, key=lambda result: (result[0].validation_mae, result[0].name))
    final_pipeline = clone(winner.pipeline).fit(features, labels)
    # Reuse the winning run so the registry and findings refer to its validation evidence.
    with mlflow.start_run(run_id=run_id):
        mlflow.log_param("refit_rows", len(features))
        version = register_champion(
            final_pipeline, features.dropna().head(5), settings["registered_model_name"],
            root / settings["artifact_path"],
        )
    findings["tracking"] = {
        "experiment_name": settings["experiment_name"],
        "best_run_id": run_id,
        "registered_model_name": settings["registered_model_name"],
        "champion_version": int(version),
    }
    temporary = findings_path.with_suffix(".yaml.tmp")
    temporary.write_text(yaml.safe_dump(findings, sort_keys=False), encoding="utf-8")
    temporary.replace(findings_path)
    print(f"Champion: {winner.name} (registry v{version}, run {run_id})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", type=Path, default=Path("params.yaml"))
    train(parser.parse_args().params)
