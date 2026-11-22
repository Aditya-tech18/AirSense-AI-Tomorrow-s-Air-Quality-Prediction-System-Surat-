"""Load the repository's data paths and quality settings."""

from pathlib import Path
from typing import Any

import yaml


def read_yaml_mapping(path: Path) -> dict[str, Any]:
    """Parse a YAML file and require a mapping at its root."""
    with path.open(encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    if not isinstance(document, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return document


def load_data_config(params: Path) -> tuple[dict[str, Path], dict[str, Any]]:
    """Resolve data paths relative to params.yaml and load the quality rules."""
    data = read_yaml_mapping(params).get("data")
    required = {"train", "later", "holdout", "quality", "findings"}
    if not isinstance(data, dict) or any(
        not isinstance(data.get(key), str) or not data[key] for key in required
    ):
        raise ValueError(f"{params} must define paths under data for {sorted(required)}")
    paths = {key: params.parent / data[key] for key in required}
    rules = read_yaml_mapping(paths["quality"])
    expected = {
        "features": list, "target": str, "categories": dict, "aliases": dict,
        "sentinel_candidates": list, "unit_factors": list,
        "minimum_stuck_days": int, "target_typo_factor": int,
    }
    for key, kind in expected.items():
        if not isinstance(rules.get(key), kind):
            raise ValueError(f"{paths['quality']}: {key} must be a {kind.__name__}")
    features = rules["features"]
    if (
        not features or any(not isinstance(name, str) for name in features)
        or len(set(features)) != len(features)
        or set(features).intersection({"record_id", "date", rules["target"]})
        or not set(rules["categories"]).issubset(features)
        or rules["minimum_stuck_days"] < 2 or rules["target_typo_factor"] <= 1
    ):
        raise ValueError(f"{paths['quality']}: inconsistent feature or detection settings")
    return paths, rules
