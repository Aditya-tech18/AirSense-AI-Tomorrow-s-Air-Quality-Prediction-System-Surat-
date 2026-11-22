"""Apply verified training repairs and prepare consistent model inputs."""

import pandas as pd

from airsense.data.quality_checks import normalize_label, target_typo_mask
from airsense.data.validation import validate_raw_schema
from airsense.features.engineering import select_features


def prepare_split(
    frame: pd.DataFrame, rules: dict, *, split: str, findings: dict | None = None
) -> tuple[pd.DataFrame, pd.Series | None, pd.DataFrame]:
    """Return aligned features, optional labels and record metadata.

    Training repairs require findings from the raw training data. Later and
    holdout preserve every row. Missing numeric features remain available to a
    downstream imputer fitted only on the training partition.
    """
    validate_raw_schema(frame, rules, split=split)
    target = rules["target"]
    cleaned = frame.copy()
    if split == "train":
        if findings is None:
            raise ValueError("Training requires verified data issue findings")
        cleaned = cleaned.drop_duplicates().copy()
    if cleaned["record_id"].duplicated().any():
        raise ValueError(f"{split}: repeated record ID after removing exact training copies")

    for column in rules["features"]:
        if column in rules["categories"]:
            cleaned[column] = cleaned[column].map(
                lambda value: normalize_label(value, column, rules), na_action="ignore"
            )
            invalid = ~cleaned[column].isin(rules["categories"][column])
            if invalid.any():
                labels = cleaned.loc[invalid, column].unique().tolist()
                raise ValueError(f"{split}: invalid categories in {column}: {labels}")
        else:
            cleaned[column] = pd.to_numeric(cleaned[column]).astype("float64")

    labels = None
    if split != "holdout":
        labels = pd.to_numeric(cleaned[target]).astype("float64")
        if labels.isna().any():
            raise ValueError(f"{split}: missing target")
    if split == "train":
        unit = findings["unit_mixup"]
        if unit["present"]:
            affected = cleaned["station_zone"].eq(unit["group_value"])
            column = unit["column"]
            cleaned.loc[affected, column] = cleaned.loc[affected, column] / unit["factor"]
        sentinel = findings["sentinel_value"]
        if sentinel["present"]:
            column = sentinel["column"]
            cleaned[column] = cleaned[column].mask(cleaned[column].eq(sentinel["value"]))
        stuck = findings["stuck_sensor"]
        if stuck["present"]:
            dates = pd.to_datetime(cleaned["date"])
            affected = dates.between(stuck["start_date"], stuck["end_date"])
            column = stuck["column"]
            cleaned[column] = cleaned[column].mask(affected)
        if findings["target_typos"]["present"]:
            factor = rules["target_typo_factor"]
            labels = labels.mask(target_typo_mask(labels, factor), labels / factor)

    metadata = cleaned.loc[:, ["record_id", "date", "station_zone"]].copy()
    metadata["date"] = pd.to_datetime(metadata["date"], format="%Y-%m-%d")
    return select_features(cleaned, rules), labels, metadata
