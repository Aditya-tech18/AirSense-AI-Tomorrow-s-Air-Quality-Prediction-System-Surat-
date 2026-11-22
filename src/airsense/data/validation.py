"""Validate source schemas without overwriting the raw audit evidence."""

import pandas as pd


def validate_raw_schema(frame: pd.DataFrame, rules: dict, *, split: str) -> None:
    """Require metadata, prediction fields and the appropriate label contract."""
    if split not in {"train", "later", "holdout"}:
        raise ValueError(f"Unknown data split: {split}")
    if frame.columns.has_duplicates:
        raise ValueError(f"{split}: duplicate column names")
    target = rules["target"]
    required = {"record_id", "date", *rules["features"]}
    if split != "holdout":
        required.add(target)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{split}: missing columns {missing}")
    if split == "holdout" and target in frame:
        raise ValueError(f"{split}: forbidden target column {target}")
    if frame.empty:
        raise ValueError(f"{split}: no records")
    ids = frame["record_id"].astype("string")
    if ids.isna().any() or ids.str.strip().eq("").any():
        raise ValueError(f"{split}: missing record ID")
    dates = pd.to_datetime(frame["date"], format="%Y-%m-%d", errors="coerce")
    if dates.isna().any():
        raise ValueError(f"{split}: missing or invalid date")
    for column in required.difference({"record_id", "date", *rules["categories"]}):
        numeric = pd.to_numeric(frame[column], errors="coerce")
        invalid = (numeric.isna() & frame[column].notna()) | numeric.isin(
            [float("inf"), float("-inf")]
        )
        if invalid.any():
            raise ValueError(f"{split}: non-finite or non-numeric value in {column}")
