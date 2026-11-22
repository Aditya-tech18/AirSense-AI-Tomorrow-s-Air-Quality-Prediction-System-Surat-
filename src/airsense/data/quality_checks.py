"""Detect the seven handbook issues from raw frames and clean references."""

from itertools import groupby

import pandas as pd

from airsense.data.validation import validate_raw_schema


def normalize_label(value: str, column: str, rules: dict) -> str:
    """Map a case, whitespace or configured spelling variant to a canonical value."""
    key = value.strip().casefold()
    canonical = {label.casefold(): label for label in rules["categories"][column]}
    return rules["aliases"].get(column, {}).get(key, canonical.get(key, value))


def _unique_result(candidates: list[dict], **empty: None) -> dict:
    if len(candidates) > 1:
        raise ValueError(f"Multiple candidates for a single findings entry: {candidates}")
    return {"present": bool(candidates), **(candidates[0] if candidates else empty)}


def _unit_candidates(train: pd.DataFrame, later: pd.DataFrame, rules: dict) -> list[dict]:
    """Compare group medians and require a candidate factor to explain the shift."""
    candidates = []
    for column in _numeric_features(rules):
        for zone, values in train.groupby("station_zone")[column]:
            reference = later.loc[later["station_zone"].eq(zone), column].median()
            observed = values.median()
            if pd.isna(reference) or reference <= 0 or observed <= 0:
                continue
            for factor in rules["unit_factors"]:
                ratio = observed / (reference * factor)
                if 0.75 <= ratio <= 1.25:
                    candidates.append({"column": column, "group_value": zone, "factor": factor})
    return candidates


def _numeric_features(rules: dict) -> list[str]:
    return [column for column in rules["features"] if column not in rules["categories"]]


def _stuck_candidates(train: pd.DataFrame, rules: dict) -> list[dict]:
    dates = pd.to_datetime(train["date"], format="%Y-%m-%d")
    candidates = []
    for column in _numeric_features(rules):
        numeric = pd.to_numeric(train[column])
        daily = numeric.groupby(dates).agg(["size", "count", "nunique", "first"])
        daily = daily.loc[daily["size"].eq(daily["count"]) & daily["nunique"].eq(1)]
        days = daily.index
        runs = zip(days, daily["first"], days.to_series().diff().ne(pd.Timedelta(days=1)).cumsum())
        for _, group in groupby(runs, key=lambda item: (item[1], item[2])):
            run = list(group)
            if len(run) >= rules["minimum_stuck_days"]:
                candidates.append({
                    "column": column,
                    "start_date": run[0][0].date().isoformat(),
                    "end_date": run[-1][0].date().isoformat(),
                })
    return candidates


def target_typo_mask(values: pd.Series, factor: int) -> pd.Series:
    """Flag an isolated high tail whose corrected values fit the main range."""
    numeric = pd.to_numeric(values)
    ordered = numeric.dropna().sort_values().reset_index(drop=True)
    for position in ordered.diff().dropna().nlargest(5).index:
        normal = ordered.iloc[:position]
        tail = ordered.iloc[position:]
        if len(tail) > max(1, len(ordered) * 0.05) or len(normal) < 2:
            continue
        if tail.iloc[0] <= 2 * normal.iloc[-1]:
            continue
        if (tail / factor).between(normal.iloc[0], normal.iloc[-1]).all():
            return numeric.ge(tail.iloc[0])
    return pd.Series(False, index=values.index)


def audit_training_data(
    train: pd.DataFrame, later: pd.DataFrame, holdout: pd.DataFrame, rules: dict
) -> dict:
    """Return diagnostics in the MP252 findings schema without changing inputs."""
    for split, frame in (("train", train), ("later", later), ("holdout", holdout)):
        validate_raw_schema(frame, rules, split=split)
    target = rules["target"]
    leak = []
    for column in sorted(set(train.columns).difference(holdout.columns, {target})):
        candidate = pd.to_numeric(train[column], errors="coerce")
        paired = pd.concat([candidate, pd.to_numeric(train[target])], axis=1).dropna()
        if len(paired) >= max(3, len(train) * 0.9):
            if abs(paired.iloc[:, 0].corr(paired.iloc[:, 1])) >= 0.99:
                leak.append({"column": column})
    sentinels = []
    for column in _numeric_features(rules):
        baseline = pd.to_numeric(later[column]).dropna()
        counts = pd.to_numeric(train[column]).value_counts()
        for value in rules["sentinel_candidates"]:
            outside = len(baseline) and not baseline.min() <= value <= baseline.max()
            if outside and counts.get(value, 0) >= 2:
                sentinels.append({"column": column, "value": value, "count": int(counts[value])})
    spelling = []
    for column in rules["categories"]:
        clean = set(later[column].dropna())
        for value in set(train[column].dropna()).difference(clean):
            if normalize_label(value, column, rules) not in clean:
                raise ValueError(f"Unmapped training category in {column}: {value!r}")
        if set(train[column].dropna()).difference(clean):
            spelling.append({"column": column, "true_levels": len(clean)})
    duplicate_count = int(train.duplicated().sum())
    typo_count = int(target_typo_mask(train[target], rules["target_typo_factor"]).sum())
    return {
        "leaking_column": _unique_result(leak, column=None),
        "unit_mixup": _unique_result(
            _unit_candidates(train, later, rules), column=None, group_value=None, factor=None
        ),
        "sentinel_value": _unique_result(sentinels, column=None, value=None, count=None),
        "duplicate_rows": {"present": bool(duplicate_count), "count": duplicate_count or None},
        "stuck_sensor": _unique_result(
            _stuck_candidates(train, rules), column=None, start_date=None, end_date=None
        ),
        "target_typos": {"present": bool(typo_count), "count": typo_count or None},
        "spelling_variants": _unique_result(spelling, column=None, true_levels=None),
    }
