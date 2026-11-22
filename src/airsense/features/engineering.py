"""Build the ordered model feature frame shared by all data splits."""

import pandas as pd


def select_features(frame: pd.DataFrame, rules: dict) -> pd.DataFrame:
    """Return an independent feature frame with the same row index and order."""
    return frame.loc[:, rules["features"]].copy()
