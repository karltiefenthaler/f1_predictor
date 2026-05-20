from __future__ import annotations

import pandas as pd

from .config import DATASET_PATH, DEFAULT_END_SEASON, DEFAULT_START_SEASON, PROCESSED_DIR
from .data_fastf1 import load_historical_fastf1
from .features import add_rolling_features
from .utils import ensure_dirs


def build_dataset(start_season: int = DEFAULT_START_SEASON, end_season: int = DEFAULT_END_SEASON) -> pd.DataFrame:
    ensure_dirs(PROCESSED_DIR)
    seasons = list(range(start_season, end_season + 1))
    df = load_historical_fastf1(seasons)
    if df.empty:
        raise RuntimeError("No historical data loaded from FastF1.")

    df = add_rolling_features(df)
    df.to_csv(DATASET_PATH, index=False)
    return df
