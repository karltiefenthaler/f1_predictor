from __future__ import annotations

import pandas as pd

from .config import MANUAL_DIR


NEWS_COLS = [
    "driver",
    "team",
    "news_risk_score",
    "car_issue_score",
    "upgrade_score",
    "penalty_score",
    "driver_focus_score",
    "notes",
]


def load_news_adjustments() -> pd.DataFrame:
    path = MANUAL_DIR / "news_adjustments.csv"
    if not path.exists():
        return pd.DataFrame(columns=NEWS_COLS)
    df = pd.read_csv(path)
    for col in NEWS_COLS:
        if col not in df.columns:
            df[col] = 0.0 if col != "notes" else ""
    return df[NEWS_COLS]
