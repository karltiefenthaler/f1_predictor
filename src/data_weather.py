from __future__ import annotations

import numpy as np
import pandas as pd


def enrich_upcoming_weather(df: pd.DataFrame) -> pd.DataFrame:
    """Fill weather fields from manual file values; API integration can be added later."""
    out = df.copy()
    weather_cols = ["rain_probability", "air_temperature", "track_temperature", "wind_speed"]
    for col in weather_cols:
        if col not in out.columns:
            out[col] = np.nan

    out["weather_risk_score"] = (
        out["rain_probability"].fillna(0) * 0.5
        + (out["wind_speed"].fillna(0) / 100) * 0.2
        + (out["track_temperature"].fillna(out["air_temperature"]).fillna(25).sub(30).abs() / 30) * 0.3
    ).clip(0, 1)
    return out
