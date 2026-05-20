from __future__ import annotations

import json
import joblib
import numpy as np
import pandas as pd

from .config import DATASET_PATH, FEATURES_PATH, MANUAL_DIR, MODEL_PATH, PREDICTIONS_PATH
from .data_news import load_news_adjustments
from .data_weather import enrich_upcoming_weather
from .features import add_rolling_features


def _prepare_upcoming() -> pd.DataFrame:
    df = pd.read_csv(MANUAL_DIR / "upcoming_race.csv")
    return enrich_upcoming_weather(df)


def predict_next_race() -> pd.DataFrame:
    model = joblib.load(MODEL_PATH)
    feature_cols = json.loads(FEATURES_PATH.read_text())

    hist = pd.read_csv(DATASET_PATH)
    hist = add_rolling_features(hist)
    latest = hist.sort_values(["season", "round"]).groupby("driver").tail(1)

    upcoming = _prepare_upcoming()
    merged = upcoming.merge(
        latest[["driver", "team", "driver_points_last_5", "team_points_last_5", "avg_finish_last_5", "avg_quali_last_5", "dnf_rate_last_10", "team_dnf_rate_last_10"]],
        on=["driver", "team"],
        how="left",
    )

    news = load_news_adjustments()
    merged = merged.merge(news, on=["driver", "team"], how="left")
    for c in ["news_risk_score", "car_issue_score", "upgrade_score", "penalty_score", "driver_focus_score"]:
        merged[c] = merged[c].fillna(0.0)

    for col in feature_cols:
        if col not in merged.columns:
            merged[col] = np.nan

    raw = model.predict_proba(merged[feature_cols])[:, 1]
    merged["raw_win_probability"] = raw
    merged["win_probability"] = raw / raw.sum() if raw.sum() > 0 else 1 / len(raw)
    merged = merged.sort_values("win_probability", ascending=False).reset_index(drop=True)
    merged["position"] = merged.index + 1

    merged["main_positive_factors"] = merged.apply(lambda r: "strong_form" if r.get("driver_points_last_5", 0) > 10 else "neutral", axis=1)
    merged["main_negative_factors"] = merged.apply(lambda r: "high_news_risk" if r.get("news_risk_score", 0) > 0.5 else "none", axis=1)

    cols = ["position", "driver", "team", "win_probability", "main_positive_factors", "main_negative_factors"]
    out = merged[cols]
    out.to_csv(PREDICTIONS_PATH, index=False)
    print(out.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    return out
