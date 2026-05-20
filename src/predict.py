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
    path = MANUAL_DIR / "upcoming_race.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing manual input file: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise RuntimeError("upcoming_race.csv is empty.")
    return enrich_upcoming_weather(df)


def _top_factors(row: pd.Series) -> tuple[str, str]:
    positives = []
    negatives = []
    if row.get("driver_points_last_5", 0) >= 15:
        positives.append("strong_recent_points")
    if row.get("avg_quali_last_5", 99) <= 5:
        positives.append("strong_recent_qualifying")
    if row.get("upgrade_score", 0) > 0.3:
        positives.append("reported_upgrade")

    if row.get("news_risk_score", 0) > 0.4:
        negatives.append("public_news_risk")
    if row.get("car_issue_score", 0) > 0.4:
        negatives.append("car_reliability_risk")
    if row.get("penalty_score", 0) > 0.3:
        negatives.append("penalty_risk")

    return ", ".join(positives) if positives else "none", ", ".join(negatives) if negatives else "none"


def predict_next_race() -> pd.DataFrame:
    model = joblib.load(MODEL_PATH)
    feature_cols = json.loads(FEATURES_PATH.read_text())

    hist = pd.read_csv(DATASET_PATH)
    if hist.empty:
        raise RuntimeError("Historical dataset is empty. Run build-dataset and train first.")

    hist = add_rolling_features(hist)
    latest_driver = hist.sort_values(["season", "round"]).groupby("driver").tail(1)
    latest_team = hist.sort_values(["season", "round"]).groupby("team").tail(1)

    upcoming = _prepare_upcoming()
    merged = upcoming.merge(
        latest_driver[["driver", "driver_points_last_5", "avg_finish_last_5", "avg_quali_last_5", "dnf_rate_last_10"]],
        on=["driver"],
        how="left",
    ).merge(
        latest_team[["team", "team_points_last_5", "team_dnf_rate_last_10"]],
        on=["team"],
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
    merged["win_probability"] = raw / raw.sum() if raw.sum() > 0 else (1.0 / len(raw))
    merged = merged.sort_values("win_probability", ascending=False).reset_index(drop=True)
    merged["position"] = merged.index + 1

    factor_strings = merged.apply(_top_factors, axis=1)
    merged["main_positive_factors"] = factor_strings.str[0]
    merged["main_negative_factors"] = factor_strings.str[1]

    grand_prix = str(merged.get("race_name", pd.Series(["Unknown Grand Prix"])).dropna().iloc[0])

    cols = ["race_name", "position", "driver", "team", "win_probability", "main_positive_factors", "main_negative_factors"]
    if "race_name" not in merged.columns:
        merged["race_name"] = grand_prix

    out = merged[cols]
    out.to_csv(PREDICTIONS_PATH, index=False)

    print(f"\nPredicted win probabilities for: {grand_prix}\n")
    print(out.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    return out
