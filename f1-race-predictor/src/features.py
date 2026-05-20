from __future__ import annotations

import numpy as np
import pandas as pd


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["season", "round", "driver"]).copy()

    out["driver_points_last_5"] = out.groupby("driver")["points"].transform(lambda s: s.shift().rolling(5, min_periods=1).mean())
    out["avg_finish_last_5"] = out.groupby("driver")["finishing_position"].transform(lambda s: s.shift().rolling(5, min_periods=1).mean())
    out["avg_quali_last_5"] = out.groupby("driver")["qualifying_position"].transform(lambda s: s.shift().rolling(5, min_periods=1).mean())
    out["dnf_rate_last_10"] = out.groupby("driver")["dnf_flag"].transform(lambda s: s.shift().rolling(10, min_periods=1).mean())

    out["team_points_last_5"] = out.groupby("team")["points"].transform(lambda s: s.shift().rolling(5, min_periods=1).mean())
    out["team_dnf_rate_last_10"] = out.groupby("team")["dnf_flag"].transform(lambda s: s.shift().rolling(10, min_periods=1).mean())

    out["circuit_driver_avg_finish"] = out.groupby(["driver", "circuit_name"])["finishing_position"].transform(lambda s: s.shift().expanding().mean())
    out["circuit_team_avg_finish"] = out.groupby(["team", "circuit_name"])["finishing_position"].transform(lambda s: s.shift().expanding().mean())

    num_cols = out.select_dtypes(include=[np.number]).columns
    out[num_cols] = out[num_cols].fillna(out[num_cols].median())
    return out
