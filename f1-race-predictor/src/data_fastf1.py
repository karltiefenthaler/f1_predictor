from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import CACHE_DIR
from .utils import warn


def _enable_fastf1_cache(cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    import fastf1

    fastf1.Cache.enable_cache(cache_dir)


def load_historical_fastf1(seasons: Iterable[int]) -> pd.DataFrame:
    """Load race+qualifying data and return one row per driver per race."""
    _enable_fastf1_cache(CACHE_DIR)
    import fastf1

    rows = []
    for season in seasons:
        try:
            schedule = fastf1.get_event_schedule(season)
        except Exception as exc:
            warn(f"Failed to load schedule for {season}: {exc}")
            continue

        for _, event in schedule.iterrows():
            rnd = int(event.get("RoundNumber", np.nan)) if not pd.isna(event.get("RoundNumber", np.nan)) else None
            if rnd is None:
                continue
            try:
                race = fastf1.get_session(season, rnd, "R")
                race.load(telemetry=False, weather=False, messages=False)
            except Exception as exc:
                warn(f"Skipping race session {season} round {rnd}: {exc}")
                continue

            quali = None
            try:
                quali = fastf1.get_session(season, rnd, "Q")
                quali.load(telemetry=False, weather=False, messages=False)
            except Exception as exc:
                warn(f"No qualifying data for {season} round {rnd}: {exc}")

            res = race.results.copy()
            qres = quali.results.copy() if quali is not None else pd.DataFrame()

            # map qualifying position and fastest lap
            q_map = {}
            if not qres.empty and "Abbreviation" in qres:
                for _, q in qres.iterrows():
                    q_map[q.get("Abbreviation")] = {
                        "qualifying_position": q.get("Position", np.nan),
                        "fastest_quali_lap_time": q.get("Q3") or q.get("Q2") or q.get("Q1"),
                    }

            pole_time = None
            if q_map:
                q_times = [v["fastest_quali_lap_time"] for v in q_map.values() if pd.notna(v["fastest_quali_lap_time"])]
                if q_times:
                    pole_time = min(q_times)

            for _, d in res.iterrows():
                drv = d.get("Abbreviation")
                q_info = q_map.get(drv, {})
                q_time = q_info.get("fastest_quali_lap_time", np.nan)
                q_gap = (q_time - pole_time).total_seconds() if (pole_time is not None and pd.notna(q_time)) else np.nan

                status = str(d.get("Status", ""))
                dnf_flag = int("Finished" not in status)

                rows.append({
                    "season": season,
                    "round": rnd,
                    "race_name": event.get("EventName"),
                    "circuit_name": event.get("Location"),
                    "track_type": event.get("EventFormat", "Unknown"),
                    "sprint_weekend": int("sprint" in str(event.get("EventFormat", "")).lower()),
                    "driver": d.get("FullName", drv),
                    "driver_code": drv,
                    "team": d.get("TeamName"),
                    "grid_position": d.get("GridPosition", np.nan),
                    "qualifying_position": q_info.get("qualifying_position", np.nan),
                    "qualifying_gap_to_pole": q_gap,
                    "finishing_position": d.get("Position", np.nan),
                    "won": int(d.get("Position", 999) == 1),
                    "points": d.get("Points", 0.0),
                    "average_race_lap_time": np.nan,
                    "fastest_race_lap_time": np.nan,
                    "fastest_quali_lap_time": q_time.total_seconds() if pd.notna(q_time) else np.nan,
                    "number_of_pit_stops": np.nan,
                    "average_tyre_life": np.nan,
                    "dnf_flag": dnf_flag,
                })

    return pd.DataFrame(rows)
