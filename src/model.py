from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .config import DATASET_PATH, FEATURE_IMPORTANCE_PATH, FEATURES_PATH, MODEL_PATH


NON_FEATURE_COLS = {"won", "race_name", "driver_code", "race_id"}


def _metrics(y_true: np.ndarray, p: np.ndarray, groups: np.ndarray) -> dict[str, float]:
    ll = log_loss(y_true, np.clip(p, 1e-8, 1 - 1e-8))
    brier = brier_score_loss(y_true, p)
    g = pd.DataFrame({"group": groups, "y": y_true, "p": p})
    top1 = float(g.loc[g.groupby("group")["p"].idxmax()].y.mean())
    top3 = float(
        g.sort_values(["group", "p"], ascending=[True, False])
        .groupby("group")
        .head(3)
        .groupby("group")
        .y.max()
        .mean()
    )
    return {"log_loss": float(ll), "brier_score": float(brier), "top1_accuracy": top1, "top3_hit_rate": top3}


def _build_preprocessor(X_train: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    cat_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [c for c in X_train.columns if c not in cat_cols]

    pre = ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median"), num_cols),
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        ("oh", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            ),
        ]
    )
    return pre, num_cols, cat_cols


def train_model() -> dict[str, Any]:
    df = pd.read_csv(DATASET_PATH)
    if df.empty:
        raise RuntimeError("Historical dataset is empty. Run build-dataset first.")

    df = df.sort_values(["season", "round", "driver"]).copy()
    df["race_id"] = df["season"].astype(str) + "_" + df["round"].astype(str)

    features = [c for c in df.columns if c not in NON_FEATURE_COLS]
    if "won" not in df.columns:
        raise RuntimeError("Dataset is missing target column 'won'.")

    unique_races = df["race_id"].unique()
    if len(unique_races) < 10:
        raise RuntimeError("Not enough races for time-based validation. Need at least 10 races.")

    split_idx = int(len(unique_races) * 0.8)
    train_races = set(unique_races[:split_idx])
    train_df = df[df["race_id"].isin(train_races)]
    test_df = df[~df["race_id"].isin(train_races)]

    X_train, y_train = train_df[features], train_df["won"]
    X_test, y_test = test_df[features], test_df["won"]

    pre, _, _ = _build_preprocessor(X_train)

    baseline = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1500, class_weight="balanced"))])
    baseline.fit(X_train, y_train)

    model_used = "random_forest"
    try:
        from xgboost import XGBClassifier

        strong = Pipeline(
            [
                ("pre", pre),
                (
                    "clf",
                    XGBClassifier(
                        n_estimators=300,
                        max_depth=5,
                        learning_rate=0.05,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        eval_metric="logloss",
                    ),
                ),
            ]
        )
        model_used = "xgboost"
    except Exception:
        strong = Pipeline(
            [
                ("pre", pre),
                ("clf", RandomForestClassifier(n_estimators=400, random_state=42, class_weight="balanced")),
            ]
        )

    calibrated = CalibratedClassifierCV(strong, method="sigmoid", cv=3)
    calibrated.fit(X_train, y_train)

    p_test = calibrated.predict_proba(X_test)[:, 1]
    metrics = _metrics(y_test.values, p_test, test_df["race_id"].values)

    joblib.dump(calibrated, MODEL_PATH)
    Path(FEATURES_PATH).write_text(json.dumps(features, indent=2))

    # Use permutation importance for readable feature-level importances
    from sklearn.inspection import permutation_importance

    importance = permutation_importance(calibrated, X_test, y_test, scoring="neg_log_loss", n_repeats=5, random_state=42)
    fi = pd.DataFrame({"feature": features, "importance": importance.importances_mean}).sort_values("importance", ascending=False)
    fi.to_csv(FEATURE_IMPORTANCE_PATH, index=False)

    return {"baseline": "logistic_regression", "strong_model": model_used, "metrics": metrics}
