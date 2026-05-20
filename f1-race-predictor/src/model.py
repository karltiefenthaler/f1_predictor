from __future__ import annotations

import json
from pathlib import Path

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


def _metrics(y_true, p, groups):
    ll = log_loss(y_true, np.clip(p, 1e-8, 1 - 1e-8))
    brier = brier_score_loss(y_true, p)
    g = pd.DataFrame({"group": groups, "y": y_true, "p": p})
    top1 = g.loc[g.groupby("group")["p"].idxmax()].y.mean()
    top3 = g.sort_values(["group", "p"], ascending=[True, False]).groupby("group").head(3).groupby("group").y.max().mean()
    return {"log_loss": ll, "brier_score": brier, "top1_accuracy": top1, "top3_hit_rate": top3}


def train_model() -> dict:
    df = pd.read_csv(DATASET_PATH)
    df["race_id"] = df["season"].astype(str) + "_" + df["round"].astype(str)

    target = "won"
    drop_cols = [target, "race_name", "driver_code", "notes"]
    features = [c for c in df.columns if c not in drop_cols]

    split_idx = int(len(df["race_id"].unique()) * 0.8)
    train_races = set(df["race_id"].unique()[:split_idx])
    train_df = df[df["race_id"].isin(train_races)]
    test_df = df[~df["race_id"].isin(train_races)]

    X_train, y_train = train_df[features], train_df[target]
    X_test, y_test = test_df[features], test_df[target]

    cat_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [c for c in features if c not in cat_cols]

    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
    ])

    baseline = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1500, class_weight="balanced"))])
    baseline.fit(X_train, y_train)

    try:
        from xgboost import XGBClassifier

        strong = Pipeline([("pre", pre), ("clf", XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, eval_metric="logloss"))])
    except Exception:
        strong = Pipeline([("pre", pre), ("clf", RandomForestClassifier(n_estimators=400, random_state=42, class_weight="balanced"))])

    calibrated = CalibratedClassifierCV(strong, method="sigmoid", cv=3)
    calibrated.fit(X_train, y_train)

    p_test = calibrated.predict_proba(X_test)[:, 1]
    metrics = _metrics(y_test.values, p_test, test_df["race_id"].values)

    joblib.dump(calibrated, MODEL_PATH)
    Path(FEATURES_PATH).write_text(json.dumps(features, indent=2))

    # Basic feature importance if available
    importances = np.zeros(len(features))
    est = calibrated.estimator.named_steps["clf"]
    if hasattr(est, "feature_importances_"):
        importances[: len(num_cols)] = est.feature_importances_[: len(num_cols)]
    elif hasattr(est, "coef_"):
        importances[: len(num_cols)] = np.abs(est.coef_[0][: len(num_cols)])

    fi = pd.DataFrame({"feature": features, "importance": importances}).sort_values("importance", ascending=False)
    fi.to_csv(FEATURE_IMPORTANCE_PATH, index=False)
    return metrics
