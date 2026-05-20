from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MANUAL_DIR = DATA_DIR / "manual"
CACHE_DIR = BASE_DIR / "cache"
MODEL_PATH = PROCESSED_DIR / "model.joblib"
FEATURES_PATH = PROCESSED_DIR / "model_features.json"
DATASET_PATH = PROCESSED_DIR / "historical_dataset.csv"
FEATURE_IMPORTANCE_PATH = PROCESSED_DIR / "feature_importance.csv"
PREDICTIONS_PATH = PROCESSED_DIR / "next_race_predictions.csv"

DEFAULT_START_SEASON = 2021
DEFAULT_END_SEASON = 2025
