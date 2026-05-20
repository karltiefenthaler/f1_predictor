# F1 Race Predictor

A beginner-friendly machine learning portfolio project that predicts **win probabilities for each driver** in the next Formula 1 Grand Prix.

## What it does
- Builds a historical driver-race dataset from FastF1 (default seasons: 2021-2025).
- Trains:
  - Baseline: Logistic Regression
  - Stronger model: XGBoost (if installed) or Random Forest fallback
- Evaluates using time-based validation:
  - log loss
  - Brier score
  - top-1 accuracy
  - top-3 hit rate
- Predicts next-race win probabilities for all entered drivers.
- Normalises probabilities so all driver probabilities sum to 100%.
- Outputs feature importance and prediction CSV files.

## Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run
```bash
python main.py build-dataset
python main.py train
python main.py predict
```

## FastF1 data usage
- Historical race and qualifying data are loaded via FastF1.
- The cache directory is created before `fastf1.Cache.enable_cache()`.
- If a session fails to load, the code warns and skips it rather than crashing.

## Weather features
- Weather can be manually provided in `data/manual/upcoming_race.csv`.
- Current implementation computes a derived `weather_risk_score` from the weather inputs.
- You can later replace this with a live weather API integration.

## Public news / external factors
- Add only publicly reported or manually known factors in:
  - `data/manual/news_adjustments.csv`
- These factors are merged into upcoming-race prediction features.
- Do not infer private information or scrape personal/private data.

## Why probabilities are normalized
Binary classifiers produce independent win likelihoods per driver. Since only one winner is possible, probabilities are normalised so all drivers sum to 1.0 (100%), giving a race-level probability distribution.

## Limitations
- Some advanced telemetry-derived features are placeholders (`NaN`) in v1.
- Quality depends on FastF1 data completeness and manual input quality.
- External factors are only as good as manual entries.
- Model predicts winner probability, not full finishing order.
