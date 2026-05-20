from __future__ import annotations

import argparse

from src.build_dataset import build_dataset
from src.model import train_model
from src.predict import predict_next_race


def main() -> None:
    parser = argparse.ArgumentParser(description="F1 next race win probability predictor")
    sub = parser.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build-dataset", help="Build historical dataset from FastF1")
    b.add_argument("--start-season", type=int, default=2021)
    b.add_argument("--end-season", type=int, default=2025)

    sub.add_parser("train", help="Train model")
    sub.add_parser("predict", help="Predict next race")

    args = parser.parse_args()

    if args.command == "build-dataset":
        df = build_dataset(args.start_season, args.end_season)
        print(f"Built dataset with {len(df)} rows")
    elif args.command == "train":
        metrics = train_model()
        print("Evaluation:", metrics)
    elif args.command == "predict":
        predict_next_race()


if __name__ == "__main__":
    main()
