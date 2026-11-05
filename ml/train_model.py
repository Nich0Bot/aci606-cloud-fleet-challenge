"""Exercise 4 training starter.

Students must complete this script. The goal is not to copy a fixed model but to
build a reproducible training pipeline and justify the chosen model.
"""
from pathlib import Path

import pandas as pd

DATA = Path("data/training.csv")

FEATURES = [
    "speed",
    "battery",
    "motor_temp",
    "vibration",
    "motor_current",
    "obstacle_distance",
]
TARGET = "status"


def main() -> None:
    if not DATA.exists():
        raise SystemExit(
            "Training dataset not found. Run: python simulator/generate_dataset.py"
        )

    df = pd.read_csv(DATA)
    print(df.head())
    print("\nClass distribution:")
    print(df[TARGET].value_counts())

    # TODO E4
    # 1. Create train/test split (stratified).
    # 2. Compare at least two classification models.
    # 3. Report confusion matrix + macro F1.
    # 4. Select the final model with a technical justification.
    # 5. Save the full preprocessing+model pipeline under models/model.joblib.
    #
    # Important: robot_id, timestamp, x and y are intentionally NOT part of the
    # baseline feature list. You may justify adding/removing features.

    print("\nTODO: implement the training pipeline for Exercise 4.")


if __name__ == "__main__":
    main()
