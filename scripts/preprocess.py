from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml
from scipy.io import arff
from sklearn.model_selection import train_test_split

TARGET_COLUMN = "Class"


def read_raw_dataset(path: Path) -> pd.DataFrame:
    data, _ = arff.loadarff(path)
    df = pd.DataFrame(data)
    if TARGET_COLUMN in df.columns:
        df[TARGET_COLUMN] = normalize_target(df[TARGET_COLUMN])
    return df


def normalize_target(series: pd.Series) -> pd.Series:
    return series.apply(
        lambda value: int(value.decode("utf-8"))
        if isinstance(value, bytes)
        else int(value)
    )


def coerce_feature_types(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    feature_columns = [column for column in result.columns if column != TARGET_COLUMN]
    for column in feature_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    if result[feature_columns].isna().any().any():
        medians = result[feature_columns].median(numeric_only=True).fillna(0.0)
        result[feature_columns] = result[feature_columns].fillna(medians)
    return result


def split_dataset(
    df: pd.DataFrame,
    *,
    test_size: float,
    random_state: int,
    stratify: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        raise ValueError("Cannot split an empty dataset")
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COLUMN}")

    target = df[TARGET_COLUMN]
    stratify_values = target if stratify and target.nunique() > 1 else None
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_values,
    )
    return train_df, test_df


def load_params(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/creditcard.arff")
    parser.add_argument("--train-output", default="data/processed/train.csv")
    parser.add_argument("--test-output", default="data/processed/test.csv")
    parser.add_argument("--params", default="params.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params = load_params(Path(args.params))
    split_params = params["split"]

    df = read_raw_dataset(Path(args.input))
    df = coerce_feature_types(df)
    train_df, test_df = split_dataset(
        df,
        test_size=float(split_params["test_size"]),
        random_state=int(split_params["random_state"]),
        stratify=bool(split_params.get("stratify", True)),
    )

    train_path = Path(args.train_output)
    test_path = Path(args.test_output)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)


if __name__ == "__main__":
    main()
