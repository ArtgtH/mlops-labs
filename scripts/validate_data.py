from __future__ import annotations

import argparse
import json
from pathlib import Path

import great_expectations as ge
import pandas as pd

TARGET_COLUMN = "Class"


def validate_frame(df: pd.DataFrame, dataset_name: str) -> dict:
    dataset = ge.dataset.PandasDataset(df)
    dataset.expect_table_row_count_to_be_between(min_value=1)
    dataset.expect_column_to_exist(TARGET_COLUMN)
    dataset.expect_column_values_to_be_in_set(TARGET_COLUMN, [0, 1])
    dataset.expect_column_values_to_not_be_null(TARGET_COLUMN)
    dataset.expect_column_values_to_not_be_null("Amount")
    dataset.expect_column_values_to_be_between("Amount", min_value=0)

    feature_columns = [column for column in df.columns if column != TARGET_COLUMN]
    dataset.expect_table_column_count_to_equal(len(feature_columns) + 1)
    for column in feature_columns:
        dataset.expect_column_values_to_not_be_null(column)

    result = dataset.validate(result_format="SUMMARY")
    return {
        "dataset": dataset_name,
        "success": bool(result["success"]),
        "statistics": result["statistics"],
        "results": [
            {
                "expectation": item["expectation_config"]["expectation_type"],
                "column": item["expectation_config"]["kwargs"].get("column"),
                "success": bool(item["success"]),
            }
            for item in result["results"]
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/processed/train.csv")
    parser.add_argument("--test", default="data/processed/test.csv")
    parser.add_argument("--output", default="reports/gx_validation.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports = [
        validate_frame(pd.read_csv(args.train), "train"),
        validate_frame(pd.read_csv(args.test), "test"),
    ]
    success = all(report["success"] for report in reports)
    output = {
        "success": success,
        "reports": reports,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    if not success:
        raise SystemExit("Great Expectations validation failed")


if __name__ == "__main__":
    main()
