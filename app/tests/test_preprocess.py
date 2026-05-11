import pandas as pd
import pytest

from preprocess import (
    coerce_numeric_columns,
    normalize_target,
    require_columns,
    split_features_target,
    stratified_train_test_split,
)


@pytest.mark.parametrize(
    ("df", "required", "raises"),
    [
        (pd.DataFrame({"Amount": [1.0], "Class": [0]}), ["Amount", "Class"], False),
        (pd.DataFrame({"Amount": [1.0]}), ["Amount", "Class"], True),
        (pd.DataFrame(), ["Amount"], True),
    ],
)
def test_require_columns(df: pd.DataFrame, required: list[str], raises: bool) -> None:
    if raises:
        with pytest.raises(ValueError):
            require_columns(df, required)
    else:
        require_columns(df, required)


@pytest.mark.parametrize(
    ("input_value", "expected"),
    [
        ("12.5", 12.5),
        ("abc", 0.0),
        (None, 0.0),
        (float("nan"), 0.0),
    ],
)
def test_coerce_numeric_columns_handles_invalid_amounts(input_value, expected) -> None:
    df = pd.DataFrame({"Amount": [input_value], "AllNaN": [None]})
    result = coerce_numeric_columns(df, ["Amount", "AllNaN"])

    assert result.loc[0, "Amount"] == expected
    assert result.loc[0, "AllNaN"] == 0.0


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([0, 1], [0, 1]),
        ([b"0", b"1"], [0, 1]),
        (["1"], [1]),
    ],
)
def test_normalize_target(values, expected) -> None:
    result = normalize_target(pd.Series(values))

    assert result.tolist() == expected


def test_split_features_target() -> None:
    df = pd.DataFrame({"Amount": [10.0], "Class": [1]})

    features, target = split_features_target(df)

    assert list(features.columns) == ["Amount"]
    assert target.tolist() == [1]


def test_stratified_train_test_split_single_class() -> None:
    df = pd.DataFrame({"Amount": [1.0, 2.0, 3.0, 4.0], "Class": [1, 1, 1, 1]})

    train_df, test_df = stratified_train_test_split(df, test_size=0.5)

    assert len(train_df) == 2
    assert len(test_df) == 2
    assert train_df["Class"].nunique() == 1
    assert test_df["Class"].nunique() == 1


def test_stratified_train_test_split_empty_dataframe() -> None:
    with pytest.raises(ValueError):
        stratified_train_test_split(pd.DataFrame(columns=["Amount", "Class"]))
