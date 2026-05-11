import pandas as pd
from sklearn.model_selection import train_test_split

TARGET_COLUMN = "Class"


def require_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def coerce_numeric_columns(
    df: pd.DataFrame,
    columns: list[str],
    *,
    fill_value: float = 0.0,
) -> pd.DataFrame:
    result = df.copy()
    for column in columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
        result[column] = result[column].fillna(fill_value)
    return result


def normalize_target(series: pd.Series) -> pd.Series:
    if series.empty:
        return series.astype(int)

    return series.apply(
        lambda value: int(value.decode("utf-8")) if isinstance(value, bytes) else int(value)
    )


def split_features_target(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
) -> tuple[pd.DataFrame, pd.Series]:
    require_columns(df, [target_column])
    features = df.drop(columns=[target_column]).copy()
    target = normalize_target(df[target_column])
    return features, target


def stratified_train_test_split(
    df: pd.DataFrame,
    *,
    target_column: str = TARGET_COLUMN,
    test_size: float = 0.2,
    random_state: int = 123,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        raise ValueError("Cannot split an empty DataFrame")

    require_columns(df, [target_column])
    target = normalize_target(df[target_column])
    stratify = target if target.nunique() > 1 else None

    return train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
