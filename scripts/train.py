from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import joblib
import mlflow
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TARGET_COLUMN = "Class"


def load_params(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COLUMN}")
    features = df.drop(columns=[TARGET_COLUMN]).copy()
    target = df[TARGET_COLUMN].astype(int)
    return features, target


def build_models(params: dict) -> dict[str, object]:
    train_cfg = params.get("train", {})
    logreg_cfg = params.get("logreg", {})
    rf_cfg = params.get("random_forest", {})
    random_seed = int(train_cfg.get("random_seed", 123))

    return {
        "logreg": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=int(logreg_cfg.get("max_iter", 1000)),
                        C=float(logreg_cfg.get("C", 1.0)),
                        class_weight="balanced",
                        random_state=random_seed,
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=int(rf_cfg.get("n_estimators", 300)),
            max_depth=rf_cfg.get("max_depth"),
            class_weight="balanced_subsample",
            random_state=random_seed,
            n_jobs=-1,
        ),
    }


def evaluate(model: object, X_valid: pd.DataFrame, y_valid: pd.Series) -> dict:
    probabilities = model.predict_proba(X_valid)[:, 1]
    predictions = model.predict(X_valid)
    return {
        "valid_recall_fraud": float(recall_score(y_valid, predictions, pos_label=1)),
        "valid_f1_fraud": float(f1_score(y_valid, predictions, pos_label=1)),
        "valid_roc_auc": float(roc_auc_score(y_valid, probabilities)),
        "valid_pr_auc": float(average_precision_score(y_valid, probabilities)),
    }


def log_run(
    model_name: str,
    model: object,
    metrics: dict,
    params: dict,
    feature_columns: list[str],
    train_rows: int,
    valid_rows: int,
) -> None:
    with mlflow.start_run(run_name=f"{model_name}_baseline"):
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("target_col", TARGET_COLUMN)
        mlflow.log_param("n_features", len(feature_columns))
        mlflow.log_param("train_rows", train_rows)
        mlflow.log_param("valid_rows", valid_rows)
        for group_name, group_params in params.items():
            if isinstance(group_params, dict):
                for key, value in group_params.items():
                    mlflow.log_param(f"{group_name}.{key}", value)
        for key, value in metrics.items():
            mlflow.log_metric(key, value)
        mlflow.sklearn.log_model(model, artifact_path="model")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/processed/train.csv")
    parser.add_argument("--test", default="data/processed/test.csv")
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--metrics-output", default="reports/metrics.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params = load_params(Path(args.params))
    experiment_name = params.get("train", {}).get("experiment_name", "fraud-baselines")
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    train_df = pd.read_csv(args.train)
    test_df = pd.read_csv(args.test)
    X_train, y_train = split_xy(train_df)
    X_valid, y_valid = split_xy(test_df)

    if list(X_train.columns) != list(X_valid.columns):
        raise ValueError("Train/test feature columns do not match")

    models = build_models(params)
    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    Path(args.metrics_output).parent.mkdir(parents=True, exist_ok=True)

    results = {}
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        metrics = evaluate(model, X_valid, y_valid)
        results[model_name] = metrics
        joblib.dump(model, models_dir / f"{model_name}.pkl")
        log_run(
            model_name,
            model,
            metrics,
            params,
            list(X_train.columns),
            len(X_train),
            len(X_valid),
        )

    best_model_name = max(
        results,
        key=lambda name: (
            results[name]["valid_recall_fraud"],
            results[name]["valid_f1_fraud"],
            results[name]["valid_pr_auc"],
        ),
    )
    joblib.dump(models[best_model_name], models_dir / "model.pkl")
    (models_dir / "feature_columns.json").write_text(
        json.dumps(list(X_train.columns), indent=2),
        encoding="utf-8",
    )

    summary = {
        "target_col": TARGET_COLUMN,
        "feature_columns": list(X_train.columns),
        "n_features": int(X_train.shape[1]),
        "train_rows": int(len(X_train)),
        "valid_rows": int(len(X_valid)),
        "results": results,
        "best_model": best_model_name,
        "best_valid_recall_fraud": results[best_model_name]["valid_recall_fraud"],
        "best_valid_f1_fraud": results[best_model_name]["valid_f1_fraud"],
    }
    Path(args.metrics_output).write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
