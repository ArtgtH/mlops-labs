from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import mlflow
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def load_params(path: str = "params.yaml") -> dict:
    p = Path(path)
    if not p.exists():
        return {
            "train": {
                "random_seed": 123,
                "experiment_name": "fraud-baselines",
            },
            "logreg": {
                "max_iter": 1000,
                "C": 1.0,
            },
            "random_forest": {
                "n_estimators": 300,
                "max_depth": None,
            },
        }

    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def detect_target_column(df: pd.DataFrame) -> str:
    for candidate in ("Class", "target", "label", "y"):
        if candidate in df.columns:
            return candidate
    return df.columns[-1]


def normalize_target(series: pd.Series) -> pd.Series:
    if series.dtype == "object":
        return series.apply(
            lambda x: int(x.decode("utf-8")) if isinstance(x, bytes) else int(x)
        )
    return series.astype(int)


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, str]:
    target_col = detect_target_column(df)
    y = normalize_target(df[target_col])
    X = df.drop(columns=[target_col]).copy()
    return X, y, target_col


def evaluate_classifier(model, X_train, y_train, X_valid, y_valid) -> dict:
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_valid)[:, 1]
    preds = model.predict(X_valid)

    metrics = {
        "valid_roc_auc": float(roc_auc_score(y_valid, proba)),
        "valid_pr_auc": float(average_precision_score(y_valid, proba)),
        "valid_accuracy": float(accuracy_score(y_valid, preds)),
    }
    return metrics


def main() -> None:
    params = load_params()

    train_cfg = params.get("train", {})
    logreg_cfg = params.get("logreg", {})
    rf_cfg = params.get("random_forest", {})

    random_seed = int(train_cfg.get("random_seed", 123))
    experiment_name = str(train_cfg.get("experiment_name", "fraud-baselines"))

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    train_df = pd.read_csv("data/processed/train.csv")
    test_df = pd.read_csv("data/processed/test.csv")

    X_train, y_train, target_col_train = split_xy(train_df)
    X_valid, y_valid, target_col_valid = split_xy(test_df)

    if list(X_train.columns) != list(X_valid.columns):
        raise ValueError("Колонки train/test не совпадают")

    if target_col_train != target_col_valid:
        raise ValueError(
            f"Target column differs: train={target_col_train}, test={target_col_valid}"
        )

    Path("models").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)

    logreg = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=int(logreg_cfg.get("max_iter", 1000)),
                    C=float(logreg_cfg.get("C", 1.0)),
                    class_weight="balanced",
                    random_state=random_seed,
                    solver="lbfgs",
                ),
            ),
        ]
    )

    random_forest = RandomForestClassifier(
        n_estimators=int(rf_cfg.get("n_estimators", 300)),
        max_depth=rf_cfg.get("max_depth", None),
        class_weight="balanced_subsample",
        random_state=random_seed,
        n_jobs=-1,
    )

    results: dict[str, dict] = {}

    with mlflow.start_run(run_name="logreg_baseline"):
        logreg_metrics = evaluate_classifier(
            logreg, X_train, y_train, X_valid, y_valid
        )

        mlflow.log_param("model_name", "LogisticRegression")
        mlflow.log_param("target_col", target_col_train)
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("valid_rows", len(X_valid))
        mlflow.log_param("random_seed", random_seed)
        mlflow.log_param("logreg.max_iter", int(logreg_cfg.get("max_iter", 1000)))
        mlflow.log_param("logreg.C", float(logreg_cfg.get("C", 1.0)))

        for k, v in logreg_metrics.items():
            mlflow.log_metric(k, v)

        joblib.dump(logreg, "models/logreg.pkl")
        results["logreg"] = logreg_metrics

    with mlflow.start_run(run_name="random_forest_baseline"):
        rf_metrics = evaluate_classifier(
            random_forest, X_train, y_train, X_valid, y_valid
        )

        mlflow.log_param("model_name", "RandomForestClassifier")
        mlflow.log_param("target_col", target_col_train)
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("valid_rows", len(X_valid))
        mlflow.log_param("random_seed", random_seed)
        mlflow.log_param(
            "random_forest.n_estimators",
            int(rf_cfg.get("n_estimators", 300)),
        )
        mlflow.log_param(
            "random_forest.max_depth",
            rf_cfg.get("max_depth", None),
        )

        for k, v in rf_metrics.items():
            mlflow.log_metric(k, v)

        joblib.dump(random_forest, "models/random_forest.pkl")
        results["random_forest"] = rf_metrics

    best_model_name = max(results, key=lambda name: results[name]["valid_roc_auc"])

    summary = {
        "target_col": target_col_train,
        "n_features": int(X_train.shape[1]),
        "train_rows": int(len(X_train)),
        "valid_rows": int(len(X_valid)),
        "results": results,
        "best_model": best_model_name,
        "best_valid_roc_auc": results[best_model_name]["valid_roc_auc"],
    }

    Path("reports/metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()