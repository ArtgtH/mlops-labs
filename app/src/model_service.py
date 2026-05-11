import json
import os
from pathlib import Path
from typing import Protocol

import joblib
import numpy as np
import pandas as pd

from schemas import FEATURE_COLUMNS, FEATURE_COUNT


class ProbabilityModel(Protocol):
    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...


class FraudModelService:
    def __init__(self, model_path: str | None = None) -> None:
        self.model_path = Path(
            model_path
            or os.getenv("MODEL_PATH", "/app/models/model.pkl")
        )
        self._model: ProbabilityModel | None = None
        self._feature_columns: list[str] | None = None

    def _load_model(self) -> ProbabilityModel | None:
        if self._model is not None:
            return self._model

        candidates = [
            self.model_path,
            Path("models/model.pkl"),
            Path("models/random_forest.pkl"),
            Path("../models/model.pkl"),
            Path("../models/random_forest.pkl"),
        ]
        for path in candidates:
            if path.exists():
                self._model = joblib.load(path)
                return self._model
        return None

    def predict_probability(self, features: list[float]) -> float:
        if len(features) != FEATURE_COUNT:
            raise ValueError(f"Expected {FEATURE_COUNT} features, got {len(features)}")

        vector = self._build_feature_frame(features)
        model = self._load_model()
        if model is None:
            return self._fallback_probability(features)

        probability = model.predict_proba(vector)[0, 1]
        return float(np.clip(probability, 0.0, 1.0))

    def _build_feature_frame(self, features: list[float]) -> pd.DataFrame:
        columns = self._load_feature_columns()
        return pd.DataFrame([features], columns=columns)

    def _load_feature_columns(self) -> list[str]:
        if self._feature_columns is not None:
            return self._feature_columns

        candidates = [
            self.model_path.with_name("feature_columns.json"),
            Path("models/feature_columns.json"),
            Path("../models/feature_columns.json"),
        ]
        for path in candidates:
            if path.exists():
                self._feature_columns = json.loads(path.read_text(encoding="utf-8"))
                return self._feature_columns

        self._feature_columns = FEATURE_COLUMNS
        return self._feature_columns

    @staticmethod
    def _fallback_probability(features: list[float]) -> float:
        amount = max(float(features[-1]), 0.0)
        score = np.log1p(amount) / 10
        return float(np.clip(score, 0.0, 1.0))
