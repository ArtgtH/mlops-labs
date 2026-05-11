from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

FEATURE_COLUMNS = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount"]
FEATURE_COUNT = len(FEATURE_COLUMNS)


class TransactionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    features: list[float] | None = Field(
        default=None,
        min_length=FEATURE_COUNT,
        max_length=FEATURE_COUNT,
        description="Feature vector: Time, V1..V28, Amount.",
    )
    Time: float | None = None
    V1: float | None = None
    V2: float | None = None
    V3: float | None = None
    V4: float | None = None
    V5: float | None = None
    V6: float | None = None
    V7: float | None = None
    V8: float | None = None
    V9: float | None = None
    V10: float | None = None
    V11: float | None = None
    V12: float | None = None
    V13: float | None = None
    V14: float | None = None
    V15: float | None = None
    V16: float | None = None
    V17: float | None = None
    V18: float | None = None
    V19: float | None = None
    V20: float | None = None
    V21: float | None = None
    V22: float | None = None
    V23: float | None = None
    V24: float | None = None
    V25: float | None = None
    V26: float | None = None
    V27: float | None = None
    V28: float | None = None
    Amount: float | None = None

    @model_validator(mode="after")
    def require_feature_vector_or_named_fields(self) -> "TransactionRequest":
        if self.features is not None:
            return self

        missing = [name for name in FEATURE_COLUMNS if getattr(self, name) is None]
        if missing:
            raise ValueError(
                "Either provide 'features' with 30 values or all named fields: "
                + ", ".join(FEATURE_COLUMNS)
            )
        return self

    def as_vector(self) -> list[float]:
        if self.features is not None:
            return self.features

        return [float(getattr(self, name)) for name in FEATURE_COLUMNS]


class PredictionResponse(BaseModel):
    fraud_probability: float
    is_fraud: bool


def example_payload() -> dict[str, Any]:
    payload = dict.fromkeys(FEATURE_COLUMNS, 0.0)
    payload["Amount"] = 42.0
    return payload
