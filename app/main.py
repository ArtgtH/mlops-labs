from fastapi import FastAPI
from fastapi.responses import JSONResponse

from model_service import FraudModelService
from routers.internal import test_router
from schemas import PredictionResponse, TransactionRequest

app = FastAPI()
model_service = FraudModelService()


@app.get("/")
async def read_root():
    return {"service": "fraud-detection-api"}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/healthcheck")
async def healthcheck():
    return await health()


@app.post("/predict", response_model=PredictionResponse)
async def predict(payload: TransactionRequest):
    probability = model_service.predict_probability(payload.as_vector())
    return JSONResponse(
        {
            "fraud_probability": probability,
            "is_fraud": probability >= 0.5,
        }
    )


app.include_router(test_router.router)
