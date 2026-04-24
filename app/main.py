import logging
import os
from typing import Any, Dict

import joblib
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.modeling import build_feature_frame, cases_to_rows

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("relevant-priors-api")

MODEL_PATH = os.getenv("MODEL_PATH", "models/relevant_priors_model.joblib")

app = FastAPI(title="Relevant Priors API", version="1.0.0")
model = None


@app.on_event("startup")
def load_model():
    global model
    model = joblib.load(MODEL_PATH)
    logger.info("Loaded model from %s", MODEL_PATH)


@app.get("/")
def health():
    return {"status": "ok", "service": "relevant-priors-api"}


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/")
async def predict_root(request: Request):
    return await _predict(request)


@app.post("/predict")
async def predict(request: Request):
    return await _predict(request)


async def _predict(request: Request):
    payload: Dict[str, Any] = await request.json()
    rows = cases_to_rows(payload)

    request_id = request.headers.get("x-request-id", "missing")
    case_count = len(payload.get("cases", []) or [])
    prior_count = len(rows)
    logger.info("request_id=%s case_count=%s prior_count=%s", request_id, case_count, prior_count)

    if not rows:
        return JSONResponse({"predictions": []})

    X = build_feature_frame(rows)
    preds = model.predict(X)

    predictions = [
        {
            "case_id": row["case_id"],
            "study_id": row["study_id"],
            "predicted_is_relevant": bool(pred),
        }
        for row, pred in zip(rows, preds)
    ]

    return JSONResponse({"predictions": predictions})
