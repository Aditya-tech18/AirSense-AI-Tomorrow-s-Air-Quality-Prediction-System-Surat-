"""Application lifecycle, validation and inference HTTP handlers."""

from contextlib import asynccontextmanager
import os
from pathlib import Path
from time import perf_counter

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from airsense.api.predictor import Model, check_contract, load_model, predict_one
from airsense.api.schemas import PredictionRequest, PredictionResponse
from airsense.logging_config import configure_logger


DEFAULT_MODEL = Path(__file__).resolve().parents[3] / "models" / "model.joblib"


def create_app(*, model: Model | None = None, model_path: str | Path | None = None) -> FastAPI:
    logger = configure_logger()
    path = Path(model_path or os.environ.get("AIRSENSE_MODEL_PATH", DEFAULT_MODEL))

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.model = None
        try:
            loaded = model if model is not None else load_model(path)
            check_contract(loaded)
            application.state.model = loaded
        except Exception as exc:
            logger.error("model_unavailable", extra={
                "status": "unavailable", "reason": type(exc).__name__,
            })
        yield
        application.state.model = None

    application = FastAPI(title="AirSense AI — MP252", lifespan=lifespan)

    @application.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError):
        # Do not echo raw input (including non-finite values) in errors or logs.
        details = [
            {key: value for key, value in error.items() if key in {"loc", "msg", "type"}}
            for error in exc.errors()
        ]
        return JSONResponse(status_code=422, content={"detail": details})

    @application.get("/health")
    def health(request: Request):
        if getattr(request.app.state, "model", None) is None:
            raise HTTPException(status_code=503, detail="Model is unavailable")
        return {"status": "ok"}

    @application.post("/predict", response_model=PredictionResponse)
    def predict(payload: PredictionRequest, request: Request):
        active_model = getattr(request.app.state, "model", None)
        if active_model is None:
            raise HTTPException(status_code=503, detail="Model is unavailable")
        started = perf_counter()
        try:
            value = predict_one(active_model, payload)
        except Exception as exc:
            logger.error("prediction_failed", extra={
                "status": "error", "reason": type(exc).__name__,
            })
            raise HTTPException(status_code=500, detail="Prediction failed") from exc
        logger.info("prediction", extra={
            "status": "ok", "latency_ms": round((perf_counter() - started) * 1000, 3),
        })
        return PredictionResponse(prediction=value)

    return application
