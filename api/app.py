import csv
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI

from api.config import Settings
from api.predict import CreditScoringModel
from api.schemas import HealthResponse, PredictionRequest, PredictionResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge le modèle au démarrage de l'application."""
    settings = Settings()
    app.state.model = CreditScoringModel(settings)
    app.state.settings = settings

    # Créer le dossier de logs si nécessaire
    settings.PREDICTIONS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    yield


app = FastAPI(
    title="Credit Scoring API",
    description="API de scoring crédit - Prédit le risque de défaut de paiement",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Vérifie que l'API et le modèle sont opérationnels."""
    model = app.state.model
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        n_features=len(model.feature_names),
        threshold=model.threshold,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """Prédit le risque de défaut pour un client."""
    start = time.perf_counter()

    result = app.state.model.predict(request.features)

    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)

    # Log de la prédiction
    if app.state.settings.LOG_PREDICTIONS:
        _log_prediction(
            request.SK_ID_CURR,
            result["probability"],
            result["prediction"],
            elapsed_ms,
        )

    return PredictionResponse(
        SK_ID_CURR=request.SK_ID_CURR,
        inference_time_ms=elapsed_ms,
        **result,
    )


@app.get("/model-info")
def model_info():
    """Retourne les métadonnées du modèle."""
    model = app.state.model
    return {
        "model_type": type(model.model).__name__,
        "n_features": len(model.feature_names),
        "threshold": model.threshold,
        "metadata": model.metadata,
    }


def _log_prediction(
    client_id: int, probability: float, prediction: int, inference_time_ms: float
):
    """Enregistre une prédiction dans le fichier de log CSV."""
    log_path = app.state.settings.PREDICTIONS_LOG_PATH
    file_exists = log_path.exists()

    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(
                [
                    "timestamp",
                    "SK_ID_CURR",
                    "probability",
                    "prediction",
                    "inference_time_ms",
                ]
            )
        writer.writerow(
            [
                datetime.now(timezone.utc).isoformat(),
                client_id,
                round(probability, 6),
                prediction,
                inference_time_ms,
            ]
        )
