import json

import joblib
import pandas as pd

from api.config import Settings


class CreditScoringModel:
    """Charge le modèle et effectue des prédictions de scoring crédit."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self.model = joblib.load(self.settings.MODEL_PATH)
        with open(self.settings.FEATURE_NAMES_PATH) as f:
            self.feature_names = json.load(f)
        with open(self.settings.METADATA_PATH) as f:
            self.metadata = json.load(f)
        self.threshold = self.settings.OPTIMAL_THRESHOLD

    def predict(self, client_data: dict[str, float]) -> dict:
        """Prédit le risque de défaut pour un client.

        Args:
            client_data: Dictionnaire feature_name -> value (données preprocessées).

        Returns:
            Dictionnaire avec probability, prediction, threshold, decision.
        """
        df = pd.DataFrame([client_data])
        df = df.reindex(columns=self.feature_names, fill_value=0)

        proba = float(self.model.predict_proba(df)[:, 1][0])
        prediction = int(proba >= self.threshold)

        return {
            "probability": round(proba, 6),
            "prediction": prediction,
            "threshold": self.threshold,
            "decision": "REFUSED" if prediction == 1 else "APPROVED",
        }
