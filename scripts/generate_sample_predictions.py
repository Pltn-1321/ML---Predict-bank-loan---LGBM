"""Génère des données de prédiction réalistes pour tester le dashboard."""

import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

OUTPUT = Path("monitoring/predictions_log.jsonl")
N_PREDICTIONS = 500
THRESHOLD = 0.494


def main():
    """Génère N_PREDICTIONS lignes de log réalistes au format JSON Lines."""
    rng = np.random.default_rng(42)

    # Simuler des probabilités réalistes (distribution bimodale)
    # ~92% de bons clients (proba basse), ~8% de mauvais (proba haute)
    n_good = int(N_PREDICTIONS * 0.92)
    n_bad = N_PREDICTIONS - n_good

    probas_good = rng.beta(2, 8, size=n_good)  # concentré vers 0
    probas_bad = rng.beta(5, 3, size=n_bad)  # concentré vers 0.6+
    probas = np.concatenate([probas_good, probas_bad])
    rng.shuffle(probas)

    # Timestamps étalés sur 48h
    now = datetime.now(tz=timezone.utc)
    start = now - timedelta(hours=48)

    rows = []
    for i, proba in enumerate(probas):
        ts = start + timedelta(seconds=random.uniform(0, 48 * 3600))
        client_id = 100001 + i
        prediction = 1 if proba >= THRESHOLD else 0
        latency = max(0.5, rng.normal(3.0, 1.2))  # ~3ms ± 1.2ms

        rows.append({
            "timestamp": ts.isoformat(),
            "SK_ID_CURR": client_id,
            "probability": round(float(proba), 6),
            "prediction": prediction,
            "inference_time_ms": round(float(latency), 2),
        })

    # Trier par timestamp
    rows.sort(key=lambda r: r["timestamp"])

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    print(f"Generated {len(rows)} predictions → {OUTPUT}")


if __name__ == "__main__":
    main()
