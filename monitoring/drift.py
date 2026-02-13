"""Simulation et détection de data drift."""

import numpy as np
import pandas as pd
from scipy import stats


def simulate_drift(
    reference_data: pd.DataFrame, drift_type: str = "none", intensity: float = 0.1
) -> pd.DataFrame:
    """Simule du data drift sur des données de référence.

    Args:
        reference_data: Données de référence (preprocessées).
        drift_type: Type de drift ("none", "gradual", "sudden", "feature_shift").
        intensity: Intensité du drift (0.0 à 1.0).

    Returns:
        DataFrame avec drift simulé.
    """
    drifted = reference_data.copy()

    if drift_type == "none":
        return drifted

    if drift_type == "gradual":
        noise = np.random.normal(0, intensity, drifted.shape)
        drifted = drifted + noise

    elif drift_type == "sudden":
        top_features = drifted.columns[:20]
        shift = drifted[top_features].std() * intensity * 5
        drifted[top_features] = drifted[top_features] + shift

    elif drift_type == "feature_shift":
        n_features = max(1, int(len(drifted.columns) * 0.1))
        selected = np.random.choice(drifted.columns, n_features, replace=False)
        drifted[selected] = drifted[selected] * (1 + intensity * 3)

    return drifted


def compute_drift_report(
    reference: pd.DataFrame, production: pd.DataFrame, top_n: int = 20
) -> pd.DataFrame:
    """Calcule les métriques de drift pour chaque feature.

    Args:
        reference: Données de référence.
        production: Données de production (potentiellement driftées).
        top_n: Nombre de features à retourner (triées par drift).

    Returns:
        DataFrame avec ks_statistic, p_value, drift_detected par feature.
    """
    common_cols = [c for c in reference.columns if c in production.columns]
    results = []

    for col in common_cols:
        ref_values = reference[col].dropna()
        prod_values = production[col].dropna()

        if len(ref_values) == 0 or len(prod_values) == 0:
            continue

        ks_stat, p_value = stats.ks_2samp(ref_values, prod_values)
        results.append(
            {
                "feature": col,
                "ks_statistic": round(ks_stat, 4),
                "p_value": round(p_value, 6),
                "drift_detected": p_value < 0.05,
            }
        )

    df = pd.DataFrame(results).sort_values("ks_statistic", ascending=False)
    return df.head(top_n).reset_index(drop=True)
