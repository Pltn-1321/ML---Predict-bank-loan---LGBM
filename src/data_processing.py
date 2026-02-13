"""Utilitaires de traitement de données pour le scoring crédit."""

import pandas as pd


def align_features(df: pd.DataFrame, expected_features: list[str], fill_value=0) -> pd.DataFrame:
    """Aligne les colonnes d'un DataFrame sur la liste de features attendues.

    Args:
        df: DataFrame d'entrée.
        expected_features: Liste ordonnée des features attendues par le modèle.
        fill_value: Valeur par défaut pour les features manquantes.

    Returns:
        DataFrame réindexé avec exactement les colonnes attendues.
    """
    return df.reindex(columns=expected_features, fill_value=fill_value)
