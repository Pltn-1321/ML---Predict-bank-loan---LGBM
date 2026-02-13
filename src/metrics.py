"""Métriques métier pour le scoring crédit."""

import numpy as np
from sklearn.metrics import confusion_matrix


def business_cost_score(y_true, y_pred, cost_fn=10, cost_fp=1):
    """Calcule le coût métier normalisé.

    Args:
        y_true: Labels réels (0 ou 1).
        y_pred: Labels prédits (0 ou 1).
        cost_fn: Coût d'un faux négatif (défaut non détecté).
        cost_fp: Coût d'un faux positif (bon client refusé).

    Returns:
        Coût métier normalisé par le nombre total de clients.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    total_cost = fn * cost_fn + fp * cost_fp
    return total_cost / len(y_true)


def find_optimal_threshold(y_true, probabilities, cost_fn=10, cost_fp=1, n_thresholds=100):
    """Trouve le seuil qui minimise le coût métier.

    Args:
        y_true: Labels réels.
        probabilities: Probabilités prédites.
        cost_fn: Coût faux négatif.
        cost_fp: Coût faux positif.
        n_thresholds: Nombre de seuils à tester.

    Returns:
        Tuple (seuil_optimal, coût_minimal).
    """
    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    best_threshold = 0.5
    best_cost = float("inf")

    for t in thresholds:
        preds = (probabilities >= t).astype(int)
        cost = business_cost_score(y_true, preds, cost_fn, cost_fp)
        if cost < best_cost:
            best_cost = cost
            best_threshold = t

    return round(best_threshold, 4), round(best_cost, 4)
