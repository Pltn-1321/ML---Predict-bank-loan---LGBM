# 🏦 Modèle de Scoring de Crédit - Projet MLOps OpenClassrooms

## Formation AI Engineer 2026 - Projet OC6

[![MLFlow](https://img.shields.io/badge/MLFlow-Tracking-blue.svg)](https://mlflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-API-009688.svg)](https://fastapi.tiangolo.com/)
[![Python](https://img.shields.io/badge/Python-3.11+-green.svg)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit-learn-ML-orange.svg)](https://scikit-learn.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-Boosting-yellow.svg)](https://lightgbm.readthedocs.io/)
[![Docker](https://img.shields.io/badge/Docker-Container-2496ED.svg)](https://www.docker.com/)
[![CI/CD](https://img.shields.io/badge/GitHub_Actions-CI%2FCD-2088FF.svg)](https://github.com/features/actions)

### 📊 **Résumé Exécutif**

**Problème métier** : Prédire le risque de défaut de paiement des clients d'une institution financière de microcrédit (Home Credit Default Risk).

**Défi principal** : Dataset massivement déséquilibré (91.9% bons clients vs 8.1% défauts → ratio **11.4:1**) + 8 tables relationnelles à agréger.

**Solution proposée** : Pipeline MLOps complet, de l'entraînement au déploiement :

- Agrégation hiérarchique de 57M+ lignes → 305 features
- **Feature "Has_History"** : capture l'absence d'historique (info critique)
- **Imputation stratégique** : 5 approches selon sémantique métier
- **Score métier personnalisé** : FN = 10× FP (priorité recall)
- **MLFlow tracking complet** : baselines, tuning, seuil optimal
- **API FastAPI** de scoring en production avec monitoring Streamlit
- **CI/CD GitHub Actions** : tests automatisés, build Docker, déploiement Render

**Résultats** : Modèle LightGBM (Val AUC = 0.7852, Business Cost = 0.4907), seuil optimal 0.494, déployé via API REST avec dashboard de monitoring.

---

## 🎯 **Objectifs du Projet**

### Partie 1 — Modélisation
1. **Ingénierie des features avancées** à partir de données relationnelles complexes
2. **Pipeline preprocessing robuste** gérant intelligemment les NaN métier
3. **Modélisation orientée business** avec score coût asymétrique (FN=10, FP=1)
4. **MLOps** : tracking expérimentations, reproductibilité, model registry
5. **Optimisation du seuil de décision** pour minimiser le coût métier

### Partie 2 — Déploiement
6. **API REST** de scoring via FastAPI
7. **Tests unitaires** automatisés (pytest, 19 tests)
8. **Dashboard de monitoring** Streamlit (scores, latence, data drift)
9. **Containerisation Docker** pour la production
10. **Pipeline CI/CD** GitHub Actions (test → build → deploy sur Render)

---

## 🏗️ **Architecture du Pipeline MLOps**

```
📥 Données Brutes (8 CSV, 57M+ lignes)
    ↓ Agrégation Hiérarchique (Notebook 01)
📊 train_aggregated.csv (307k × 305 features)
    ↓ Preprocessing + Feature Engineering (Notebook 02)
⚙️ train_preprocessed.csv (307k × 419 features, 0 NaN, scalé)
    ↓ Modeling + MLFlow (Notebook 03)
🚀 Meilleur Modèle LightGBM (tracké MLFlow, seuil 0.494)
    ↓ Export modèle (scripts/export_model.py)
📦 artifacts/ (model.pkl, scaler.pkl, feature_names.json)
    ↓ API FastAPI + Docker
🌐 API REST /predict → probabilité + décision (APPROVED/REFUSED)
    ↓ CI/CD GitHub Actions
☁️ Déploiement automatique sur Render
```

---

## 📁 **Structure du Projet**

```
OC6_MLOPS/
├── api/                           # API de scoring (FastAPI)
│   ├── app.py                     # Routes (/health, /predict, /model-info)
│   ├── predict.py                 # Chargement modèle + inférence
│   ├── schemas.py                 # Schémas Pydantic request/response
│   └── config.py                  # Configuration (seuil, chemins)
├── artifacts/                     # Modèle exporté (commité dans git)
│   ├── model.pkl                  # LightGBM (joblib)
│   ├── scaler.pkl                 # StandardScaler
│   ├── feature_names.json         # 419 features attendues
│   └── model_metadata.json        # Seuil, coûts, metadata
├── monitoring/                    # Dashboard Streamlit + drift
│   ├── dashboard.py               # Dashboard 5 onglets (prediction, scores, latence, drift, modèle)
│   ├── drift.py                   # Simulation drift + KS test
│   └── predictions_log.jsonl      # Log des prédictions (JSON Lines, généré par l'API)
├── tests/                         # Tests unitaires (pytest, 19 tests)
│   ├── test_api.py                # Tests endpoints API (7 tests)
│   ├── test_predict.py            # Tests logique de prédiction (4 tests)
│   └── test_drift.py              # Tests détection de drift (8 tests)
├── src/                           # Code modulaire réutilisable
│   ├── data_processing.py         # Alignement features
│   └── metrics.py                 # Score métier (FN=10, FP=1)
├── notebooks/                     # Pipeline en 3 étapes
│   ├── 01_EDA.ipynb               # EDA + Agrégation
│   ├── 02_preprocessing_and_feature_engineering.ipynb
│   └── 03_modeling_with_MLFLOW.ipynb
├── scripts/                       # Scripts utilitaires
│   ├── export_model.py            # Export depuis MLflow → artifacts/
│   └── generate_sample_predictions.py  # Génère des prédictions de démo (500 lignes)
├── .github/workflows/ci-cd.yml   # GitHub Actions (test → build → deploy)
├── Dockerfile                     # Image Docker production
├── docker-compose.yml             # API + Dashboard local
├── main.py                        # Point d'entrée uvicorn
├── pyproject.toml                 # Dépendances (uv)
└── README.md
```

---

## 🚀 **Installation & Exécution**

### Prérequis

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (recommandé) ou pip

### Installation

```bash
git clone <votre-repo>
cd OC6_MLOPS
uv sync          # ou pip install -e .
```

### Lancer l'API

```bash
uv run python main.py
# API disponible sur http://localhost:8000
# Documentation Swagger : http://localhost:8000/docs
```

### Lancer les tests

```bash
uv run pytest tests/ -v
```

### Lancer le dashboard de monitoring

```bash
uv run streamlit run monitoring/dashboard.py
# Dashboard disponible sur http://localhost:8501
```

### Lancer avec Docker

```bash
# API seule
docker build -t credit-scoring .
docker run -p 8000:8000 credit-scoring

# API + Dashboard
docker compose up
```

### Lancer le pipeline notebooks

```bash
jupyter notebook notebooks/01_EDA.ipynb
jupyter notebook notebooks/02_preprocessing_and_feature_engineering.ipynb
jupyter notebook notebooks/03_modeling_with_MLFLOW.ipynb
mlflow ui   # http://localhost:5000
```

---

## 📝 **Méthodologie Détaillée par Notebook**

### **Notebook 01 : EDA & Agrégation Hiérarchique** 🔍

**Objectifs** :

- Charger 8 tables (57M+ lignes total)
- Analyser relations : `application_train ← bureau ← bureau_balance`, `previous_application ← POS/CC/Installments`
- Créer dataset plat pour ML

**Innovations** :

- **Agrégation en cascade** : `bureau_balance` (27M) → `bureau` → client
- 183 features créées : 45 bureau + 138 previous_application
- **Statistiques riches** : min/max/mean/sum + one-hot catégorielles
- **Visualisations avancées** : 5 graphiques EDA (âge, corrélations, EXT_SOURCE, ratios, bureau)

**Résultats** :

```
307,511 clients × 305 features
Déséquilibre : 91.9% bons vs 8.1% défauts (11.4:1)
250/305 colonnes NaN (normal : absence historique)
Outputs : train_aggregated.csv + test_aggregated.csv
```

### **Notebook 02 : Preprocessing & Feature Engineering Avancé** ⚙️

**Objectifs** :

- Gérer 250 colonnes NaN intelligemment
- Créer features métier prédictives
- Préparer données scalées pour ML

**Innovations Clés** :

1. **Feature "Has_History"** :

   ```
   HAS_BUREAU, HAS_PREV_APP, HAS_CREDIT_CARD, HAS_POS_CASH, HAS_INSTALLMENTS
   Créées AVANT imputation → capture "aucun historique = info métier"
   ```

2. **Imputation Stratégique (5 règles sémantiques)** :
   | Type Colonne | Stratégie | Exemple | Rationale |
   |------------------|---------------|--------------------------|-----------|
   | Montants (AMT*) | 0 | AMT_CREDIT_SUM → 0 | Pas de crédit = 0€ |
   | Comptages (CNT*) | 0 | SK_ID_BUREAU_COUNT → 0 | 0 occurrence |
   | Dates (DAYS*) | -999 | DAYS_BIRTH → -999 | Sentinelle |
   | Moyennes (MEAN*) | Médiane | EXT_SOURCE_MEAN → median | Robuste outliers |
   | Autres | Médiane | - | Défaut conservateur |

3. **Feature Engineering Métier (11 nouvelles)** :
   - CREDIT_INCOME_RATIO, ANNUITY_INCOME_RATIO
   - AGE_YEARS, EMPLOYMENT_YEARS
   - EXT_SOURCE_MEAN, EXT_SOURCE_PROD
   - INCOME_PER_PERSON, CHILDREN_RATIO
   - BUREAU_DEBT_INCOME_RATIO

**Résultats** :

```
307k × 419 features | 0 NaN | 0 Inf | Scalé (mean=0, std=1)
Scaler.pkl sauvegardé (production-ready)
```

### **Notebook 03 : Modeling MLOps avec MLFlow** 🎯

**Objectifs** :

- Baselines + tuning avec tracking MLFlow
- Score métier asymétrique (FN=10, FP=1)
- Optimisation du seuil de décision

**Approche** :

1. **Score Métier Personnalisé** :

   ```python
   coût_total = (FN × 10) + FP    # Recall prioritaire
   ```

2. **5 Baselines Comparées** :
   | Modèle | Avantages | CV Business Cost |
   |------------------|------------------------|------------------|
   | Logistic Reg (balanced) | Linéaire, rapide | Baseline |
   | Logistic Reg (non-balanced) | Référence | Pire |
   | Random Forest | Non-linéaire | Moyen |
   | XGBoost | Gradient Boosting | Bon |
   | **LightGBM** | **Gradient Boosting, rapide** | **Meilleur** |

3. **Hyperparameter Tuning** : GridSearchCV sur LightGBM
4. **Seuil Optimal** : 0.494 (vs 0.5 défaut) → minimise le coût métier
5. **Évaluation sur validation set** : AUC = 0.7852, Business Cost = 0.4907
6. **MLFlow Complet** : paramètres, métriques, matrices confusion, modèles loggés, model registry

---

## 📊 **Métriques Clés**

```
Dataset : 307k train | 48k test | 11.4:1 imbalance
Features : 122 orig → 305 agrégées → 419 finales
Meilleur Modèle : LightGBM Tuned
Seuil Optimal : 0.494 (vs 0.5 défaut)
Val AUC : 0.7852
Val Business Cost : 0.4907
Tests : 19/19 passent
```

---

## 🌐 **API de Scoring**

L'API FastAPI expose le modèle en production :

| Endpoint | Méthode | Description |
|-----------|---------|-------------|
| `/health` | GET | Status de l'API + modèle chargé |
| `/predict` | POST | Prédiction de scoring (proba + décision) |
| `/model-info` | GET | Metadata du modèle |

**Exemple de requête :**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"SK_ID_CURR": 100001, "features": {"AMT_CREDIT": 0.5, "AMT_ANNUITY": -0.3}}'
```

**Réponse :**
```json
{
  "SK_ID_CURR": 100001,
  "probability": 0.38,
  "prediction": 0,
  "threshold": 0.494,
  "decision": "APPROVED",
  "inference_time_ms": 2.5
}
```

---

## 📈 **Dashboard Monitoring**

Dashboard Streamlit avec 5 onglets :

1. **Prediction** — Scoring client interactif (par ID ou saisie manuelle de features)
2. **Scores & Décisions** — Distribution des probabilités, taux de refus, répartition approuvés/refusés
3. **Performance API** — Latence (P50, P95, max), évolution temporelle
4. **Data Drift** — Simulation de drift (graduel/soudain/feature shift), test KS par feature, rapport Evidently AI
5. **Modèle** — Metadata, seuil optimal, coûts métier, configuration complète

### Log des prédictions (`monitoring/predictions_log.jsonl`)

Chaque appel à `/predict` est enregistré au format **JSON Lines** dans `monitoring/predictions_log.jsonl`. Chaque ligne contient :

| Champ | Description |
|-------|-------------|
| `timestamp` | Horodatage UTC (ISO 8601) |
| `SK_ID_CURR` | Identifiant client |
| `probability` | Probabilité de défaut (0-1) |
| `prediction` | Décision binaire (0=approved, 1=refused) |
| `inference_time_ms` | Temps d'inférence du modèle en ms |

Ce fichier alimente les onglets **Scores & Décisions** et **Performance API** du dashboard.

### Générer des données de démo

Pour tester le dashboard sans lancer l'API, un script génère 500 prédictions réalistes :

```bash
uv run python scripts/generate_sample_predictions.py
```

Le script simule une distribution bimodale (92% bons clients, 8% défauts) avec des timestamps répartis sur 48h et des latences réalistes (~3ms).

---

## 🔄 **CI/CD**

Pipeline GitHub Actions en 3 étapes :
1. **Test** — `ruff check` (linting) + `pytest` (19 tests unitaires)
2. **Build** — `docker build` + test `/health` dans le container
3. **Deploy** — Déploiement automatique sur Render (push main uniquement)

---

## 💡 **Points Forts Méthodologiques**

| Innovation                | Impact Métier/Business                     |
| ------------------------- | ------------------------------------------ |
| **Has_History features**  | "Nouveau client" = risque → info critique  |
| **Imputation sémantique** | Respecte logique bancaire (0€=pas crédit)  |
| **Score FN=10×FP**        | Recall prioritaire (perte >> manque gain)  |
| **Seuil 0.494**           | Minimise le coût métier vs 0.5 par défaut  |
| **No Data Leakage**       | Scaler fit train only                      |
| **MLFlow end-to-end**     | Reproductible, auditable, production-ready |
| **API + monitoring**      | Modèle déployé avec suivi en production    |
| **CI/CD automatisé**      | Tests + build + deploy à chaque push       |

---

## 👨‍💻 **Auteur & Licence**

**Auteur** : Pierre Pluton
**Formation** : OpenClassrooms AI Engineer 2026 - Projet OC6 MLOps
**Date** : Février 2026

**Licence** : MIT License

```
© 2026 Pierre Pluton.
```

---

**Contact** : pierre.pluton@outlook.fr | pierre@thoughtside.com
