"""
Entraînement et évaluation du modèle de détection de fraude.

Étapes :
  1. Encodage des variables catégorielles (Merchant_Type, Device_Type) par
     one-hot encoding, ajusté UNIQUEMENT sur le train set (évite toute fuite
     de catégories du test set), puis appliqué de façon cohérente partout
     (comparaison, entraînement final, transactions individuelles).
  2. Split stratifié train/test : le taux de fraude (quel qu'il soit) est
     préservé des deux côtés.
  3. Comparaison de 4 approches de gestion du déséquilibre sur le train set
     (validation croisée stratifiée), toutes évaluées en PR-AUC :
       - RandomForest brut (aucun traitement du déséquilibre)
       - RandomForest + class_weight='balanced_subsample'
       - RandomForest + SMOTE (ratio ADAPTATIF, cf. _smote_sampling_strategy :
         un ratio complet 1:1 si le jeu est petit ou déjà proche de l'équilibre,
         un ratio modéré sinon pour rester exécutable sur un gros volume
         réellement déséquilibré — appliqué UNIQUEMENT dans les plis
         d'entraînement pour éviter toute fuite de données)
       - Régression logistique + class_weight='balanced' (référence interprétable)
  4. La meilleure stratégie RandomForest est ré-entraînée sur tout le train set :
       - une version "explainer_model" (non calibrée) pour SHAP/LIME
       - une version calibrée (Platt scaling) pour les probabilités affichées
  5. Évaluation finale sur le test set (jamais ré-échantillonné).

Note méthodologique : ce jeu de données particulier (data/data.csv) s'est
avéré parfaitement équilibré (50/50) lors de l'inspection — voir la remarque
dans src/data.py et le README. La comparaison de stratégies reste calculée et
affichée pour rester fidèle au cahier des charges et parce que le code doit
rester valable sur un jeu réellement déséquilibré, mais l'écart entre
stratégies sera probablement faible ici, ce qui est normal et attendu.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data import CATEGORICAL_COLUMNS

RANDOM_STATE = 42

RF_PARAMS = dict(
    n_estimators=200,
    max_depth=12,
    min_samples_leaf=2,
    n_jobs=-1,
    random_state=RANDOM_STATE,
)

# Plafond de lignes pour la phase de COMPARAISON (recherche rapide) sur de
# GROS volumes réellement déséquilibrés ; sans effet sur un petit jeu comme
# data/data.csv (10 000 lignes). Le modèle final est toujours ré-entraîné sur
# l'intégralité du train set.
COMPARISON_SAMPLE_CAP = 40_000


@dataclass
class StrategyResult:
    name: str
    pr_auc_mean: float
    pr_auc_std: float
    roc_auc_mean: float


def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    """One-hot encoding des colonnes catégorielles, passthrough pour le reste.
    Ajusté UNIQUEMENT sur X_train (par l'appelant, via .fit)."""
    numeric_cols = [c for c in X_train.columns if c not in CATEGORICAL_COLUMNS]
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLUMNS),
            ("num", "passthrough", numeric_cols),
        ]
    )


def encode_features(preprocessor: ColumnTransformer, X: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
    """Applique le préprocesseur (fit=True pour l'ajuster sur ce X, sinon transform seul)
    et retourne un DataFrame avec des noms de colonnes explicites (utile pour SHAP/LIME)."""
    arr = preprocessor.fit_transform(X) if fit else preprocessor.transform(X)
    names = preprocessor.get_feature_names_out()
    clean_names = [n.replace("cat__", "").replace("num__", "") for n in names]
    return pd.DataFrame(arr, columns=clean_names, index=X.index)


def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.25):
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE)


def _smote_sampling_strategy(y: pd.Series) -> float | str:
    """Ratio SMOTE adaptatif : rééquilibrage complet (1:1) si le jeu est petit
    ou déjà proche de l'équilibre (rapide dans ce cas) ; ratio modéré (10 % de
    la classe majoritaire) sur un gros volume réellement très déséquilibré,
    pour rester exécutable en quelques minutes (cf. dataset ULB classique,
    284 807 lignes, ~0,17 % de fraude)."""
    counts = y.value_counts()
    minority_ratio = counts.min() / counts.max()
    if len(y) > 100_000 and minority_ratio < 0.05:
        return 0.1
    return "auto"


def _candidate_pipelines(y_train_for_smote_ratio: pd.Series) -> dict[str, Pipeline]:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline

    smote_ratio = _smote_sampling_strategy(y_train_for_smote_ratio)

    return {
        "RandomForest (brut, sans traitement)": Pipeline(
            [("clf", RandomForestClassifier(**RF_PARAMS))]
        ),
        "RandomForest + class_weight='balanced'": Pipeline(
            [("clf", RandomForestClassifier(**{**RF_PARAMS, "class_weight": "balanced_subsample"}))]
        ),
        "RandomForest + SMOTE": ImbPipeline(
            [
                ("smote", SMOTE(sampling_strategy=smote_ratio, random_state=RANDOM_STATE)),
                ("clf", RandomForestClassifier(**RF_PARAMS)),
            ]
        ),
        "Régression logistique + class_weight='balanced' (baseline interprétable)": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE)),
            ]
        ),
    }


def _cap_for_comparison(X_train: pd.DataFrame, y_train: pd.Series, cap: int = COMPARISON_SAMPLE_CAP):
    if len(X_train) <= cap:
        return X_train, y_train
    fraud_idx = y_train[y_train == 1].index
    legit_idx = y_train[y_train == 0].index
    n_legit_keep = max(cap - len(fraud_idx), len(fraud_idx))
    rng = np.random.default_rng(RANDOM_STATE)
    legit_keep = rng.choice(legit_idx, size=min(n_legit_keep, len(legit_idx)), replace=False)
    keep_idx = np.concatenate([fraud_idx.to_numpy(), legit_keep])
    return X_train.loc[keep_idx], y_train.loc[keep_idx]


def _clone_pipeline(pipeline):
    from sklearn.base import clone
    return clone(pipeline)


def compare_imbalance_strategies(X_train_enc: pd.DataFrame, y_train: pd.Series, n_splits: int = 3) -> list[StrategyResult]:
    X_train_enc, y_train = _cap_for_comparison(X_train_enc, y_train)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    results = []

    for name, pipeline in _candidate_pipelines(y_train).items():
        pr_scores, roc_scores = [], []
        for train_idx, val_idx in skf.split(X_train_enc, y_train):
            X_tr, X_val = X_train_enc.iloc[train_idx], X_train_enc.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            model = _clone_pipeline(pipeline)
            model.fit(X_tr, y_tr)
            proba = model.predict_proba(X_val)[:, 1]
            pr_scores.append(average_precision_score(y_val, proba))
            roc_scores.append(roc_auc_score(y_val, proba))
        results.append(StrategyResult(name, float(np.mean(pr_scores)), float(np.std(pr_scores)), float(np.mean(roc_scores))))

    return sorted(results, key=lambda r: r.pr_auc_mean, reverse=True)


def select_production_strategy(results: list[StrategyResult]) -> tuple[str, str | None]:
    """Sélectionne la meilleure stratégie RandomForest (SHAP TreeExplainer homogène,
    capture d'interactions non linéaires) ; note de transparence si la régression
    logistique dominait le classement complet."""
    rf_results = sorted([r for r in results if r.name.startswith("RandomForest")],
                         key=lambda r: r.pr_auc_mean, reverse=True)
    if not rf_results:
        raise RuntimeError("Aucune stratégie RandomForest disponible.")
    chosen = rf_results[0]
    overall_best = results[0]
    note = None
    if overall_best.name != chosen.name and not overall_best.name.startswith("RandomForest"):
        note = (
            f"Remarque méthodologique : « {overall_best.name} » obtient un PR-AUC "
            f"légèrement supérieur ({overall_best.pr_auc_mean:.3f} vs {chosen.pr_auc_mean:.3f}), "
            f"généralement dans le bruit de la validation croisée. « {chosen.name} » est "
            f"néanmoins retenu pour la production (interactions non linéaires, SHAP homogène)."
        )
    return chosen.name, note


def train_final_model(X_train_enc: pd.DataFrame, y_train: pd.Series, chosen_strategy: str) -> tuple[object, object]:
    pipelines = _candidate_pipelines(y_train)
    base_pipeline = pipelines[chosen_strategy]

    explainer_model = _clone_pipeline(base_pipeline)
    explainer_model.fit(X_train_enc, y_train)

    calibrated_model = CalibratedClassifierCV(_clone_pipeline(base_pipeline), method="sigmoid", cv=3)
    calibrated_model.fit(X_train_enc, y_train)

    return explainer_model, calibrated_model


@dataclass
class EvalMetrics:
    pr_auc: float
    roc_auc: float
    brier: float
    precision_at_threshold: float
    recall_at_threshold: float
    threshold: float
    confusion: np.ndarray
    pr_curve: tuple
    roc_curve: tuple
    calibration_curve: tuple


def evaluate(y_true: pd.Series, probas: np.ndarray, threshold: float = 0.5) -> EvalMetrics:
    preds = (probas >= threshold).astype(int)
    precision = precision_score(y_true, preds, zero_division=0)
    recall = recall_score(y_true, preds, zero_division=0)
    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    prec_curve, rec_curve, _ = precision_recall_curve(y_true, probas)
    fpr, tpr, _ = roc_curve(y_true, probas)
    frac_pos, mean_pred = calibration_curve(y_true, probas, n_bins=10, strategy="quantile")
    return EvalMetrics(
        pr_auc=float(average_precision_score(y_true, probas)),
        roc_auc=float(roc_auc_score(y_true, probas)),
        brier=float(brier_score_loss(y_true, probas)),
        precision_at_threshold=float(precision), recall_at_threshold=float(recall), threshold=threshold,
        confusion=cm, pr_curve=(prec_curve.tolist(), rec_curve.tolist()),
        roc_curve=(fpr.tolist(), tpr.tolist()), calibration_curve=(mean_pred.tolist(), frac_pos.tolist()),
    )


def false_positive_rate(y_true: pd.Series, probas: np.ndarray, threshold: float = 0.5) -> float:
    """Taux de faux positifs = FP / (FP + TN) — métrique de suivi explicitement
    demandée (gouvernance : surveiller les faux positifs)."""
    preds = (probas >= threshold).astype(int)
    tn = int(((preds == 0) & (y_true == 0)).sum())
    fp = int(((preds == 1) & (y_true == 0)).sum())
    return fp / (fp + tn) if (fp + tn) else 0.0


def best_threshold_by_fbeta(y_true: pd.Series, probas: np.ndarray, beta: float = 2.0) -> float:
    prec, rec, thresholds = precision_recall_curve(y_true, probas)
    prec, rec = prec[:-1], rec[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        fbeta = (1 + beta**2) * (prec * rec) / (beta**2 * prec + rec)
    fbeta = np.nan_to_num(fbeta)
    if len(fbeta) == 0:
        return 0.5
    return float(thresholds[int(np.argmax(fbeta))])
