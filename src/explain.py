"""
Explicabilité du modèle : SHAP (TreeExplainer) et LIME, plus une métrique de
"fidélité des explications".

Choix méthodologique important : SHAP et LIME expliquent tous les deux
`explainer_model` (la version NON calibrée du modèle retenu), afin de comparer
les deux méthodes sur un même objet et de calculer une métrique de fidélité
cohérente. Le modèle CALIBRÉ, lui, sert uniquement à produire la probabilité
finale affichée à l'utilisateur (le calibrage recale les probabilités mais ne
change pratiquement pas le classement des transactions ni l'importance
relative des variables).

Définition de la "fidélité des explications" retenue ici :
Il n'existe pas d'explication "vraie" de référence pour juger une explication
XAI dans l'absolu. Une mesure pragmatique et largement utilisée consiste à
évaluer la COHÉRENCE entre deux méthodes indépendantes (SHAP et LIME) : plus
elles s'accordent sur les variables qui comptent le plus pour une prédiction
donnée, plus on peut avoir confiance dans l'explication produite. On calcule
ici l'indice de Jaccard entre le top-5 des variables les plus contributives
selon SHAP et selon LIME, moyenné sur un échantillon de transactions.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer


@dataclass
class LocalExplanation:
    features: list[str]
    values: list[float]        # valeur brute de la variable pour cette transaction
    contributions: list[float] # contribution signée (SHAP) ou poids (LIME)


def make_shap_explainer(explainer_model, background_X: pd.DataFrame):
    """Construit un TreeExplainer SHAP à partir du modèle non calibré.
    `explainer_model` est soit un RandomForestClassifier direct, soit un
    Pipeline imblearn se terminant par un RandomForestClassifier (étape 'clf')."""
    tree_model = explainer_model
    if hasattr(explainer_model, "named_steps") and "clf" in getattr(explainer_model, "named_steps", {}):
        tree_model = explainer_model.named_steps["clf"]
    return shap.TreeExplainer(tree_model)


def _raw_shap_values(explainer, X: pd.DataFrame) -> np.ndarray:
    """Retourne les valeurs SHAP pour la classe positive (fraude), shape (n, n_features)."""
    raw = explainer.shap_values(X)
    if isinstance(raw, list):  # ancienne API shap : une matrice par classe
        return raw[1]
    if isinstance(raw, np.ndarray) and raw.ndim == 3:  # API récente : (n, features, classes)
        return raw[:, :, 1]
    return raw


def global_shap_summary(explainer, X_sample: pd.DataFrame) -> pd.DataFrame:
    """Importance globale = moyenne des |valeurs SHAP| par variable, sur un échantillon."""
    shap_vals = _raw_shap_values(explainer, X_sample)
    mean_abs = np.abs(shap_vals).mean(axis=0)
    out = pd.DataFrame({"feature": X_sample.columns, "mean_abs_shap": mean_abs})
    return out.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def local_shap(explainer, x_row: pd.DataFrame, top_k: int = 8) -> LocalExplanation:
    """Explication SHAP locale pour UNE transaction (x_row : DataFrame à 1 ligne)."""
    shap_vals = _raw_shap_values(explainer, x_row)[0]
    order = np.argsort(-np.abs(shap_vals))[:top_k]
    feats = list(x_row.columns[order])
    contribs = [float(shap_vals[i]) for i in order]
    values = [float(x_row.iloc[0, i]) for i in order]
    return LocalExplanation(features=feats, values=values, contributions=contribs)


def make_lime_explainer(X_train: pd.DataFrame, feature_names: list[str]) -> LimeTabularExplainer:
    """Les colonnes issues d'un one-hot encoding (préfixes Merchant_Type_ /
    Device_Type_, valeurs 0/1) sont déclarées explicitement catégorielles
    auprès de LIME, plutôt que discrétisées comme des variables continues —
    ce qui produit des explications plus lisibles ("Device_Type_POS = 1")."""
    categorical_idx = [
        i for i, f in enumerate(feature_names)
        if f.startswith("Merchant_Type_") or f.startswith("Device_Type_")
    ]
    return LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=feature_names,
        categorical_features=categorical_idx,
        class_names=["Légitime", "Fraude"],
        mode="classification",
        discretize_continuous=True,
        random_state=42,
    )


def local_lime(
    lime_explainer: LimeTabularExplainer,
    predict_proba_fn,
    x_row: pd.DataFrame,
    top_k: int = 8,
    num_samples: int = 2000,
) -> LocalExplanation:
    """Explication LIME locale pour UNE transaction."""
    exp = lime_explainer.explain_instance(
        x_row.values[0],
        predict_proba_fn,
        num_features=top_k,
        num_samples=num_samples,
        labels=(1,),
    )
    pairs = exp.as_list(label=1)  # [(description_condition, poids), ...]
    feats = [p[0] for p in pairs]
    contribs = [float(p[1]) for p in pairs]
    return LocalExplanation(features=feats, values=[float("nan")] * len(feats), contributions=contribs)


def _extract_base_feature(lime_feature_desc: str, known_features: list[str]) -> str | None:
    """LIME renvoie des descriptions du type '-0.12 < V14 <= 0.45' : on retrouve le
    nom de variable brut pour pouvoir le comparer aux variables SHAP.
    Important : on teste les noms les plus longs en premier ('V14' avant 'V1'),
    sinon 'V1' matcherait à tort comme sous-chaîne de 'V10'..'V19'."""
    for f in sorted(known_features, key=len, reverse=True):
        if f in lime_feature_desc:
            return f
    return None


def explanation_fidelity(
    explainer_model,
    calibrated_predict_proba_fn,
    X_sample: pd.DataFrame,
    feature_names: list[str],
    shap_explainer=None,
    lime_explainer: LimeTabularExplainer | None = None,
    top_k: int = 5,
    lime_num_samples: int = 700,
) -> tuple[float, list[dict]]:
    """
    Calcule la fidélité moyenne (accord SHAP/LIME, indice de Jaccard sur le top-k
    variables) sur un échantillon de transactions. Retourne (score_moyen, détails).
    """
    if shap_explainer is None:
        shap_explainer = shap.TreeExplainer(
            explainer_model.named_steps["clf"] if hasattr(explainer_model, "named_steps") else explainer_model
        )
    if lime_explainer is None:
        lime_explainer = make_lime_explainer(X_sample, feature_names)

    scores = []
    details = []
    for i in range(len(X_sample)):
        row = X_sample.iloc[[i]]
        shap_exp = local_shap(shap_explainer, row, top_k=top_k)
        lime_exp = local_lime(
            lime_explainer, calibrated_predict_proba_fn, row, top_k=top_k, num_samples=lime_num_samples
        )

        shap_top = set(shap_exp.features)
        lime_top = {
            f for f in (_extract_base_feature(desc, feature_names) for desc in lime_exp.features) if f
        }
        if shap_top or lime_top:
            jaccard = len(shap_top & lime_top) / len(shap_top | lime_top)
        else:
            jaccard = 0.0
        scores.append(jaccard)
        details.append({"index": int(X_sample.index[i]), "jaccard": jaccard,
                         "shap_top": sorted(shap_top), "lime_top": sorted(lime_top)})

    return float(np.mean(scores)) if scores else 0.0, details
