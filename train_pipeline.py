"""
Pipeline complet d'entraînement du détecteur de fraude.

Usage : python3 train_pipeline.py

Étapes : chargement des données (data/data.csv) -> feature engineering ->
encodage des variables catégorielles -> split stratifié -> comparaison des
stratégies de gestion du déséquilibre -> sélection et entraînement du modèle
final (+ calibration) -> évaluation (incluant le taux de faux positifs) ->
explicabilité (SHAP global, fidélité SHAP/LIME) -> export des artefacts.
"""
from __future__ import annotations

import json
import sys
import time

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from src.data import engineer_features, load_dataset
from src.explain import (
    explanation_fidelity,
    global_shap_summary,
    local_shap,
    make_lime_explainer,
    make_shap_explainer,
)
from src.model import (
    best_threshold_by_fbeta,
    build_preprocessor,
    compare_imbalance_strategies,
    encode_features,
    evaluate,
    false_positive_rate,
    select_production_strategy,
    split_data,
    train_final_model,
)

MODELS_DIR = "models"
SHAP_SAMPLE_SIZE = 1200
FIDELITY_SAMPLE_SIZE = 60
DEMO_SAMPLE_SIZE = 60


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def stratified_explain_sample(X: pd.DataFrame, y: pd.Series, n: int, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    fraud_idx = y[y == 1].index
    legit_idx = y[y == 0].index
    n_fraud = min(len(fraud_idx), n // 2)
    n_legit = n - n_fraud
    fraud_keep = rng.choice(fraud_idx, size=n_fraud, replace=False) if n_fraud > 0 else np.array([])
    legit_keep = rng.choice(legit_idx, size=min(n_legit, len(legit_idx)), replace=False)
    keep = np.concatenate([fraud_keep, legit_keep])
    return X.loc[keep]


def main():
    t_start = time.time()
    log("=== Pipeline de détection de fraude — démarrage ===")

    # 1) Données
    df, info = load_dataset()
    log(f"Données : source={info.source} | n={info.n_rows} | fraudes={info.n_fraud} ({info.fraud_rate:.4%})")
    if info.source == "synthetic":
        log("⚠ data/data.csv introuvable -> jeu SYNTHÉTIQUE généré (même schéma). "
            "Déposez le vrai fichier au même emplacement pour l'utiliser à la place.")
    if abs(info.fraud_rate - 0.5) < 0.02:
        log("⚠ REMARQUE IMPORTANTE : ce corpus est quasiment parfaitement équilibré "
            "(~50/50), ce qui contredit l'hypothèse d'un fort déséquilibre du cahier "
            "des charges. La comparaison de stratégies de déséquilibre reste calculée "
            "ci-dessous (le code reste valable sur données réellement déséquilibrées), "
            "mais un écart marqué entre stratégies n'est PAS attendu ici. Voir la section "
            "'Biais du jeu de données' du README / de l'application pour le détail.")

    X, y, feature_names_raw = engineer_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    log(f"Split : train={X_train.shape} (fraude={int(y_train.sum())}) | "
        f"test={X_test.shape} (fraude={int(y_test.sum())})")

    # 2) Encodage (one-hot ajusté sur le train uniquement)
    preprocessor = build_preprocessor(X_train)
    X_train_enc = encode_features(preprocessor, X_train, fit=True)
    X_test_enc = encode_features(preprocessor, X_test, fit=False)
    feature_names = list(X_train_enc.columns)
    log(f"Encodage : {len(feature_names)} colonnes finales -> {feature_names}")

    # 3) Comparaison des stratégies de gestion du déséquilibre
    t0 = time.time()
    comparison = compare_imbalance_strategies(X_train_enc, y_train)
    log(f"Comparaison des stratégies terminée en {time.time()-t0:.1f}s :")
    for r in comparison:
        log(f"   {r.name:65s} PR-AUC={r.pr_auc_mean:.4f} ± {r.pr_auc_std:.4f} | ROC-AUC={r.roc_auc_mean:.4f}")

    chosen_strategy, strategy_note = select_production_strategy(comparison)
    log(f"Stratégie retenue pour la production : {chosen_strategy}")
    if strategy_note:
        log(f"Note : {strategy_note}")

    # 4) Entraînement final
    t0 = time.time()
    explainer_model, calibrated_model = train_final_model(X_train_enc, y_train, chosen_strategy)
    log(f"Entraînement final terminé en {time.time()-t0:.1f}s")

    # 5) Évaluation (incluant le taux de faux positifs, suivi de gouvernance demandé)
    test_probas = calibrated_model.predict_proba(X_test_enc)[:, 1]
    metrics_default = evaluate(y_test, test_probas, threshold=0.5)
    fpr_default = false_positive_rate(y_test, test_probas, threshold=0.5)
    best_thr = best_threshold_by_fbeta(y_test, test_probas, beta=2.0)
    metrics_best = evaluate(y_test, test_probas, threshold=best_thr)
    fpr_best = false_positive_rate(y_test, test_probas, threshold=best_thr)
    log(f"Test @0.5   -> PR-AUC={metrics_default.pr_auc:.4f} ROC-AUC={metrics_default.roc_auc:.4f} "
        f"precision={metrics_default.precision_at_threshold:.4f} recall={metrics_default.recall_at_threshold:.4f} "
        f"FPR={fpr_default:.4f}")
    log(f"Test @seuil F2={best_thr:.3f} -> precision={metrics_best.precision_at_threshold:.4f} "
        f"recall={metrics_best.recall_at_threshold:.4f} FPR={fpr_best:.4f}")

    # 6) Explicabilité
    t0 = time.time()
    shap_explainer = make_shap_explainer(explainer_model, X_train_enc)
    shap_sample = stratified_explain_sample(X_test_enc, y_test, SHAP_SAMPLE_SIZE)
    global_shap = global_shap_summary(shap_explainer, shap_sample)
    log(f"SHAP global calculé en {time.time()-t0:.1f}s sur {len(shap_sample)} transactions")
    log("Top 5 variables (importance globale SHAP) : " + ", ".join(global_shap.head(5)["feature"].tolist()))

    t0 = time.time()

    def calibrated_predict_proba(Xarr):
        return calibrated_model.predict_proba(pd.DataFrame(Xarr, columns=feature_names))

    fidelity_sample = stratified_explain_sample(X_test_enc, y_test, FIDELITY_SAMPLE_SIZE, seed=11)
    lime_explainer = make_lime_explainer(X_train_enc, feature_names)
    fidelity_score, _ = explanation_fidelity(
        explainer_model, calibrated_predict_proba, fidelity_sample, feature_names,
        shap_explainer=shap_explainer, lime_explainer=lime_explainer,
    )
    log(f"Fidélité des explications (accord SHAP/LIME, Jaccard@5) = {fidelity_score:.3f} "
        f"(en {time.time()-t0:.1f}s sur {len(fidelity_sample)} transactions)")

    # 7) Échantillon de démonstration (pour l'app + la démo HTML)
    t0 = time.time()
    demo_sample = stratified_explain_sample(X_test_enc, y_test, DEMO_SAMPLE_SIZE, seed=23)
    demo_records = []
    for idx in demo_sample.index:
        row = X_test_enc.loc[[idx]]
        proba = float(calibrated_model.predict_proba(row)[:, 1][0])
        local_exp = local_shap(shap_explainer, row, top_k=6)
        demo_records.append({
            "id": int(idx),
            "amount": float(df.loc[idx, "Amount"]),
            "hour": int(df.loc[idx, "Hour_of_Day"]),
            "merchant_type": str(df.loc[idx, "Merchant_Type"]),
            "device_type": str(df.loc[idx, "Device_Type"]),
            "is_international": int(df.loc[idx, "Is_International"]),
            "true_label": int(y_test.loc[idx]),
            "probability": proba,
            "top_features": [
                {"feature": f, "value": v, "contribution": c}
                for f, v, c in zip(local_exp.features, local_exp.values, local_exp.contributions)
            ],
        })
    log(f"Échantillon de démonstration ({len(demo_records)} transactions) préparé en {time.time()-t0:.1f}s")

    # 8) Sauvegarde des artefacts
    import os
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(explainer_model, f"{MODELS_DIR}/explainer_model.joblib")
    joblib.dump(calibrated_model, f"{MODELS_DIR}/calibrated_model.joblib")
    joblib.dump(shap_explainer, f"{MODELS_DIR}/shap_explainer.joblib")
    joblib.dump(preprocessor, f"{MODELS_DIR}/preprocessor.joblib")
    X_test_enc.to_pickle(f"{MODELS_DIR}/X_test_enc.pkl")
    y_test.to_pickle(f"{MODELS_DIR}/y_test.pkl")
    X_train_enc.to_pickle(f"{MODELS_DIR}/X_train_enc.pkl")
    y_train.to_pickle(f"{MODELS_DIR}/y_train.pkl")
    df.to_pickle(f"{MODELS_DIR}/raw_df.pkl")
    global_shap.to_csv(f"{MODELS_DIR}/global_shap_summary.csv", index=False)

    metadata = {
        "dataset": {
            "source": info.source, "n_rows": info.n_rows,
            "n_fraud": info.n_fraud, "fraud_rate": info.fraud_rate,
        },
        "split": {
            "n_train": len(X_train_enc), "n_fraud_train": int(y_train.sum()),
            "n_test": len(X_test_enc), "n_fraud_test": int(y_test.sum()),
        },
        "feature_names": feature_names,
        "strategy_comparison": [
            {"name": r.name, "pr_auc_mean": r.pr_auc_mean, "pr_auc_std": r.pr_auc_std, "roc_auc_mean": r.roc_auc_mean}
            for r in comparison
        ],
        "chosen_strategy": chosen_strategy,
        "strategy_note": strategy_note,
        "metrics_threshold_0.5": {
            "pr_auc": metrics_default.pr_auc, "roc_auc": metrics_default.roc_auc, "brier": metrics_default.brier,
            "precision": metrics_default.precision_at_threshold, "recall": metrics_default.recall_at_threshold,
            "false_positive_rate": fpr_default,
            "confusion_matrix": metrics_default.confusion.tolist(),
            "pr_curve": metrics_default.pr_curve, "roc_curve": metrics_default.roc_curve,
            "calibration_curve": metrics_default.calibration_curve,
        },
        "best_threshold_f2": best_thr,
        "metrics_best_threshold": {
            "precision": metrics_best.precision_at_threshold, "recall": metrics_best.recall_at_threshold,
            "false_positive_rate": fpr_best, "confusion_matrix": metrics_best.confusion.tolist(),
        },
        "explanation_fidelity_jaccard5": fidelity_score,
        "demo_sample": demo_records,
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_training_time_seconds": round(time.time() - t_start, 1),
        "near_perfect_separation_warning": bool(metrics_default.pr_auc > 0.995 and metrics_default.roc_auc > 0.995),
    }
    with open(f"{MODELS_DIR}/metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    log(f"Artefacts sauvegardés dans {MODELS_DIR}/")
    log(f"=== Pipeline terminé en {time.time()-t_start:.1f}s ===")
    return metadata


if __name__ == "__main__":
    main()
