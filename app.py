"""
Détecteur de fraude carte bancaire explicable — interface Streamlit.

Lancement : streamlit run app.py

Au tout premier lancement (aucun artefact dans models/), l'application
entraîne le modèle elle-même (rapide : moins d'une minute sur data/data.csv,
10 000 lignes) puis met tout en cache sur disque.
"""
from __future__ import annotations

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data import DEFAULT_DATA_PATH, EXPECTED_COLUMNS, engineer_features, validate_schema
from src.explain import local_lime, local_shap, make_lime_explainer
from src.llm_explain import DEFAULT_MODEL, RISK_REVIEW_NOTICE, generate_llm_explanation, humanize_feature
from src.model import build_preprocessor, encode_features, false_positive_rate

MODELS_DIR = "models"

PALETTE = {
    "legit": "#2F8F7A", "fraud": "#D6432F", "warn": "#E8A33C", "steel": "#5B7A9D",
}

st.set_page_config(
    page_title="Détecteur de fraude carte bancaire explicable",
    page_icon="🕵️", layout="wide", initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { font-family: 'IBM Plex Sans', -apple-system, sans-serif; }
    div[data-testid="stMetricValue"] { font-family: 'IBM Plex Mono', monospace; }
    .governance-banner {
        background: #FDF2E9; border-left: 4px solid #E8A33C; border-radius: 6px;
        padding: 10px 16px; margin-bottom: 16px; font-size: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------------------
# Chargement / entraînement (mis en cache)
# --------------------------------------------------------------------------------------

def artifacts_exist() -> bool:
    return os.path.exists(f"{MODELS_DIR}/metadata.json") and os.path.exists(f"{MODELS_DIR}/calibrated_model.joblib")


@st.cache_resource(show_spinner=False)
def get_artifacts(retrain_token: int = 0):
    if not artifacts_exist():
        import train_pipeline
        train_pipeline.main()

    with open(f"{MODELS_DIR}/metadata.json", encoding="utf-8") as f:
        metadata = json.load(f)

    explainer_model = joblib.load(f"{MODELS_DIR}/explainer_model.joblib")
    calibrated_model = joblib.load(f"{MODELS_DIR}/calibrated_model.joblib")
    shap_explainer = joblib.load(f"{MODELS_DIR}/shap_explainer.joblib")
    preprocessor = joblib.load(f"{MODELS_DIR}/preprocessor.joblib")
    X_test_enc = pd.read_pickle(f"{MODELS_DIR}/X_test_enc.pkl")
    y_test = pd.read_pickle(f"{MODELS_DIR}/y_test.pkl")
    X_train_enc = pd.read_pickle(f"{MODELS_DIR}/X_train_enc.pkl")
    raw_df = pd.read_pickle(f"{MODELS_DIR}/raw_df.pkl")
    global_shap = pd.read_csv(f"{MODELS_DIR}/global_shap_summary.csv")

    lime_explainer = make_lime_explainer(X_train_enc, metadata["feature_names"])
    test_probas = calibrated_model.predict_proba(X_test_enc)[:, 1]

    return dict(
        metadata=metadata, explainer_model=explainer_model, calibrated_model=calibrated_model,
        shap_explainer=shap_explainer, lime_explainer=lime_explainer, preprocessor=preprocessor,
        X_test_enc=X_test_enc, y_test=y_test, X_train_enc=X_train_enc, raw_df=raw_df,
        global_shap=global_shap, test_probas=test_probas,
    )


def calibrated_predict_proba_factory(calibrated_model, feature_names):
    def _fn(Xarr):
        return calibrated_model.predict_proba(pd.DataFrame(Xarr, columns=feature_names))
    return _fn


# --------------------------------------------------------------------------------------
# Composants réutilisables
# --------------------------------------------------------------------------------------

def kpi_row(items):
    cols = st.columns(len(items))
    for col, (label, value, delta) in zip(cols, items):
        with col:
            st.metric(label, value, delta)


def risk_badge(proba: float, threshold: float) -> str:
    if proba >= max(threshold, 0.7):
        return "🔴 Risque élevé"
    if proba >= threshold:
        return "🟠 À surveiller"
    return "🟢 Risque faible"


def shap_bar_chart(features, values, title=""):
    colors = [PALETTE["fraud"] if v > 0 else PALETTE["legit"] for v in values]
    order = np.argsort(values)
    fig = go.Figure(go.Bar(
        x=[values[i] for i in order], y=[features[i] for i in order], orientation="h",
        marker_color=[colors[i] for i in order],
        text=[f"{values[i]:+.3f}" for i in order], textposition="outside",
    ))
    fig.update_layout(title=title, xaxis_title="Contribution (→ risque de fraude)", height=340,
                       margin=dict(l=10, r=10, t=40, b=10), showlegend=False, plot_bgcolor="white")
    return fig


def governance_banner():
    st.markdown(
        f'<div class="governance-banner">⚖️ <b>Gouvernance :</b> {RISK_REVIEW_NOTICE} '
        f'Ce tableau de bord produit des <b>alertes</b>, jamais des décisions automatiques.</div>',
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------------------

st.sidebar.title("🕵️ Détecteur de fraude")
st.sidebar.caption("Classification déséquilibrée & explicabilité — équipes conformité")

if "retrain_token" not in st.session_state:
    st.session_state.retrain_token = 0
if "selected_tx" not in st.session_state:
    st.session_state.selected_tx = None

with st.sidebar.expander("📁 Charger un autre fichier de transactions", expanded=False):
    st.caption(
        f"Schéma attendu : `{', '.join(EXPECTED_COLUMNS)}`. "
        f"Déposez votre fichier ici pour ré-entraîner le pipeline dessus."
    )
    uploaded = st.file_uploader("data.csv", type=["csv"])
    if uploaded is not None:
        try:
            df_up = pd.read_csv(uploaded)
            ok, msg = validate_schema(df_up)
            if not ok:
                st.error(f"Schéma invalide : {msg}")
            else:
                st.success(f"Fichier valide : {len(df_up):,} lignes, {int(df_up['Class'].sum())} fraudes "
                           f"({df_up['Class'].mean():.2%}).")
                if st.button("🔁 Ré-entraîner sur ce fichier", type="primary"):
                    os.makedirs("data", exist_ok=True)
                    df_up.to_csv(DEFAULT_DATA_PATH, index=False)
                    marker = DEFAULT_DATA_PATH + ".synthetic_marker"
                    if os.path.exists(marker):
                        os.remove(marker)
                    for fn in os.listdir(MODELS_DIR):
                        if fn != ".gitkeep":
                            os.remove(os.path.join(MODELS_DIR, fn))
                    st.session_state.retrain_token += 1
                    get_artifacts.clear()
                    st.rerun()
        except Exception as e:
            st.error(f"Impossible de lire ce fichier : {e}")

with st.sidebar.expander("🤖 Rapports générés par IA", expanded=False):
    st.caption(
        "Fournissez votre propre clé API Anthropic pour générer les rapports "
        "d'alerte avec Claude. Sans clé, un générateur par règles (même ton, "
        "mêmes garanties de gouvernance) prend automatiquement le relais."
    )
    api_key_input = st.text_input("Clé ANTHROPIC_API_KEY", type="password", value="")
    st.caption(f"Modèle utilisé : `{DEFAULT_MODEL}`")

st.sidebar.divider()
st.sidebar.caption(
    "⚠️ Licence du dataset Kaggle à revalider avant toute diffusion : "
    "[fiche Kaggle](https://www.kaggle.com/datasets/somnathpaul71/credit-card-fraud-detection-new)."
)

# --------------------------------------------------------------------------------------
# Chargement
# --------------------------------------------------------------------------------------

if not artifacts_exist():
    st.info("🔧 Premier lancement : entraînement du pipeline en cours (moins d'une minute)...")

with st.spinner("Chargement du modèle..."):
    A = get_artifacts(st.session_state.retrain_token)

meta = A["metadata"]
feature_names = meta["feature_names"]
calibrated_predict_proba = calibrated_predict_proba_factory(A["calibrated_model"], feature_names)

st.sidebar.divider()
st.sidebar.subheader("⚙️ Seuil d'alerte")
default_threshold = float(meta.get("best_threshold_f2", 0.5))
threshold = st.sidebar.slider(
    "Score à partir duquel une transaction est signalée",
    min_value=0.0, max_value=1.0, value=round(default_threshold, 3), step=0.01,
)
st.sidebar.caption(f"Seuil suggéré (optimise le rappel des fraudes) : **{default_threshold:.1%}**.")

if meta["dataset"]["source"] == "synthetic":
    st.sidebar.warning("📊 Données **synthétiques** (data/data.csv introuvable).")
else:
    st.sidebar.success(f"📊 Entraîné sur {meta['dataset']['n_rows']:,} transactions fournies (data.csv).")

# --------------------------------------------------------------------------------------
# En-tête
# --------------------------------------------------------------------------------------

st.title("Détecteur de fraude carte bancaire explicable")
st.caption("Classification déséquilibrée · SHAP/LIME · calibration · rapports d'alerte en langage métier")
governance_banner()

if meta.get("near_perfect_separation_warning"):
    st.warning(
        "🔎 **Constat important sur les données** : ce corpus permet une séparation quasi "
        "parfaite entre fraude et transactions légitimes (PR-AUC/ROC-AUC ≈ 1,0 — voir l'onglet "
        "**Biais du jeu de données**). C'est inhabituel pour de la fraude bancaire réelle et "
        "suggère un jeu de démonstration/pédagogique plutôt que des transactions brutes non "
        "filtrées. Les résultats affichés sont réels (aucune donnée n'a été truquée), mais ne "
        "doivent pas être présentés comme représentatifs de la difficulté d'un cas réel."
    )

tab_overview, tab_perf, tab_explain, tab_explorer, tab_bias, tab_method = st.tabs(
    ["📊 Vue d'ensemble", "🎯 Performance", "🔍 Explicabilité", "📋 Explorateur",
     "⚖️ Biais & gouvernance", "ℹ️ Méthodologie"]
)

raw_df = A["raw_df"]
y_test = A["y_test"]
probas = A["test_probas"]

# --------------------------------------------------------------------------------------
# Onglet Vue d'ensemble
# --------------------------------------------------------------------------------------
with tab_overview:
    ds = meta["dataset"]
    kpi_row([
        ("Transactions", f"{ds['n_rows']:,}".replace(",", " "), None),
        ("Fraudes", f"{ds['n_fraud']:,}", None),
        ("Taux de fraude", f"{ds['fraud_rate']:.2%}", None),
        ("Source", "Fichier fourni" if ds["source"] == "real_upload" else "Synthétique", None),
    ])

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Montant par classe")
        fig = go.Figure()
        for cls, name, color in [(0, "Légitime", PALETTE["legit"]), (1, "Fraude", PALETTE["fraud"])]:
            fig.add_trace(go.Box(y=raw_df.loc[raw_df.Class == cls, "Amount"], name=name, marker_color=color))
        fig.update_yaxes(type="log", title="Montant (€, échelle log)")
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, width='stretch')
    with c2:
        st.subheader("Fréquence de transaction récente par classe")
        fig = go.Figure()
        for cls, name, color in [(0, "Légitime", PALETTE["legit"]), (1, "Fraude", PALETTE["fraud"])]:
            fig.add_trace(go.Box(y=raw_df.loc[raw_df.Class == cls, "Transaction_Frequency"], name=name, marker_color=color))
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="Transactions récentes")
        st.plotly_chart(fig, width='stretch')
        st.caption("La variable la plus corrélée à la fraude dans ce jeu (voir onglet Biais).")

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Taux de fraude par catégorie de commerçant")
        grp = raw_df.groupby("Merchant_Type")["Class"].mean().sort_values(ascending=False)
        fig = go.Figure(go.Bar(x=grp.index, y=grp.values, marker_color=PALETTE["steel"],
                                text=[f"{v:.0%}" for v in grp.values], textposition="outside"))
        fig.update_layout(height=300, yaxis_title="Taux de fraude", margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, width='stretch')
    with c4:
        st.subheader("Taux de fraude par type d'appareil")
        grp = raw_df.groupby("Device_Type")["Class"].mean().sort_values(ascending=False)
        fig = go.Figure(go.Bar(x=grp.index, y=grp.values, marker_color=PALETTE["warn"],
                                text=[f"{v:.0%}" for v in grp.values], textposition="outside"))
        fig.update_layout(height=300, yaxis_title="Taux de fraude", margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, width='stretch')

    st.subheader("International vs domestique")
    grp = raw_df.groupby("Is_International")["Class"].agg(["mean", "count"])
    fig = go.Figure(go.Bar(
        x=["Domestique", "International"], y=[grp.loc[0, "mean"], grp.loc[1, "mean"]],
        marker_color=[PALETTE["legit"], PALETTE["fraud"]],
        text=[f"{grp.loc[0,'mean']:.0%} (n={grp.loc[0,'count']})", f"{grp.loc[1,'mean']:.0%} (n={grp.loc[1,'count']})"],
        textposition="outside",
    ))
    fig.update_layout(height=280, yaxis_title="Taux de fraude", margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width='stretch')

# --------------------------------------------------------------------------------------
# Onglet Performance
# --------------------------------------------------------------------------------------
with tab_perf:
    preds = (probas >= threshold).astype(int)
    tp = int(((preds == 1) & (y_test == 1)).sum())
    fp = int(((preds == 1) & (y_test == 0)).sum())
    fn = int(((preds == 0) & (y_test == 1)).sum())
    tn = int(((preds == 0) & (y_test == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    fpr_current = fp / (fp + tn) if (fp + tn) else 0.0

    m5 = meta["metrics_threshold_0.5"]
    kpi_row([
        ("PR-AUC (test)", f"{m5['pr_auc']:.4f}", None),
        ("ROC-AUC (test)", f"{m5['roc_auc']:.4f}", None),
        (f"Précision des alertes @{threshold:.0%}", f"{precision:.1%}", None),
        (f"Rappel des fraudes @{threshold:.0%}", f"{recall:.1%}", None),
    ])
    kpi_row([
        ("Taux de faux positifs (FPR)", f"{fpr_current:.2%}", None),
        ("Brier score (calibration)", f"{m5['brier']:.4f}", None),
        ("Fidélité SHAP/LIME", f"{meta['explanation_fidelity_jaccard5']:.3f}", None),
        ("Faux positifs (nombre)", f"{fp}", None),
    ])
    st.caption(
        "Le **taux de faux positifs** est suivi explicitement (exigence de gouvernance : "
        "surveiller les faux positifs) — chaque faux positif est une alerte inutile transmise "
        "aux équipes conformité, à minimiser sans sacrifier le rappel des vraies fraudes."
    )

    st.divider()
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Précision-rappel")
        prec_c, rec_c = m5["pr_curve"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rec_c, y=prec_c, mode="lines", line_color=PALETTE["steel"]))
        fig.add_trace(go.Scatter(x=[recall], y=[precision], mode="markers",
                                  marker=dict(size=12, color=PALETTE["fraud"]), name=f"seuil={threshold:.2f}"))
        fig.update_layout(xaxis_title="Rappel", yaxis_title="Précision", height=310, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, width='stretch')
    with c2:
        st.subheader("Courbe ROC")
        fpr_c, tpr_c = m5["roc_curve"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr_c, y=tpr_c, mode="lines", line_color=PALETTE["steel"]))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="lightgrey")))
        fig.update_layout(xaxis_title="Taux de faux positifs", yaxis_title="Taux de vrais positifs",
                           height=310, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
        st.plotly_chart(fig, width='stretch')
    with c3:
        st.subheader("Calibration")
        mean_pred, frac_pos = m5["calibration_curve"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=mean_pred, y=frac_pos, mode="lines+markers", line_color=PALETTE["warn"]))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="lightgrey")))
        fig.update_layout(xaxis_title="Score prédit moyen", yaxis_title="Fréquence réelle de fraude",
                           height=310, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
        st.plotly_chart(fig, width='stretch')

    st.subheader(f"Matrice de confusion @ seuil {threshold:.0%}")
    cm = np.array([[tn, fp], [fn, tp]])
    fig = go.Figure(go.Heatmap(
        z=cm, x=["Prédit légitime", "Prédit fraude"], y=["Réel légitime", "Réel fraude"],
        colorscale=[[0, "#F4F6F5"], [1, PALETTE["steel"]]], text=cm, texttemplate="%{text}", showscale=False,
    ))
    fig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width='stretch')

    st.subheader("Comparaison des stratégies de gestion du déséquilibre")
    comp_df = pd.DataFrame(meta["strategy_comparison"])
    comp_df.columns = ["Stratégie", "PR-AUC (moyenne)", "PR-AUC (écart-type)", "ROC-AUC (moyenne)"]
    st.dataframe(comp_df.style.format({
        "PR-AUC (moyenne)": "{:.4f}", "PR-AUC (écart-type)": "±{:.4f}", "ROC-AUC (moyenne)": "{:.4f}"
    }), width='stretch', hide_index=True)
    st.success(f"✅ Stratégie retenue pour la production : **{meta['chosen_strategy']}**")
    if meta.get("strategy_note"):
        st.info(meta["strategy_note"])
    if comp_df["PR-AUC (moyenne)"].nunique() == 1:
        st.caption(
            "⚠️ Toutes les stratégies obtiennent un score quasi identique : ce corpus est "
            "quasiment équilibré (50/50, cf. onglet Biais) et très fortement séparable — la "
            "gestion du déséquilibre n'apporte donc pas de gain visible ICI, bien que le code "
            "reste pleinement fonctionnel sur un jeu réellement déséquilibré."
        )

# --------------------------------------------------------------------------------------
# Onglet Explicabilité
# --------------------------------------------------------------------------------------
with tab_explain:
    st.subheader("Importance globale des variables (SHAP)")
    gs = A["global_shap"].head(12)
    fig = go.Figure(go.Bar(x=gs["mean_abs_shap"][::-1], y=gs["feature"][::-1], orientation="h", marker_color=PALETTE["steel"]))
    fig.update_layout(height=380, xaxis_title="Impact moyen sur le score de risque", margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width='stretch')

    st.divider()
    st.subheader("Rapport d'alerte pour une transaction précise")

    X_test_enc = A["X_test_enc"]
    options = list(X_test_enc.index)
    default_idx = st.session_state.selected_tx if st.session_state.selected_tx in options else options[0]
    selected = st.selectbox(
        "Choisir une transaction (jeu de test)", options=options,
        index=options.index(default_idx),
        format_func=lambda i: f"#{i} — {raw_df.loc[i,'Amount']:.2f}€ — "
                               f"{'FRAUDE réelle' if y_test.loc[i]==1 else 'légitime réelle'}",
    )
    st.session_state.selected_tx = selected

    row = X_test_enc.loc[[selected]]
    proba = float(A["calibrated_model"].predict_proba(row)[:, 1][0])
    decision = "🚩 SIGNALÉE pour vérification" if proba >= threshold else "✅ Laissée passer"

    kpi_row([
        ("Montant", f"{raw_df.loc[selected,'Amount']:.2f} €", None),
        ("Heure", f"{int(raw_df.loc[selected,'Hour_of_Day'])}h", None),
        ("Score de risque (calibré)", f"{proba:.1%}", None),
        ("Statut", decision, None),
    ])
    c_meta1, c_meta2, c_meta3 = st.columns(3)
    c_meta1.caption(f"Commerçant : **{raw_df.loc[selected,'Merchant_Type']}**")
    c_meta2.caption(f"Appareil : **{raw_df.loc[selected,'Device_Type']}**")
    c_meta3.caption(f"International : **{'Oui' if raw_df.loc[selected,'Is_International']==1 else 'Non'}**")
    st.caption(risk_badge(proba, threshold))

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Contribution locale — SHAP**")
        shap_exp = local_shap(A["shap_explainer"], row, top_k=8)
        st.plotly_chart(shap_bar_chart(shap_exp.features, shap_exp.contributions), width='stretch')
    with c2:
        st.markdown("**Contribution locale — LIME**")
        with st.spinner("Calcul de l'explication LIME..."):
            lime_exp = local_lime(A["lime_explainer"], calibrated_predict_proba, row, top_k=8, num_samples=1500)
        fig = go.Figure(go.Bar(
            x=lime_exp.contributions[::-1], y=lime_exp.features[::-1], orientation="h",
            marker_color=[PALETTE["fraud"] if v > 0 else PALETTE["legit"] for v in lime_exp.contributions[::-1]],
        ))
        fig.update_layout(height=340, xaxis_title="Poids LIME", margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, width='stretch')

    st.markdown("**En langage clair :**")
    for f, v, c in zip(shap_exp.features[:4], shap_exp.values[:4], shap_exp.contributions[:4]):
        st.write(("🔺 " if c > 0 else "🔻 ") + humanize_feature(f, v, c))

    st.divider()
    st.markdown("**✨ Rapport d'alerte en langage naturel (équipe conformité)**")
    if st.button("Générer le rapport", type="primary"):
        top_feats = [{"feature": f, "value": v, "contribution": c}
                     for f, v, c in zip(shap_exp.features[:5], shap_exp.values[:5], shap_exp.contributions[:5])]
        with st.spinner("Rédaction du rapport..."):
            result = generate_llm_explanation(
                amount=float(raw_df.loc[selected, "Amount"]), hour=int(raw_df.loc[selected, "Hour_of_Day"]),
                probability=proba, threshold=threshold, decision=decision,
                top_features=top_feats, api_key=api_key_input or None,
            )
        badge = "🤖 Rédigé par Claude" if result.source == "llm" else "📐 Rédigé par le générateur par règles (pas de clé API valide)"
        st.info(f"**{badge}**\n\n{result.text}")
        if result.error and result.source == "template":
            st.caption(f"Détail technique : {result.error}")

# --------------------------------------------------------------------------------------
# Onglet Explorateur
# --------------------------------------------------------------------------------------
with tab_explorer:
    st.subheader("Explorateur des transactions du jeu de test")

    disp = pd.DataFrame({
        "id": X_test_enc.index,
        "Montant (€)": raw_df.loc[X_test_enc.index, "Amount"].values,
        "Heure": raw_df.loc[X_test_enc.index, "Hour_of_Day"].values,
        "Commerçant": raw_df.loc[X_test_enc.index, "Merchant_Type"].values,
        "Appareil": raw_df.loc[X_test_enc.index, "Device_Type"].values,
        "International": np.where(raw_df.loc[X_test_enc.index, "Is_International"].values == 1, "Oui", "Non"),
        "Classe réelle": np.where(y_test.values == 1, "Fraude", "Légitime"),
        "Score de risque": probas,
    })
    disp["Alerte"] = np.where(disp["Score de risque"] >= threshold, "🚩 Signalée", "—")

    c1, c2 = st.columns(2)
    with c1:
        label_filter = st.multiselect("Filtrer par classe réelle", ["Fraude", "Légitime"], default=["Fraude", "Légitime"])
    with c2:
        min_proba = st.slider("Score de risque minimal", 0.0, 1.0, 0.0, 0.05)

    filtered = disp[disp["Classe réelle"].isin(label_filter) & (disp["Score de risque"] >= min_proba)]
    filtered = filtered.sort_values("Score de risque", ascending=False)
    st.dataframe(filtered.style.format({"Montant (€)": "{:.2f}", "Score de risque": "{:.1%}"}),
                 width='stretch', hide_index=True, height=420)
    st.caption(f"{len(filtered):,} transaction(s) affichée(s) sur {len(disp):,}.")

    pick = st.number_input("Ouvrir dans l'onglet Explicabilité (identifiant)",
                            min_value=int(X_test_enc.index.min()), max_value=int(X_test_enc.index.max()), step=1,
                            value=int(filtered.iloc[0]["id"]) if len(filtered) else int(X_test_enc.index[0]))
    if st.button("🔎 Analyser cette transaction"):
        st.session_state.selected_tx = int(pick)
        st.info("Ouvrez l'onglet **🔍 Explicabilité** pour voir le détail.")

# --------------------------------------------------------------------------------------
# Onglet Biais & gouvernance
# --------------------------------------------------------------------------------------
with tab_bias:
    st.subheader("⚖️ Gouvernance du système")
    st.markdown(f"""
- **Aucun blocage automatique.** {RISK_REVIEW_NOTICE} Le tableau de bord et les rapports
  générés (IA ou repli par règles) sont conçus pour **toujours** formuler une recommandation
  de revue humaine — jamais une action déjà exécutée. C'est garanti au niveau du code (le
  gabarit de repli applique cette règle indépendamment de tout LLM), pas seulement par
  consigne au modèle de langage.
- **Suivi des faux positifs.** Le taux de faux positifs (FPR) est calculé et affiché en continu
  dans l'onglet *Performance*, à tout seuil choisi — {fp} faux positifs sur {int(m5['confusion_matrix'][0][0])+fp:,}
  transactions légitimes du jeu de test au seuil actuel ({threshold:.0%}), soit un FPR de
  **{fpr_current:.2%}**. En production, ce taux devrait être surveillé dans la durée et par
  segment (type de commerçant, appareil...) pour détecter toute dérive.
""")

    st.divider()
    st.subheader("📄 Biais et limites connues du jeu de données")
    st.markdown(f"""
1. **Corpus quasiment équilibré (50/50), pas déséquilibré.** `data/data.csv` contient
   {ds['n_rows']:,} transactions, exactement {ds['n_fraud']:,} fraudes ({ds['fraud_rate']:.1%}).
   Ceci contredit l'hypothèse initiale du cahier des charges (« conserver le déséquilibre réel
   des classes », ~0,17 % attendu pour le dataset ULB classique). Ce n'est **pas** le corpus
   complet du dataset de référence historique : il s'agit d'une variante Kaggle différente,
   vraisemblablement ré-échantillonnée et/ou en partie synthétique à des fins pédagogiques.

2. **Séparation quasi parfaite entre classes.** Le modèle atteint un PR-AUC et un ROC-AUC
   proches de 1,0 (voir onglet Performance). Plusieurs catégories (`Device_Type = POS`,
   `Merchant_Type ∈ {{Retail, Food}}`) affichent un taux de fraude **exactement égal à 0 %**, et
   `Transaction_Frequency` corrèle à **0,857** avec la fraude — un niveau de signal inhabituel
   pour des données de fraude bancaire réelles (généralement bruitées, avec beaucoup de
   recouvrement entre classes). Ce niveau de performance ne doit **pas** être extrapolé à un
   contexte réel : en production, des PR-AUC de 0,7-0,85 sont plus représentatifs d'un bon
   modèle sur données réellement déséquilibrées et bruitées.

3. **Fenêtre et représentativité inconnues.** Le fichier ne documente pas la période de
   collecte, la zone géographique, ni l'établissement d'origine — impossible de vérifier la
   représentativité temporelle (saisonnalité, jours fériés) ou démographique.

4. **Biais d'étiquetage potentiel.** Comme pour tout jeu de fraude, les labels reflètent ce que
   le processus d'investigation existant a détecté et confirmé — une fraude non détectée par
   ce processus serait, par construction, étiquetée à tort comme légitime.

5. **Variables anonymisées (V1-V5) non auditables.** Leur contenu réel étant inconnu, il est
   impossible de vérifier si elles encodent indirectement des caractéristiques sensibles
   (biais potentiel non détectable avec les informations disponibles).

6. **Split aléatoire stratifié, pas temporel** — standard pour ce type de benchmark, mais un
   split temporel (entraîner sur le passé, tester sur le futur) serait plus rigoureux pour
   évaluer un déploiement réel.
""")
    st.caption("Cette section répond explicitement à l'exigence du cahier des charges : "
               "« documenter les biais du jeu de données ».")

# --------------------------------------------------------------------------------------
# Onglet Méthodologie
# --------------------------------------------------------------------------------------
with tab_method:
    st.markdown(f"""
### Pipeline
1. **Données** — `data/data.csv` : `Hour_of_Day, Amount, V1..V5, Merchant_Type,
   Location_Distance, Transaction_Frequency, Is_International, Device_Type, Class`.
   Schéma validé automatiquement à l'upload (colonnes + catégories attendues).
2. **Feature engineering** — log-transformation du montant (`Amount_log`) ; les variables
   catégorielles (`Merchant_Type`, `Device_Type`) sont encodées en one-hot, ajusté
   **uniquement sur le train set** (aucune fuite de catégories du test set).
3. **Gestion du déséquilibre** — 4 stratégies comparées par validation croisée stratifiée
   (PR-AUC) sur le corpus complet : RandomForest brut / pondéré / + SMOTE (ratio adaptatif
   selon la taille et le déséquilibre réel des données), et une régression logistique pondérée
   comme référence interprétable. Voir onglet *Performance* pour les résultats — l'écart entre
   stratégies est ici minime car ce corpus est quasi équilibré (cf. onglet *Biais*).
4. **Calibration** — `CalibratedClassifierCV` (sigmoid/Platt scaling).
5. **Explicabilité** — SHAP (`TreeExplainer`) et LIME, tous deux sur le modèle **non calibré**
   pour rester comparables ; la **fidélité des explications** = accord SHAP/LIME (Jaccard@5,
   score actuel : {meta['explanation_fidelity_jaccard5']:.3f}).
6. **Rapport d'alerte en langage naturel** — généré via Claude à partir des facteurs SHAP
   **traduits en langage métier** (pas de jargon technique), avec repli déterministe garanti
   respectant la même règle de gouvernance (jamais de blocage automatique suggéré).

### Métriques suivies (cahier des charges)
| Demandée | Où |
|---|---|
| PR-AUC / ROC-AUC | Onglet *Performance* |
| Recall des fraudes / précision des alertes | Onglet *Performance* (seuil interactif) |
| Fidélité des explications | Onglet *Performance* + *Explicabilité* |
| Faux positifs (gouvernance) | Onglet *Performance* + *Biais & gouvernance* |

Voir l'onglet **⚖️ Biais & gouvernance** pour les limites du jeu de données et les garanties
de gouvernance (aucun blocage automatique, revue humaine systématique).
""")
    st.caption(f"Modèle entraîné le {meta.get('trained_at','?')} · "
               f"temps d'entraînement total : {meta.get('total_training_time_seconds','?')}s")
