"""
Génération d'un rapport d'alerte en langage naturel, à destination des
équipes CONFORMITÉ (non techniques) — pas des data scientists.

Contraintes explicites du cahier des charges, appliquées ici :
  - Ton non technique, pas de jargon IA/statistique dans le texte produit
    (pas de "SHAP", "modèle", "algorithme", "probabilité calibrée au sens
    statistique"...) — le rapport doit se lire comme rédigé par un collègue,
    pas par une IA. Les facteurs SHAP servent de FONDEMENT au rapport, sans
    être nommés comme tels.
  - Le score de risque (calibré) est mentionné en langage clair ("niveau de
    risque estimé"), pas comme une "probabilité calibrée par CalibratedClassifierCV".
  - GOUVERNANCE : ce système ne bloque JAMAIS automatiquement une transaction.
    Toute recommandation est formulée comme une recommandation DE REVUE
    HUMAINE, jamais comme une action déjà prise ou à déclencher seule. Cette
    règle est imposée aussi bien dans le prompt LLM que dans le repli
    déterministe — elle ne dépend donc pas de la bonne volonté du LLM.

Deux modes :
  1. LLM (Claude) si une clé API est disponible (ANTHROPIC_API_KEY, ou fournie
     explicitement depuis la barre latérale de l'application).
  2. Repli déterministe (template) sinon, ou si l'appel échoue — l'application
     reste 100 % fonctionnelle dans tous les cas.

Nous ne fabriquons jamais de clé API : c'est à la personne qui exécute le
projet de fournir la sienne si elle souhaite les rapports réellement
rédigés par un LLM.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass

# Modèle Claude utilisé par défaut. Configurable via la variable d'environnement
# ANTHROPIC_MODEL. Cf. https://docs.claude.com/en/docs/about-claude/models
DEFAULT_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")

RISK_REVIEW_NOTICE = (
    "Ce système ne bloque jamais automatiquement une transaction : cette "
    "alerte doit être examinée par un analyste avant toute action."
)


@dataclass
class ExplanationResult:
    text: str
    source: str  # "llm" ou "template"
    error: str | None = None


def humanize_feature(feature: str, value: float, contribution: float) -> str:
    """Traduit un facteur technique (nom de variable encodée + valeur) en une
    phrase compréhensible par un non-technicien, sans jargon de variable."""
    up = contribution > 0

    if feature == "Transaction_Frequency":
        return f"un nombre de transactions récentes {'élevé' if up else 'peu élevé'} pour ce client ({value:.0f} sur la période récente)"
    if feature == "Location_Distance":
        return f"une opération réalisée {'loin' if up else 'près'} du lieu habituel du client (environ {value:.0f} km d'écart)"
    if feature == "Hour_of_Day":
        return f"un horaire {'inhabituel' if up else 'habituel'} pour ce type d'opération ({int(value)}h)"
    if feature == "Amount_log":
        amount = math.expm1(value)
        return f"un montant {'élevé' if up else 'dans la norme'} ({amount:,.2f} €)".replace(",", " ")
    if feature == "Is_International":
        return ("une opération réalisée à l'étranger" if value >= 0.5
                else "une opération réalisée dans le pays du client")
    if feature.startswith("Merchant_Type_"):
        cat = feature.replace("Merchant_Type_", "")
        present = value >= 0.5
        return (f"un achat chez un commerçant de catégorie « {cat} »" if present
                else f"un commerçant qui n'est pas de catégorie « {cat} » (catégorie habituellement plus sûre)" if not up
                else f"une catégorie de commerçant différente de « {cat} »")
    if feature.startswith("Device_Type_"):
        cat = feature.replace("Device_Type_", "")
        present = value >= 0.5
        return (f"une opération réalisée depuis un appareil de type « {cat} »" if present
                else f"une opération qui n'a pas été réalisée sur un terminal de type « {cat} » (habituellement plus sûr)" if not up
                else f"un type de terminal différent de « {cat} »")
    if feature.startswith("V") and feature[1:].isdigit():
        return f"un indicateur comportemental interne ({feature}) {'nettement supérieur' if up else 'nettement inférieur'} au comportement habituel"
    return f"{feature} = {value:.2f}"


def _format_factors(top_features: list[dict]) -> str:
    lines = []
    for f in top_features:
        phrase = humanize_feature(f["feature"], f["value"], f["contribution"])
        sens = "fait monter le niveau de risque estimé" if f["contribution"] > 0 else "fait plutôt baisser le niveau de risque estimé"
        lines.append(f"- {phrase} → {sens}")
    return "\n".join(lines)


def _risk_level(probability: float, threshold: float) -> str:
    if probability >= max(threshold, 0.7):
        return "élevé"
    if probability >= threshold:
        return "à surveiller"
    return "faible"


def build_prompt(
    amount: float, hour: int, probability: float, threshold: float,
    decision: str, top_features: list[dict],
) -> str:
    niveau = _risk_level(probability, threshold)
    return f"""Tu rédiges, pour l'équipe CONFORMITÉ d'une banque (lectorat non technique,
métier — pas des data scientists), un COURT rapport d'alerte sur une
transaction jugée à risque par notre système de surveillance.

RÈGLES DE TON (impératives) :
- Écris en français, 4 à 6 phrases, comme un collègue qui explique un dossier
  à l'oral — pas comme une IA, pas de ton robotique.
- N'utilise AUCUN jargon technique ou statistique : pas de "SHAP", "modèle",
  "algorithme", "score de probabilité calibré", "feature", "variable",
  "machine learning", "intelligence artificielle". Ne te présente jamais
  comme une IA.
- Reste concret et orienté décision métier.

RÈGLE DE GOUVERNANCE (impérative, non négociable) :
- Ce système NE BLOQUE JAMAIS automatiquement une transaction. Ta
  recommandation finale doit TOUJOURS être formulée comme une recommandation
  DE REVUE HUMAINE (ex. "à transmettre à un analyste pour vérification"),
  JAMAIS comme une action déjà prise, déjà décidée, ou à déclencher seule
  (ne dis jamais "la transaction a été bloquée" ni "bloquez cette transaction").

Éléments du dossier :
- Montant : {amount:.2f} €
- Heure : {hour}h
- Niveau de risque estimé : {niveau} ({probability:.0%})
- Statut de l'alerte : {decision}

Facteurs ayant motivé cette alerte (déjà en langage clair, à reformuler
naturellement, ne pas les recopier tels quels en liste) :
{_format_factors(top_features)}

Termine par une recommandation d'action proportionnée AU NIVEAU DE RISQUE et
UNE SEULE phrase rappelant que ceci est une estimation, pas une certitude."""


def template_explanation(
    amount: float, hour: int, probability: float, threshold: float,
    decision: str, top_features: list[dict],
) -> str:
    """Génération déterministe (repli sans LLM) — même ton non technique et
    même règle de gouvernance, garantis par construction (pas de dépendance
    au comportement d'un LLM)."""
    niveau = _risk_level(probability, threshold)
    factors_up = [f for f in top_features if f["contribution"] > 0][:2]
    factors_down = [f for f in top_features if f["contribution"] <= 0][:1]

    phrases_up = [humanize_feature(f["feature"], f["value"], f["contribution"]) for f in factors_up]
    phrases_down = [humanize_feature(f["feature"], f["value"], f["contribution"]) for f in factors_down]

    factor_text = ""
    if phrases_up:
        factor_text += "Ce qui a surtout attiré l'attention : " + " et ".join(phrases_up) + ". "
    if phrases_down:
        factor_text += "À l'inverse, " + phrases_down[0] + " joue plutôt en faveur du client. "

    reco = {
        "élevé": "Recommandation : dossier à transmettre en priorité à un analyste pour vérification avant toute décision.",
        "à surveiller": "Recommandation : contrôle standard recommandé avant validation par un analyste.",
        "faible": "Recommandation : aucune action prioritaire, la transaction peut suivre son circuit habituel.",
    }[niveau]

    return (
        f"Transaction de {amount:.2f} € à {hour}h — niveau de risque estimé {niveau} ({probability:.0%}). "
        f"{factor_text}"
        f"{reco} {RISK_REVIEW_NOTICE} "
        f"Il s'agit d'une estimation, pas d'une certitude : le jugement de l'analyste reste déterminant."
    )


def generate_llm_explanation(
    amount: float, hour: int, probability: float, threshold: float,
    decision: str, top_features: list[dict],
    api_key: str | None = None, model: str = DEFAULT_MODEL,
) -> ExplanationResult:
    """Tente un appel à l'API Claude ; repli automatique sur le template en cas d'échec."""
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return ExplanationResult(
            text=template_explanation(amount, hour, probability, threshold, decision, top_features),
            source="template", error="Aucune clé ANTHROPIC_API_KEY configurée.",
        )
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=key)
        prompt = build_prompt(amount, hour, probability, threshold, decision, top_features)
        response = client.messages.create(
            model=model, max_tokens=400, temperature=0.4,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in response.content if getattr(b, "type", "") == "text").strip()
        if not text:
            raise ValueError("Réponse vide de l'API.")
        if RISK_REVIEW_NOTICE not in text:
            text += f"\n\n{RISK_REVIEW_NOTICE}"
        return ExplanationResult(text=text, source="llm", error=None)
    except Exception as exc:
        return ExplanationResult(
            text=template_explanation(amount, hour, probability, threshold, decision, top_features),
            source="template",
            error=f"Appel LLM indisponible ({type(exc).__name__}: {exc}) — repli sur le générateur par règles.",
        )
