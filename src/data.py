"""
Chargement et préparation des données pour le détecteur de fraude.

Schéma RÉEL du fichier fourni (data/data.csv, dataset Kaggle "Credit Card
Fraud Detection [new]") :

    Hour_of_Day, Amount, V1, V2, V3, V4, V5, Merchant_Type, Location_Distance,
    Transaction_Frequency, Is_International, Device_Type, Class

- Hour_of_Day            : heure de la transaction (0-23), déjà fournie
- Amount                 : montant de la transaction
- V1..V5                 : composantes anonymisées (type PCA), seulement 5 ici
                           (à ne pas confondre avec le dataset ULB classique
                           qui en a 28 — ce dataset "new" est une variante
                           différente, avec des variables métier explicites
                           en plus, ce qui est un vrai atout pour l'explicabilité)
- Merchant_Type          : catégorie du marchand (catégorielle : Electronics,
                           Online, Travel, Retail, Food)
- Location_Distance      : distance (km) par rapport au lieu habituel du client
- Transaction_Frequency  : nombre de transactions récentes du client
- Is_International       : transaction internationale (0/1)
- Device_Type            : type d'appareil (catégorielle : Mobile, Desktop, POS)
- Class                  : 0 = légitime, 1 = fraude

⚠️ CONSTAT IMPORTANT (à documenter comme biais du jeu de données, cf. README) :
Ce fichier contient exactement 10 000 lignes, avec exactement 5 000 fraudes /
5 000 légitimes (50 % / 50 %). Ceci contredit l'hypothèse initiale du cahier
des charges ("conserver le déséquilibre réel des classes", ~0,17 % attendu
pour le dataset ULB classique). Ce n'est PAS le corpus complet du dataset
ULB original : il s'agit d'une variante Kaggle différente ("...-new"),
vraisemblablement ré-échantillonnée et/ou en partie synthétique à des fins
pédagogiques (plusieurs catégories affichent un taux de fraude EXACTEMENT
égal à 0 %, et une variable — Transaction_Frequency — corrèle à 0,857 avec
la classe : un niveau de signal inhabituellement élevé et "propre" pour des
données de fraude réelles). Le pipeline reste 100 % capable de traiter un
jeu réellement déséquilibré (la comparaison de stratégies reste implémentée
et utile), mais les résultats sur CE fichier ne doivent pas être présentés
comme représentatifs d'un cas de fraude bancaire réel à fort déséquilibre.

Un générateur synthétique de secours (même schéma) est conservé pour le cas
où data/data.csv serait absent.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

DEFAULT_DATA_PATH = "data/data.csv"

CATEGORICAL_COLUMNS = ["Merchant_Type", "Device_Type"]
NUMERIC_COLUMNS = [
    "Hour_of_Day", "Amount", "V1", "V2", "V3", "V4", "V5",
    "Location_Distance", "Transaction_Frequency", "Is_International",
]
EXPECTED_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS + ["Class"]

MERCHANT_TYPES = ["Electronics", "Online", "Travel", "Retail", "Food"]
DEVICE_TYPES = ["Mobile", "Desktop", "POS"]

DEFAULT_SYNTHETIC_N = 10_000


@dataclass
class DatasetInfo:
    source: str          # "real_upload" ou "synthetic"
    path: str
    n_rows: int
    n_fraud: int
    fraud_rate: float


def validate_schema(df: pd.DataFrame) -> tuple[bool, str]:
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        return False, f"Colonnes manquantes : {', '.join(missing)}"
    if df["Class"].dropna().nunique() > 2:
        return False, "La colonne 'Class' doit être binaire (0 = légitime, 1 = fraude)."
    unknown_merchant = set(df["Merchant_Type"].dropna().unique()) - set(MERCHANT_TYPES)
    unknown_device = set(df["Device_Type"].dropna().unique()) - set(DEVICE_TYPES)
    if unknown_merchant:
        return False, f"Catégories Merchant_Type inconnues : {unknown_merchant}"
    if unknown_device:
        return False, f"Catégories Device_Type inconnues : {unknown_device}"
    return True, "OK"


def generate_synthetic_dataset(n_samples: int = DEFAULT_SYNTHETIC_N, seed: int = 42) -> pd.DataFrame:
    """Génère un jeu de données synthétique au schéma de data/data.csv (fallback
    utilisé uniquement si ce fichier est absent). Reproduit qualitativement
    (mais pas exactement) le type de signal observé dans le vrai fichier :
    quelques catégories de marchand/appareil moins risquées, une fréquence de
    transaction corrélée au risque, et un chevauchement partiel (pas parfait)
    entre classes, plus réaliste qu'une séparation à 0 % exacte."""
    rng = np.random.default_rng(seed)
    n = n_samples
    y = rng.integers(0, 2, size=n)  # 50/50 par defaut, comme le vrai fichier
    fraud = y == 1

    hour = rng.integers(0, 24, size=n)
    freq = np.where(fraud, rng.poisson(9, size=n), rng.poisson(3, size=n)).clip(0, 25)
    v1 = rng.normal(0, 1.3, size=n)
    v2 = np.where(fraud, rng.normal(-2.6, 1.5, size=n), rng.normal(0.3, 1.4, size=n))
    v3 = np.where(fraud, rng.normal(-1.6, 1.2, size=n), rng.normal(0.1, 1.1, size=n))
    v4 = np.where(fraud, rng.normal(1.9, 1.4, size=n), rng.normal(0.2, 1.3, size=n))
    v5 = rng.normal(0, 1.3, size=n)
    is_intl = np.where(fraud, rng.random(n) < 0.6, rng.random(n) < 0.2).astype(int)
    loc_dist = np.where(fraud, rng.exponential(140, size=n), rng.exponential(45, size=n))
    amount = rng.lognormal(mean=np.where(fraud, 5.4, 4.6), sigma=1.1)

    merchant = rng.choice(MERCHANT_TYPES, size=n, p=[0.20, 0.34, 0.25, 0.10, 0.11])
    device = rng.choice(DEVICE_TYPES, size=n, p=[0.56, 0.28, 0.16])
    # quelques categories moins a risque (sans etre a 0% exactement, plus realiste)
    low_risk_merchant = np.isin(merchant, ["Retail", "Food"])
    low_risk_device = device == "POS"
    y_final = fraud & ~(low_risk_merchant & (rng.random(n) < 0.85)) & ~(low_risk_device & (rng.random(n) < 0.85))
    y_final = y_final.astype(int)

    df = pd.DataFrame({
        "Hour_of_Day": hour, "Amount": np.round(amount, 2),
        "V1": v1, "V2": v2, "V3": v3, "V4": v4, "V5": v5,
        "Merchant_Type": merchant, "Location_Distance": np.round(loc_dist, 2),
        "Transaction_Frequency": freq, "Is_International": is_intl,
        "Device_Type": device, "Class": y_final,
    })
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def load_dataset(path: str = DEFAULT_DATA_PATH, force_synthetic: bool = False) -> tuple[pd.DataFrame, DatasetInfo]:
    marker_path = path + ".synthetic_marker"
    is_cached_synthetic = os.path.exists(marker_path)

    if not force_synthetic and os.path.exists(path) and not is_cached_synthetic:
        df = pd.read_csv(path)
        ok, msg = validate_schema(df)
        if not ok:
            raise ValueError(f"Le fichier {path} ne respecte pas le schéma attendu : {msg}")
        info = DatasetInfo("real_upload", path, len(df), int(df["Class"].sum()), float(df["Class"].mean()))
        return df, info

    if not force_synthetic and os.path.exists(path) and is_cached_synthetic:
        df = pd.read_csv(path)
        info = DatasetInfo("synthetic", path, len(df), int(df["Class"].sum()), float(df["Class"].mean()))
        return df, info

    df = generate_synthetic_dataset()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path, index=False)
    with open(marker_path, "w") as f:
        f.write("synthetic")
    info = DatasetInfo("synthetic", path, len(df), int(df["Class"].sum()), float(df["Class"].mean()))
    return df, info


def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Feature engineering : log1p(Amount) pour redistribuer une variable très
    asymétrique. Hour_of_Day est déjà fourni. Les colonnes catégorielles
    (Merchant_Type, Device_Type) sont conservées en dtype 'category' — elles
    seront encodées (one-hot) dans le pipeline de modélisation (src/model.py),
    pas ici, pour garder cette fonction indépendante du choix d'encodage."""
    out = df.copy()
    out["Amount_log"] = np.log1p(out["Amount"].clip(lower=0))
    for c in CATEGORICAL_COLUMNS:
        out[c] = out[c].astype("category")

    feature_cols = (
        ["Hour_of_Day", "Amount_log", "V1", "V2", "V3", "V4", "V5",
         "Location_Distance", "Transaction_Frequency", "Is_International"]
        + CATEGORICAL_COLUMNS
    )
    X = out[feature_cols].copy()
    y = out["Class"].astype(int)
    return X, y, feature_cols
