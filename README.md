# Détecteur de fraude carte bancaire

Pipeline complet de détection de fraude — classification, calibration,
explicabilité (SHAP/LIME) et **rapports d'alerte en langage naturel pour les
équipes conformité** — avec interface graphique (Streamlit) et image Docker.

> **Domaine :** Finance et risque. Une banque veut détecter les transactions
> frauduleuses et **expliquer chaque décision aux équipes conformité**.

---

## À lire en premier : constats sur les données fournies

Ce projet a été **entraîné sur le fichier réel fourni** (`data/data.csv`,
10 000 lignes, variante Kaggle *Credit Card Fraud Detection — new* :
[lien de la fiche](https://www.kaggle.com/datasets/somnathpaul71/credit-card-fraud-detection-new)
— *licence à revalider avant toute diffusion, comme indiqué dans le cahier
des charges*). L'inspection de ce fichier a révélé deux écarts importants
par rapport aux hypothèses initiales du cahier des charges, qu'il est
essentiel de connaître avant d'interpréter les résultats :

1. **Le corpus est quasiment parfaitement équilibré (50 % / 50 %)**, et non
   déséquilibré comme l'énoncé le suppose (« conserver le déséquilibre réel
   des classes », ~0,17 % attendu pour le dataset ULB classique). Ce n'est
   **pas** le corpus complet du dataset historique : `data.csv` est une
   variante Kaggle différente ("...-new"), probablement ré-échantillonnée
   et/ou en partie synthétique à des fins pédagogiques.
2. **Le modèle atteint une séparation quasi parfaite** (PR-AUC = ROC-AUC ≈
   0,9999...). Certaines catégories (`Device_Type = POS`,
   `Merchant_Type ∈ {Retail, Food}`) affichent un taux de fraude **exactement
   égal à 0 %**, et `Transaction_Frequency` seule corrèle à **0,857** avec la
   fraude — un niveau de signal inhabituellement "propre" pour de la fraude
   bancaire réelle (généralement bruitée, avec beaucoup de recouvrement entre
   classes).

**Ces résultats sont réels et non truqués** (aucune donnée n'a été modifiée
ni le signal artificiellement renforcé), mais **ne doivent pas être présentés
comme représentatifs de la difficulté d'un cas de fraude réel** — voir
l'onglet **⚖️ Biais & gouvernance** de l'application pour une discussion
complète ; `src/data.py` reste capable de gérer un jeu réellement
déséquilibré si vous en fournissez un (la comparaison de stratégies de
rééquilibrage reste pleinement fonctionnelle).

**Format attendu si vous fournissez votre propre fichier** (déposé dans
`data/data.csv`, ou téléversé depuis la barre latérale de l'application) :

```
Hour_of_Day, Amount, V1, V2, V3, V4, V5, Merchant_Type,
Location_Distance, Transaction_Frequency, Is_International, Device_Type, Class
```

---

## Fonctionnalités

- **Feature engineering** : log-transformation du montant ; encodage one-hot
  des variables catégorielles (`Merchant_Type`, `Device_Type`), ajusté
  uniquement sur le train set.
- **Gestion du déséquilibre** : comparaison de 4 stratégies (RandomForest
  brut / pondéré / + SMOTE à ratio adaptatif, régression logistique pondérée),
  sur le corpus complet, sans sous-échantillonnage.
- **Calibration** : `CalibratedClassifierCV` (Platt scaling).
- **Explicabilité** : SHAP (`TreeExplainer`, global + local) et LIME
  (catégorielles gérées explicitement), avec métrique de **fidélité des
  explications** (accord SHAP/LIME).
- **Rapports d'alerte en langage naturel** — générés via Claude, **rédigés
  pour un public non technique** (équipes conformité) : facteurs SHAP
  traduits en langage métier, aucun jargon IA/statistique, repli déterministe
  automatique si aucune clé API n'est configurée.
- **Gouvernance intégrée** : le système ne recommande **jamais** un blocage
  automatique (garanti au niveau du code, pas seulement du prompt) ; le taux
  de faux positifs est suivi explicitement à tout seuil choisi.
- **Interface graphique** (Streamlit, 6 onglets) + **démo HTML autonome**
  (`demo/demo.html`, ouvrable directement dans un navigateur, avec rapport IA
  en direct).
- **Conteneurisation Docker** (`Dockerfile` + `docker-compose.yml`).

---

## Installation et lancement

### Option A — Python directement

```bash
python3 -m venv venv && source venv/bin/activate   # optionnel mais recommandé
pip install -r requirements.txt
streamlit run app.py
```

Placez votre fichier de transactions dans `data/data.csv` avant de lancer
l'application (sinon un jeu synthétique de secours, au même schéma, est
généré automatiquement). Premier lancement : moins d'une minute avec 10 000
lignes ; les lancements suivants sont instantanés (artefacts mis en cache
dans `models/`).

### Option B — Docker

```bash
docker compose up --build
```

puis ouvrez [http://localhost:8501](http://localhost:8501). Les dossiers
`./data` et `./models` sont montés en volume (persistance + dépôt facile de
votre propre fichier). Pour activer les rapports générés par Claude, éditez
`docker-compose.yml` et renseignez `ANTHROPIC_API_KEY`.

> ⚠️ Le `Dockerfile`/`docker-compose.yml` suivent les pratiques standard
> (image slim, cache des dépendances, utilisateur non-root, healthcheck) mais
> **n'ont pas pu être testés par une construction réelle** dans
> l'environnement de génération de ce projet (Docker indisponible dans ce
> sandbox). Merci de valider `docker build` / `docker compose up` de votre
> côté avant toute mise en production.

### Rapports générés par IA (optionnel)

Barre latérale → « 🤖 Rapports générés par IA » → collez votre clé
`ANTHROPIC_API_KEY` (ou définissez-la en variable d'environnement avant de
lancer). Sans clé, un générateur déterministe (même ton, mêmes garanties de
gouvernance) prend le relais automatiquement. Modèle par défaut :
`claude-sonnet-5` (configurable via `ANTHROPIC_MODEL` ; voir
[docs.claude.com](https://docs.claude.com/en/docs/about-claude/models)).

---

## Architecture du projet

```
fraud_detector/
├── app.py                    # Interface Streamlit (point d'entrée principal)
├── train_pipeline.py         # Pipeline d'entraînement complet (CLI)
├── Dockerfile / docker-compose.yml / .dockerignore
├── requirements.txt
├── src/
│   ├── data.py                # Chargement, validation de schéma, génération synthétique de secours
│   ├── model.py                # Encodage, split, comparaison des stratégies, calibration, FPR
│   ├── explain.py              # SHAP, LIME (catégorielles gérées), métrique de fidélité
│   └── llm_explain.py          # Rapport en langage métier (Claude + repli déterministe), règle de gouvernance
├── data/                      # data.csv (le vôtre, ou généré automatiquement)
├── models/                    # Artefacts entraînés (générés au premier lancement)
└── demo/
    └── demo.html               # Démo interactive autonome
```

---

## Méthodologie

### 1. Données et split
Split stratifié train/test (75/25), taux de fraude préservé des deux côtés,
**aucun sous-échantillonnage** du corpus avant entraînement. Encodage one-hot
des variables catégorielles ajusté uniquement sur le train set.

### 2. Comparaison des stratégies de gestion du déséquilibre

| Stratégie | PR-AUC (moyenne ± écart-type) | ROC-AUC |
|---|---|---|
| **RandomForest brut** ✅ retenu | 0,99998 ± 0,00002 | 0,99998 |
| RandomForest + SMOTE | 0,99998 ± 0,00002 | 0,99998 |
| RandomForest + `class_weight='balanced'` | 0,99998 ± 0,00002 | 0,99998 |
| Régression logistique pondérée | 0,99996 ± 0,00002 | 0,99996 |

*Toutes les stratégies obtiennent un score quasi identique — attendu, ce
corpus étant quasi équilibré et très fortement séparable (cf. ci-dessus). Le
code de comparaison reste pleinement valable sur un jeu réellement
déséquilibré ; le RandomForest brut est retenu ici simplement parce
qu'aucune stratégie de rééquilibrage n'apporte de gain mesurable sur CE
fichier.*

### 3. Calibration
`CalibratedClassifierCV` (sigmoid/Platt, 3 plis).

### 4. Explicabilité
SHAP (`TreeExplainer`) et LIME sur le modèle non calibré. Les variables
catégorielles one-hot sont déclarées explicitement à LIME (`categorical_features`)
pour des explications plus lisibles ("Device_Type_POS = 1" plutôt qu'un
seuillage continu arbitraire). **Fidélité des explications (Jaccard SHAP/LIME
@5) = 0,452.**

### 5. Rapport d'alerte en langage naturel
Les facteurs SHAP sont d'abord **traduits en phrases en langage métier**
(`humanize_feature`, ex. *"une opération réalisée à l'étranger"* plutôt que
*"Is_International = 1, SHAP = +0.19"*) avant d'être transmis au prompt —
cela garantit un rapport lisible même par le générateur de repli (sans LLM),
et rend le LLM moins susceptible d'halluciner du jargon technique.

---

## Gouvernance (exigences du cahier des charges)

- **Ne jamais bloquer automatiquement sans revue humaine** : chaque rapport
  (généré par IA ou par le repli déterministe) se termine systématiquement
  par une recommandation de **revue humaine**, jamais par une action déjà
  décidée. Cette règle est câblée dans le code du générateur par règles
  (`RISK_REVIEW_NOTICE` dans `src/llm_explain.py`) — elle ne dépend donc pas
  du bon comportement du LLM, même si le prompt le lui demande aussi
  explicitement.
- **Surveiller les faux positifs** : le taux de faux positifs (FPR) est
  calculé et affiché en continu (onglet *Performance*), à tout seuil choisi.
  Sur ce corpus, FPR = 0,24 % au seuil 0,5 (6 faux positifs sur ~2 500
  transactions de test) — voir onglet *Biais & gouvernance* pour la mise en
  contexte (un FPR aussi bas reflète surtout la séparabilité inhabituelle du
  jeu de données, pas une garantie de performance en production).
- **Documenter les biais du jeu de données** : section dédiée dans
  l'application (onglet *⚖️ Biais & gouvernance*) et ci-dessous.

### Biais et limites documentés

1. Corpus quasi équilibré (50/50), pas représentatif du déséquilibre réel de
   la fraude bancaire.
2. Séparation quasi parfaite (PR-AUC/ROC-AUC ≈ 1,0) — signal inhabituellement
   propre, suggérant un jeu pédagogique/ré-échantillonné plutôt que des
   transactions brutes.
3. Période, zone géographique et établissement d'origine non documentés —
   représentativité non vérifiable.
4. Biais d'étiquetage potentiel (les labels reflètent ce que le processus
   d'investigation existant a détecté).
5. Variables `V1..V5` anonymisées, non auditables (impossible de vérifier
   l'absence de proxy pour des caractéristiques sensibles).
6. Split aléatoire stratifié plutôt que temporel.

---

## Résultats

| Métrique | Seuil 0,5 | Seuil optimal F2 = 0,146 |
|---|---|---|
| PR-AUC | 0,9999776 | — |
| ROC-AUC | 0,9999776 | — |
| Précision des alertes | 99,76 % | 99,29 % |
| Rappel des fraudes | 99,76 % | 100,00 % |
| Taux de faux positifs | 0,24 % | 0,72 % |
| Fidélité des explications (Jaccard SHAP/LIME@5) | 0,452 | — |

**Rappel :** ces chiffres, bien que réels, caractérisent un corpus
atypiquement "propre" — à recalculer/réinterpréter avec prudence sur des
données de production réelles (voir section Biais ci-dessus).

---

## Métriques suivies

| Demandée | Où |
|---|---|
| PR-AUC / ROC-AUC | Onglet *Performance*, `metadata.json` |
| Recall des fraudes / précision des alertes | Onglet *Performance* (seuil interactif) |
| Fidélité des explications | Onglet *Performance* + *Explicabilité* |
| Faux positifs (gouvernance) | Onglet *Performance* + *Biais & gouvernance* |

---