# Détecteur de fraude carte bancaire explicable — image Docker du dashboard.
#
# Construction : docker build -t fraud-detector .
# Lancement    : docker run -p 8501:8501 -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models fraud-detector
# (ou, plus simple : docker compose up --build)
#
# Remarque : cette image n'a pas pu être testée par une construction réelle
# dans l'environnement de génération (Docker non disponible dans ce sandbox) ;
# elle suit les pratiques standard (image slim, dépendances mises en cache,
# utilisateur non-root, healthcheck) — merci de valider `docker build` de
# votre côté avant mise en production.

FROM python:3.12-slim

LABEL maintainer="fraud-detector-project" \
      description="Détecteur de fraude carte bancaire explicable — dashboard Streamlit"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# Dépendances système minimales (certaines libs scientifiques en ont besoin
# pour compiler si aucune wheel précompilée n'est disponible pour l'archi cible)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Étape séparée pour profiter du cache Docker : les dépendances ne sont
# réinstallées que si requirements.txt change, pas à chaque modification de code.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN mkdir -p data models

# Utilisateur non-root (bonne pratique sécurité)
RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

# Premier lancement : le pipeline s'entraîne automatiquement s'il n'y a pas
# encore d'artefacts dans /app/models (voir app.py) — comptez ~1 minute avec
# data/data.csv (10 000 lignes) monté en volume, plus si vous fournissez un
# fichier plus volumineux.
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
