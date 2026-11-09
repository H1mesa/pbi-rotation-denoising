FROM python:3.12-slim

WORKDIR /app

# Les dépendances d'abord : cette couche est réutilisée tant que
# requirements.txt ne change pas.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Par défaut, le conteneur lance l'ensemble des tests automatisés.
CMD ["python", "-m", "pytest", "-v"]
