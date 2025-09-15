# Dockerfile pour Streamlit
FROM python:3.11-slim

# Définir le répertoire de travail
WORKDIR /app

# Copier les requirements du projet principal
COPY requirements.txt .

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copier l'application Streamlit
COPY app.py .

# Créer les dossiers nécessaires
RUN mkdir -p deploy build data

# Exposer le port Streamlit
EXPOSE 8501

# Variables d'environnement pour Streamlit
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Commande de démarrage
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]