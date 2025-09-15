#!/bin/bash

# Script de démarrage pour le projet ML Pipeline
echo "🚀 Démarrage du ML Pipeline Project"

# Vérifier si Docker est installé
if ! command -v docker &> /dev/null; then
    echo "❌ Docker n'est pas installé. Veuillez installer Docker pour continuer."
    exit 1
fi

# Vérifier si Docker Compose est installé
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose n'est pas installé. Veuillez installer Docker Compose pour continuer."
    exit 1
fi

# Créer le fichier .env s'il n'existe pas
if [ ! -f .env ]; then
    echo "📝 Création du fichier .env..."
    cp .env.example .env
    echo "⚠️  Veuillez éditer le fichier .env avec vos paramètres avant de continuer."
    echo "   Notamment le token API_TOKEN pour sécuriser votre API."
    read -p "Appuyez sur Entrée pour continuer..."
fi

# Construire et démarrer les services
echo "🔨 Construction des images Docker..."
docker-compose build

echo "🚀 Démarrage des services..."
docker-compose up -d

# Attendre que les services soient prêts
echo "⏳ Attente du démarrage des services..."
sleep 10

# Vérifier l'état des services
echo "📊 État des services:"
docker-compose ps

echo ""
echo "✅ Services démarrés avec succès!"
echo ""
echo "🌐 Accès aux services:"
echo "   - API FastAPI: http://localhost:8000"
echo "   - Documentation API: http://localhost:8000/docs"
echo "   - Dashboard Streamlit: http://localhost:8501"
echo ""
echo "📋 Commandes utiles:"
echo "   - Voir les logs: docker-compose logs -f"
echo "   - Arrêter les services: docker-compose down"
echo "   - Redémarrer: docker-compose restart"
echo ""
echo "🔧 Pour le développement:"
echo "   - Logs API: docker-compose logs -f api"
echo "   - Logs Streamlit: docker-compose logs -f streamlit"