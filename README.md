# ML Pipeline CI/CD Project

Ce projet implémente un pipeline CI/CD pour des modèles de machine learning, permettant d'automatiser les processus de test, d'évaluation et de comparaison des performances des modèles.

## 🎯 Objectif du Projet

Automatisation et optimisation du cycle de vie des modèles de Machine Learning avec MLOps : conception et implémentation d'un pipeline CI/CD basé sur YAML.

## 🏗️ Architecture

- **API FastAPI** : Service REST pour servir le modèle v_best avec authentification
- **Dashboard Streamlit** : Interface web pour visualiser les performances des modèles
- **Pipeline ML** : Scripts pour build, test, evaluate, compare et deploy des modèles
- **Configuration YAML** : Gestion centralisée des paramètres
- **Système v_best** : Conservation automatique du meilleur modèle

## 🚀 Démarrage Rapide avec Docker

### Prérequis
- Docker et Docker Compose installés
- Au moins 4GB de RAM disponible

### 1. Configuration
```bash
# Cloner le projet
git clone <votre-repo>
cd ml-pipeline-project

# Éditer .env avec vos paramètres
```

### 2. Lancement des services
```bash
# Démarrer tous les services
docker-compose up -d

# Voir les logs
docker-compose logs -f

# Arrêter les services
docker-compose down
```

### 3. Accès aux services
- **API FastAPI** : http://localhost:8000
  - Documentation : http://localhost:8000/docs
  - Health check : http://localhost:8000/health
- **Dashboard Streamlit** : http://localhost:8501

## 📋 Fonctionnalités

### API FastAPI
- **POST /predict** : Prédictions avec le modèle v_best (authentifié)
- **GET /health** : Vérification de l'état de l'API
- **GET /model/info** : Informations sur le modèle chargé
- **POST /model/reload** : Rechargement du modèle (authentifié)

### Dashboard Streamlit
- Visualisation des métriques de performance
- Comparaison entre versions de modèles
- Graphiques et visualisations interactives
- Historique des performances

### Pipeline ML
- Support pour régression et classification
- Configuration via fichiers YAML
- Système de versioning automatique
- Comparaison et sélection du meilleur modèle (v_best)

## 🔧 Développement Local

### Sans Docker
```bash
# Installation des dépendances
pip install -r requirements.txt

# API FastAPI
cd api
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# Dashboard Streamlit
cd app
streamlit run app.py --server.port 8501
```

### Avec Docker (développement)
```bash
# Build et démarrage en mode développement
docker-compose up --build

# Rebuild d'un service spécifique
docker-compose build api
docker-compose up -d api
```

## 📊 Types de Modèles Supportés

### Classification
- **Métriques** : Accuracy, Precision, Recall, F1-score
- **Visualisations** : Matrice de confusion, Courbe ROC, Courbe Précision-Rappel

### Régression
- **Métriques** : MSE, MAE, RMSE
- **Visualisations** : Scatter plot, Distribution des erreurs

## 🔐 Authentification

L'API utilise une authentification par token. Configurez votre token dans le fichier `.env` :

```bash
API_TOKEN=your-secure-token-here
```

Utilisez le token dans vos requêtes :
```bash
curl -H "Authorization: Bearer your-token" http://localhost:8000/predict
```

## 📁 Structure du Projet

```
ml-pipeline-project/
├── api/                    # API FastAPI
│   ├── main.py            # Application principale
│   ├── model_loader.py    # Chargement des modèles
│   ├── auth.py            # Authentification
│   ├── requirements.txt   # Dépendances API
│   └── Dockerfile         # Docker API
├── app/                   # Dashboard Streamlit
│   ├── app.py            # Application Streamlit
│   └── Dockerfile        # Docker Streamlit
├── src/                   # Scripts du pipeline ML
├── models/               # Configuration des modèles
├── data/                 # Données d'entraînement
├── deploy/               # Modèles déployés
├── build/                # Artefacts de build
├── docker-compose.yml    # Configuration Docker
├── .env.example         # Variables d'environnement
└── README.md           # Ce fichier
```

## 🔄 Pipeline CI/CD

### Étapes du Pipeline
1. **Build** : Construction et entraînement du modèle
2. **Test** : Test du modèle sur des données non vues
3. **Evaluate** : Évaluation des performances
4. **Compare** : Comparaison avec le modèle v_best
5. **Deploy** : Déploiement du meilleur modèle

### Configuration YAML
```yaml
model:
  name: "ClassifierModel"
  type: "classification"
  version: "2.0.0"

data:
  train_path: "data/classification_train.csv"
  test_path: "data/classification_test.csv"
  validation_split: 0.2

evaluation:
  metrics:
    - accuracy
    - precision
    - recall
    - f1
```

## 🐳 Commandes Docker Utiles

```bash
# Voir les conteneurs en cours
docker-compose ps

# Logs d'un service spécifique
docker-compose logs api
docker-compose logs streamlit

# Redémarrer un service
docker-compose restart api

# Accéder au shell d'un conteneur
docker-compose exec api bash

# Nettoyer les volumes
docker-compose down -v
```

## 🚨 Dépannage

### Problèmes Courants

1. **Port déjà utilisé**
   ```bash
   # Changer les ports dans docker-compose.yml
   ports:
     - "8001:8000"  # API
     - "8502:8501"  # Streamlit
   ```

2. **Modèle non trouvé**
   - Vérifiez que le dossier `deploy/` contient les fichiers du modèle
   - Exécutez le pipeline ML pour générer un modèle

3. **Erreur d'authentification**
   - Vérifiez le token dans `.env`
   - Assurez-vous que le fichier `.env` est bien monté dans le conteneur

### Logs et Monitoring
```bash
# Voir tous les logs
docker-compose logs -f

# Logs avec timestamp
docker-compose logs -f -t

# Monitoring des ressources
docker stats
```

## 📈 Métriques et Monitoring

Le système surveille automatiquement :
- Performances des modèles en temps réel
- Métriques de validation et de test
- Comparaisons entre versions
- État de santé des services

## 🤝 Contribution

1. Fork le projet
2. Créer une branche feature (`git checkout -b feature/AmazingFeature`)
3. Commit les changements (`git commit -m 'Add AmazingFeature'`)
4. Push vers la branche (`git push origin feature/AmazingFeature`)
5. Ouvrir une Pull Request

## 📄 Licence

Ce projet est développé dans le cadre d'un mémoire de Master en MLOps.

## 📞 Support

Pour toute question ou problème :
- Ouvrir une issue sur GitHub
- Consulter la documentation API : http://localhost:8000/docs
- Vérifier les logs : `docker-compose logs`

---

**Développé dans le cadre d'un projet académique pour démontrer l'automatisation des workflows de machine learning avec système v_best.**