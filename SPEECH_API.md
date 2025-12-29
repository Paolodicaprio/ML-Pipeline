# 🌐 API FastAPI - Guide de Présentation

## Vue d'ensemble de l'API

L'API FastAPI expose le modèle ML v_best via une interface REST sécurisée, permettant aux applications externes d'effectuer des prédictions en temps réel.

---

## 📍 Endpoints Disponibles

### 1. **GET `/health`** - Vérification de l'état de santé
**Objectif:** Vérifier que l'API fonctionne correctement et que le modèle est chargé.

**Utilisation:**
- Surveillance de la disponibilité de l'API
- Vérification du statut du modèle avant d'effectuer des prédictions
- Monitoring automatisé par des outils de supervision

**Réponse:**
```json
{
  "status": "ok",
  "timestamp": "2025-12-17T13:07:17.123456",
  "model_loaded": true,
  "model_info": {
    "model_name": "MonModele",
    "model_version": "1.0.0",
    "model_type": "classification"
  }
}
```

**Pourquoi c'est important:**
- Permet de détecter rapidement si l'API est opérationnelle
- Indique si le modèle est chargé en mémoire
- Fournit des informations de base sur le modèle déployé

---

### 2. **POST `/predict`** - Effectuer des prédictions
**Objectif:** Soumettre des données au modèle et obtenir des prédictions.

**Authentification:** ✅ **Requise** (Token Bearer)

**Utilisation:**
- Prédictions en temps réel pour les applications clientes
- Intégration dans des workflows automatisés
- Scoring de nouveaux échantillons

**Format de requête:**
```json
{
  "data": {
    "feature1": 1.0,
    "feature2": 2.0,
    "feature3": 3.0
  }
}
```

**Ou pour des prédictions multiples:**
```json
{
  "data": [
    {"feature1": 1.0, "feature2": 2.0, "feature3": 3.0},
    {"feature1": 1.5, "feature2": 2.5, "feature3": 3.5}
  ]
}
```

**Réponse:**
```json
{
  "predictions": [0, 1],
  "probabilities": [[0.8, 0.2], [0.3, 0.7]],
  "model_info": {
    "model_name": "MonModele",
    "model_version": "1.0.0"
  },
  "prediction_timestamp": "2025-12-17T13:07:17.123456",
  "input_shape": [2, 3]
}
```

**Pourquoi c'est important:**
- C'est le cœur de l'API : permet d'utiliser le modèle ML en production
- Authentification requise pour sécuriser l'accès
- Retourne les prédictions avec les probabilités (pour classification)
- Traite des requêtes individuelles ou par batch

**Exemple d'utilisation avec curl:**
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Authorization: Bearer votre-token-secret" \
     -H "Content-Type: application/json" \
     -d '{
       "data": {
         "feature1": 1.0,
         "feature2": 2.0,
         "feature3": 3.0
       }
     }'
```

---

### 3. **GET `/model/info`** - Informations sur le modèle
**Objectif:** Obtenir des informations détaillées sur le modèle actuellement déployé.

**Authentification:** ⚠️ **Optionnelle** (plus de détails avec authentification)

**Utilisation:**
- Consulter les métadonnées du modèle
- Vérifier la version déployée
- Obtenir les métriques de performance
- Accéder aux paramètres d'entraînement (avec authentification)

**Réponse (sans authentification):**
```json
{
  "model_name": "MonModele",
  "model_type": "classification",
  "model_version": "1.0.0",
  "last_loaded": "2025-12-17T13:00:00.000000",
  "validation_metrics": {
    "accuracy": 0.95,
    "precision": 0.93,
    "recall": 0.94
  }
}
```

**Réponse (avec authentification):**
```json
{
  "model_name": "MonModele",
  "model_type": "classification",
  "model_version": "1.0.0",
  "last_loaded": "2025-12-17T13:00:00.000000",
  "training_params": {
    "max_depth": 10,
    "n_estimators": 100,
    "learning_rate": 0.001
  },
  "validation_metrics": {
    "accuracy": 0.95,
    "precision": 0.93,
    "recall": 0.94
  },
  "v_best_info": {
    "is_v_best": true,
    "comparison_date": "2025-12-17T13:00:00.000000"
  }
}
```

**Pourquoi c'est important:**
- Permet de tracer quelle version du modèle est en production
- Fournit les métriques de performance pour validation
- Accès public limité pour transparence, accès complet pour les administrateurs
- Utile pour la documentation et l'audit

---

### 4. **POST `/model/reload`** - Recharger le modèle
**Objectif:** Recharger le modèle depuis le disque sans redémarrer l'API.

**Authentification:** ✅ **Requise** (Token Bearer)

**Utilisation:**
- Mise à jour du modèle après un nouveau déploiement
- Correction d'un modèle défaillant
- Rechargement après modification des fichiers

**Réponse:**
```json
{
  "success": true,
  "message": "Model reloaded successfully",
  "timestamp": "2025-12-17T13:07:17.123456"
}
```

**Pourquoi c'est important:**
- Permet de mettre à jour le modèle sans interruption de service
- Évite les temps d'arrêt lors des déploiements
- Authentification requise pour sécuriser cette opération critique
- Utile pour le déploiement continu (CD)

**Exemple d'utilisation:**
```bash
curl -X POST "http://localhost:8000/model/reload" \
     -H "Authorization: Bearer votre-token-secret"
```

---

### 5. **GET `/`** - Page d'accueil
**Objectif:** Fournir une vue d'ensemble de l'API.

**Utilisation:**
- Point d'entrée pour découvrir l'API
- Liste des endpoints disponibles
- Liens vers la documentation

**Réponse:**
```json
{
  "message": "ML Pipeline API",
  "version": "1.0.0",
  "docs": "/docs",
  "health": "/health",
  "endpoints": {
    "predict": "/predict (POST, requires auth)",
    "model_info": "/model/info (GET, optional auth)",
    "model_reload": "/model/reload (POST, requires auth)",
    "health": "/health (GET, public)"
  }
}
```

---

## 🔐 Authentification

L'API utilise un système d'authentification par **Bearer Token**.

**Configuration:**
1. Créer un fichier `.env` à la racine du projet
2. Définir votre token secret:
   ```
   API_TOKEN=votre-token-secret-ici
   ```

**Utilisation dans les requêtes:**
```
Authorization: Bearer votre-token-secret-ici
```

**Endpoints nécessitant l'authentification:**
- `/predict` - Obligatoire
- `/model/reload` - Obligatoire
- `/model/info` - Optionnel (plus de détails avec auth)

---

## 📊 Architecture de l'API

```
┌─────────────────────────────────────────────┐
│           Client Applications                │
│  (Web Apps, Mobile Apps, Scripts, etc.)     │
└─────────────────┬───────────────────────────┘
                  │ HTTP/REST
                  ▼
┌─────────────────────────────────────────────┐
│              FastAPI Server                  │
│  ┌─────────────────────────────────────┐   │
│  │  Authentication Middleware           │   │
│  └─────────────────────────────────────┘   │
│  ┌─────────────────────────────────────┐   │
│  │  Model Loader (Singleton)            │   │
│  │  - Charge le modèle v_best           │   │
│  │  - Gère les prédictions              │   │
│  │  - Cache le modèle en mémoire        │   │
│  └─────────────────────────────────────┘   │
└─────────────────┬───────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────┐
│         deploy/v_best/model.pkl              │
│         deploy/model_metadata.json           │
└─────────────────────────────────────────────┘
```

---

## 🚀 Avantages de cette API

1. **Séparation des préoccupations:** Le modèle ML est isolé dans un service dédié
2. **Scalabilité:** Peut être déployé sur plusieurs instances pour gérer la charge
3. **Sécurité:** Authentification pour protéger l'accès au modèle
4. **Monitoring:** Endpoint `/health` pour surveillance automatisée
5. **Flexibilité:** Rechargement du modèle sans redémarrage
6. **Documentation:** Swagger UI automatique sur `/docs`
7. **Validation:** Pydantic valide automatiquement les entrées/sorties

---

## 📖 Documentation Interactive

L'API génère automatiquement une documentation interactive accessible via:
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

Ces interfaces permettent de:
- Tester tous les endpoints directement depuis le navigateur
- Voir les schémas de requête/réponse
- Comprendre les paramètres requis
- Essayer l'authentification

---

## 🔧 Configuration

Variables d'environnement (fichier `.env`):
```bash
API_TOKEN=votre-token-secret-ici
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=True
```

---

## 📝 Résumé

L'API FastAPI sert de **pont entre votre modèle ML et les applications clientes**. Elle:

1. **Expose le modèle v_best** via des endpoints REST sécurisés
2. **Gère l'authentification** pour protéger l'accès
3. **Fournit des informations** sur le modèle déployé
4. **Permet le rechargement** du modèle sans interruption
5. **Surveille la santé** du service
6. **Valide les données** en entrée et sortie
7. **Documente automatiquement** tous les endpoints

Cette architecture permet une **intégration facile** du modèle ML dans n'importe quelle application, tout en maintenant la **sécurité**, la **performance** et la **maintenabilité**.