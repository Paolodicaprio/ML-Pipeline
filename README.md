# ML Pipeline CI/CD Project

Ce projet implémente un pipeline CI/CD pour des modèles de machine learning, permettant d'automatiser les processus de test, d'évaluation et de comparaison des performances des modèles.

## Fonctionnalités

- Support pour les modèles de régression et de classification
- Configuration de pipeline via des fichiers YAML
- Automatisation des tests et de l'évaluation des modèles
- Calcul automatique des métriques appropriées selon le type de modèle
- Interface web pour visualiser les résultats (via Streamlit)
- Intégration avec GitHub Actions

## Structure du projet

- `src/`: Code source pour le pipeline CI/CD
- `models/`: Modèles de machine learning pour les tests
- `workflows/`: Fichiers de configuration GitHub Actions
- `app/`: Application Streamlit pour visualiser les résultats
- `tests/`: Tests unitaires pour le pipeline

## Utilisation

1. Créer un modèle de machine learning (régression ou classification)
2. Configurer le pipeline via le fichier YAML approprié
3. Pousser les modifications vers le dépôt GitHub
4. Le pipeline s'exécute automatiquement et évalue le modèle
5. Visualiser les résultats via l'interface web

## Types de modèles supportés

- **Régression**: Modèles qui produisent des sorties continues
  - Métriques: MSE, MAE, RMSE
- **Classification**: Modèles qui produisent des sorties discrètes
  - Métriques: Précision, Rappel, F1-score