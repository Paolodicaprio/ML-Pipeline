# 📊 Application Streamlit - Guide de Présentation

## Vue d'ensemble de l'Application

L'application Streamlit est un **tableau de bord interactif** qui permet de visualiser et d'analyser les résultats du pipeline ML. Elle offre une interface conviviale pour explorer les performances des modèles, comparer les versions, et surveiller le data drift.

---

## 🎯 Les 6 Onglets Principaux

### 1. 📈 **Tableau de bord** (Dashboard)

**Objectif:** Vue d'ensemble rapide des performances du modèle déployé.

**Contenu:**
- **Informations sur le modèle:**
  - Nom du modèle
  - Type (classification ou régression)
  - Version actuelle

- **Métriques principales:**
  - Pour la classification: Accuracy, Precision, Recall, F1-Score
  - Pour la régression: MSE, MAE, RMSE, R²
  - Affichage sous forme de cartes métriques colorées

- **Visualisations:**
  - **Classification:**
    - Matrice de confusion
    - Courbe ROC
    - Courbe Précision-Rappel
  - **Régression:**
    - Graphique de dispersion (Prédictions vs Valeurs réelles)
    - Histogramme de distribution des erreurs

**Pourquoi c'est important:**
- Donne une vue instantanée de la performance du modèle
- Permet de vérifier rapidement si le modèle fonctionne correctement
- Identifie visuellement les problèmes potentiels (ex: classes mal prédites)

**Cas d'usage:**
- Vérification quotidienne des performances
- Présentation des résultats aux parties prenantes
- Validation après déploiement

---

### 2. 🔍 **Détails du modèle**

**Objectif:** Exploration approfondie des paramètres et de l'historique du modèle.

**Contenu:**
- **Paramètres du modèle:**
  - Configuration complète en format JSON
  - Hyperparamètres utilisés (max_depth, n_estimators, etc.)
  - Métadonnées (date de création, auteur, etc.)

- **Paramètres d'entraînement:**
  - Epochs, batch_size, learning_rate
  - Configuration de validation
  - Stratégie de séparation des données

- **Historique des métriques:**
  - Tableau comparatif Validation vs Test
  - Graphique en barres pour visualiser les différences
  - Identification des métriques stables ou instables

**Pourquoi c'est important:**
- Permet de comprendre comment le modèle a été construit
- Facilite la reproduction des résultats
- Aide à identifier les paramètres optimaux
- Détecte le surapprentissage (overfitting) en comparant validation et test

**Cas d'usage:**
- Audit du modèle
- Optimisation des hyperparamètres
- Documentation technique
- Debugging de problèmes de performance

---

### 3. 🔄 **Comparaison**

**Objectif:** Comparer le modèle actuel avec le modèle v_best précédent.

**Contenu:**
- **Informations sur les modèles:**
  - Modèle actuel: nom, version, type
  - Modèle v_best: version précédente (si existe)

- **Résultat de la comparaison:**
  - ✅ Amélioration: Le nouveau modèle est meilleur
  - ❌ Pas d'amélioration: Le v_best précédent est conservé
  - ℹ️ Premier modèle: Pas de v_best précédent

- **Statut v_best:**
  - 🌟 Nouveau v_best: Ce modèle devient la référence
  - 📦 V_best conservé: L'ancien modèle reste la référence

- **Comparaison des métriques:**
  - Tableau détaillé: Actuel vs V_Best
  - Différence absolue et en pourcentage
  - Indicateur d'amélioration (✅/❌) par métrique
  - Graphique en barres pour visualisation

**Pourquoi c'est important:**
- **Système v_best:** Garantit que seuls les meilleurs modèles sont déployés
- Évite les régressions de performance
- Fournit une traçabilité des améliorations
- Justifie les décisions de déploiement

**Logique de décision:**
- Un modèle devient v_best s'il est meilleur sur **plus de 50% des métriques**
- Sinon, le v_best précédent est conservé
- Cette approche évite les faux positifs et garantit une amélioration réelle

**Cas d'usage:**
- Validation avant déploiement en production
- Suivi de l'évolution des performances
- Justification des mises à jour de modèle
- Reporting aux équipes métier

---

### 4. 🚨 **Monitoring Drift**

**Objectif:** Surveiller les changements dans la distribution des données entre l'entraînement et la production.

**Contenu:**
- **Statut Global du Drift:**
  - ✅ OK: < 20% des features montrent un drift
  - ⚠️ ATTENTION: 20-50% des features montrent un drift
  - 🚨 ALARMANT: > 50% des features montrent un drift

- **Résumé des Métriques:**
  - Nombre total de caractéristiques
  - Nombre de caractéristiques avec drift détecté
  - Pourcentage de drift
  - Nombre de nouveaux échantillons analysés

- **Détails par Caractéristique:**
  - Tableau complet avec:
    - Drift détecté (OUI/NON)
    - Score de drift (p-value)
    - Moyenne de référence vs nouvelle moyenne
    - Décalage moyen
    - Tests statistiques (KS p-value, MW p-value)

- **Visualisation des Scores de Drift:**
  - Graphique en barres montrant le score de drift par feature
  - Couleur rouge pour drift détecté, vert pour OK
  - Ligne de seuil de détection (0.05)

- **Informations sur le Rapport:**
  - Date de génération
  - Nombre d'échantillons de référence
  - Seuil de détection utilisé
  - Métadonnées du modèle

- **Recommandations:**
  - Actions suggérées selon le statut (OK, ATTENTION, ALARMANT)
  - Guidance pour le réentraînement
  - Conseils de surveillance

- **Rapports Disponibles:**
  - Liste des rapports de drift générés
  - Accès aux rapports historiques

**Pourquoi c'est important:**
- **Data Drift:** Les données en production changent souvent avec le temps
- Un modèle entraîné sur d'anciennes données peut devenir obsolète
- La détection précoce du drift permet d'anticiper les problèmes
- Évite la dégradation silencieuse des performances

**Tests Statistiques Utilisés:**
- **Kolmogorov-Smirnov (KS):** Détecte les changements de distribution
- **Mann-Whitney U (MW):** Détecte les différences de médiane
- **Drift Score:** Minimum des deux p-values (approche conservatrice)

**Cas d'usage:**
- Surveillance continue en production
- Planification du réentraînement
- Détection d'anomalies dans les données
- Validation de la qualité des données
- Alertes automatiques

**Workflow typique:**
1. Exécuter `python run_full_pipeline.py`
2. Le rapport de drift est généré automatiquement
3. Consulter l'onglet Monitoring Drift dans Streamlit
4. Analyser le statut global et les features affectées
5. Prendre des décisions (continuer, surveiller, réentraîner)

---

### 5. 📚 **Guide d'utilisation**

**Objectif:** Documentation complète pour utiliser le pipeline ML.

**Contenu:**

**1. Prérequis:**
- Logiciels requis (Python, Docker, Git)
- Configuration système minimale

**2. Installation:**
- Clonage du repository
- Création de l'environnement virtuel
- Installation des dépendances

**3. Configuration YAML:**
- Structure du fichier de configuration
- Exemples pour classification et régression
- Paramètres disponibles

**4. Préparation des Données:**
- Format CSV attendu
- Structure des données
- Conseils de qualité

**5. Exécution du Pipeline:**
- **Pipeline Complet:** `python run_full_pipeline.py`
- **Étapes Individuelles:** build, test, evaluate, compare, deploy, drift
- **Déploiement Docker:** docker-compose

**6. Services API et Dashboard:**
- Configuration de l'authentification
- Utilisation de l'API (exemples curl)
- Lancement du Dashboard Streamlit

**7. Surveillance du Drift:**
- Génération du rapport
- Interprétation des résultats
- Actions recommandées

**8. Dépannage:**
- Problèmes courants et solutions
- Rapport de drift non visible
- Erreurs de modèle
- Problèmes d'authentification API
- Conflits de ports
- Dépendances manquantes

**Pourquoi c'est important:**
- Permet aux nouveaux utilisateurs de démarrer rapidement
- Documentation centralisée et accessible
- Réduit les questions de support
- Facilite l'adoption du pipeline

**Cas d'usage:**
- Onboarding de nouveaux membres de l'équipe
- Référence rapide pour les commandes
- Résolution de problèmes
- Formation et tutoriels

---

### 6. ℹ️ **À propos**

**Objectif:** Informations sur le projet et ses technologies.

**Contenu:**
- **Version:** Numéro de version de l'application
- **Description:** Vue d'ensemble du pipeline ML
- **Fonctionnalités principales:**
  - Visualisation des métriques
  - Analyse détaillée des modèles
  - Comparaison de versions
  - Surveillance du drift
  - Guide d'utilisation

- **Technologies utilisées:**
  - Python, Streamlit, FastAPI
  - Docker, Scikit-learn
  - Pandas, NumPy, Matplotlib, Seaborn

- **Système v_best:**
  - Explication du concept
  - Garantie de qualité
  - Processus de sélection

- **Surveillance du Drift:**
  - Importance de la détection
  - Système d'alertes automatiques
  - Seuils configurables

- **Informations légales:**
  - Auteur
  - Licence

**Pourquoi c'est important:**
- Fournit le contexte du projet
- Crédite les technologies utilisées
- Explique les concepts clés (v_best, drift)
- Informations de contact et licence

---

## 🎨 Design et Expérience Utilisateur

**Caractéristiques de l'interface:**
- **Navigation intuitive:** Barre latérale avec icônes
- **Visualisations riches:** Graphiques interactifs
- **Code couleur:** Vert (succès), Orange (attention), Rouge (alerte)
- **Responsive:** S'adapte à différentes tailles d'écran
- **Rafraîchissement:** Bouton pour recharger les données
- **Expandeurs:** Sections repliables pour plus de détails

---

## 🔄 Workflow Typique d'Utilisation

```
1. Exécuter le pipeline
   └─> python run_full_pipeline.py

2. Lancer Streamlit
   └─> streamlit run app/app.py

3. Consulter le Tableau de bord
   └─> Vérifier les métriques principales

4. Analyser les Détails du modèle
   └─> Examiner les paramètres et l'historique

5. Vérifier la Comparaison
   └─> Confirmer que le modèle est v_best

6. Surveiller le Drift
   └─> Analyser le statut et les features affectées

7. Décider des actions
   └─> Continuer, surveiller, ou réentraîner
```

---

## 🚀 Avantages de l'Application Streamlit

1. **Accessibilité:** Interface web accessible depuis n'importe quel navigateur
2. **Interactivité:** Exploration dynamique des données et visualisations
3. **Centralisation:** Toutes les informations en un seul endroit
4. **Pas de code:** Les utilisateurs non-techniques peuvent analyser les résultats
5. **Temps réel:** Rafraîchissement facile des données
6. **Visualisations riches:** Graphiques professionnels et informatifs
7. **Documentation intégrée:** Guide d'utilisation inclus dans l'application

---

## 📊 Cas d'Usage par Profil

**Data Scientist:**
- Onglet "Détails du modèle" pour analyser les paramètres
- Onglet "Comparaison" pour valider les améliorations
- Onglet "Monitoring Drift" pour planifier le réentraînement

**Manager / Product Owner:**
- Onglet "Tableau de bord" pour vue d'ensemble rapide
- Onglet "Comparaison" pour justifier les déploiements
- Onglet "À propos" pour comprendre le système

**DevOps / MLOps:**
- Onglet "Monitoring Drift" pour alertes et surveillance
- Onglet "Guide d'utilisation" pour déploiement
- Onglet "Détails du modèle" pour debugging

**Nouveau membre de l'équipe:**
- Onglet "Guide d'utilisation" pour démarrer
- Onglet "À propos" pour comprendre le contexte
- Tous les onglets pour exploration

---

## 🔧 Configuration et Lancement

**Lancement de l'application:**
```bash
# Depuis la racine du projet
streamlit run app/app.py

# Avec un port spécifique
streamlit run app/app.py --server.port 8501
```

**Accès:**
- URL locale: http://localhost:8501
- L'application se lance automatiquement dans le navigateur

**Rafraîchissement des données:**
- Bouton "🔄 Rafraîchir les données" dans l'onglet Monitoring Drift
- Ou relancer l'application après génération de nouveaux rapports

---

## 📝 Résumé

L'application Streamlit est le **tableau de bord central** du pipeline ML. Elle:

1. **Visualise les performances** du modèle déployé
2. **Analyse en profondeur** les paramètres et l'historique
3. **Compare les versions** pour garantir la qualité (système v_best)
4. **Surveille le drift** pour anticiper les problèmes
5. **Guide les utilisateurs** avec une documentation intégrée
6. **Informe sur le projet** et ses technologies

Cette application transforme des **données techniques complexes** en **informations visuelles accessibles**, permettant à tous les membres de l'équipe de comprendre et de suivre les performances du modèle ML en production.

**Philosophie:** Rendre le Machine Learning **transparent**, **compréhensible** et **actionnable** pour tous.