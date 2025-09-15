"""
Application Streamlit pour visualiser les résultats des modèles de machine learning.
"""
import os
import sys
import json
import base64
import datetime
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from PIL import Image

# Configuration de la page
st.set_page_config(
    page_title="ML Pipeline - Tableau de bord",
    page_icon="📊",
    layout="wide"
)

# Titre et description
st.title("📊 ML Pipeline - Tableau de bord")
st.write("""
Cette application présente les résultats des modèles de machine learning exécutés par le pipeline CI/CD.
Elle permet de visualiser les performances des modèles, de comparer les versions et de suivre l'évolution des métriques.
""")

# Fonction pour charger les données
@st.cache_data
def load_data(path):
    """Charge les données JSON depuis un fichier."""
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        st.error(f"Erreur lors du chargement des données: {e}")
        return None

# Fonction pour convertir une image base64 en image PIL
def base64_to_image(base64_str):
    """Convertit une chaîne base64 en image PIL."""
    try:
        img_data = base64.b64decode(base64_str)
        return Image.open(BytesIO(img_data))
    except Exception as e:
        st.error(f"Erreur lors de la conversion de l'image: {e}")
        return None

# Fonction pour charger le dernier rapport de drift
@st.cache_data
def load_drift_report():
    """Charge le dernier rapport de drift."""
    drift_path = "reports/drift/latest_drift_report.json"
    if os.path.exists(drift_path):
        return load_data(drift_path)
    return None

# Fonction pour lister les rapports de drift disponibles
def list_drift_reports():
    """Liste tous les rapports de drift disponibles."""
    drift_dir = "reports/drift"
    if not os.path.exists(drift_dir):
        return []
    
    reports = []
    for file in os.listdir(drift_dir):
        if file.endswith('.json') and file.startswith('drift_report_'):
            reports.append(file)
    
    return sorted(reports, reverse=True)

# Barre latérale pour la navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Choisir une page",
    ["📈 Tableau de bord", "🔍 Détails du modèle", "🔄 Comparaison", "🚨 Monitoring Drift", "📚 Guide d'utilisation", "ℹ️ À propos"]
)

# Page: Tableau de bord
if page == "📈 Tableau de bord":
    st.header("📈 Tableau de bord")
    
    reports_path = "deploy"
    if os.path.exists(reports_path):
        st.success("Modèles déployés trouvés!")
        
        metadata = load_data(os.path.join(reports_path, "model_metadata.json"))
        if metadata:
            st.subheader("📋 Informations sur le modèle")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Nom du modèle", metadata['model_name'])
            with col2:
                st.metric("Type de modèle", metadata['model_type'].capitalize())
            with col3:
                st.metric("Version", metadata['model_version'])
            
            evaluation = load_data(os.path.join(reports_path, "evaluation_report.json"))
            if evaluation:
                st.subheader("📊 Métriques principales")
                metrics = evaluation['metrics']
                
                cols = st.columns(len(metrics))
                for i, (name, value) in enumerate(metrics.items()):
                    with cols[i]:
                        if name.lower() in ['mse', 'mae', 'rmse']:
                            st.metric(name.upper(), f"{value:.4f}", "Erreur")
                        else:
                            st.metric(name.capitalize(), f"{value:.4f}", "")
                
                if 'visualizations' in evaluation:
                    st.subheader("📉 Visualisations")
                    visualizations = evaluation['visualizations']
                    
                    if metadata['model_type'] == 'classification':
                        if 'confusion_matrix' in visualizations:
                            st.image(base64_to_image(visualizations['confusion_matrix']), caption="Matrice de confusion")
                        
                        if 'roc_curve' in visualizations and 'pr_curve' in visualizations:
                            col1, col2 = st.columns(2)
                            with col1:
                                st.image(base64_to_image(visualizations['roc_curve']), caption="Courbe ROC")
                            with col2:
                                st.image(base64_to_image(visualizations['pr_curve']), caption="Courbe Précision-Rappel")
                                
                    elif metadata['model_type'] == 'regression':
                        if 'scatter_plot' in visualizations and 'error_histogram' in visualizations:
                            col1, col2 = st.columns(2)
                            with col1:
                                st.image(base64_to_image(visualizations['scatter_plot']), caption="Prédictions vs Valeurs réelles")
                            with col2:
                                st.image(base64_to_image(visualizations['error_histogram']), caption="Distribution des erreurs")
        else:
            st.warning("Aucune métadonnée de modèle trouvée.")
    else:
        st.warning("Aucun modèle déployé trouvé. Veuillez exécuter le pipeline complet pour déployer un modèle.")

# Page: Détails du modèle
elif page == "🔍 Détails du modèle":
    st.header("🔍 Détails du modèle")
    
    deploy_path = "deploy"
    if os.path.exists(deploy_path):
        metadata = load_data(os.path.join(deploy_path, "model_metadata.json"))
        if metadata:
            st.subheader("🔎 Paramètres du modèle")
            st.json(metadata)
            
            if 'training_params' in metadata:
                st.subheader("⚙️ Paramètres d'entraînement")
                for param, value in metadata['training_params'].items():
                    st.text(f"{param}: {value}")
                    
            st.subheader("📝 Historique des métriques")
            if 'validation_metrics' in metadata:
                validation_metrics = metadata['validation_metrics']
                df_val = pd.DataFrame({
                    'Métrique': list(validation_metrics.keys()),
                    'Validation': list(validation_metrics.values())
                })
                
                evaluation = load_data(os.path.join(deploy_path, "evaluation_report.json"))
                if evaluation and 'metrics' in evaluation:
                    test_metrics = evaluation['metrics']
                    df_val['Test'] = [test_metrics.get(metric, None) for metric in validation_metrics.keys()]
                
                st.dataframe(df_val)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                df_melt = pd.melt(df_val, id_vars=['Métrique'], value_vars=['Validation', 'Test'])
                sns.barplot(x='Métrique', y='value', hue='variable', data=df_melt)
                plt.title('Comparaison des métriques de validation et de test')
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
        else:
            st.warning("Aucune métadonnée de modèle trouvée.")
    else:
        st.warning("Aucun modèle déployé trouvé. Veuillez exécuter le pipeline complet pour déployer un modèle.")

# Page: Comparaison
elif page == "🔄 Comparaison":
    st.header("🔄 Comparaison des modèles")
    
    deploy_path = "deploy"
    if os.path.exists(deploy_path):
        comparison_path = os.path.join("build", "comparison_report.json")
        if os.path.exists(comparison_path):
            comparison = load_data(comparison_path)
            if comparison:
                st.subheader("📋 Informations sur les modèles")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Modèle actuel**")
                    st.text(f"Nom: {comparison['current_model']['name']}")
                    st.text(f"Version: {comparison['current_model']['version']}")
                    st.text(f"Type: {comparison['current_model']['type']}")
                
                with col2:
                    st.markdown("**Modèle v_best**")
                    if 'v_best_model' in comparison and comparison['v_best_model']['version']:
                        st.text(f"Version: {comparison['v_best_model']['version']}")
                    else:
                        st.text("Pas de version v_best précédente")
                
                st.subheader("🏆 Résultat de la comparaison")
                if comparison['overall_improvement'] is None:
                    st.info("Pas de modèle v_best précédent pour comparaison")
                elif comparison['overall_improvement']:
                    st.success("✅ Le modèle actuel est meilleur que le v_best précédent!")
                else:
                    st.error("❌ Le modèle actuel n'est pas meilleur que le v_best précédent")
                
                if comparison.get('is_new_v_best', False):
                    st.success("🌟 Ce modèle est devenu le nouveau v_best!")
                else:
                    st.info("📦 Le modèle v_best précédent a été conservé")
                
                if comparison['metrics_comparison']:
                    st.subheader("📊 Comparaison des métriques")
                    
                    metrics_data = []
                    for metric, values in comparison['metrics_comparison'].items():
                        if values['v_best'] is not None:
                            metrics_data.append({
                                'Métrique': metric,
                                'Actuel': values['current'],
                                'V_Best': values['v_best'],
                                'Différence': values['absolute_diff'],
                                'Différence (%)': values['percentage_diff'],
                                'Amélioration': '✅' if values['is_improvement'] else '❌'
                            })
                        else:
                            metrics_data.append({
                                'Métrique': metric,
                                'Actuel': values['current'],
                                'V_Best': 'N/A',
                                'Différence': 'N/A',
                                'Différence (%)': 'N/A',
                                'Amélioration': 'Premier modèle'
                            })
                    
                    if metrics_data:
                        df_comparison = pd.DataFrame(metrics_data)
                        st.dataframe(df_comparison)
                        
                        comparable_data = [row for row in metrics_data if row['V_Best'] != 'N/A']
                        if comparable_data:
                            fig, ax = plt.subplots(figsize=(10, 6))
                            df_comparable = pd.DataFrame(comparable_data)
                            df_melt = pd.melt(
                                df_comparable, 
                                id_vars=['Métrique'], 
                                value_vars=['Actuel', 'V_Best'],
                                var_name='Version',
                                value_name='Valeur'
                            )
                            sns.barplot(x='Métrique', y='Valeur', hue='Version', data=df_melt)
                            plt.title('Comparaison des métriques entre versions')
                            plt.xticks(rotation=45)
                            plt.tight_layout()
                            st.pyplot(fig)
            else:
                st.warning("Impossible de charger le rapport de comparaison.")
        else:
            st.info("Aucun rapport de comparaison disponible. Exécutez le pipeline avec au moins deux versions de modèle pour générer une comparaison.")
    else:
        st.warning("Aucun modèle déployé trouvé. Veuillez exécuter le pipeline complet pour déployer un modèle.")

# Page: Monitoring Drift
elif page == "🚨 Monitoring Drift":
    st.header("🚨 Monitoring du Data Drift")
    
    st.write("""
    Cette page présente les résultats de la surveillance du Data Drift, qui détecte les changements 
    dans la distribution des données entre l'entraînement et la production.
    """)
    
    drift_report = load_drift_report()
    
    if drift_report:
        st.subheader("📊 Statut Global du Drift")
        
        global_status = drift_report.get('globalstatus', 'INCONNU')
        summary = drift_report.get('summary', {})
        
        if global_status == "OK":
            st.success(f"✅ Statut: {global_status}")
        elif global_status == "ATTENTION":
            st.warning(f"⚠️ Statut: {global_status}")
        elif global_status == "ALARMANT":
            st.error(f"🚨 Statut: {global_status}")
        else:
            st.info(f"ℹ️ Statut: {global_status}")
        
        st.subheader("📈 Résumé des Métriques")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Caractéristiques Totales", summary.get('total_features', 0))
        with col2:
            st.metric("Drift Détecté", summary.get('drift_detected_count', 0))
        with col3:
            st.metric("Pourcentage de Drift", f"{summary.get('drift_percentage', 0):.1f}%")
        with col4:
            st.metric("Nouveaux Échantillons", summary.get('new_samples', 0))
        
        st.subheader("🔍 Détails par Caractéristique")
        
        feature_results = drift_report.get('feature_results', {})
        if feature_results:
            drift_data = []
            for feature, results in feature_results.items():
                drift_data.append({
                    'Caractéristique': feature,
                    'Drift Détecté': '🚨 OUI' if results.get('drift_detected', False) else '✅ NON',
                    'Score de Drift': f"{results.get('drift_score', 0):.4f}",
                    'Moyenne Réf.': f"{results.get('ref_mean', 0):.4f}",
                    'Moyenne Nouvelle': f"{results.get('new_mean', 0):.4f}",
                    'Décalage': f"{results.get('mean_shift', 0):.4f}",
                    'KS p-value': f"{results.get('ks_p_value', 0):.4f}",
                    'MW p-value': f"{results.get('mw_p_value', 0):.4f}"
                })
            
            df_drift = pd.DataFrame(drift_data)
            st.dataframe(df_drift, use_container_width=True)
            
            st.subheader("📊 Visualisation des Scores de Drift")
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            features = list(feature_results.keys())
            drift_scores = [feature_results[f].get('drift_score', 0) for f in features]
            drift_detected = [feature_results[f].get('drift_detected', False) for f in features]
            
            colors = ['red' if detected else 'green' for detected in drift_detected]
            
            bars = ax.bar(features, drift_scores, color=colors, alpha=0.7)
            ax.axhline(y=0.05, color='orange', linestyle='--', label='Seuil de détection (0.05)')
            
            ax.set_xlabel('Caractéristiques')
            ax.set_ylabel('Score de Drift (p-value)')
            ax.set_title('Scores de Drift par Caractéristique')
            ax.set_yscale('log')
            plt.xticks(rotation=45, ha='right')
            plt.legend()
            plt.tight_layout()
            
            st.pyplot(fig)
        
        st.subheader("ℹ️ Informations sur le Rapport")
        
        col1, col2 = st.columns(2)
        with col1:
            st.text(f"Généré le: {drift_report.get('timestamp', 'Inconnu')}")
            st.text(f"Échantillons de référence: {summary.get('reference_samples', 0)}")
            st.text(f"Seuil de détection: {summary.get('drift_threshold', 0.05)}")
        
        with col2:
            metadata = drift_report.get('metadata', {})
            if metadata:
                model_config = metadata.get('model_config', {})
                st.text(f"Modèle: {model_config.get('name', 'Inconnu')}")
                st.text(f"Version: {model_config.get('version', 'Inconnu')}")
                st.text(f"Type: {model_config.get('type', 'Inconnu')}")
        
        st.subheader("💡 Recommandations")
        
        if global_status == "ALARMANT":
            st.error("""
            **Action Immédiate Requise:**
            - Plus de 50% des caractéristiques montrent un drift significatif
            - Considérer un réentraînement du modèle
            - Vérifier la qualité des nouvelles données
            - Analyser les causes du changement de distribution
            """)
        elif global_status == "ATTENTION":
            st.warning("""
            **Surveillance Renforcée:**
            - 20-50% des caractéristiques montrent un drift
            - Surveiller de près l'évolution
            - Préparer un plan de réentraînement
            - Analyser les caractéristiques affectées
            """)
        else:
            st.success("""
            **Situation Normale:**
            - Moins de 20% des caractéristiques montrent un drift
            - Continuer la surveillance régulière
            - Le modèle peut continuer à être utilisé
            """)
    
    else:
        st.warning("""
        Aucun rapport de drift disponible. 
        
        Pour générer un rapport de drift:
        1. Exécutez le pipeline complet avec: `python src/pipeline_with_drift.py`
        2. Ou exécutez directement: `python src/monitor_drift.py`
        
        Les rapports seront sauvegardés dans le dossier `reports/drift/`.
        """)

# Page: Guide d'utilisation
elif page == "📚 Guide d'utilisation":
    st.header("📚 Guide d'utilisation du Pipeline ML")
    
    st.write("""
    Ce guide vous explique comment utiliser ce pipeline ML pour automatiser vos propres modèles.
    Suivez ces étapes pour intégrer votre modèle dans le système v_best avec surveillance du drift.
    """)
    
    st.subheader("📋 Table des Matières")
    st.markdown("""
    1. [Prérequis](#prérequis)
    2. [Structure du Projet](#structure-du-projet)
    3. [Configuration YAML](#configuration-yaml)
    4. [Préparation des Données](#préparation-des-données)
    5. [Exécution du Pipeline](#exécution-du-pipeline)
    6. [Services API et Dashboard](#services-api-et-dashboard)
    7. [Surveillance du Drift](#surveillance-du-drift)
    8. [Intégration GitHub Actions](#intégration-github-actions)
    9. [Dépannage](#dépannage)
    """)
    
    # Section 1: Prérequis
    st.subheader("1. 🔧 Prérequis")
    
    st.markdown("""
    **Logiciels requis:**
    - Python 3.8+ 
    - Docker et Docker Compose
    - Git
    - Au moins 4GB de RAM disponible
    
    **Dépendances Python:**
    """)
    
    st.code("""
pip install -r requirements.txt

# Principales dépendances:
# - scikit-learn
# - pandas
# - numpy
# - fastapi
# - streamlit
# - pyyaml
# - matplotlib
# - seaborn
    """, language="bash")
    
    # Section 2: Structure du Projet
    st.subheader("2. 📁 Structure du Projet")
    
    st.markdown("**Voici la structure de dossiers que vous devez respecter:**")
    
    st.code("""
ml-pipeline-project/
├── api/                    # API FastAPI
│   ├── main.py            # Application principale
│   ├── model_loader.py    # Chargement des modèles
│   ├── auth.py            # Authentification
│   └── requirements.txt   # Dépendances API
├── app/                   # Dashboard Streamlit
│   └── app.py            # Application Streamlit
├── src/                   # Scripts du pipeline ML
│   ├── build_model.py    # Construction du modèle
│   ├── test_model.py     # Tests du modèle
│   ├── evaluate_model.py # Évaluation
│   ├── compare_models.py # Comparaison v_best
│   ├── deploy_model.py   # Déploiement
│   └── monitor_drift.py  # Surveillance drift
├── models/               # Configuration des modèles
│   └── model_config.yml  # Configuration YAML
├── data/                 # Données d'entraînement
├── deploy/               # Modèles déployés
├── build/                # Artefacts de build
├── reports/              # Rapports de drift
├── docker-compose.yml    # Configuration Docker
├── .env                  # Variables d'environnement
└── run_full_pipeline.py  # Script principal
    """, language="text")
    
    # Section 3: Configuration YAML
    st.subheader("3. ⚙️ Configuration YAML")
    
    st.markdown("**Le fichier `models/model_config.yml` est le cœur de votre configuration:**")
    
    tab1, tab2 = st.tabs(["🎯 Classification", "📈 Régression"])
    
    with tab1:
        st.markdown("**Exemple pour un modèle de classification:**")
        st.code("""
model:
  name: "MonModeleClassification"
  type: "classification"
  version: "1.0.0"

data:
  train_path: "data/mon_dataset_train.csv"
  test_path: "data/mon_dataset_test.csv"  # Optionnel
  test_split: 0.2                         # Si pas de test_path
  validation_split: 0.2

parameters:
  max_depth: 10
  n_estimators: 100
  random_state: 42

training:
  epochs: 100
  batch_size: 32
  learning_rate: 0.001

evaluation:
  metrics:
    - accuracy
    - precision
    - recall
    - f1

drift_monitoring:
  enabled: true
  threshold: 0.05
  methods:
    - kolmogorov_smirnov
    - mann_whitney
        """, language="yaml")
    
    with tab2:
        st.markdown("**Exemple pour un modèle de régression:**")
        st.code("""
model:
  name: "MonModeleRegression"
  type: "regression"
  version: "1.0.0"

data:
  train_path: "data/mon_dataset_train.csv"
  test_path: "data/mon_dataset_test.csv"  # Optionnel
  test_split: 0.2                         # Si pas de test_path
  validation_split: 0.2

parameters:
  max_depth: 15
  n_estimators: 200
  random_state: 42

training:
  epochs: 150
  batch_size: 64
  learning_rate: 0.01

evaluation:
  metrics:
    - mse
    - mae
    - rmse

drift_monitoring:
  enabled: true
  threshold: 0.05
  methods:
    - kolmogorov_smirnov
    - mann_whitney
        """, language="yaml")
    
    # Section 4: Préparation des Données
    st.subheader("4. 📊 Préparation des Données")
    
    st.markdown("""
    **Format des données:**
    - Format CSV avec en-têtes
    - Dernière colonne = variable cible
    - Pas de valeurs manquantes
    - Variables numériques uniquement
    
    **Structure attendue:**
    """)
    
    st.code("""
feature1,feature2,feature3,target
1.2,3.4,5.6,0
2.1,4.3,6.5,1
3.0,5.2,7.4,0
...
    """, language="csv")
    
    st.markdown("""
    **Conseils:**
    - Minimum 1000 échantillons recommandé
    - Équilibrage des classes pour la classification
    - Normalisation/standardisation si nécessaire
    - Validation de la qualité des données
    """)
    
    # Section 5: Exécution du Pipeline
    st.subheader("5. 🚀 Exécution du Pipeline")
    
    st.markdown("**Méthodes d'exécution disponibles:**")
    
    exec_tab1, exec_tab2, exec_tab3 = st.tabs(["🎯 Pipeline Complet", "🔧 Étapes Individuelles", "🐳 Docker"])
    
    with exec_tab1:
        st.markdown("**Exécution complète du pipeline:**")
        st.code("""
# Exécution complète avec toutes les étapes
python run_full_pipeline.py

# Avec options avancées
python run_full_pipeline.py --config models/model_config.yml --verbose

# Sans Docker (développement)
python run_full_pipeline.py --skip-docker

# Sans surveillance du drift
python run_full_pipeline.py --skip-drift
        """, language="bash")
        
        st.markdown("**Le pipeline exécute automatiquement:**")
        st.markdown("""
        1. ✅ Vérification des dépendances
        2. 📊 Génération de données (si nécessaire)
        3. 🏗️ Construction du modèle
        4. 🧪 Tests du modèle
        5. 📈 Évaluation des performances
        6. 🔄 Comparaison avec v_best
        7. 🚨 Surveillance du drift
        8. 🚀 Déploiement
        9. 🐳 Lancement des services Docker
        """)
    
    with exec_tab2:
        st.markdown("**Exécution étape par étape:**")
        st.code("""
# 1. Construction du modèle
python src/build_model.py

# 2. Test du modèle
python src/test_model.py

# 3. Évaluation
python src/evaluate_model.py

# 4. Comparaison avec v_best
python src/compare_models.py

# 5. Surveillance du drift
python src/monitor_drift.py --config models/model_config.yml --new-data data/new_data.csv

# 6. Déploiement
python src/deploy_model.py
        """, language="bash")
        
        st.info("💡 **Conseil:** L'exécution étape par étape est utile pour le débogage et le développement.")
    
    with exec_tab3:
        st.markdown("**Déploiement avec Docker:**")
        st.code("""
# Lancement complète des services
docker-compose up -d

# Construction et lancement
docker-compose up -d --build

# Voir les logs
docker-compose logs -f

# Arrêter les services
docker-compose down

# Services individuels
docker-compose up -d api        # API seulement
docker-compose up -d streamlit  # Dashboard seulement
        """, language="bash")
        
        st.markdown("**Services disponibles:**")
        st.markdown("""
        - **API FastAPI**: http://localhost:8000
        - **Dashboard Streamlit**: http://localhost:8501
        - **Documentation API**: http://localhost:8000/docs
        """)
    
    # Section 6: Services API et Dashboard
    st.subheader("6. 🌐 Services API et Dashboard")
    
    st.markdown("**Configuration de l'authentification:**")
    st.code("""
# Créer le fichier .env
echo "API_TOKEN=votre-token-secret-ici" > .env
echo "API_HOST=0.0.0.0" >> .env
echo "API_PORT=8000" >> .env
echo "DEBUG=True" >> .env
    """, language="bash")
    
    st.markdown("**Utilisation de l'API:**")
    st.code("""
# Test de santé
curl http://localhost:8000/health

# Prédiction (avec authentification)
curl -X POST "http://localhost:8000/predict" \\
     -H "Authorization: Bearer votre-token" \\
     -H "Content-Type: application/json" \\
     -d '{
       "data": {
         "feature1": 1.0,
         "feature2": 2.0,
         "feature3": 3.0
       }
     }'

# Informations sur le modèle
curl http://localhost:8000/model/info

# Rechargement du modèle
curl -X POST "http://localhost:8000/model/reload" \\
     -H "Authorization: Bearer votre-token"
    """, language="bash")
    
    # Section 7: Surveillance du Drift
    st.subheader("7. 🚨 Surveillance du Drift")
    
    st.markdown("**Configuration du monitoring du drift:**")

# Page: À propos
elif page == "ℹ️ À propos":
    st.header("ℹ️ À propos")
    
    st.markdown("""
    ## ML Pipeline - Tableau de bord
    
    **Version:** 1.0.0
    
    **Description:**
    Cette application Streamlit fait partie d'un pipeline CI/CD complet pour le machine learning.
    Elle permet de visualiser les résultats des modèles, de comparer les performances et de surveiller
    le data drift en production.
    
    **Fonctionnalités principales:**
    - 📊 Visualisation des métriques de performance
    - 🔍 Analyse détaillée des modèles
    - 🔄 Comparaison entre versions de modèles
    - 🚨 Surveillance du data drift
    - 📚 Guide d'utilisation complet
    
    **Technologies utilisées:**
    - Python 3.8+
    - Streamlit
    - FastAPI
    - Docker
    - Scikit-learn
    - Pandas, NumPy, Matplotlib
    
    **Auteur:** Votre Nom
    **Contact:** votre.email@example.com
    **Licence:** MIT
    """)