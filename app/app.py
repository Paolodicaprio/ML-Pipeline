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

# Barre latérale pour la navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Choisir une page",
    ["📈 Tableau de bord", "🔍 Détails du modèle", "🔄 Comparaison", "ℹ️ À propos"]
)

# Page: Tableau de bord
if page == "📈 Tableau de bord":
    st.header("📈 Tableau de bord")
    
    # Rechercher les rapports disponibles
    reports_path = "deploy"
    if os.path.exists(reports_path):
        st.success("Modèles déployés trouvés!")
        
        # Charger les métadonnées du modèle
        metadata = load_data(os.path.join(reports_path, "model_metadata.json"))
        if metadata:
            # Afficher les informations du modèle
            st.subheader("📋 Informations sur le modèle")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Nom du modèle", metadata['model_name'])
            with col2:
                st.metric("Type de modèle", metadata['model_type'].capitalize())
            with col3:
                st.metric("Version", metadata['model_version'])
            
            # Charger le rapport d'évaluation
            evaluation = load_data(os.path.join(reports_path, "evaluation_report.json"))
            if evaluation:
                # Afficher les métriques principales
                st.subheader("📊 Métriques principales")
                metrics = evaluation['metrics']
                
                # Créer une mise en page en colonnes pour les métriques
                cols = st.columns(len(metrics))
                for i, (name, value) in enumerate(metrics.items()):
                    with cols[i]:
                        if name.lower() in ['mse', 'mae', 'rmse']:  # Métriques d'erreur (plus bas = mieux)
                            st.metric(name.upper(), f"{value:.4f}", "Erreur")
                        else:  # Métriques de performance (plus haut = mieux)
                            st.metric(name.capitalize(), f"{value:.4f}", "")
                
                # Visualisations
                if 'visualizations' in evaluation:
                    st.subheader("📉 Visualisations")
                    visualizations = evaluation['visualizations']
                    
                    # Afficher les visualisations selon le type de modèle
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
    
    # Vérifier si un modèle est déployé
    deploy_path = "deploy"
    if os.path.exists(deploy_path):
        # Charger les métadonnées du modèle
        metadata = load_data(os.path.join(deploy_path, "model_metadata.json"))
        if metadata:
            # Informations détaillées sur le modèle
            st.subheader("🔎 Paramètres du modèle")
            st.json(metadata)
            
            # Afficher les paramètres d'entraînement
            if 'training_params' in metadata:
                st.subheader("⚙️ Paramètres d'entraînement")
                for param, value in metadata['training_params'].items():
                    st.text(f"{param}: {value}")
                    
            # Historique des métriques
            st.subheader("📝 Historique des métriques")
            if 'validation_metrics' in metadata:
                validation_metrics = metadata['validation_metrics']
                df_val = pd.DataFrame({
                    'Métrique': list(validation_metrics.keys()),
                    'Validation': list(validation_metrics.values())
                })
                
                # Charger le rapport d'évaluation pour les métriques de test
                evaluation = load_data(os.path.join(deploy_path, "evaluation_report.json"))
                if evaluation and 'metrics' in evaluation:
                    test_metrics = evaluation['metrics']
                    df_val['Test'] = [test_metrics.get(metric, None) for metric in validation_metrics.keys()]
                
                # Afficher le tableau des métriques
                st.dataframe(df_val)
                
                # Graphique des métriques
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
    
    # Vérifier si un rapport de comparaison est disponible
    deploy_path = "deploy"
    if os.path.exists(deploy_path):
        # Tenter de charger un rapport de comparaison
        comparison_path = os.path.join("build", "comparison_report.json")
        if os.path.exists(comparison_path):
            comparison = load_data(comparison_path)
            if comparison:
                # Informations sur le modèle actuel et v_best
                st.subheader("📋 Informations sur les modèles")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Modèle actuel**")
                    st.text(f"Nom: {comparison['current_model']['name']}")
                    st.text(f"Version: {comparison['current_model']['version']}")
                    st.text(f"Type: {comparison['current_model']['type']}")
                
                with col2:
                    st.markdown("**Modèle v_best**")
                    # Utiliser v_best_model au lieu de previous_model
                    if 'v_best_model' in comparison and comparison['v_best_model']['version']:
                        st.text(f"Version: {comparison['v_best_model']['version']}")
                    else:
                        st.text("Pas de version v_best précédente")
                
                # Résultat global de la comparaison
                st.subheader("🏆 Résultat de la comparaison")
                if comparison['overall_improvement'] is None:
                    st.info("Pas de modèle v_best précédent pour comparaison")
                elif comparison['overall_improvement']:
                    st.success("✅ Le modèle actuel est meilleur que le v_best précédent!")
                else:
                    st.error("❌ Le modèle actuel n'est pas meilleur que le v_best précédent")
                
                # Afficher le statut v_best
                if comparison.get('is_new_v_best', False):
                    st.success("🌟 Ce modèle est devenu le nouveau v_best!")
                else:
                    st.info("📦 Le modèle v_best précédent a été conservé")
                
                # Détails de la comparaison des métriques
                if comparison['metrics_comparison']:
                    st.subheader("📊 Comparaison des métriques")
                    
                    # Créer un DataFrame pour la comparaison
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
                            # Premier modèle, pas de comparaison
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
                        
                        # Graphique de comparaison (seulement si on a des données v_best)
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

# Page: À propos
elif page == "ℹ️ À propos":
    st.header("ℹ️ À propos")
    
    st.markdown("""
    ## ML Pipeline CI/CD avec système v_best
    
    Cette application fait partie d'un projet de pipeline CI/CD pour les modèles de machine learning.
    
    ### Fonctionnalités
    
    - **Automatisation**: Construction, test, évaluation et déploiement automatisés des modèles
    - **Métriques**: Calcul automatique des métriques pertinentes selon le type de modèle
    - **Système v_best**: Comparaison des performances et conservation du meilleur modèle
    - **Visualisation**: Visualisation des résultats et des métriques
    
    ### Types de modèles supportés
    
    - **Régression**: Modèles qui produisent des sorties continues
        - Métriques: MSE, MAE, RMSE
    - **Classification**: Modèles qui produisent des sorties discrètes
        - Métriques: Précision, Rappel, F1-score
        
    ### Pipeline GitHub Actions
    
    Le pipeline comprend les étapes suivantes:
    
    1. **Build**: Construction et entraînement du modèle
    2. **Test**: Test du modèle sur des données non vues
    3. **Evaluate**: Évaluation des performances du modèle
    4. **Compare**: Comparaison avec le modèle v_best
    5. **Deploy**: Déploiement du modèle depuis v_best
    
    ### Système v_best
    
    - Le conteneur `v_best` conserve toujours la meilleure version du modèle
    - Chaque nouveau modèle est comparé avec v_best
    - Seuls les modèles avec >50% de métriques améliorées remplacent v_best
    - Le déploiement se fait toujours depuis v_best
    """)

# Pied de page
st.sidebar.markdown("---")
st.sidebar.info(
    "Cette application a été développée dans le cadre d'un projet académique "
    "pour démontrer l'automatisation des workflows de machine learning avec système v_best."
)
st.sidebar.text(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d')}")