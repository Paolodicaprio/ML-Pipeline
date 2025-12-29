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
        with open(path, 'r', encoding='utf-8') as f:
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
    """Charge le dernier rapport de drift (plus robuste).

    Stratégie:
    1) Essayer des chemins explicites `latest_drift_report.json`
    2) Si introuvable, chercher tous les fichiers `drift_report_*.json` et choisir le plus récent
    3) Enregistrer dans `st.session_state['last_drift_path']` le chemin réussi pour faciliter le debug
    4) Signaler les erreurs de parsing pour aider au diagnostic
    """
    # Réinitialiser la trace précédente
    try:
        st.session_state.pop('last_drift_path', None)
    except Exception:
        pass

    # Support pour exécution dans Docker ou environnements conteneurisés
    drift_env = os.environ.get('DRIFT_REPORT_DIR')
    container_app_path = "/app/reports/drift"

    possible_files = [
        r"E:\WORK\Mémoire\ml-pipeline-project\reports\drift\latest_drift_report.json",
        os.path.normpath("../reports/drift/latest_drift_report.json"),
        os.path.normpath("reports/drift/latest_drift_report.json"),
        os.path.join(os.getcwd(), "reports", "drift", "latest_drift_report.json"),
    ]

    # Si l'utilisateur a défini une variable d'environnement pointant vers le dossier des rapports, l'insérer en tête
    if drift_env:
        possible_files.insert(0, os.path.join(drift_env, 'latest_drift_report.json'))

    # Vérifier aussi le chemin usuel à l'intérieur du container
    possible_files.insert(0, os.path.join(container_app_path, 'latest_drift_report.json'))

    for p in possible_files:
        try:
            if os.path.exists(p):
                with open(p, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                st.session_state['last_drift_path'] = os.path.abspath(p)
                return data
        except Exception as e:
            # Laisser l'utilisateur voir l'erreur dans le dashboard
            st.error(f"Erreur lors du chargement du fichier {p}: {e}")
            continue

    # Si aucun `latest_drift_report.json`, chercher tous les rapports et choisir le plus récent (mtime)
    drift_dirs = [
        r"E:\WORK\Mémoire\ml-pipeline-project\reports\drift",
        os.path.normpath("../reports/drift"),
        os.path.normpath("reports/drift"),
        os.path.join(os.getcwd(), "reports", "drift"),
        container_app_path,
    ]

    # Si variable d'environnement définie, l'ajouter aussi
    if drift_env:
        drift_dirs.insert(0, drift_env)

    candidates = []
    for d in drift_dirs:
        if os.path.exists(d) and os.path.isdir(d):
            try:
                for f in os.listdir(d):
                    if f.endswith('.json') and (f.startswith('drift_report_') or f == 'latest_drift_report.json'):
                        candidates.append(os.path.join(d, f))
            except Exception:
                continue

    if candidates:
        # Trier par date de modification, décroissant
        candidates.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        for c in candidates:
            try:
                with open(c, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                st.session_state['last_drift_path'] = os.path.abspath(c)
                return data
            except Exception as e:
                # Continuer si un fichier est corrompu
                st.error(f"Erreur lors du parsing du fichier {c}: {e}")
                continue

    return None

# Fonction pour lister les rapports de drift disponibles
def list_drift_reports():
    """Liste tous les rapports de drift disponibles, triés par date de modification (plus récent d'abord)."""
    drift_dirs = [
        r"E:\WORK\Mémoire\ml-pipeline-project\reports\drift",
        os.path.normpath("../reports/drift"),
        os.path.normpath("reports/drift"),
        os.path.join(os.getcwd(), "reports", "drift"),
    ]

    all_reports = []
    for drift_dir in drift_dirs:
        if os.path.exists(drift_dir) and os.path.isdir(drift_dir):
            try:
                for file in os.listdir(drift_dir):
                    if file.endswith('.json') and (file.startswith('drift_report_') or file == 'latest_drift_report.json'):
                        fullpath = os.path.join(drift_dir, file)
                        try:
                            mtime = os.path.getmtime(fullpath)
                        except Exception:
                            mtime = 0
                        all_reports.append((fullpath, mtime))
            except Exception:
                continue

    if all_reports:
        # Trier par mtime décroissant
        all_reports.sort(key=lambda x: x[1], reverse=True)
        # Retourner juste les noms de fichiers (ou chemins si vous préférez)
        return [os.path.basename(r[0]) for r in all_reports]

    return []

# Barre latérale pour la navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Choisir une page",
    ["📈 Tableau de bord", "🔍 Détails du modèle", "🔄 Comparaison", "🚨 Monitoring Drift", "📚 Guide d'utilisation", "ℹ️ À propos"]
)

# Page: Tableau de bord
if page == "📈 Tableau de bord":
    st.header("📈 Tableau de bord")
    
    # Chercher le dossier deploy dans le parent ou local
    reports_path_parent = "../deploy"
    reports_path_local = "deploy"
    
    reports_path = None
    if os.path.exists(reports_path_parent):
        reports_path = reports_path_parent
    elif os.path.exists(reports_path_local):
        reports_path = reports_path_local
    
    if reports_path and os.path.exists(reports_path):
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
    
    deploy_path_parent = "../deploy"
    deploy_path_local = "deploy"
    
    deploy_path = None
    if os.path.exists(deploy_path_parent):
        deploy_path = deploy_path_parent
    elif os.path.exists(deploy_path_local):
        deploy_path = deploy_path_local
    
    if deploy_path and os.path.exists(deploy_path):
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
    
    deploy_path_parent = "../deploy"
    deploy_path_local = "deploy"
    build_path_parent = "../build"
    build_path_local = "build"
    
    deploy_path = deploy_path_parent if os.path.exists(deploy_path_parent) else deploy_path_local
    build_path = build_path_parent if os.path.exists(build_path_parent) else build_path_local
    
    if os.path.exists(deploy_path):
        comparison_path = os.path.join(build_path, "comparison_report.json")
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
    
    # Bouton pour rafraîchir les données
    if st.button("🔄 Rafraîchir les données"):
        st.cache_data.clear()
        st.rerun()
    
    drift_report = load_drift_report()
    
    if drift_report:
        st.subheader("📊 Statut Global du Drift")
        
        global_status = drift_report.get('global_status', 'INCONNU')
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
            # Indiquer le fichier chargé pour faciliter le debug
            last_path = st.session_state.get('last_drift_path')
            if last_path:
                st.text(f"Rapport chargé depuis: {last_path}")
            else:
                st.text("Rapport chargé depuis: Inconnu")

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
        
        # Afficher les rapports disponibles
        st.subheader("📂 Rapports de Drift Disponibles")
        available_reports = list_drift_reports()
        if available_reports:
            st.info(f"{len(available_reports)} rapport(s) de drift disponible(s)")
            with st.expander("Voir la liste des rapports"):
                for report in available_reports[:10]:  # Afficher les 10 plus récents
                    st.text(f"• {report}")
        else:
            st.warning("Aucun rapport de drift trouvé dans le dossier reports/drift/")
    
    else:
        st.warning("""
        Aucun rapport de drift disponible. 
        
        **Pour générer un rapport de drift:**
        
        1. **Méthode 1 - Pipeline complet** (recommandé):
           ```bash
           python run_full_pipeline.py
           ```
           Cette commande exécute tout le pipeline incluant la génération du rapport de drift.
        
        2. **Méthode 2 - Drift uniquement**:
           ```bash
           python src/monitor_drift.py --config models/model_config.yml --new-data data/classification_test.csv
           ```
           Cette commande génère uniquement le rapport de drift.
        
        Les rapports seront sauvegardés dans le dossier `reports/drift/`.
        
        **Vérification:**
        - Assurez-vous que le dossier `reports/drift/` existe
        - Vérifiez que le fichier `latest_drift_report.json` est présent
        - Cliquez sur le bouton "🔄 Rafraîchir les données" ci-dessus après génération
        """)
        
        # Afficher le chemin de recherche pour debug
        with st.expander("🔧 Informations de débogage"):
            st.text("Chemins recherchés:")
            st.text("1. E:\\WORK\\Mémoire\\ml-pipeline-project\\reports\\drift\\latest_drift_report.json")
            st.text("2. ../reports/drift/latest_drift_report.json")
            st.text("3. reports/drift/latest_drift_report.json")
            st.text("4. Dernier fichier drift_report_*.json dans reports/drift/")
            
            st.text("\nRépertoire de travail actuel:")
            st.text(os.getcwd())
            
            # Vérifier tous les dossiers possibles
            drift_dirs = [
                r"E:\WORK\Mémoire\ml-pipeline-project\reports\drift",
                "../reports/drift",
                "reports/drift"
            ]
            
            for drift_dir in drift_dirs:
                st.text(f"\nFichiers dans {drift_dir} :")
                if os.path.exists(drift_dir):
                    try:
                        files = os.listdir(drift_dir)
                        for f in files[:10]:
                            st.text(f"  • {f}")
                    except Exception as e:
                        st.text(f"  Erreur: {e}")
                else:
                    st.text("  Dossier non trouvé")

            # Infos additionnelles pour le debug: dernier chemin tenté/existant
            last = st.session_state.get('last_drift_path')
            if last:
                st.text(f"Dernier rapport chargé: {last}")
                st.text(f"Chemin existe: {os.path.exists(last)}")
            else:
                st.text("Aucun rapport chargé récemment (session vide)")

            # Afficher variable d'environnement et détection container
            drift_env = os.environ.get('DRIFT_REPORT_DIR')
            st.text(f"DRIFT_REPORT_DIR: {drift_env}")
            in_container = os.path.exists('/.dockerenv') or os.getcwd() == '/app'
            st.text(f"Exécution en container détectée: {in_container}")

            # Présence du fichier latest_drift_report.json (chemin absolu)
            abs_path = r"E:\\WORK\\Mémoire\\ml-pipeline-project\\reports\\drift\\latest_drift_report.json"
            st.text(f"Existence du chemin absolu {abs_path}: {os.path.exists(abs_path)}")

            # Suggestion si exécution dans container sans dossier monté
            if in_container and not (os.path.exists('/app/reports/drift') or os.path.exists(abs_path)):
                st.warning("Le dossier 'reports/drift' n'est pas monté dans le container.\n" \
                           "Si vous utilisez Docker, ajoutez ce volume dans votre docker-compose.yml:\n" \
                           "  volumes:\n" \
                           "    - ./reports:/app/reports:ro\n" \
                           "Puis redémarrez le service streamlit ou exécutez Streamlit localement depuis la racine du projet.")

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
    2. [Installation](#installation)
    3. [Configuration YAML](#configuration-yaml)
    4. [Préparation des Données](#préparation-des-données)
    5. [Exécution du Pipeline](#exécution-du-pipeline)
    6. [Services API et Dashboard](#services-api-et-dashboard)
    7. [Surveillance du Drift](#surveillance-du-drift)
    8. [Dépannage](#dépannage)
    """)
    
    # Section 1: Prérequis
    st.subheader("1. 🔧 Prérequis")
    
    st.markdown("""
    **Logiciels requis:**
    - Python 3.8+ 
    - Docker et Docker Compose (optionnel)
    - Git
    - Au moins 4GB de RAM disponible
    """)
    
    # Section 2: Installation
    st.subheader("2. 📦 Installation")
    
    st.markdown("**Étape 1: Cloner le projet**")
    st.code("""
# Cloner le repository
git clone <votre-repo>
cd ml-pipeline-project
    """, language="bash")
    
    st.markdown("**Étape 2: Créer un environnement virtuel**")
    st.code("""
# Créer l'environnement virtuel
python -m venv venv

# Activer l'environnement
# Sur Windows:
venv\\Scripts\\activate
# Sur Linux/Mac:
source venv/bin/activate
    """, language="bash")
    
    st.markdown("**Étape 3: Installer les dépendances**")
    st.code("""
pip install -r requirements.txt
    """, language="bash")
    
    st.info("💡 **Astuce**: Le pipeline vérifie automatiquement les dépendances et installe les packages manquants lors de l'exécution.")
    
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
  algorithm: "random_forest"

data:
  train_path: "data/mon_dataset_train.csv"
  test_path: "data/mon_dataset_test.csv"  # Optionnel
  test_split: 0.2                         # Si pas de test_path
  validation_split: 0.2

parameters:
  max_depth: 10
  n_estimators: 100
  min_samples_split: 2
  min_samples_leaf: 1

training:
  epochs: 100
  batch_size: 32
  learning_rate: 0.001
  validation_split: 0.2

evaluation:
  metrics:
    - accuracy
    - precision
    - recall
    - f1
        """, language="yaml")
    
    with tab2:
        st.markdown("**Exemple pour un modèle de régression:**")
        st.code("""
model:
  name: "MonModeleRegression"
  type: "regression"
  version: "1.0.0"
  algorithm: "random_forest"

data:
  train_path: "data/mon_dataset_train.csv"
  test_path: "data/mon_dataset_test.csv"  # Optionnel
  test_split: 0.2                         # Si pas de test_path
  validation_split: 0.2

parameters:
  max_depth: 15
  n_estimators: 200
  min_samples_split: 5
  min_samples_leaf: 2

training:
  epochs: 150
  batch_size: 64
  learning_rate: 0.01
  validation_split: 0.2

evaluation:
  metrics:
    - mse
    - mae
    - rmse
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
        7. 🚀 Déploiement
        8. 🚨 Surveillance du drift
        9. 🐳 Lancement des services Docker (optionnel)
        """)
        
        st.success("✅ **Résultat**: Un modèle déployé, des rapports complets, une API et un dashboard opérationnels!")
    
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

# 5. Déploiement
python src/deploy_model.py

# 6. Surveillance du drift
python src/monitor_drift.py --config models/model_config.yml --new-data data/new_data.csv
        """, language="bash")
        
        st.info("💡 **Conseil:** L'exécution étape par étape est utile pour le débogage et le développement.")
    
    with exec_tab3:
        st.markdown("**Déploiement avec Docker:**")
        st.code("""
# Lancement complet des services
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
    
    st.markdown("**Lancement du Dashboard:**")
    st.code("""
# Depuis la racine du projet
streamlit run app/app.py

# Avec un port spécifique
streamlit run app/app.py --server.port 8501
    """, language="bash")
    
    # Section 7: Surveillance du Drift
    st.subheader("7. 🚨 Surveillance du Drift")
    
    st.markdown("""
    **Le data drift** désigne les changements dans la distribution des données entre l'entraînement et la production.
    
    **Génération du rapport de drift:**
    """)
    
    st.code("""
# Méthode 1: Avec le pipeline complet (recommandé)
python run_full_pipeline.py

# Méthode 2: Drift uniquement
python src/monitor_drift.py --config models/model_config.yml --new-data data/new_data.csv
    """, language="bash")
    
    st.markdown("""
    **Interprétation des résultats:**
    
    - **OK (< 20% drift)**: Situation normale, continuer la surveillance
    - **ATTENTION (20-50% drift)**: Surveillance renforcée, préparer le réentraînement
    - **ALARMANT (> 50% drift)**: Réentraînement immédiat recommandé
    
    **Rapports générés:**
    - `reports/drift/drift_report_YYYYMMDD_HHMMSS.json` - Rapport JSON
    - `reports/drift/drift_report_YYYYMMDD_HHMMSS.html` - Rapport HTML avec visualisations
    - `/images/DriftVisualization.jpg` - Graphiques
    - `reports/drift/latest_drift_report.json` - Dernier rapport (utilisé par le dashboard)
    """)
    
    # Section 8: Dépannage
    st.subheader("8. 🚨 Dépannage")
    
    st.markdown("**Problèmes courants et solutions:**")
    
    with st.expander("❌ Le rapport de drift n'apparaît pas dans le dashboard"):
        st.markdown("""
        **Solutions:**
        1. Vérifiez que le dossier `reports/drift/` existe
        2. Exécutez `python run_full_pipeline.py` pour générer le rapport
        3. Vérifiez que `latest_drift_report.json` est présent dans `reports/drift/`
        4. Cliquez sur "🔄 Rafraîchir les données" dans l'onglet Monitoring Drift
        5. Relancez Streamlit: `streamlit run app/app.py`
        """)
    
    with st.expander("❌ Erreur: Modèle non trouvé"):
        st.markdown("""
        **Solutions:**
        1. Vérifiez que le dossier `deploy/` contient les fichiers du modèle
        2. Exécutez le pipeline complet: `python run_full_pipeline.py`
        3. Vérifiez les logs dans `pipeline_execution.log`
        """)
    
    with st.expander("❌ Erreur d'authentification API"):
        st.markdown("""
        **Solutions:**
        1. Vérifiez que le fichier `.env` existe et contient `API_TOKEN`
        2. Utilisez le bon token dans vos requêtes
        3. Vérifiez que le token est au format: `Authorization: Bearer votre-token`
        """)
    
    with st.expander("❌ Port déjà utilisé"):
        st.markdown("""
        **Solutions:**
        1. Changez le port dans `docker-compose.yml`:
           ```yaml
           ports:
             - "8001:8000"  # API
             - "8502:8501"  # Streamlit
           ```
        2. Ou arrêtez le service utilisant le port:
           ```bash
           # Windows
           netstat -ano | findstr :8000
           taskkill /PID <PID> /F
           
           # Linux/Mac
           lsof -i :8000
           kill -9 <PID>
           ```
        """)
    
    with st.expander("❌ Dépendances manquantes"):
        st.markdown("""
        **Solutions:**
        1. Réinstallez les dépendances: `pip install -r requirements.txt`
        2. Le pipeline installe automatiquement les packages manquants
        3. Vérifiez votre version de Python: `python --version` (doit être 3.8+)
        """)
    
    st.success("✅ **Besoin d'aide supplémentaire?** Consultez les logs dans `pipeline_execution.log` ou ouvrez une issue sur GitHub.")

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
    - Pandas, NumPy, Matplotlib, Seaborn
    
    **Système v_best:**
    Le système v_best est une innovation majeure qui garantit que seuls les meilleurs modèles
    sont déployés en production. Chaque nouveau modèle est automatiquement comparé au v_best
    existant, et ne le remplace que s'il est meilleur sur plus de 50% des métriques.
    
    **Surveillance du Drift:**
    Le système surveille en continu les changements de distribution des données en production
    et génère des alertes automatiques lorsque le drift dépasse les seuils configurés.
    
    **Auteur:** Projet MLOps - Pipeline ML Automatisé
    **Licence:** MIT
    """)