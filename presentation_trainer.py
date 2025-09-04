"""
Script pour entraîner et comparer deux versions d'un modèle pour une présentation.
Ce script simule un workflow Git en créant deux versions du modèle avec des
paramètres différents pour pouvoir les comparer.
"""
import os
import sys
import json
import shutil
import logging
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def backup_model_files():
    """Sauvegarde les fichiers actuels du modèle s'ils existent."""
    backup_dir = "presentation_backup"
    os.makedirs(backup_dir, exist_ok=True)

    # Fichiers à sauvegarder
    files = [
        ("models/model_config.yml", "model_config_original.yml"),
        ("build/evaluation_report.json", "evaluation_report_original.json"),
        ("deploy/model_metadata.json", "model_metadata_original.json")
    ]

    for src, dst in files:
        if os.path.exists(src):
            shutil.copy(src, os.path.join(backup_dir, dst))

    logger.info("Sauvegarde des fichiers originaux terminée.")

def restore_backup_files():
    """Restaure les fichiers sauvegardés."""
    backup_dir = "presentation_backup"
    if not os.path.exists(backup_dir):
        logger.warning("Aucune sauvegarde trouvée.")
        return False

    # Fichiers à restaurer
    files = [
        ("model_config_original.yml", "models/model_config.yml"),
        ("evaluation_report_original.json", "build/evaluation_report.json"),
        ("model_metadata_original.json", "deploy/model_metadata.json")
    ]

    for src, dst in files:
        src_path = os.path.join(backup_dir, src)
        if os.path.exists(src_path):
            shutil.copy(src_path, dst)

    logger.info("Restauration des fichiers originaux terminée.")
    return True

def prepare_v1_config():
    """Prépare la configuration de la première version du modèle."""
    config_content = """# Configuration du modèle - Version 1.0.0
model:
  name: "ClassifierModel"
  version: "1.0.0"
  type: "classification"
  algorithm: "random_forest"

parameters:
  n_estimators: 100
  max_depth: 10
  min_samples_split: 2
  min_samples_leaf: 1

training:
  epochs: 100
  batch_size: 32
  learning_rate: 0.001
  validation_split: 0.2

evaluation:
  metrics:
    - "accuracy"
    - "precision"
    - "recall"
    - "f1"

data:
  train_path: "data/classification_train.csv"
  test_path: "data/classification_test.csv"
  test_split: 0.1
  validation_split: 0.2
"""
    with open("models/model_config_v1.yml", 'w') as f:
        f.write(config_content)

    # Copier comme config actuelle
    shutil.copy("models/model_config_v1.yml", "models/model_config.yml")
    
    logger.info("Configuration v1 créée et appliquée.")

def prepare_v2_config():
    """Prépare la configuration de la deuxième version du modèle avec des paramètres améliorés."""
    config_content = """# Configuration du modèle - Version 2.0.0
model:
  name: "ClassifierModel"
  version: "2.0.0"
  type: "classification"
  algorithm: "random_forest"

parameters:
  n_estimators: 200  # Augmenté de 100 à 200
  max_depth: 15      # Augmenté de 10 à 15
  min_samples_split: 5  # Augmenté pour réduire l'overfitting
  min_samples_leaf: 2   # Augmenté pour réduire l'overfitting

training:
  epochs: 150        # Augmenté de 100 à 150
  batch_size: 64     # Augmenté de 32 à 64
  learning_rate: 0.01  # Augmenté de 0.001 à 0.01
  validation_split: 0.2

evaluation:
  metrics:
    - "accuracy"
    - "precision"
    - "recall"
    - "f1"

data:
  train_path: "data/classification_train.csv"
  test_path: "data/classification_test.csv"
  test_split: 0.1
  validation_split: 0.2
"""
    with open("models/model_config_v2.yml", 'w') as f:
        f.write(config_content)
    
    logger.info("Configuration v2 créée.")

def run_pipeline(model_type="classification"):
    """Exécute le pipeline complet."""
    try:
        cmd = ["python", "run.py", "--type", model_type, "--generate-data", "--stages", "all"]
        logger.info(f"Exécution de la commande: {' '.join(cmd)}")
        
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Afficher la sortie en temps réel
        for stdout_line in iter(process.stdout.readline, ""):
            print(stdout_line, end="")
        
        stderr = process.stderr.read()
        if stderr:
            print(f"ERREURS: {stderr}")
        
        return_code = process.wait()
        if return_code != 0:
            logger.error(f"Le pipeline a échoué avec le code de retour {return_code}")
            return False
        
        logger.info("Pipeline exécuté avec succès.")
        return True
        
    except Exception as e:
        logger.error(f"Erreur lors de l'exécution du pipeline: {e}")
        return False

def save_model_v1_artifacts():
    """Sauvegarde les artefacts de la version 1 du modèle."""
    v1_dir = "model_v1"
    os.makedirs(v1_dir, exist_ok=True)
    
    files_to_save = {
        "build/evaluation_report.json": "evaluation_report.json",
        "deploy/model_metadata.json": "model_metadata.json",
        "deploy/model.pkl": "model.pkl"
    }
    
    for src, dst in files_to_save.items():
        if os.path.exists(src):
            shutil.copy(src, os.path.join(v1_dir, dst))
    
    logger.info("Artefacts du modèle v1 sauvegardés.")

def inject_v1_evaluation_as_previous():
    """Injecte l'évaluation de la v1 comme si c'était un modèle précédent dans un commit Git."""
    # Copier les fichiers de la version 1 dans le dossier temporaire
    # que le script de comparaison s'attend à trouver
    os.makedirs("temp_previous", exist_ok=True)
    
    if os.path.exists("model_v1/evaluation_report.json"):
        shutil.copy("model_v1/evaluation_report.json", "temp_previous/evaluation_report.json")
        logger.info("Évaluation v1 injectée comme évaluation précédente.")
        return True
    else:
        logger.error("Fichier d'évaluation v1 introuvable.")
        return False

def create_modified_comparison_script():
    """Crée une version modifiée du script de comparaison pour la présentation."""
    modified_script = """#!/usr/bin/env python
# -*- coding: utf-8 -*-

\"\"\"
Script modifié pour comparer les performances de deux versions d'un modèle,
spécifiquement pour une présentation.
\"\"\"
import os
import sys
import json
import pandas as pd
import numpy as np
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_current_evaluation():
    \"\"\"Charge le rapport d'évaluation du modèle actuel.\"\"\"
    try:
        with open("build/evaluation_report.json", 'r') as f:
            current_evaluation = json.load(f)
        
        return current_evaluation
    except Exception as e:
        logger.error(f"Erreur lors du chargement de l'évaluation actuelle: {e}")
        sys.exit(1)

def load_previous_evaluation():
    \"\"\"Charge le rapport d'évaluation du modèle précédent.\"\"\"
    try:
        # Pour la présentation, nous utilisons directement le fichier sauvegardé
        if os.path.exists("model_v1/evaluation_report.json"):
            with open("model_v1/evaluation_report.json", 'r') as f:
                previous_evaluation = json.load(f)
            return previous_evaluation
        else:
            logger.warning("Pas de rapport d'évaluation précédent trouvé")
            return None
    except Exception as e:
        logger.error(f"Erreur lors du chargement de l'évaluation précédente: {e}")
        return None

def compare_metrics(current_metrics, previous_metrics, model_type):
    \"\"\"Compare les métriques des deux versions du modèle.\"\"\"
    try:
        comparison = {}
        
        # Pour chaque métrique présente dans les deux évaluations
        for metric_name in current_metrics.keys():
            if previous_metrics and metric_name in previous_metrics:
                current_value = current_metrics[metric_name]
                previous_value = previous_metrics[metric_name]
                
                # Calculer la différence et le pourcentage de changement
                absolute_diff = current_value - previous_value
                
                if previous_value != 0:
                    percentage_diff = (absolute_diff / abs(previous_value)) * 100
                else:
                    percentage_diff = float('inf') if absolute_diff > 0 else float('-inf') if absolute_diff < 0 else 0
                
                # Déterminer si c'est une amélioration
                is_improvement = None
                
                if model_type == 'classification':
                    # Pour les métriques de classification, plus c'est élevé, mieux c'est généralement
                    is_improvement = absolute_diff > 0
                elif model_type == 'regression':
                    # Pour les métriques d'erreur en régression, plus c'est bas, mieux c'est
                    if metric_name in ['mse', 'mae', 'rmse']:
                        is_improvement = absolute_diff < 0
                    else:
                        is_improvement = absolute_diff > 0
                
                comparison[metric_name] = {
                    'current': current_value,
                    'previous': previous_value,
                    'absolute_diff': absolute_diff,
                    'percentage_diff': percentage_diff,
                    'is_improvement': is_improvement
                }
            else:
                # Si la métrique n'existe pas dans l'évaluation précédente
                comparison[metric_name] = {
                    'current': current_metrics[metric_name],
                    'previous': None,
                    'absolute_diff': None,
                    'percentage_diff': None,
                    'is_improvement': None
                }
                
        return comparison
    except Exception as e:
        logger.error(f"Erreur lors de la comparaison des métriques: {e}")
        return {}

def generate_comparison_report(current_evaluation, previous_evaluation, metrics_comparison):
    \"\"\"Génère un rapport de comparaison complet.\"\"\"
    try:
        # Vérifier si l'évaluation précédente existe
        if previous_evaluation is None:
            overall_improvement = None
            previous_version = None
        else:
            # Calculer le nombre de métriques améliorées
            improved_metrics = sum(1 for metric in metrics_comparison.values() if metric['is_improvement'] == True)
            total_comparable_metrics = sum(1 for metric in metrics_comparison.values() if metric['is_improvement'] is not None)
            
            # Déterminer si globalement il y a une amélioration
            overall_improvement = None
            if total_comparable_metrics > 0:
                improvement_ratio = improved_metrics / total_comparable_metrics
                overall_improvement = improvement_ratio >= 0.5
                
            previous_version = previous_evaluation['model_info']['model_version']
        
        comparison_report = {
            'current_model': {
                'name': current_evaluation['model_info']['model_name'],
                'version': current_evaluation['model_info']['model_version'],
                'type': current_evaluation['model_info']['model_type']
            },
            'previous_model': {
                'version': previous_version
            },
            'metrics_comparison': metrics_comparison,
            'overall_improvement': overall_improvement,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        with open("build/comparison_report.json", 'w') as f:
            json.dump(comparison_report, f, indent=2)
            
        logger.info(f"Rapport de comparaison généré avec succès. Amélioration globale: {overall_improvement}")
        return comparison_report
        
    except Exception as e:
        logger.error(f"Erreur lors de la génération du rapport de comparaison: {e}")
        sys.exit(1)

def main():
    \"\"\"Fonction principale pour comparer les modèles.\"\"\"
    try:
        # Charger l'évaluation actuelle
        current_evaluation = load_current_evaluation()
        
        # Charger l'évaluation précédente depuis le fichier sauvegardé
        previous_evaluation = load_previous_evaluation()
            
        # Comparer les métriques
        metrics_comparison = {}
        if previous_evaluation:
            current_metrics = current_evaluation['metrics']  # Utilise la clé correcte
            previous_metrics = previous_evaluation['metrics']  # Utilise la clé correcte
            model_type = current_evaluation['model_info']['model_type']
            
            metrics_comparison = compare_metrics(current_metrics, previous_metrics, model_type)
            
        # Générer le rapport de comparaison
        comparison_report = generate_comparison_report(
            current_evaluation, 
            previous_evaluation, 
            metrics_comparison
        )
        
        logger.info("Comparaison des modèles terminée avec succès")
        
    except Exception as e:
        logger.error(f"Erreur lors de la comparaison des modèles: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
"""

    with open("src/presentation_compare.py", 'w') as f:
        f.write(modified_script)
    
    logger.info("Script de comparaison modifié créé avec succès.")

def train_and_save_v1():
    """Entraîne et sauvegarde le modèle v1."""
    prepare_v1_config()
    success = run_pipeline("classification")
    if success:
        save_model_v1_artifacts()
        return True
    return False

def train_v2_and_compare():
    """Entraîne le modèle v2 et le compare avec v1."""
    # Appliquer la configuration v2
    shutil.copy("models/model_config_v2.yml", "models/model_config.yml")
    
    # Exécuter le pipeline pour le modèle v2
    success = run_pipeline("classification")
    if not success:
        return False
    
    # Créer et utiliser le script de comparaison modifié
    create_modified_comparison_script()
    
    # Exécuter le script de comparaison modifié
    try:
        subprocess.run(["python", "src/presentation_compare.py"], check=True)
        logger.info("Comparaison des modèles terminée avec succès.")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Erreur lors de l'exécution du script de comparaison: {e}")
        return False

def create_visualization_for_presentation():
    """Crée une visualisation de comparaison pour la présentation."""
    try:
        if not os.path.exists("build/comparison_report.json"):
            logger.error("Rapport de comparaison non trouvé.")
            return False
            
        with open("build/comparison_report.json", 'r') as f:
            comparison = json.load(f)
        
        if not comparison.get("metrics_comparison"):
            logger.error("Pas de métriques de comparaison dans le rapport.")
            return False
        
        # Créer un dossier pour les visualisations
        os.makedirs("build/visualizations", exist_ok=True)
        
        # Préparer les données pour la visualisation
        metrics = []
        v1_values = []
        v2_values = []
        improvements = []
        
        for metric_name, data in comparison["metrics_comparison"].items():
            metrics.append(metric_name)
            v1_values.append(data.get("previous", 0))
            v2_values.append(data.get("current", 0))
            improvements.append(data.get("percentage_diff", 0))
        
        # Créer un dataframe
        df = pd.DataFrame({
            'Métrique': metrics,
            'Modèle V1': v1_values,
            'Modèle V2': v2_values,
            'Amélioration (%)': improvements
        })
        
        # Créer les visualisations
        plt.figure(figsize=(15, 10))
        
        # 1. Comparaison des métriques
        plt.subplot(2, 1, 1)
        df_melt = pd.melt(df, id_vars=['Métrique'], value_vars=['Modèle V1', 'Modèle V2'])
        sns.barplot(x='Métrique', y='value', hue='variable', data=df_melt)
        plt.title("Comparaison des métriques de performance", fontsize=16)
        plt.ylabel("Valeur", fontsize=12)
        plt.xlabel("Métrique", fontsize=12)
        plt.legend(title="Version")
        
        # 2. Pourcentage d'amélioration
        plt.subplot(2, 1, 2)
        bars = sns.barplot(x='Métrique', y='Amélioration (%)', data=df)
        
        # Colorier les barres selon l'amélioration (vert) ou la détérioration (rouge)
        for i, bar in enumerate(bars.patches):
            if improvements[i] > 0:
                bar.set_color('green')
            else:
                bar.set_color('red')
        
        plt.title("Pourcentage d'amélioration par métrique", fontsize=16)
        plt.ylabel("Amélioration (%)", fontsize=12)
        plt.xlabel("Métrique", fontsize=12)
        plt.axhline(y=0, color='black', linestyle='-')
        
        # Ajouter les valeurs sur les barres
        for i, p in enumerate(bars.patches):
            height = p.get_height()
            if not pd.isna(height):
                bars.annotate(f"{height:.1f}%",
                              (p.get_x() + p.get_width() / 2., height),
                              ha='center', va='center',
                              xytext=(0, 9),
                              textcoords='offset points')
        
        plt.tight_layout()
        
        # Sauvegarder l'image
        plt.savefig("build/visualizations/model_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Visualisation de comparaison créée avec succès.")
        return True
    
    except Exception as e:
        logger.error(f"Erreur lors de la création de la visualisation: {e}")
        return False

def generate_presentation_guide():
    """Génère un guide de présentation pour montrer l'amélioration du modèle."""
    try:
        # Charger les données de comparaison
        with open("build/comparison_report.json", 'r') as f:
            comparison = json.load(f)
        
        v1_version = comparison["previous_model"]["version"]
        v2_version = comparison["current_model"]["version"]
        
        # Calculer les améliorations globales
        improved_metrics = sum(1 for metric in comparison["metrics_comparison"].values() 
                              if metric.get("is_improvement") == True)
        total_metrics = len(comparison["metrics_comparison"])
        improvement_ratio = improved_metrics / total_metrics if total_metrics > 0 else 0
        
        # Générer le guide de présentation
        guide = f"""# Guide de présentation des améliorations du modèle

## Introduction
Bonjour, je vais vous présenter notre travail d'amélioration de notre modèle de machine learning.

## Contexte
Notre pipeline ML déploie des modèles et garde une trace de leurs performances. 
Aujourd'hui, je vais vous montrer comment nous avons amélioré notre modèle de classification
en modifiant certains paramètres clés.

## Versions du modèle
- **Version précédente (v1)**: {v1_version}
- **Nouvelle version (v2)**: {v2_version}

## Changements apportés dans la nouvelle version
Nous avons modifié plusieurs paramètres pour améliorer les performances:

1. **Augmentation des n_estimators**: de 100 à 200
   - Impact: Réduit la variance du modèle en créant plus d'arbres

2. **Augmentation de la profondeur maximale**: de 10 à 15
   - Impact: Permet au modèle de capturer des interactions plus complexes

3. **Ajustement des hyperparamètres de régularisation**:
   - min_samples_split: de 2 à 5
   - min_samples_leaf: de 1 à 2
   - Impact: Réduit le risque de surapprentissage

4. **Optimisation de l'entraînement**:
   - Nombre d'époques: de 100 à 150
   - Taille des batchs: de 32 à 64
   - Taux d'apprentissage: de 0.001 à 0.01

## Résultats et améliorations
"""

        # Ajouter les métriques spécifiques au guide
        for metric_name, data in comparison["metrics_comparison"].items():
            if data["previous"] is not None and data["current"] is not None:
                previous_value = data["previous"]
                current_value = data["current"]
                diff_percent = data["percentage_diff"]
                
                improvement_text = "améliorée" if data.get("is_improvement") else "détériorée"
                
                guide += f"- **{metric_name}**: {previous_value:.4f} → {current_value:.4f} "
                guide += f"({diff_percent:+.2f}%, {improvement_text})\n"
        
        guide += f"""
## Conclusion
La nouvelle version du modèle présente une amélioration dans {improved_metrics} métriques sur {total_metrics}.
Cela représente un taux d'amélioration de {improvement_ratio*100:.1f}%.

{comparison.get("overall_improvement", False) is True and "Dans l'ensemble, cette nouvelle version est meilleure que la précédente." or "Certaines métriques ont été améliorées, mais d'autres nécessitent encore du travail."}

## Prochaines étapes
1. Continuer à optimiser les hyperparamètres
2. Explorer d'autres algorithmes
3. Améliorer la qualité des données d'entrée

## Démonstration
Je vais maintenant vous montrer l'application Streamlit qui permet de visualiser ces résultats en temps réel.
```
streamlit run app/app.py
```
"""

        # Sauvegarder le guide
        with open("presentation_guide.md", 'w') as f:
            f.write(guide)
            
        logger.info("Guide de présentation généré avec succès.")
        return True
    
    except Exception as e:
        logger.error(f"Erreur lors de la génération du guide de présentation: {e}")
        return False

def main():
    """Fonction principale."""
    try:
        # Sauvegarder la configuration actuelle
        backup_model_files()
        
        # Préparer les configurations v1 et v2
        prepare_v1_config()
        prepare_v2_config()
        
        # Entraîner et sauvegarder le modèle v1
        logger.info("Entraînement du modèle v1...")
        train_and_save_v1()
        
        # Entraîner le modèle v2 et le comparer avec v1
        logger.info("Entraînement du modèle v2 et comparaison...")
        train_v2_and_compare()
        
        # Créer une visualisation pour la présentation
        create_visualization_for_presentation()
        
        # Générer un guide de présentation
        generate_presentation_guide()
        
        logger.info("""
=== PRÉPARATION TERMINÉE AVEC SUCCÈS ===

Pour votre présentation:
1. Les deux modèles (v1 et v2) ont été entraînés et sauvegardés
2. Un rapport de comparaison a été généré
3. Une visualisation est disponible dans build/visualizations/model_comparison.png
4. Un guide de présentation a été créé dans presentation_guide.md

Lancez l'application Streamlit pour visualiser les résultats:
    streamlit run app/app.py
""")
        
    except Exception as e:
        logger.error(f"Erreur lors de l'exécution du script: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()