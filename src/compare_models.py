"""
Script pour comparer les performances de deux versions d'un modèle et gérer le conteneur v_best.
"""
import os
import sys
import json
import pandas as pd
import numpy as np
import logging
import subprocess
import git
import shutil
import yaml

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_current_evaluation():
    """Charge le rapport d'évaluation du modèle actuel."""
    try:
        with open("build/evaluation_report.json", 'r') as f:
            current_evaluation = json.load(f)
        
        return current_evaluation
    except Exception as e:
        logger.error(f"Erreur lors du chargement de l'évaluation actuelle: {e}")
        sys.exit(1)

def load_v_best_evaluation():
    """Charge le rapport d'évaluation du modèle v_best s'il existe."""
    try:
        v_best_path = "deploy/v_best/evaluation_report.json"
        if os.path.exists(v_best_path):
            with open(v_best_path, 'r') as f:
                v_best_evaluation = json.load(f)
            return v_best_evaluation
        else:
            logger.info("Aucun modèle v_best existant trouvé")
            return None
    except Exception as e:
        logger.error(f"Erreur lors du chargement de l'évaluation v_best: {e}")
        return None

def get_current_commit():
    """Récupère le hash du commit actuel."""
    try:
        repo = git.Repo('.')
        return repo.head.commit.hexsha
    except Exception as e:
        logger.error(f"Erreur lors de la récupération du commit actuel: {e}")
        return "unknown_commit"

def compare_metrics(current_metrics, v_best_metrics, model_type):
    """Compare les métriques des deux versions du modèle."""
    try:
        comparison = {}
        
        # Pour chaque métrique présente dans les deux évaluations
        for metric_name in current_metrics.keys():
            if v_best_metrics and metric_name in v_best_metrics:
                current_value = current_metrics[metric_name]
                v_best_value = v_best_metrics[metric_name]
                
                # Calculer la différence et le pourcentage de changement
                absolute_diff = current_value - v_best_value
                
                if v_best_value != 0:
                    percentage_diff = (absolute_diff / abs(v_best_value)) * 100
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
                    'v_best': v_best_value,
                    'absolute_diff': absolute_diff,
                    'percentage_diff': percentage_diff,
                    'is_improvement': is_improvement
                }
            else:
                # Si la métrique n'existe pas dans l'évaluation v_best
                comparison[metric_name] = {
                    'current': current_metrics[metric_name],
                    'v_best': None,
                    'absolute_diff': None,
                    'percentage_diff': None,
                    'is_improvement': None
                }
                
        return comparison
    except Exception as e:
        logger.error(f"Erreur lors de la comparaison des métriques: {e}")
        return {}

def is_model_better(metrics_comparison):
    """Détermine si le modèle actuel est meilleur que v_best."""
    try:
        # Si aucun modèle v_best n'existe, le nouveau modèle devient automatiquement le meilleur
        if not any(metric['v_best'] is not None for metric in metrics_comparison.values()):
            return True
        
        # Calculer le nombre de métriques améliorées
        improved_metrics = sum(1 for metric in metrics_comparison.values() if metric['is_improvement'] == True)
        total_comparable_metrics = sum(1 for metric in metrics_comparison.values() if metric['is_improvement'] is not None)
        
        # Déterminer si globalement il y a une amélioration
        if total_comparable_metrics > 0:
            improvement_ratio = improved_metrics / total_comparable_metrics
            return improvement_ratio > 0.5  # Plus de 50% des métriques doivent être améliorées
        
        return False
    except Exception as e:
        logger.error(f"Erreur lors de la détermination de l'amélioration: {e}")
        return False

def update_v_best(current_evaluation):
    """Met à jour le conteneur v_best avec le nouveau modèle."""
    try:
        # Créer le répertoire v_best s'il n'existe pas
        os.makedirs("deploy/v_best", exist_ok=True)
        
        # Copier les fichiers du modèle actuel vers v_best
        shutil.copy("build/model.pkl", "deploy/v_best/model.pkl")
        shutil.copy("build/model_metadata.json", "deploy/v_best/model_metadata.json")
        shutil.copy("build/evaluation_report.json", "deploy/v_best/evaluation_report.json")
        
        # Copier la configuration du modèle
        if os.path.exists("models/model_config.yml"):
            shutil.copy("models/model_config.yml", "deploy/v_best/model_config.yml")
        
        # Créer les métadonnées v_best
        v_best_metadata = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'commit_id': get_current_commit(),
            'model_info': current_evaluation['model_info'],
            'metrics': current_evaluation['metrics'],
            'promoted_from_build': pd.Timestamp.now().isoformat()
        }
        
        with open("deploy/v_best/v_best_metadata.json", 'w') as f:
            json.dump(v_best_metadata, f, indent=2)
        
        logger.info("✅ Nouveau modèle déployé comme v_best")
        return True
        
    except Exception as e:
        logger.error(f"Erreur lors de la mise à jour de v_best: {e}")
        return False

def generate_comparison_report(current_evaluation, v_best_evaluation, metrics_comparison, is_better):
    """Génère un rapport de comparaison complet."""
    try:
        # Vérifier si l'évaluation v_best existe
        if v_best_evaluation is None:
            overall_improvement = True  # Premier modèle
            v_best_version = None
        else:
            overall_improvement = is_better
            v_best_version = v_best_evaluation['model_info']['model_version']
        
        comparison_report = {
            'current_model': {
                'name': current_evaluation['model_info']['model_name'],
                'version': current_evaluation['model_info']['model_version'],
                'type': current_evaluation['model_info']['model_type']
            },
            'v_best_model': {
                'version': v_best_version
            },
            'metrics_comparison': metrics_comparison,
            'overall_improvement': overall_improvement,
            'is_new_v_best': is_better,
            'timestamp': pd.Timestamp.now().isoformat(),
            'commit_id': get_current_commit()
        }
        
        with open("build/comparison_report.json", 'w') as f:
            json.dump(comparison_report, f, indent=2)
            
        if is_better:
            logger.info(f"✅ Nouveau modèle déployé comme v_best. Amélioration globale: {overall_improvement}")
        else:
            logger.info(f"⚠️ Ancien modèle conservé comme v_best. Amélioration globale: {overall_improvement}")
            
        return comparison_report
        
    except Exception as e:
        logger.error(f"Erreur lors de la génération du rapport de comparaison: {e}")
        sys.exit(1)

def main():
    """Fonction principale pour comparer les modèles et gérer v_best."""
    try:
        # Charger l'évaluation actuelle
        current_evaluation = load_current_evaluation()
        
        # Charger l'évaluation v_best
        v_best_evaluation = load_v_best_evaluation()
            
        # Comparer les métriques
        metrics_comparison = {}
        if v_best_evaluation:
            current_metrics = current_evaluation['metrics']
            v_best_metrics = v_best_evaluation['metrics']
            model_type = current_evaluation['model_info']['model_type']
            
            metrics_comparison = compare_metrics(current_metrics, v_best_metrics, model_type)
        else:
            # Premier modèle, pas de comparaison possible
            current_metrics = current_evaluation['metrics']
            for metric_name, value in current_metrics.items():
                metrics_comparison[metric_name] = {
                    'current': value,
                    'v_best': None,
                    'absolute_diff': None,
                    'percentage_diff': None,
                    'is_improvement': None
                }
            
        # Déterminer si le nouveau modèle est meilleur
        is_better = is_model_better(metrics_comparison)
        
        # Mettre à jour v_best si nécessaire
        if is_better:
            success = update_v_best(current_evaluation)
            if not success:
                logger.error("Échec de la mise à jour de v_best")
                sys.exit(1)
        else:
            logger.info("⚠️ Ancien modèle conservé comme v_best")
        
        # Générer le rapport de comparaison
        comparison_report = generate_comparison_report(
            current_evaluation, 
            v_best_evaluation, 
            metrics_comparison,
            is_better
        )
        
        logger.info("Comparaison des modèles terminée avec succès")
        
    except Exception as e:
        logger.error(f"Erreur lors de la comparaison des modèles: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()