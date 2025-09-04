#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script modifié pour comparer les performances de deux versions d'un modèle,
spécifiquement pour une présentation.
"""
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
    """Charge le rapport d'évaluation du modèle actuel."""
    try:
        with open("build/evaluation_report.json", 'r') as f:
            current_evaluation = json.load(f)
        
        return current_evaluation
    except Exception as e:
        logger.error(f"Erreur lors du chargement de l'évaluation actuelle: {e}")
        sys.exit(1)

def load_previous_evaluation():
    """Charge le rapport d'évaluation du modèle précédent."""
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
    """Compare les métriques des deux versions du modèle."""
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
    """Génère un rapport de comparaison complet."""
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
    """Fonction principale pour comparer les modèles."""
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
