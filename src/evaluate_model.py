"""
Script pour évaluer les performances d'un modèle de machine learning.
"""
import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import seaborn as sns
import logging
import base64
from io import BytesIO

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Créer le répertoire pour les visualisations
os.makedirs("build/visualizations", exist_ok=True)

def load_test_results():
    """Charge les résultats des tests."""
    try:
        with open("build/test_results.json", 'r') as f:
            test_results = json.load(f)
        
        with open("build/prediction_results.json", 'r') as f:
            prediction_results = json.load(f)
        
        return test_results, prediction_results
    except Exception as e:
        logger.error(f"Erreur lors du chargement des résultats de test: {e}")
        sys.exit(1)

def generate_classification_visualizations(prediction_results):
    """Génère des visualisations pour un modèle de classification."""
    try:
        y_true = np.array(prediction_results['true_values'])
        y_pred = np.array(prediction_results['predictions'])
        
        visualizations = {}
        
        # Matrice de confusion
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(10, 8))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=ax, cmap=plt.cm.Blues)
        plt.title('Matrice de confusion')
        plt.tight_layout()
        
        # Sauvegarder l'image
        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        visualizations['confusion_matrix'] = image_base64
        
        # Si classification binaire, générer ROC et PR curves
        unique_classes = np.unique(y_true)
        if len(unique_classes) == 2:
            # ROC Curve
            fig, ax = plt.subplots(figsize=(10, 8))
            fpr, tpr, _ = roc_curve(y_true, y_pred)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Courbe ROC')
            plt.legend(loc='lower right')
            
            # Sauvegarder l'image
            buf = BytesIO()
            plt.savefig(buf, format='png')
            plt.close(fig)
            
            image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            visualizations['roc_curve'] = image_base64
            
            # Precision-Recall Curve
            fig, ax = plt.subplots(figsize=(10, 8))
            precision, recall, _ = precision_recall_curve(y_true, y_pred)
            
            plt.plot(recall, precision)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Courbe Précision-Rappel')
            
            # Sauvegarder l'image
            buf = BytesIO()
            plt.savefig(buf, format='png')
            plt.close(fig)
            
            image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            visualizations['pr_curve'] = image_base64
        
        # Sauvegarder les visualisations
        with open("build/visualizations/classification_visualizations.json", 'w') as f:
            json.dump(visualizations, f)
            
        return visualizations
        
    except Exception as e:
        logger.error(f"Erreur lors de la génération des visualisations pour la classification: {e}")
        sys.exit(1)

def generate_regression_visualizations(prediction_results):
    """Génère des visualisations pour un modèle de régression."""
    try:
        y_true = np.array(prediction_results['true_values'])
        y_pred = np.array(prediction_results['predictions'])
        
        visualizations = {}
        
        # Diagramme de dispersion: Prédictions vs Réalité
        fig, ax = plt.subplots(figsize=(10, 8))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
        plt.xlabel('Valeurs réelles')
        plt.ylabel('Prédictions')
        plt.title('Prédictions vs Valeurs réelles')
        
        # Sauvegarder l'image
        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        visualizations['scatter_plot'] = image_base64
        
        # Histogramme des erreurs
        errors = y_pred - y_true
        
        fig, ax = plt.subplots(figsize=(10, 8))
        plt.hist(errors, bins=30, alpha=0.7, color='blue')
        plt.axvline(x=0, color='r', linestyle='--')
        plt.xlabel('Erreur de prédiction')
        plt.ylabel('Fréquence')
        plt.title('Distribution des erreurs de prédiction')
        
        # Sauvegarder l'image
        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        visualizations['error_histogram'] = image_base64
        
        # Sauvegarder les visualisations
        with open("build/visualizations/regression_visualizations.json", 'w') as f:
            json.dump(visualizations, f)
            
        return visualizations
        
    except Exception as e:
        logger.error(f"Erreur lors de la génération des visualisations pour la régression: {e}")
        sys.exit(1)

def generate_evaluation_report(test_results, visualizations):
    """Génère un rapport d'évaluation complet."""
    try:
        evaluation_report = {
            'model_info': {
                'model_name': test_results['model_name'],
                'model_version': test_results['model_version'],
                'model_type': test_results['model_type']
            },
            'metrics': test_results['test_metrics'],
            'visualizations': visualizations,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        with open("build/evaluation_report.json", 'w') as f:
            json.dump(evaluation_report, f)
            
        logger.info(f"Rapport d'évaluation généré avec succès")
        return evaluation_report
        
    except Exception as e:
        logger.error(f"Erreur lors de la génération du rapport d'évaluation: {e}")
        sys.exit(1)

def main():
    """Fonction principale pour évaluer le modèle."""
    try:
        # Charger les résultats des tests
        test_results, prediction_results = load_test_results()
        
        # Générer les visualisations selon le type de modèle
        if test_results['model_type'] == 'classification':
            visualizations = generate_classification_visualizations(prediction_results)
        elif test_results['model_type'] == 'regression':
            visualizations = generate_regression_visualizations(prediction_results)
        else:
            logger.error(f"Type de modèle non supporté: {test_results['model_type']}")
            sys.exit(1)
            
        # Générer le rapport d'évaluation
        evaluation_report = generate_evaluation_report(test_results, visualizations)
        
        logger.info("Évaluation du modèle terminée avec succès")
        
    except Exception as e:
        logger.error(f"Erreur lors de l'évaluation du modèle: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()