#!/usr/bin/env python3
"""
Script pour tester un modèle de machine learning construit.
"""
import os
import sys
import pickle
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_model_and_metadata():
    """Charge le modèle et ses métadonnées."""
    try:
        # Charger le modèle
        with open("build/model.pkl", 'rb') as f:
            model = pickle.load(f)
        
        # Charger les métadonnées
        with open("build/model_metadata.json", 'r') as f:
            metadata = json.load(f)
        
        return model, metadata
    except Exception as e:
        logger.error(f"Erreur lors du chargement du modèle ou des métadonnées: {e}")
        sys.exit(1)

def load_test_data():
    """Charge les données de test."""
    try:
        X_test = np.load("build/X_test.npy")
        y_test = np.load("build/y_test.npy")
        return X_test, y_test
    except Exception as e:
        logger.error(f"Erreur lors du chargement des données de test: {e}")
        sys.exit(1)

def test_classification_model(model, metadata, X_test, y_test):
    """Teste un modèle de classification."""
    try:
        # Prédictions sur l'ensemble de test
        y_pred = model.predict(X_test)
        
        # Calculer les métriques
        metrics = {}
        for metric_name in metadata['validation_metrics'].keys():
            if metric_name == 'accuracy':
                metrics['accuracy'] = float(accuracy_score(y_test, y_pred))
            elif metric_name == 'precision':
                metrics['precision'] = float(precision_score(y_test, y_pred, average='weighted'))
            elif metric_name == 'recall':
                metrics['recall'] = float(recall_score(y_test, y_pred, average='weighted'))
            elif metric_name == 'f1':
                metrics['f1'] = float(f1_score(y_test, y_pred, average='weighted'))
        
        # Sauvegarder les résultats des prédictions pour comparaison future
        results = {
            'true_values': y_test.tolist(),
            'predictions': y_pred.tolist()
        }
        
        # Sauvegarder les métriques et résultats
        test_results = {
            'model_type': metadata['model_type'],
            'model_name': metadata['model_name'],
            'model_version': metadata['model_version'],
            'test_metrics': metrics,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        with open("build/test_results.json", 'w') as f:
            json.dump(test_results, f)
            
        with open("build/prediction_results.json", 'w') as f:
            json.dump(results, f)
        
        logger.info(f"Test du modèle de classification terminé. Métriques de test: {metrics}")
        return metrics
        
    except Exception as e:
        logger.error(f"Erreur lors du test du modèle de classification: {e}")
        sys.exit(1)

def test_regression_model(model, metadata, X_test, y_test):
    """Teste un modèle de régression."""
    try:
        # Prédictions sur l'ensemble de test
        y_pred = model.predict(X_test)
        
        # Calculer les métriques
        metrics = {}
        for metric_name in metadata['validation_metrics'].keys():
            if metric_name == 'mse':
                metrics['mse'] = float(mean_squared_error(y_test, y_pred))
            elif metric_name == 'mae':
                metrics['mae'] = float(mean_absolute_error(y_test, y_pred))
            elif metric_name == 'rmse':
                metrics['rmse'] = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        
        # Sauvegarder les résultats des prédictions pour comparaison future
        results = {
            'true_values': y_test.tolist(),
            'predictions': y_pred.tolist()
        }
        
        # Sauvegarder les métriques et résultats
        test_results = {
            'model_type': metadata['model_type'],
            'model_name': metadata['model_name'],
            'model_version': metadata['model_version'],
            'test_metrics': metrics,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        with open("build/test_results.json", 'w') as f:
            json.dump(test_results, f)
            
        with open("build/prediction_results.json", 'w') as f:
            json.dump(results, f)
        
        logger.info(f"Test du modèle de régression terminé. Métriques de test: {metrics}")
        return metrics
        
    except Exception as e:
        logger.error(f"Erreur lors du test du modèle de régression: {e}")
        sys.exit(1)

def main():
    """Fonction principale pour tester le modèle."""
    try:
        # Charger le modèle et les métadonnées
        model, metadata = load_model_and_metadata()
        
        # Charger les données de test
        X_test, y_test = load_test_data()
        
        # Tester le modèle selon son type
        if metadata['model_type'] == 'classification':
            metrics = test_classification_model(model, metadata, X_test, y_test)
        elif metadata['model_type'] == 'regression':
            metrics = test_regression_model(model, metadata, X_test, y_test)
        else:
            logger.error(f"Type de modèle non supporté: {metadata['model_type']}")
            sys.exit(1)
            
        logger.info("Test du modèle terminé avec succès")
        
    except Exception as e:
        logger.error(f"Erreur lors du test du modèle: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()