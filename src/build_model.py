#!/usr/bin/env python3
"""
Script pour construire un modèle de machine learning à partir de la configuration YAML.
"""
import os
import sys
import yaml
import pickle
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Créer le répertoire build s'il n'existe pas
os.makedirs("build", exist_ok=True)

def load_config(config_path="models/model_config.yml"):
    """Charge la configuration du modèle depuis un fichier YAML."""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        logger.error(f"Erreur lors du chargement de la configuration: {e}")
        sys.exit(1)

def load_data(config):
    """Charge et prépare les données selon la configuration."""
    try:
        # Charger les données d'entraînement
        data_path = config['data']['train_path']
        df = pd.read_csv(data_path)
        
        # Séparer les caractéristiques et la cible
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        
        # Diviser les données si test_path n'est pas spécifié
        if not config['data'].get('test_path'):
            test_size = config['data']['test_split']
            val_size = config['data']['validation_split']
            
            # Calcul des proportions pour la division
            val_ratio = val_size / (1 - test_size)
            
            # Première division: train+val et test
            X_train_val, X_test, y_train_val, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            
            # Deuxième division: train et validation
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val, test_size=val_ratio, random_state=42
            )
            
            # Sauvegarder les ensembles de données
            np.save("build/X_train.npy", X_train)
            np.save("build/y_train.npy", y_train)
            np.save("build/X_val.npy", X_val)
            np.save("build/y_val.npy", y_val)
            np.save("build/X_test.npy", X_test)
            np.save("build/y_test.npy", y_test)
            
            return X_train, y_train, X_val, y_val
        else:
            # Utiliser l'ensemble de test fourni
            test_path = config['data']['test_path']
            test_df = pd.read_csv(test_path)
            
            X_test = test_df.iloc[:, :-1].values
            y_test = test_df.iloc[:, -1].values
            
            # Diviser l'ensemble d'entraînement en train et validation
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=config['data']['validation_split'], random_state=42
            )
            
            # Sauvegarder les ensembles de données
            np.save("build/X_train.npy", X_train)
            np.save("build/y_train.npy", y_train)
            np.save("build/X_val.npy", X_val)
            np.save("build/y_val.npy", y_val)
            np.save("build/X_test.npy", X_test)
            np.save("build/y_test.npy", y_test)
            
            return X_train, y_train, X_val, y_val
    except Exception as e:
        logger.error(f"Erreur lors du chargement des données: {e}")
        sys.exit(1)

def build_classification_model(config, X_train, y_train, X_val, y_val):
    """Construit un modèle de classification."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    try:
        # Créer et entraîner le modèle
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=config['parameters'].get('max_depth', None),
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # Évaluer le modèle sur l'ensemble de validation
        y_pred = model.predict(X_val)
        
        # Calculer les métriques
        metrics = {}
        for metric in config['evaluation']['metrics']:
            if metric == 'accuracy':
                metrics['accuracy'] = float(accuracy_score(y_val, y_pred))
            elif metric == 'precision':
                metrics['precision'] = float(precision_score(y_val, y_pred, average='weighted'))
            elif metric == 'recall':
                metrics['recall'] = float(recall_score(y_val, y_pred, average='weighted'))
            elif metric == 'f1':
                metrics['f1'] = float(f1_score(y_val, y_pred, average='weighted'))
        
        # Sauvegarder les métriques de validation
        with open("build/validation_metrics.json", 'w') as f:
            json.dump(metrics, f)
        
        # Sauvegarder le modèle
        with open("build/model.pkl", 'wb') as f:
            pickle.dump(model, f)
            
        logger.info(f"Modèle de classification construit avec succès. Métriques de validation: {metrics}")
        return model, metrics
    
    except Exception as e:
        logger.error(f"Erreur lors de la construction du modèle de classification: {e}")
        sys.exit(1)

def build_regression_model(config, X_train, y_train, X_val, y_val):
    """Construit un modèle de régression."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    
    try:
        # Créer et entraîner le modèle
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=config['parameters'].get('max_depth', None),
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # Évaluer le modèle sur l'ensemble de validation
        y_pred = model.predict(X_val)
        
        # Calculer les métriques
        metrics = {}
        for metric in config['evaluation']['metrics']:
            if metric == 'mse':
                metrics['mse'] = float(mean_squared_error(y_val, y_pred))
            elif metric == 'mae':
                metrics['mae'] = float(mean_absolute_error(y_val, y_pred))
            elif metric == 'rmse':
                metrics['rmse'] = float(np.sqrt(mean_squared_error(y_val, y_pred)))
        
        # Sauvegarder les métriques de validation
        with open("build/validation_metrics.json", 'w') as f:
            json.dump(metrics, f)
        
        # Sauvegarder le modèle
        with open("build/model.pkl", 'wb') as f:
            pickle.dump(model, f)
            
        logger.info(f"Modèle de régression construit avec succès. Métriques de validation: {metrics}")
        return model, metrics
    
    except Exception as e:
        logger.error(f"Erreur lors de la construction du modèle de régression: {e}")
        sys.exit(1)

def main():
    """Fonction principale pour construire le modèle."""
    try:
        # Charger la configuration
        config = load_config()
        
        # Journaliser les informations de configuration
        logger.info(f"Configuration chargée: Type de modèle = {config['model']['type']}")
        
        # Charger et préparer les données
        X_train, y_train, X_val, y_val = load_data(config)
        
        # Construire le modèle selon le type
        if config['model']['type'] == 'classification':
            model, metrics = build_classification_model(config, X_train, y_train, X_val, y_val)
        elif config['model']['type'] == 'regression':
            model, metrics = build_regression_model(config, X_train, y_train, X_val, y_val)
        else:
            logger.error(f"Type de modèle non supporté: {config['model']['type']}")
            sys.exit(1)
        
        # Sauvegarder les métadonnées du modèle
        metadata = {
            'model_type': config['model']['type'],
            'model_name': config['model']['name'],
            'model_version': config['model']['version'],
            'training_params': config['training'],
            'validation_metrics': metrics,
            'build_timestamp': pd.Timestamp.now().isoformat()
        }
        
        with open("build/model_metadata.json", 'w') as f:
            json.dump(metadata, f)
            
        logger.info("Construction du modèle terminée avec succès")
        
    except Exception as e:
        logger.error(f"Erreur lors de la construction du modèle: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()