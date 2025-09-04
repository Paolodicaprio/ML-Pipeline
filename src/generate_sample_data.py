"""
Script pour générer des données synthétiques pour tester le pipeline.
"""
import os
import sys
import numpy as np
import pandas as pd
import argparse
import logging
from sklearn.datasets import make_regression, make_classification

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_regression_data(n_samples=1000, n_features=10, noise=0.1, test_size=0.2):
    """
    Génère des données synthétiques pour un problème de régression.
    
    Args:
        n_samples: Nombre d'échantillons à générer
        n_features: Nombre de caractéristiques
        noise: Niveau de bruit
        test_size: Proportion de données pour le test
        
    Returns:
        Deux DataFrames (train et test)
    """
    # Générer les données
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        noise=noise,
        random_state=42
    )
    
    # Créer un DataFrame
    feature_names = [f"feature_{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    # Diviser en train et test
    n_test = int(n_samples * test_size)
    n_train = n_samples - n_test
    
    train_df = df.iloc[:n_train]
    test_df = df.iloc[n_train:]
    
    return train_df, test_df

def generate_classification_data(n_samples=1000, n_features=10, n_classes=2, test_size=0.2):
    """
    Génère des données synthétiques pour un problème de classification.
    
    Args:
        n_samples: Nombre d'échantillons à générer
        n_features: Nombre de caractéristiques
        n_classes: Nombre de classes
        test_size: Proportion de données pour le test
        
    Returns:
        Deux DataFrames (train et test)
    """
    # Générer les données
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        n_informative=max(1, n_features - 2),  # Moins que le nombre total de caractéristiques
        random_state=42
    )
    
    # Créer un DataFrame
    feature_names = [f"feature_{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    # Diviser en train et test
    n_test = int(n_samples * test_size)
    n_train = n_samples - n_test
    
    train_df = df.iloc[:n_train]
    test_df = df.iloc[n_train:]
    
    return train_df, test_df

def main():
    """Fonction principale pour générer des données."""
    parser = argparse.ArgumentParser(description='Générateur de données synthétiques pour machine learning')
    parser.add_argument('--type', type=str, choices=['regression', 'classification'], required=True,
                        help='Type de données à générer')
    parser.add_argument('--samples', type=int, default=1000, help='Nombre d\'échantillons')
    parser.add_argument('--features', type=int, default=10, help='Nombre de caractéristiques')
    parser.add_argument('--classes', type=int, default=2, help='Nombre de classes (uniquement pour la classification)')
    parser.add_argument('--noise', type=float, default=0.1, help='Niveau de bruit (uniquement pour la régression)')
    parser.add_argument('--test-size', type=float, default=0.2, help='Proportion des données pour le test')
    parser.add_argument('--output-dir', type=str, default='data', help='Répertoire de sortie')
    
    args = parser.parse_args()
    
    try:
        # Créer le répertoire de sortie s'il n'existe pas
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Générer les données selon le type spécifié
        if args.type == 'regression':
            train_df, test_df = generate_regression_data(
                n_samples=args.samples,
                n_features=args.features,
                noise=args.noise,
                test_size=args.test_size
            )
            logger.info(f"Données de régression générées: {len(train_df)} échantillons d'entraînement, {len(test_df)} échantillons de test")
        else:  # classification
            train_df, test_df = generate_classification_data(
                n_samples=args.samples,
                n_features=args.features,
                n_classes=args.classes,
                test_size=args.test_size
            )
            logger.info(f"Données de classification générées: {len(train_df)} échantillons d'entraînement, {len(test_df)} échantillons de test")
        
        # Sauvegarder les données
        train_path = os.path.join(args.output_dir, f"{args.type}_train.csv")
        test_path = os.path.join(args.output_dir, f"{args.type}_test.csv")
        
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        logger.info(f"Données sauvegardées dans {train_path} et {test_path}")
        
    except Exception as e:
        logger.error(f"Erreur lors de la génération des données: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()