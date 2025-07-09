#!/usr/bin/env python3
"""
Exemple d'un modèle de régression simple.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class RegressionModel:
    """Modèle de régression pour prédire des valeurs continues."""
    
    def __init__(self, model_type='linear'):
        """
        Initialise le modèle de régression.
        
        Args:
            model_type: Type de modèle ('linear' ou 'random_forest')
        """
        if model_type == 'linear':
            self.model = LinearRegression()
        elif model_type == 'random_forest':
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Type de modèle non supporté: {model_type}")
            
        self.model_type = model_type
        self.is_fitted = False
        
    def fit(self, X, y):
        """
        Entraîne le modèle sur les données fournies.
        
        Args:
            X: Caractéristiques d'entraînement
            y: Cible d'entraînement
        """
        self.model.fit(X, y)
        self.is_fitted = True
        
    def predict(self, X):
        """
        Fait des prédictions à partir des caractéristiques fournies.
        
        Args:
            X: Caractéristiques pour la prédiction
            
        Returns:
            Prédictions du modèle
        """
        if not self.is_fitted:
            raise RuntimeError("Le modèle doit être entraîné avant de faire des prédictions")
            
        return self.model.predict(X)
        
    def evaluate(self, X, y_true):
        """
        Évalue les performances du modèle.
        
        Args:
            X: Caractéristiques pour l'évaluation
            y_true: Vraies valeurs cibles
            
        Returns:
            Dictionnaire contenant les métriques d'évaluation
        """
        if not self.is_fitted:
            raise RuntimeError("Le modèle doit être entraîné avant d'être évalué")
            
        y_pred = self.predict(X)
        
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }
        
        return metrics
        
    def save(self, filepath):
        """
        Sauvegarde le modèle dans un fichier.
        
        Args:
            filepath: Chemin où sauvegarder le modèle
        """
        if not self.is_fitted:
            raise RuntimeError("Le modèle doit être entraîné avant d'être sauvegardé")
            
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
            
    @classmethod
    def load(cls, filepath):
        """
        Charge un modèle depuis un fichier.
        
        Args:
            filepath: Chemin du fichier modèle
            
        Returns:
            Instance du modèle chargé
        """
        import pickle
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
            
        return model

# Fonction de démonstration
def generate_synthetic_regression_data(n_samples=1000, n_features=5, noise=0.1):
    """
    Génère des données synthétiques pour la régression.
    
    Args:
        n_samples: Nombre d'échantillons
        n_features: Nombre de caractéristiques
        noise: Niveau de bruit
        
    Returns:
        X, y: Caractéristiques et cibles
    """
    np.random.seed(42)
    
    # Générer des coefficients aléatoires
    coef = np.random.randn(n_features)
    
    # Générer des caractéristiques aléatoires
    X = np.random.randn(n_samples, n_features)
    
    # Générer la cible avec du bruit
    y = np.dot(X, coef) + noise * np.random.randn(n_samples)
    
    return X, y

# Exemple d'utilisation
if __name__ == "__main__":
    # Générer des données synthétiques
    X, y = generate_synthetic_regression_data()
    
    # Diviser en ensembles d'entraînement et de test
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Créer et entraîner le modèle
    model = RegressionModel(model_type='random_forest')
    model.fit(X_train, y_train)
    
    # Évaluer le modèle
    metrics = model.evaluate(X_test, y_test)
    print("Métriques d'évaluation:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")