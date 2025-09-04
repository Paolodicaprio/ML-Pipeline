"""
Exemple d'un modèle de classification simple.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class ClassificationModel:
    """Modèle de classification pour prédire des catégories."""
    
    def __init__(self, model_type='random_forest'):
        """
        Initialise le modèle de classification.
        
        Args:
            model_type: Type de modèle ('random_forest' ou 'svm')
        """
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == 'svm':
            self.model = SVC(probability=True, random_state=42)
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
        
    def predict_proba(self, X):
        """
        Prédit les probabilités de chaque classe.
        
        Args:
            X: Caractéristiques pour la prédiction
            
        Returns:
            Probabilités des classes
        """
        if not self.is_fitted:
            raise RuntimeError("Le modèle doit être entraîné avant de faire des prédictions")
            
        return self.model.predict_proba(X)
        
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
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted')
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
def generate_synthetic_classification_data(n_samples=1000, n_features=5, n_classes=2):
    """
    Génère des données synthétiques pour la classification.
    
    Args:
        n_samples: Nombre d'échantillons
        n_features: Nombre de caractéristiques
        n_classes: Nombre de classes
        
    Returns:
        X, y: Caractéristiques et cibles
    """
    np.random.seed(42)
    
    # Générer des centres de classes aléatoires
    centers = np.random.randn(n_classes, n_features) * 2
    
    # Générer des caractéristiques aléatoires pour chaque classe
    X = np.vstack([
        np.random.randn(n_samples // n_classes, n_features) + centers[i]
        for i in range(n_classes)
    ])
    
    # Générer les étiquettes de classe
    y = np.hstack([
        np.full(n_samples // n_classes, i)
        for i in range(n_classes)
    ])
    
    # Mélanger les données
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    return X, y

# Exemple d'utilisation
if __name__ == "__main__":
    # Générer des données synthétiques
    X, y = generate_synthetic_classification_data(n_classes=3)
    
    # Diviser en ensembles d'entraînement et de test
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Créer et entraîner le modèle
    model = ClassificationModel(model_type='random_forest')
    model.fit(X_train, y_train)
    
    # Évaluer le modèle
    metrics = model.evaluate(X_test, y_test)
    print("Métriques d'évaluation:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")