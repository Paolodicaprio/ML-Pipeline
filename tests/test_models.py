"""
Tests unitaires pour les modèles de machine learning.
"""
import os
import sys
import unittest
import numpy as np
from sklearn.datasets import make_regression, make_classification

# Ajouter le répertoire parent au path pour l'import des modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.regression_model import RegressionModel
from models.classification_model import ClassificationModel

class TestRegressionModel(unittest.TestCase):
    """Tests pour le modèle de régression."""
    
    def setUp(self):
        """Préparation des tests."""
        # Générer des données synthétiques pour les tests
        X, y = make_regression(n_samples=100, n_features=5, random_state=42)
        self.X_train, self.X_test = X[:80], X[80:]
        self.y_train, self.y_test = y[:80], y[80:]
    
    def test_initialization(self):
        """Teste l'initialisation du modèle."""
        model = RegressionModel(model_type='linear')
        self.assertEqual(model.model_type, 'linear')
        self.assertFalse(model.is_fitted)
        
        model = RegressionModel(model_type='random_forest')
        self.assertEqual(model.model_type, 'random_forest')
        
        with self.assertRaises(ValueError):
            RegressionModel(model_type='invalid_type')
    
    def test_fit_predict(self):
        """Teste l'entraînement et la prédiction."""
        model = RegressionModel(model_type='linear')
        model.fit(self.X_train, self.y_train)
        
        self.assertTrue(model.is_fitted)
        
        # Vérifier que les prédictions ont la bonne forme
        predictions = model.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.X_test))
    
    def test_evaluate(self):
        """Teste l'évaluation du modèle."""
        model = RegressionModel(model_type='linear')
        model.fit(self.X_train, self.y_train)
        
        # Évaluer le modèle
        metrics = model.evaluate(self.X_test, self.y_test)
        
        # Vérifier que les métriques existent
        self.assertIn('mse', metrics)
        self.assertIn('rmse', metrics)
        self.assertIn('mae', metrics)
        self.assertIn('r2', metrics)
        
        # Vérifier que les valeurs sont du bon type
        self.assertIsInstance(metrics['mse'], float)
        self.assertIsInstance(metrics['rmse'], float)
        self.assertIsInstance(metrics['mae'], float)
        self.assertIsInstance(metrics['r2'], float)
        
        # Vérifier que les métriques sont cohérentes
        self.assertGreaterEqual(metrics['mse'], 0)
        self.assertGreaterEqual(metrics['rmse'], 0)
        self.assertGreaterEqual(metrics['mae'], 0)
        self.assertLessEqual(metrics['r2'], 1)
    
    def test_save_load(self):
        """Teste la sauvegarde et le chargement du modèle."""
        # Créer et entraîner un modèle
        model = RegressionModel(model_type='linear')
        model.fit(self.X_train, self.y_train)
        
        # Sauvegarder le modèle
        save_path = 'test_regression_model.pkl'
        model.save(save_path)
        
        # Vérifier que le fichier existe
        self.assertTrue(os.path.exists(save_path))
        
        # Charger le modèle
        loaded_model = RegressionModel.load(save_path)
        
        # Vérifier que le modèle chargé est du bon type
        self.assertIsInstance(loaded_model, RegressionModel)
        self.assertEqual(loaded_model.model_type, 'linear')
        self.assertTrue(loaded_model.is_fitted)
        
        # Vérifier que les prédictions sont cohérentes
        original_predictions = model.predict(self.X_test)
        loaded_predictions = loaded_model.predict(self.X_test)
        np.testing.assert_allclose(original_predictions, loaded_predictions)
        
        # Nettoyer
        if os.path.exists(save_path):
            os.remove(save_path)

class TestClassificationModel(unittest.TestCase):
    """Tests pour le modèle de classification."""
    
    def setUp(self):
        """Préparation des tests."""
        # Générer des données synthétiques pour les tests
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        self.X_train, self.X_test = X[:80], X[80:]
        self.y_train, self.y_test = y[:80], y[80:]
    
    def test_initialization(self):
        """Teste l'initialisation du modèle."""
        model = ClassificationModel(model_type='random_forest')
        self.assertEqual(model.model_type, 'random_forest')
        self.assertFalse(model.is_fitted)
        
        model = ClassificationModel(model_type='svm')
        self.assertEqual(model.model_type, 'svm')
        
        with self.assertRaises(ValueError):
            ClassificationModel(model_type='invalid_type')
    
    def test_fit_predict(self):
        """Teste l'entraînement et la prédiction."""
        model = ClassificationModel(model_type='random_forest')
        model.fit(self.X_train, self.y_train)
        
        self.assertTrue(model.is_fitted)
        
        # Vérifier que les prédictions ont la bonne forme
        predictions = model.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.X_test))
        
        # Vérifier que predict_proba retourne des probabilités
        probas = model.predict_proba(self.X_test)
        self.assertEqual(probas.shape[0], len(self.X_test))
        self.assertEqual(probas.shape[1], 2)  # Binaire, donc 2 classes
        np.testing.assert_allclose(np.sum(probas, axis=1), np.ones(len(self.X_test)))
    
    def test_evaluate(self):
        """Teste l'évaluation du modèle."""
        model = ClassificationModel(model_type='random_forest')
        model.fit(self.X_train, self.y_train)
        
        # Évaluer le modèle
        metrics = model.evaluate(self.X_test, self.y_test)
        
        # Vérifier que les métriques existent
        self.assertIn('accuracy', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1', metrics)
        
        # Vérifier que les valeurs sont du bon type
        self.assertIsInstance(metrics['accuracy'], float)
        self.assertIsInstance(metrics['precision'], float)
        self.assertIsInstance(metrics['recall'], float)
        self.assertIsInstance(metrics['f1'], float)
        
        # Vérifier que les métriques sont dans l'intervalle [0, 1]
        for metric_name, value in metrics.items():
            self.assertGreaterEqual(value, 0)
            self.assertLessEqual(value, 1)
    
    def test_save_load(self):
        """Teste la sauvegarde et le chargement du modèle."""
        # Créer et entraîner un modèle
        model = ClassificationModel(model_type='random_forest')
        model.fit(self.X_train, self.y_train)
        
        # Sauvegarder le modèle
        save_path = 'test_classification_model.pkl'
        model.save(save_path)
        
        # Vérifier que le fichier existe
        self.assertTrue(os.path.exists(save_path))
        
        # Charger le modèle
        loaded_model = ClassificationModel.load(save_path)
        
        # Vérifier que le modèle chargé est du bon type
        self.assertIsInstance(loaded_model, ClassificationModel)
        self.assertEqual(loaded_model.model_type, 'random_forest')
        self.assertTrue(loaded_model.is_fitted)
        
        # Vérifier que les prédictions sont cohérentes
        original_predictions = model.predict(self.X_test)
        loaded_predictions = loaded_model.predict(self.X_test)
        np.testing.assert_allclose(original_predictions, loaded_predictions)
        
        # Nettoyer
        if os.path.exists(save_path):
            os.remove(save_path)

if __name__ == '__main__':
    unittest.main()