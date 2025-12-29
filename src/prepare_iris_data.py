"""
Script pour préparer le dataset Iris depuis sklearn.
Génère les fichiers CSV pour l'entraînement et le test.
"""
import os
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def prepare_iris_dataset():
    """
    Charge le dataset Iris depuis sklearn et le prépare pour le pipeline.
    
    Structure du dataset Iris:
    - 150 échantillons (50 par classe)
    - 4 features: sepal length, sepal width, petal length, petal width
    - 3 classes: Setosa (0), Versicolor (1), Virginica (2)
    """
    try:
        logger.info("Chargement du dataset Iris depuis sklearn...")
        
        # Charger le dataset Iris
        iris = load_iris()
        X = iris.data
        y = iris.target
        
        # Créer un DataFrame avec les noms de colonnes
        feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = y
        
        logger.info(f"Dataset chargé: {len(df)} échantillons, {len(feature_names)} features, {len(np.unique(y))} classes")
        logger.info(f"Distribution des classes: {np.bincount(y)}")
        
        # Diviser en train et test (80/20)
        train_df, test_df = train_test_split(
            df, 
            test_size=0.2, 
            random_state=42, 
            stratify=y  # Maintenir la distribution des classes
        )
        
        logger.info(f"Train set: {len(train_df)} échantillons")
        logger.info(f"Test set: {len(test_df)} échantillons")
        
        # Créer le dossier data s'il n'existe pas
        os.makedirs("data", exist_ok=True)
        
        # Sauvegarder les datasets
        train_path = "data/iris_train.csv"
        test_path = "data/iris_test.csv"
        
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        logger.info(f"✅ Données d'entraînement sauvegardées: {train_path}")
        logger.info(f"✅ Données de test sauvegardées: {test_path}")
        
        # Sauvegarder aussi le dataset complet pour référence
        full_path = "data/iris_full.csv"
        df.to_csv(full_path, index=False)
        logger.info(f"✅ Dataset complet sauvegardé: {full_path}")
        
        # Afficher des statistiques
        logger.info("\n📊 Statistiques du dataset:")
        logger.info(f"\nTrain set - Distribution des classes:")
        logger.info(train_df['target'].value_counts().sort_index())
        logger.info(f"\nTest set - Distribution des classes:")
        logger.info(test_df['target'].value_counts().sort_index())
        
        logger.info("\n📈 Statistiques descriptives (Train set):")
        logger.info(train_df.describe())
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de la préparation du dataset Iris: {e}")
        return False

def main():
    """Fonction principale."""
    logger.info("=" * 60)
    logger.info("PRÉPARATION DU DATASET IRIS")
    logger.info("=" * 60)
    
    success = prepare_iris_dataset()
    
    if success:
        logger.info("\n" + "=" * 60)
        logger.info("✅ Dataset Iris préparé avec succès!")
        logger.info("=" * 60)
        logger.info("\nProchaines étapes:")
        logger.info("1. Vérifiez les fichiers dans le dossier data/")
        logger.info("2. Lancez le pipeline complet: python run_full_pipeline.py")
        logger.info("3. Ou lancez étape par étape:")
        logger.info("   - python src/build_model.py")
        logger.info("   - python src/test_model.py")
        logger.info("   - python src/evaluate_model.py")
        logger.info("   - python src/compare_models.py")
        logger.info("   - python src/deploy_model.py")
    else:
        logger.error("\n❌ Échec de la préparation du dataset Iris")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())