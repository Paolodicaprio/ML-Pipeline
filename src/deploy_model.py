#!/usr/bin/env python3
"""
Script pour déployer un modèle de machine learning.
"""
import os
import sys
import json
import shutil
import logging
import datetime

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_comparison_report():
    """Charge le rapport de comparaison des modèles."""
    try:
        with open("build/comparison_report.json", 'r') as f:
            comparison_report = json.load(f)
        
        return comparison_report
    except Exception as e:
        logger.error(f"Erreur lors du chargement du rapport de comparaison: {e}")
        sys.exit(1)

def should_deploy(comparison_report):
    """Détermine si le modèle doit être déployé en fonction de la comparaison."""
    try:
        # Si c'est le premier modèle ou s'il y a une amélioration globale
        if comparison_report['overall_improvement'] is None or comparison_report['overall_improvement']:
            return True
        else:
            logger.warning("Le modèle n'est pas meilleur que la version précédente, le déploiement est annulé")
            return False
    except Exception as e:
        logger.error(f"Erreur lors de la détermination du déploiement: {e}")
        return False

def deploy_model():
    """Déploie le modèle."""
    try:
        # Créer un répertoire de déploiement si nécessaire
        os.makedirs("deploy", exist_ok=True)
        
        # Copier le modèle dans le répertoire de déploiement
        shutil.copy("build/model.pkl", "deploy/model.pkl")
        
        # Copier les métadonnées du modèle
        shutil.copy("build/model_metadata.json", "deploy/model_metadata.json")
        
        # Copier le rapport d'évaluation
        shutil.copy("build/evaluation_report.json", "deploy/evaluation_report.json")
        
        # Créer un fichier d'information sur le déploiement
        deployment_info = {
            'deployment_timestamp': datetime.datetime.now().isoformat(),
            'deployed_from_commit': os.environ.get('GITHUB_SHA', 'local_deployment'),
            'deployed_by': os.environ.get('GITHUB_ACTOR', 'local_user')
        }
        
        with open("deploy/deployment_info.json", 'w') as f:
            json.dump(deployment_info, f)
            
        logger.info("Modèle déployé avec succès")
        return True
        
    except Exception as e:
        logger.error(f"Erreur lors du déploiement du modèle: {e}")
        return False

def main():
    """Fonction principale pour déployer le modèle."""
    try:
        # Charger le rapport de comparaison
        comparison_report = load_comparison_report()
        
        # Vérifier si le modèle doit être déployé
        if should_deploy(comparison_report):
            success = deploy_model()
            
            if success:
                logger.info("Déploiement du modèle terminé avec succès")
            else:
                logger.error("Échec du déploiement du modèle")
                sys.exit(1)
        else:
            logger.info("Le déploiement du modèle est ignoré")
            
    except Exception as e:
        logger.error(f"Erreur lors du déploiement du modèle: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()