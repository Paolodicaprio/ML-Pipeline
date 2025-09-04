"""
Script pour déployer un modèle de machine learning basé sur v_best.
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

def check_v_best_exists():
    """Vérifie si un modèle v_best existe."""
    v_best_path = "deploy/v_best"
    required_files = ["model.pkl", "model_metadata.json", "evaluation_report.json"]
    
    if not os.path.exists(v_best_path):
        return False
    
    for file in required_files:
        if not os.path.exists(os.path.join(v_best_path, file)):
            return False
    
    return True

def deploy_from_v_best():
    """Déploie le modèle depuis le conteneur v_best."""
    try:
        # Vérifier que v_best existe
        if not check_v_best_exists():
            logger.error("Aucun modèle v_best trouvé pour le déploiement")
            return False
        
        # Créer le répertoire de déploiement principal s'il n'existe pas
        os.makedirs("deploy", exist_ok=True)
        
        # Copier les fichiers depuis v_best vers le répertoire de déploiement principal
        v_best_files = [
            "model.pkl",
            "model_metadata.json", 
            "evaluation_report.json",
            "model_config.yml"
        ]
        
        for file in v_best_files:
            src_path = os.path.join("deploy/v_best", file)
            dst_path = os.path.join("deploy", file)
            
            if os.path.exists(src_path):
                shutil.copy(src_path, dst_path)
                logger.info(f"Copié {file} depuis v_best vers deploy/")
        
        # Créer un fichier d'information sur le déploiement
        deployment_info = {
            'deployment_timestamp': datetime.datetime.now().isoformat(),
            'deployed_from': 'v_best',
            'deployed_from_commit': os.environ.get('GITHUB_SHA', 'local_deployment'),
            'deployed_by': os.environ.get('GITHUB_ACTOR', 'local_user')
        }
        
        # Charger les métadonnées v_best pour obtenir des informations supplémentaires
        v_best_metadata_path = "deploy/v_best/v_best_metadata.json"
        if os.path.exists(v_best_metadata_path):
            with open(v_best_metadata_path, 'r') as f:
                v_best_metadata = json.load(f)
            deployment_info['v_best_commit_id'] = v_best_metadata.get('commit_id', 'unknown')
            deployment_info['v_best_timestamp'] = v_best_metadata.get('timestamp', 'unknown')
        
        with open("deploy/deployment_info.json", 'w') as f:
            json.dump(deployment_info, f, indent=2)
            
        logger.info("🚀 Modèle déployé avec succès depuis v_best")
        return True
        
    except Exception as e:
        logger.error(f"Erreur lors du déploiement depuis v_best: {e}")
        return False

def deploy_current_model():
    """Déploie le modèle actuel (fallback si v_best n'existe pas)."""
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
            'deployed_from': 'current_build',
            'deployed_from_commit': os.environ.get('GITHUB_SHA', 'local_deployment'),
            'deployed_by': os.environ.get('GITHUB_ACTOR', 'local_user'),
            'note': 'Deployed current model as no v_best was available'
        }
        
        with open("deploy/deployment_info.json", 'w') as f:
            json.dump(deployment_info, f, indent=2)
            
        logger.info("🚀 Modèle actuel déployé (aucun v_best disponible)")
        return True
        
    except Exception as e:
        logger.error(f"Erreur lors du déploiement du modèle actuel: {e}")
        return False

def main():
    """Fonction principale pour déployer le modèle."""
    try:
        # Charger le rapport de comparaison
        comparison_report = load_comparison_report()
        
        # Toujours déployer depuis v_best si disponible
        if check_v_best_exists():
            success = deploy_from_v_best()
            
            if success:
                # Afficher des informations sur le modèle déployé
                with open("deploy/v_best/v_best_metadata.json", 'r') as f:
                    v_best_metadata = json.load(f)
                
                logger.info(f"📊 Modèle v_best déployé:")
                logger.info(f"   - Version: {v_best_metadata['model_info']['model_version']}")
                logger.info(f"   - Type: {v_best_metadata['model_info']['model_type']}")
                logger.info(f"   - Commit: {v_best_metadata['commit_id'][:8]}...")
                logger.info(f"   - Métriques: {v_best_metadata['metrics']}")
                
                if comparison_report.get('is_new_v_best', False):
                    logger.info("✨ Ce déploiement utilise le nouveau modèle v_best")
                else:
                    logger.info("📦 Ce déploiement utilise le modèle v_best existant")
                    
            else:
                logger.error("Échec du déploiement depuis v_best")
                sys.exit(1)
        else:
            # Fallback: déployer le modèle actuel
            logger.warning("Aucun modèle v_best trouvé, déploiement du modèle actuel")
            success = deploy_current_model()
            
            if not success:
                logger.error("Échec du déploiement du modèle actuel")
                sys.exit(1)
        
        logger.info("Déploiement terminé avec succès")
            
    except Exception as e:
        logger.error(f"Erreur lors du déploiement du modèle: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()