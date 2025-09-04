"""
Script pour entraîner et déployer un nouveau modèle avec des paramètres différents
pour permettre la comparaison entre deux versions.
"""
import os
import sys
import subprocess
import shutil
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def backup_current_model():
    """Sauvegarde le modèle actuellement déployé pour permettre la comparaison."""
    if os.path.exists("deploy/model_metadata.json"):
        # Créer un répertoire pour la sauvegarde si nécessaire
        os.makedirs("model_history", exist_ok=True)
        
        # Copier les fichiers du modèle actuellement déployé
        if os.path.exists("deploy/model.pkl"):
            shutil.copy("deploy/model.pkl", "model_history/previous_model.pkl")
        if os.path.exists("deploy/model_metadata.json"):
            shutil.copy("deploy/model_metadata.json", "model_history/previous_metadata.json")
        if os.path.exists("deploy/evaluation_report.json"):
            shutil.copy("deploy/evaluation_report.json", "model_history/previous_evaluation.json")
        
        logger.info("Sauvegarde du modèle actuel terminée.")
        return True
    else:
        logger.warning("Aucun modèle déployé trouvé. Aucune sauvegarde effectuée.")
        return False

def switch_model_config(config_path):
    """Change la configuration du modèle utilisée pour l'entraînement."""
    if not os.path.exists(config_path):
        logger.error(f"Le fichier de configuration {config_path} n'existe pas.")
        return False
    
    # Copier la nouvelle configuration en tant que configuration principale
    shutil.copy(config_path, "models/model_config.yml")
    logger.info(f"Configuration du modèle mise à jour avec {config_path}.")
    return True

def run_pipeline(model_type):
    """Exécute le pipeline complet avec un type de modèle spécifié."""
    try:
        cmd = ["python", "run.py", "--type", model_type, "--generate-data", "--stages", "all"]
        logger.info(f"Exécution de la commande: {' '.join(cmd)}")
        
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Afficher la sortie en temps réel
        for stdout_line in iter(process.stdout.readline, ""):
            print(stdout_line, end="")
        
        # Capturer les erreurs
        stderr = process.stderr.read()
        if stderr:
            print(f"ERREURS: {stderr}")
        
        return_code = process.wait()
        if return_code != 0:
            logger.error(f"Le pipeline a échoué avec le code de retour {return_code}")
            return False
        
        logger.info("Pipeline exécuté avec succès.")
        return True
        
    except Exception as e:
        logger.error(f"Erreur lors de l'exécution du pipeline: {e}")
        return False

def main():
    """Fonction principale pour entraîner un nouveau modèle et permettre la comparaison."""
    # Vérifier si un modèle est déjà déployé
    has_previous_model = backup_current_model()
    
    # Utiliser la nouvelle configuration
    config_path = "models/model_config_v2.yml"
    if not switch_model_config(config_path):
        sys.exit(1)
    
    # Exécuter le pipeline avec la nouvelle configuration
    model_type = "classification"  # Le type est défini dans le fichier de configuration
    if not run_pipeline(model_type):
        sys.exit(1)
    
    # Générer un rapport de comparaison si un modèle précédent existe
    if has_previous_model:
        try:
            # Appeler directement le script de comparaison
            subprocess.run(["python", "src/compare_models.py"], check=True)
            logger.info("Comparaison des modèles terminée avec succès.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Erreur lors de la comparaison des modèles: {e}")
    
    logger.info("Entraînement et déploiement du nouveau modèle terminés avec succès.")
    logger.info("Vous pouvez maintenant visualiser la comparaison dans l'application Streamlit.")

if __name__ == "__main__":
    main()