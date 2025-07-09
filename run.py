#!/usr/bin/env python3
"""
Script pour exécuter le pipeline ML manuellement.
"""
import os
import sys
import argparse
import subprocess
import logging
import yaml

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_command(cmd, description=None):
    """Exécute une commande shell et affiche le résultat."""
    if description:
        logger.info(f"Exécution de: {description}")
    
    try:
        process = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            shell=True
        )
        
        if process.returncode == 0:
            logger.info("Commande exécutée avec succès")
            return True, process.stdout
        else:
            logger.error(f"Erreur lors de l'exécution de la commande: {process.stderr}")
            return False, process.stderr
    except Exception as e:
        logger.error(f"Exception lors de l'exécution de la commande: {e}")
        return False, str(e)

def setup_environment():
    """Prépare l'environnement pour l'exécution du pipeline."""
    # Créer les répertoires nécessaires
    os.makedirs("data", exist_ok=True)
    os.makedirs("build", exist_ok=True)
    os.makedirs("deploy", exist_ok=True)
    
    # Installer les dépendances
    success, output = run_command("pip install -r requirements.txt", "Installation des dépendances")
    return success

def generate_data(data_type):
    """Génère des données synthétiques."""
    cmd = f"python src/generate_sample_data.py --type {data_type} --samples 1000 --features 10"
    if data_type == 'classification':
        cmd += " --classes 2"
    
    success, output = run_command(cmd, f"Génération des données de {data_type}")
    return success

def update_config(data_type):
    """Met à jour le fichier de configuration du modèle."""
    try:
        # Charger la configuration actuelle
        with open("models/model_config.yml", 'r') as file:
            config = yaml.safe_load(file)
        
        # Mettre à jour le type de modèle et les chemins de données
        config['model']['type'] = data_type
        config['data']['train_path'] = f"data/{data_type}_train.csv"
        config['data']['test_path'] = f"data/{data_type}_test.csv"
        
        # Sauvegarder la configuration mise à jour
        with open("models/model_config.yml", 'w') as file:
            yaml.dump(config, file, default_flow_style=False)
            
        logger.info(f"Configuration mise à jour pour le type de modèle: {data_type}")
        return True
    except Exception as e:
        logger.error(f"Erreur lors de la mise à jour de la configuration: {e}")
        return False

def run_pipeline(stages):
    """Exécute les étapes du pipeline spécifiées."""
    pipeline_stages = {
        "build": {
            "cmd": "python src/build_model.py",
            "description": "Construction du modèle"
        },
        "test": {
            "cmd": "python src/test_model.py",
            "description": "Test du modèle"
        },
        "evaluate": {
            "cmd": "python src/evaluate_model.py",
            "description": "Évaluation du modèle"
        },
        "compare": {
            "cmd": "python src/compare_models.py",
            "description": "Comparaison des modèles"
        },
        "deploy": {
            "cmd": "python src/deploy_model.py",
            "description": "Déploiement du modèle"
        }
    }
    
    # Si stages est None ou vide, exécuter toutes les étapes
    if not stages:
        stages = list(pipeline_stages.keys())
    
    # Exécuter les étapes demandées
    for stage in stages:
        if stage in pipeline_stages:
            cmd = pipeline_stages[stage]["cmd"]
            desc = pipeline_stages[stage]["description"]
            
            success, output = run_command(cmd, desc)
            if not success:
                logger.error(f"Échec de l'étape {stage}, arrêt du pipeline")
                return False
        else:
            logger.warning(f"Étape inconnue: {stage}, ignorée")
    
    return True

def run_app():
    """Lance l'application Streamlit."""
    cmd = "streamlit run app/app.py"
    success, output = run_command(cmd, "Lancement de l'application Streamlit")
    return success

def main():
    """Fonction principale du script."""
    parser = argparse.ArgumentParser(description='Exécuteur de pipeline ML')
    parser.add_argument('--type', type=str, choices=['regression', 'classification'], default='classification',
                        help='Type de modèle à construire')
    parser.add_argument('--stages', type=str, nargs='+', choices=['build', 'test', 'evaluate', 'compare', 'deploy', 'all'],
                        help='Étapes du pipeline à exécuter')
    parser.add_argument('--generate-data', action='store_true', help='Générer des données synthétiques')
    parser.add_argument('--run-app', action='store_true', help='Lancer l\'application Streamlit')
    
    args = parser.parse_args()
    
    # Convertir 'all' en None pour exécuter toutes les étapes
    stages = args.stages
    if stages and 'all' in stages:
        stages = None
    
    try:
        # Préparer l'environnement
        if not setup_environment():
            logger.error("Échec de la préparation de l'environnement")
            sys.exit(1)
        
        # Générer des données si demandé
        if args.generate_data:
            if not generate_data(args.type):
                logger.error("Échec de la génération des données")
                sys.exit(1)
            
            # Mettre à jour la configuration pour utiliser les données générées
            if not update_config(args.type):
                logger.error("Échec de la mise à jour de la configuration")
                sys.exit(1)
        
        # Exécuter les étapes du pipeline si spécifiées
        if stages:
            if not run_pipeline(stages):
                logger.error("Échec du pipeline")
                sys.exit(1)
        
        # Lancer l'application si demandé
        if args.run_app:
            if not run_app():
                logger.error("Échec du lancement de l'application")
                sys.exit(1)
        
        logger.info("Exécution terminée avec succès")
        
    except Exception as e:
        logger.error(f"Erreur lors de l'exécution: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()