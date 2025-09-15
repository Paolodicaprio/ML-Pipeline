#!/usr/bin/env python3
"""
Script principal pour exécuter le pipeline ML complet en une seule commande.
Automatise tout le workflow de l'entraînement au déploiement.
"""

import os
import sys
import json
import yaml
import subprocess
import argparse
from datetime import datetime
from pathlib import Path
import logging

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('pipeline_execution.log')
    ]
)
logger = logging.getLogger(__name__)

class MLPipelineOrchestrator:
    """Orchestrateur principal du pipeline ML."""
    
    def __init__(self, config_path="models/model_config.yml"):
        """Initialise l'orchestrateur."""
        self.config_path = config_path
        self.config = self.load_config()
        self.start_time = datetime.now()
        self.execution_log = []
        
        # Créer les dossiers nécessaires
        self.create_directories()
        
    def load_config(self):
        """Charge la configuration du modèle."""
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            logger.info(f"Configuration chargée depuis {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Erreur lors du chargement de la configuration: {e}")
            sys.exit(1)
    
    def create_directories(self):
        """Crée les dossiers nécessaires."""
        directories = [
            'build', 'data', 'deploy', 'model_history', 
            'reports/drift', 'build/visualizations'
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
        
        logger.info("Dossiers créés avec succès")
    
    def execute_step(self, step_name, command, description, critical=True):
        """Exécute une étape du pipeline."""
        logger.info(f"[RUNNING] {step_name}: {description}")
        
        try:
            # Enregistrer le début de l'étape
            step_start = datetime.now()
            
            # Exécuter la commande
            if isinstance(command, list):
                result = subprocess.run(command, capture_output=True, text=True, check=True)
            else:
                result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
            
            # Calculer la durée
            duration = datetime.now() - step_start
            
            # Enregistrer le succès
            self.execution_log.append({
                'step': step_name,
                'status': 'SUCCESS',
                'duration': duration.total_seconds(),
                'description': description,
                'timestamp': step_start.isoformat()
            })
            
            logger.info(f"[SUCCESS] {step_name} terminé avec succès ({duration.total_seconds():.2f}s)")
            
            if result.stdout:
                logger.debug(f"Sortie: {result.stdout}")
            
            return True, result.stdout
            
        except subprocess.CalledProcessError as e:
            duration = datetime.now() - step_start
            
            # Enregistrer l'échec
            self.execution_log.append({
                'step': step_name,
                'status': 'FAILED',
                'duration': duration.total_seconds(),
                'description': description,
                'error': str(e),
                'stderr': e.stderr,
                'timestamp': step_start.isoformat()
            })
            
            logger.error(f"[ERROR] {step_name} échoué: {e}")
            if e.stderr:
                logger.error(f"Erreur: {e.stderr}")
            
            if critical:
                logger.error("Arrêt du pipeline en raison d'une erreur critique")
                self.generate_failure_report()
                sys.exit(1)
            
            return False, e.stderr
    
    def check_dependencies(self):
        """Vérifie les dépendances Python."""
        logger.info("[CHECK] Vérification des dépendances...")
        
        required_packages = [
            'numpy', 'pandas', 'scikit-learn', 'matplotlib', 
            'seaborn', 'pyyaml', 'streamlit', 'fastapi'
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
                logger.debug(f"[SUCCESS] {package}: OK")
            except ImportError:
                missing_packages.append(package)
                logger.warning(f"[ERROR] {package}: MANQUANT")
        
        if missing_packages:
            logger.error(f"Packages manquants: {', '.join(missing_packages)}")
            logger.info("Installation des packages manquants...")
            
            install_cmd = f"pip install {' '.join(missing_packages)}"
            success, _ = self.execute_step(
                "INSTALL_DEPS", 
                install_cmd, 
                "Installation des dépendances manquantes"
            )
            
            if not success:
                return False
        
        logger.info("[SUCCESS] Toutes les dépendances sont disponibles")
        return True
    
    def generate_sample_data(self):
        """Génère des données d'exemple si nécessaire."""
        model_type = self.config.get('model', {}).get('type', 'classification')
        train_path = self.config.get('data', {}).get('train_path', 'data/train.csv')
        
        if not os.path.exists(train_path):
            logger.info(f"Génération de données d'exemple pour {model_type}...")
            
            cmd = f"python src/generate_sample_data.py --type {model_type} --samples 1000 --features 10"
            return self.execute_step(
                "GENERATE_DATA", 
                cmd, 
                f"Génération de données d'exemple ({model_type})"
            )
        else:
            logger.info(f"Données d'entraînement trouvées: {train_path}")
            return True, "Data already exists"
    
    def build_model(self):
        """Construit et entraîne le modèle."""
        return self.execute_step(
            "BUILD_MODEL",
            "python src/build_model.py",
            "Construction et entraînement du modèle"
        )
    
    def test_model(self):
        """Teste le modèle."""
        return self.execute_step(
            "TEST_MODEL",
            "python src/test_model.py",
            "Test du modèle sur les données de test"
        )
    
    def monitor_drift(self):
        """Surveille le data drift."""
        # Ajouter le chemin des données de test par défaut
        return self.execute_step(
            "MONITOR_DRIFT",
            f"python src/monitor_drift.py --config {self.config_path} --new-data data/classification_test.csv",
            "Surveillance du data drift",
            critical=False  # Non critique pour permettre la continuation
        )
    
    def evaluate_model(self):
        """Évalue les performances du modèle."""
        return self.execute_step(
            "EVALUATE_MODEL",
            "python src/evaluate_model.py",
            "Évaluation des performances du modèle"
        )
    
    def compare_models(self):
        """Compare avec le modèle v_best."""
        return self.execute_step(
            "COMPARE_MODELS",
            "python src/compare_models.py",
            "Comparaison avec le modèle v_best"
        )
    
    def deploy_model(self):
        """Déploie le modèle."""
        return self.execute_step(
            "DEPLOY_MODEL",
            "python src/deploy_model.py",
            "Déploiement du modèle"
        )
    
    def run_docker_services(self):
        """Lance les services Docker."""
        logger.info("[DOCKER] Lancement des services Docker...")
        
        # Vérifier si Docker est disponible
        try:
            subprocess.run(["docker", "--version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("Docker n'est pas disponible, services non lancés")
            return False, "Docker not available"
        
        # Arrêter les services existants
        self.execute_step(
            "DOCKER_DOWN",
            "docker-compose down",
            "Arrêt des services Docker existants",
            critical=False
        )
        
        # Construire et lancer les services
        return self.execute_step(
            "DOCKER_UP",
            "docker-compose up -d --build",
            "Construction et lancement des services Docker",
            critical=False
        )
    
    def generate_success_report(self):
        """Génère un rapport de succès."""
        end_time = datetime.now()
        total_duration = end_time - self.start_time
        
        report = {
            'pipeline_status': 'SUCCESS',
            'start_time': self.start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'total_duration_seconds': total_duration.total_seconds(),
            'execution_steps': self.execution_log,
            'model_info': self.config.get('model', {}),
            'summary': {
                'total_steps': len(self.execution_log),
                'successful_steps': len([s for s in self.execution_log if s['status'] == 'SUCCESS']),
                'failed_steps': len([s for s in self.execution_log if s['status'] == 'FAILED'])
            }
        }
        
        # Sauvegarder le rapport
        report_path = f"pipeline_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"[REPORT] Rapport de pipeline généré: {report_path}")
        return report_path
    
    def generate_failure_report(self):
        """Génère un rapport d'échec."""
        end_time = datetime.now()
        total_duration = end_time - self.start_time
        
        report = {
            'pipeline_status': 'FAILED',
            'start_time': self.start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'total_duration_seconds': total_duration.total_seconds(),
            'execution_steps': self.execution_log,
            'failure_analysis': {
                'failed_steps': [s for s in self.execution_log if s['status'] == 'FAILED'],
                'last_successful_step': next((s for s in reversed(self.execution_log) if s['status'] == 'SUCCESS'), None)
            }
        }
        
        # Sauvegarder le rapport
        report_path = f"pipeline_failure_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.error(f"[REPORT] Rapport d'échec généré: {report_path}")
        return report_path
    
    def run_full_pipeline(self, skip_docker=False, skip_drift=False):
        """Exécute le pipeline complet."""
        logger.info("[START] Démarrage du pipeline ML complet")
        logger.info(f"Configuration: {self.config_path}")
        logger.info(f"Modèle: {self.config.get('model', {}).get('name', 'Unknown')}")
        
        try:
            # 1. Vérification des dépendances
            if not self.check_dependencies():
                logger.error("Échec de la vérification des dépendances")
                return False
            
            # 2. Génération des données si nécessaire
            success, _ = self.generate_sample_data()
            if not success:
                return False
            
            # 3. Construction du modèle
            success, _ = self.build_model()
            if not success:
                return False
            
            # 4. Test du modèle
            success, _ = self.test_model()
            if not success:
                return False
            
            # 5. Surveillance du drift (optionnel)
            if not skip_drift:
                success, _ = self.monitor_drift()
                # Continue même si le drift échoue
            
            # 6. Évaluation du modèle
            success, _ = self.evaluate_model()
            if not success:
                return False
            
            # 7. Comparaison avec v_best
            success, _ = self.compare_models()
            if not success:
                return False
            
            # 8. Déploiement
            success, _ = self.deploy_model()
            if not success:
                return False
            
            # 9. Lancement des services Docker (optionnel)
            if not skip_docker:
                self.run_docker_services()
            
            # Génération du rapport de succès
            report_path = self.generate_success_report()
            
            logger.info("[CELEBRATE] Pipeline ML exécuté avec succès!")
            logger.info(f"[REPORT] Rapport: {report_path}")
            logger.info(f"[TIME] Durée totale: {(datetime.now() - self.start_time).total_seconds():.2f}s")
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur inattendue dans le pipeline: {e}")
            self.generate_failure_report()
            return False

def main():
    """Fonction principale."""
    parser = argparse.ArgumentParser(description='Orchestrateur de Pipeline ML')
    parser.add_argument('--config', type=str, default='models/model_config.yml', 
                       help='Chemin vers le fichier de configuration')
    parser.add_argument('--skip-docker', action='store_true', 
                       help='Ignorer le lancement des services Docker')
    parser.add_argument('--skip-drift', action='store_true', 
                       help='Ignorer la surveillance du drift')
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='Mode verbose (debug)')
    
    args = parser.parse_args()
    
    # Configuration du niveau de logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Vérifier que le fichier de configuration existe
    if not os.path.exists(args.config):
        logger.error(f"Fichier de configuration non trouvé: {args.config}")
        sys.exit(1)
    
    # Créer l'orchestrateur
    orchestrator = MLPipelineOrchestrator(args.config)
    
    # Exécuter le pipeline
    success = orchestrator.run_full_pipeline(
        skip_docker=args.skip_docker,
        skip_drift=args.skip_drift
    )
    
    if success:
        logger.info("[SUCCESS] Pipeline terminé avec succès")
        sys.exit(0)
    else:
        logger.error("[ERROR] Pipeline échoué")
        sys.exit(1)

if __name__ == "__main__":
    main()