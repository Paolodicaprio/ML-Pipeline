#!/usr/bin/env python3
"""
Pipeline ML intégré avec surveillance du data drift.
Version optimisée qui intègre le monitoring de drift dans le workflow principal.
"""

import os
import sys
import json
import yaml
import logging
from datetime import datetime
from pathlib import Path

# Ajouter le dossier parent au path pour les imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.build_model import ModelBuilder
from src.test_model import ModelTester
from src.evaluate_model import ModelEvaluator
from src.compare_models import ModelComparator
from src.deploy_model import ModelDeployer
from src.monitor_drift import DataDriftMonitor

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class IntegratedMLPipeline:
    """Pipeline ML intégré avec surveillance du drift."""
    
    def __init__(self, config_path="models/model_config.yml"):
        """Initialise le pipeline intégré."""
        self.config_path = config_path
        self.config = self.load_config()
        self.drift_monitor = DataDriftMonitor(config_path)
        
        # Créer les dossiers nécessaires
        self.create_directories()
        
    def load_config(self):
        """Charge la configuration."""
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            return config
        except Exception as e:
            logger.error(f"Erreur lors du chargement de la configuration: {e}")
            sys.exit(1)
    
    def create_directories(self):
        """Crée les dossiers nécessaires."""
        directories = [
            'build', 'deploy', 'reports/drift', 
            'build/visualizations', 'model_history'
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def run_drift_check(self, new_data_path=None):
        """Exécute la vérification du drift."""
        logger.info("🔍 Vérification du data drift...")
        
        try:
            # Exécuter la surveillance du drift
            drift_ok = self.drift_monitor.monitor_drift(new_data_path)
            
            if not drift_ok:
                logger.warning("⚠️ Drift significatif détecté!")
                
                # Charger le rapport de drift pour plus de détails
                drift_report_path = "reports/drift/latest_drift_report.json"
                if os.path.exists(drift_report_path):
                    with open(drift_report_path, 'r') as f:
                        drift_report = json.load(f)
                    
                    global_status = drift_report.get('global_status', 'UNKNOWN')
                    drift_percentage = drift_report.get('summary', {}).get('drift_percentage', 0)
                    
                    logger.info(f"Statut du drift: {global_status}")
                    logger.info(f"Pourcentage de caractéristiques affectées: {drift_percentage:.1f}%")
                    
                    # Décision basée sur le niveau de drift
                    if global_status == 'ALARMANT':
                        logger.error("🚨 Drift critique détecté - Arrêt du pipeline")
                        return False, "CRITICAL_DRIFT"
                    elif global_status == 'ATTENTION':
                        logger.warning("⚠️ Drift modéré détecté - Pipeline continue avec surveillance")
                        return True, "MODERATE_DRIFT"
                else:
                    logger.warning("Rapport de drift non disponible")
                    return True, "DRIFT_REPORT_MISSING"
            else:
                logger.info("✅ Aucun drift significatif détecté")
                return True, "NO_DRIFT"
                
        except Exception as e:
            logger.error(f"Erreur lors de la vérification du drift: {e}")
            return True, "DRIFT_CHECK_FAILED"  # Continue le pipeline
    
    def run_model_pipeline(self):
        """Exécute le pipeline de modèle standard."""
        logger.info("🏗️ Exécution du pipeline de modèle...")
        
        try:
            # 1. Construction du modèle
            logger.info("1️⃣ Construction du modèle...")
            builder = ModelBuilder(self.config_path)
            if not builder.build_model():
                logger.error("Échec de la construction du modèle")
                return False
            
            # 2. Test du modèle
            logger.info("2️⃣ Test du modèle...")
            tester = ModelTester(self.config_path)
            if not tester.test_model():
                logger.error("Échec des tests du modèle")
                return False
            
            # 3. Évaluation du modèle
            logger.info("3️⃣ Évaluation du modèle...")
            evaluator = ModelEvaluator(self.config_path)
            if not evaluator.evaluate_model():
                logger.error("Échec de l'évaluation du modèle")
                return False
            
            # 4. Comparaison avec v_best
            logger.info("4️⃣ Comparaison avec v_best...")
            comparator = ModelComparator(self.config_path)
            if not comparator.compare_models():
                logger.error("Échec de la comparaison des modèles")
                return False
            
            # 5. Déploiement
            logger.info("5️⃣ Déploiement du modèle...")
            deployer = ModelDeployer(self.config_path)
            if not deployer.deploy_model():
                logger.error("Échec du déploiement du modèle")
                return False
            
            logger.info("✅ Pipeline de modèle terminé avec succès")
            return True
            
        except Exception as e:
            logger.error(f"Erreur dans le pipeline de modèle: {e}")
            return False
    
    def generate_integrated_report(self, drift_status, pipeline_success):
        """Génère un rapport intégré du pipeline avec drift."""
        try:
            timestamp = datetime.now().isoformat()
            
            # Charger les rapports existants
            reports = {}
            
            # Rapport de drift
            drift_report_path = "reports/drift/latest_drift_report.json"
            if os.path.exists(drift_report_path):
                with open(drift_report_path, 'r') as f:
                    reports['drift'] = json.load(f)
            
            # Rapport de comparaison
            comparison_report_path = "build/comparison_report.json"
            if os.path.exists(comparison_report_path):
                with open(comparison_report_path, 'r') as f:
                    reports['comparison'] = json.load(f)
            
            # Rapport d'évaluation
            evaluation_report_path = "build/evaluation_report.json"
            if os.path.exists(evaluation_report_path):
                with open(evaluation_report_path, 'r') as f:
                    reports['evaluation'] = json.load(f)
            
            # Créer le rapport intégré
            integrated_report = {
                'timestamp': timestamp,
                'pipeline_status': 'SUCCESS' if pipeline_success else 'FAILED',
                'drift_status': drift_status,
                'config': self.config,
                'reports': reports,
                'summary': {
                    'drift_check': drift_status,
                    'model_pipeline': 'SUCCESS' if pipeline_success else 'FAILED',
                    'overall_status': 'SUCCESS' if pipeline_success and drift_status not in ['CRITICAL_DRIFT'] else 'FAILED'
                }
            }
            
            # Ajouter des recommandations
            recommendations = []
            
            if drift_status == 'CRITICAL_DRIFT':
                recommendations.extend([
                    "Drift critique détecté - Réentraînement immédiat recommandé",
                    "Analyser les causes du changement de distribution des données",
                    "Vérifier la qualité des nouvelles données"
                ])
            elif drift_status == 'MODERATE_DRIFT':
                recommendations.extend([
                    "Drift modéré détecté - Surveillance renforcée recommandée",
                    "Planifier un réentraînement dans un futur proche",
                    "Analyser les caractéristiques affectées"
                ])
            elif drift_status == 'NO_DRIFT':
                recommendations.append("Aucun drift détecté - Continuer la surveillance régulière")
            
            if pipeline_success:
                recommendations.append("Pipeline exécuté avec succès - Modèle prêt pour la production")
            else:
                recommendations.append("Échec du pipeline - Vérifier les logs pour identifier les problèmes")
            
            integrated_report['recommendations'] = recommendations
            
            # Sauvegarder le rapport
            report_path = f"reports/integrated_pipeline_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_path, 'w') as f:
                json.dump(integrated_report, f, indent=2)
            
            # Sauvegarder aussi comme dernier rapport
            latest_report_path = "reports/latest_integrated_report.json"
            with open(latest_report_path, 'w') as f:
                json.dump(integrated_report, f, indent=2)
            
            logger.info(f"📊 Rapport intégré généré: {report_path}")
            return report_path
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération du rapport intégré: {e}")
            return None
    
    def run_integrated_pipeline(self, new_data_path=None, force_pipeline=False):
        """Exécute le pipeline intégré avec surveillance du drift."""
        logger.info("🚀 Démarrage du pipeline ML intégré avec surveillance du drift")
        
        start_time = datetime.now()
        
        try:
            # 1. Vérification du drift
            drift_ok, drift_status = self.run_drift_check(new_data_path)
            
            # 2. Décision de continuer le pipeline
            if not drift_ok and not force_pipeline:
                logger.error("Pipeline arrêté en raison d'un drift critique")
                self.generate_integrated_report(drift_status, False)
                return False
            
            # 3. Exécution du pipeline de modèle
            pipeline_success = self.run_model_pipeline()
            
            # 4. Génération du rapport intégré
            report_path = self.generate_integrated_report(drift_status, pipeline_success)
            
            # 5. Résumé final
            end_time = datetime.now()
            duration = end_time - start_time
            
            logger.info("=" * 60)
            logger.info("📊 RÉSUMÉ DU PIPELINE INTÉGRÉ")
            logger.info("=" * 60)
            logger.info(f"⏱️ Durée totale: {duration.total_seconds():.2f}s")
            logger.info(f"🔍 Statut du drift: {drift_status}")
            logger.info(f"🏗️ Pipeline de modèle: {'SUCCESS' if pipeline_success else 'FAILED'}")
            logger.info(f"📊 Rapport: {report_path}")
            
            if pipeline_success and drift_status != 'CRITICAL_DRIFT':
                logger.info("🎉 Pipeline intégré terminé avec succès!")
                return True
            else:
                logger.error("❌ Pipeline intégré échoué")
                return False
                
        except Exception as e:
            logger.error(f"Erreur inattendue dans le pipeline intégré: {e}")
            self.generate_integrated_report("ERROR", False)
            return False

def main():
    """Fonction principale."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Pipeline ML intégré avec surveillance du drift')
    parser.add_argument('--config', type=str, default='models/model_config.yml',
                       help='Chemin vers le fichier de configuration')
    parser.add_argument('--new-data', type=str, 
                       help='Chemin vers les nouvelles données pour la vérification du drift')
    parser.add_argument('--force', action='store_true',
                       help='Forcer l\'exécution du pipeline même en cas de drift critique')
    
    args = parser.parse_args()
    
    # Vérifier que le fichier de configuration existe
    if not os.path.exists(args.config):
        logger.error(f"Fichier de configuration non trouvé: {args.config}")
        sys.exit(1)
    
    # Créer et exécuter le pipeline
    pipeline = IntegratedMLPipeline(args.config)
    success = pipeline.run_integrated_pipeline(
        new_data_path=args.new_data,
        force_pipeline=args.force
    )
    
    if success:
        logger.info("✅ Pipeline intégré terminé avec succès")
        sys.exit(0)
    else:
        logger.error("❌ Pipeline intégré échoué")
        sys.exit(1)

if __name__ == "__main__":
    main()