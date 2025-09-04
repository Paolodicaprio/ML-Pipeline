"""
Script amélioré pour entraîner et comparer deux modèles différents
"""
import os
import sys
import subprocess
import shutil
import json
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def backup_current_model():
    """Sauvegarde le modèle actuellement déployé."""
    if os.path.exists("deploy/model_metadata.json"):
        # Créer un répertoire pour la sauvegarde si nécessaire
        os.makedirs("model_backup", exist_ok=True)
        
        # Copier les fichiers du modèle actuellement déployé
        if os.path.exists("deploy/model.pkl"):
            shutil.copy("deploy/model.pkl", "model_backup/model_v1.pkl")
        if os.path.exists("deploy/model_metadata.json"):
            shutil.copy("deploy/model_metadata.json", "model_backup/model_metadata_v1.json")
        if os.path.exists("deploy/evaluation_report.json"):
            shutil.copy("deploy/evaluation_report.json", "model_backup/evaluation_report_v1.json")
        
        logger.info("Sauvegarde du modèle v1 terminée.")
        return True
    else:
        logger.warning("Aucun modèle déployé trouvé. Aucune sauvegarde effectuée.")
        return False

def train_first_model(model_type="regression"):
    """Entraîne le premier modèle avec la configuration standard."""
    # Copier la configuration originale
    if os.path.exists("models/model_config_v2.yml"):
        # Sauvegarde au cas où
        if os.path.exists("models/model_config.yml"):
            shutil.copy("models/model_config.yml", "models/model_config_original.yml")
    else:
        logger.warning("Configuration v2 non trouvée. Utilisation de la configuration existante.")
    
    # Utiliser la configuration originale
    if os.path.exists("models/model_config_original.yml"):
        shutil.copy("models/model_config_original.yml", "models/model_config.yml")
        logger.info("Configuration du modèle v1 restaurée.")
    
    # Exécuter le pipeline pour le premier modèle
    run_pipeline(model_type)
    
    # Sauvegarder le premier modèle
    backup_current_model()

def train_second_model(model_type="classification"):
    """Entraîne le second modèle avec la configuration v2."""
    if not os.path.exists("models/model_config_v2.yml"):
        logger.error("Le fichier de configuration v2 n'existe pas.")
        return False
    
    # Copier la nouvelle configuration en tant que configuration principale
    shutil.copy("models/model_config_v2.yml", "models/model_config.yml")
    logger.info("Configuration du modèle mise à jour pour v2.")
    
    # Exécuter le pipeline pour le second modèle
    run_pipeline(model_type)
    
    # Sauvegarder le deuxième modèle
    os.makedirs("model_backup", exist_ok=True)
    if os.path.exists("deploy/model.pkl"):
        shutil.copy("deploy/model.pkl", "model_backup/model_v2.pkl")
    if os.path.exists("deploy/model_metadata.json"):
        shutil.copy("deploy/model_metadata.json", "model_backup/model_metadata_v2.json")
    if os.path.exists("deploy/evaluation_report.json"):
        shutil.copy("deploy/evaluation_report.json", "model_backup/evaluation_report_v2.json")
    
    logger.info("Sauvegarde du modèle v2 terminée.")
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

def generate_comparison_report():
    """Génère un rapport de comparaison entre les deux modèles."""
    try:
        # Vérifier que les deux modèles existent
        if (not os.path.exists("model_backup/model_metadata_v1.json") or 
            not os.path.exists("model_backup/model_metadata_v2.json")):
            logger.error("Les métadonnées des deux modèles n'existent pas.")
            return False
        
        # Charger les métadonnées et rapports d'évaluation
        with open("model_backup/model_metadata_v1.json", 'r') as f:
            metadata_v1 = json.load(f)
        with open("model_backup/model_metadata_v2.json", 'r') as f:
            metadata_v2 = json.load(f)
        
        with open("model_backup/evaluation_report_v1.json", 'r') as f:
            eval_v1 = json.load(f)
        with open("model_backup/evaluation_report_v2.json", 'r') as f:
            eval_v2 = json.load(f)
        
        # Créer le rapport de comparaison
        comparison = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "models_compared": [
                {
                    "name": metadata_v1.get("model_name", "Model V1"),
                    "version": metadata_v1.get("model_version", "1.0.0"),
                    "type": metadata_v1.get("model_type", "Unknown")
                },
                {
                    "name": metadata_v2.get("model_name", "Model V2"),
                    "version": metadata_v2.get("model_version", "2.0.0"),
                    "type": metadata_v2.get("model_type", "Unknown")
                }
            ],
            "metrics_comparison": {}
        }
        
        # Comparer les métriques de test
        model_type = metadata_v1.get("model_type")
        if model_type == "classification":
            metrics = ["accuracy", "precision", "recall", "f1"]
        else:
            metrics = ["mse", "mae", "rmse"]
        
        # Ajouter les métriques au rapport de comparaison
        for metric in metrics:
            if metric in eval_v1.get("test_metrics", {}) and metric in eval_v2.get("test_metrics", {}):
                val_v1 = eval_v1["test_metrics"][metric]
                val_v2 = eval_v2["test_metrics"][metric]
                
                # Calcul de l'amélioration (pour les métriques où une valeur plus élevée est meilleure)
                if metric in ["accuracy", "precision", "recall", "f1"]:
                    improvement = ((val_v2 - val_v1) / val_v1) * 100 if val_v1 > 0 else float("inf")
                    better = val_v2 > val_v1
                # Pour les métriques d'erreur, une valeur plus basse est meilleure
                else:
                    improvement = ((val_v1 - val_v2) / val_v1) * 100 if val_v1 > 0 else float("inf")
                    better = val_v2 < val_v1
                
                comparison["metrics_comparison"][metric] = {
                    "model_v1": val_v1,
                    "model_v2": val_v2,
                    "improvement_percent": improvement,
                    "improved": better
                }
        
        # Calculer l'amélioration globale (moyenne des améliorations)
        improvements = [v["improvement_percent"] for v in comparison["metrics_comparison"].values() 
                      if not pd.isna(v["improvement_percent"]) and v["improvement_percent"] != float("inf")]
        
        if improvements:
            comparison["overall_improvement"] = sum(improvements) / len(improvements)
        else:
            comparison["overall_improvement"] = None
        
        # Sauvegarder le rapport de comparaison
        with open("build/comparison_report.json", 'w') as f:
            json.dump(comparison, f, indent=2)
        
        # Générer également des visualisations pour le rapport
        generate_comparison_visualizations(comparison)
        
        logger.info(f"Rapport de comparaison généré avec succès. Amélioration globale: {comparison['overall_improvement']}")
        return True
        
    except Exception as e:
        logger.error(f"Erreur lors de la génération du rapport de comparaison: {e}")
        return False

def generate_comparison_visualizations(comparison):
    """Génère des visualisations pour comparer les deux modèles."""
    try:
        metrics = comparison["metrics_comparison"]
        
        # Créer un DataFrame pour la visualisation
        data = []
        for metric, values in metrics.items():
            data.append({
                "Métrique": metric,
                "Modèle V1": values["model_v1"],
                "Modèle V2": values["model_v2"],
                "Amélioration (%)": values["improvement_percent"]
            })
        
        df = pd.DataFrame(data)
        
        # Créer un répertoire pour les visualisations
        os.makedirs("build/visualizations", exist_ok=True)
        
        # Graphique de comparaison des métriques
        plt.figure(figsize=(12, 6))
        
        # Barplot pour comparer les métriques
        ax = plt.subplot(121)
        df_melt = pd.melt(df, id_vars=["Métrique"], value_vars=["Modèle V1", "Modèle V2"])
        sns.barplot(x="Métrique", y="value", hue="variable", data=df_melt, ax=ax)
        plt.title("Comparaison des métriques entre modèles")
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Graphique pour l'amélioration
        ax2 = plt.subplot(122)
        sns.barplot(x="Métrique", y="Amélioration (%)", data=df, ax=ax2)
        plt.axhline(y=0, color='r', linestyle='-')
        plt.title("Pourcentage d'amélioration")
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Sauvegarder la figure
        plt.savefig("build/visualizations/metrics_comparison.png", dpi=300, bbox_inches="tight")
        
        # Sauvegarder aussi au format base64 pour l'application Streamlit
        import io
        import base64
        
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode()
        
        # Sauvegarder l'image encodée en base64
        with open("build/visualizations/metrics_comparison_base64.txt", 'w') as f:
            f.write(img_str)
        
        plt.close()
        
        return True
        
    except Exception as e:
        logger.error(f"Erreur lors de la génération des visualisations: {e}")
        return False

def main():
    """Fonction principale pour entraîner et comparer deux modèles."""
    try:
        # Entraîner le premier modèle (régression)
        logger.info("Entraînement du premier modèle (régression)...")
        train_first_model("regression")
        
        # Entraîner le second modèle (classification)
        logger.info("Entraînement du second modèle (classification)...")
        train_second_model("classification")
        
        # Générer le rapport de comparaison
        logger.info("Génération du rapport de comparaison...")
        generate_comparison_report()
        
        logger.info("Entraînement et comparaison des modèles terminés avec succès.")
        logger.info("Vous pouvez maintenant visualiser la comparaison dans l'application Streamlit.")
        
    except Exception as e:
        logger.error(f"Erreur lors de l'exécution du script: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()