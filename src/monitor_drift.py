"""
Script de surveillance du Data Drift pour le pipeline ML.
VERSION CORRIGÉE avec gestion améliorée des erreurs et des chemins
"""
import os
import sys
import json
import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Backend non-interactif pour éviter les erreurs
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataDriftMonitor:
    """Classe pour surveiller le drift des données."""
    
    def __init__(self, config_path="models/model_config.yml"):
        """Initialise le moniteur de drift."""
        self.config = self.load_config(config_path)
        self.drift_threshold = 0.05  # Seuil p-value pour détecter le drift
        self.reports_dir = "reports/drift"
        os.makedirs(self.reports_dir, exist_ok=True)
        
    def load_config(self, config_path):
        """Charge la configuration du modèle."""
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            return config
        except Exception as e:
            logger.error(f"Erreur lors du chargement de la configuration: {e}")
            sys.exit(1)
    
    def load_reference_data(self):
        """Charge les données de référence (données d'entraînement)."""
        try:
            train_path = self.config['data']['train_path']
            df = pd.read_csv(train_path)
            
            # Séparer les caractéristiques et la cible
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]
            
            logger.info(f"Données de référence chargées: {X.shape[0]} échantillons, {X.shape[1]} caractéristiques")
            return X, y
        except Exception as e:
            logger.error(f"Erreur lors du chargement des données de référence: {e}")
            return None, None
    
    def load_new_data(self, new_data_path):
        """Charge les nouvelles données pour comparaison."""
        try:
            if not new_data_path or not os.path.exists(new_data_path):
                # Si pas de nouvelles données, utiliser une partie des données de test
                logger.info("Pas de nouvelles données trouvées, utilisation des données de test pour simulation")
                test_path = self.config['data'].get('test_path')
                if test_path and os.path.exists(test_path):
                    df = pd.read_csv(test_path)
                    # Prendre seulement une partie pour simuler un batch
                    df = df.sample(n=min(100, len(df)), random_state=42)
                else:
                    logger.warning("Aucune donnée de test disponible pour simulation")
                    return None, None
            else:
                df = pd.read_csv(new_data_path)
            
            # Séparer les caractéristiques et la cible
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]
            
            logger.info(f"Nouvelles données chargées: {X.shape[0]} échantillons, {X.shape[1]} caractéristiques")
            return X, y
        except Exception as e:
            logger.error(f"Erreur lors du chargement des nouvelles données: {e}")
            return None, None
    
    def calculate_drift_metrics(self, X_ref, X_new):
        """Calcule les métriques de drift entre les deux datasets."""
        drift_results = {}
        
        # Vérifier que les colonnes correspondent
        if list(X_ref.columns) != list(X_new.columns):
            logger.error("Les colonnes des datasets ne correspondent pas")
            return None
        
        for column in X_ref.columns:
            try:
                # Test de Kolmogorov-Smirnov pour détecter les changements de distribution
                ks_stat, ks_p_value = stats.ks_2samp(X_ref[column], X_new[column])
                
                # Test de Mann-Whitney U pour les différences de médiane
                mw_stat, mw_p_value = stats.mannwhitneyu(X_ref[column], X_new[column], alternative='two-sided')
                
                # Statistiques descriptives
                ref_mean = X_ref[column].mean()
                new_mean = X_new[column].mean()
                ref_std = X_ref[column].std()
                new_std = X_new[column].std()
                
                # Calcul du drift score (combinaison des p-values)
                drift_score = min(ks_p_value, mw_p_value)
                drift_detected = drift_score < self.drift_threshold
                
                drift_results[column] = {
                    'ks_statistic': float(ks_stat),
                    'ks_p_value': float(ks_p_value),
                    'mw_statistic': float(mw_stat),
                    'mw_p_value': float(mw_p_value),
                    'drift_score': float(drift_score),
                    'drift_detected': bool(drift_detected),
                    'ref_mean': float(ref_mean),
                    'new_mean': float(new_mean),
                    'ref_std': float(ref_std),
                    'new_std': float(new_std),
                    'mean_shift': float(abs(new_mean - ref_mean) / ref_std) if ref_std > 0 else 0.0
                }
                
            except Exception as e:
                logger.warning(f"Erreur lors du calcul du drift pour {column}: {e}")
                continue
        
        return drift_results
    
    def generate_drift_visualizations(self, X_ref, X_new, drift_results):
        """Génère les visualisations du drift."""
        try:
            # Créer une figure avec plusieurs sous-graphiques
            n_features = len(X_ref.columns)
            n_cols = 3
            n_rows = (n_features + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
            if n_rows == 1:
                axes = axes.reshape(1, -1)
            
            for i, column in enumerate(X_ref.columns):
                row = i // n_cols
                col = i % n_cols
                ax = axes[row, col]
                
                # Histogrammes comparatifs
                ax.hist(X_ref[column], alpha=0.7, label='Référence', bins=30, density=True)
                ax.hist(X_new[column], alpha=0.7, label='Nouvelles données', bins=30, density=True)
                
                # Ajouter les informations de drift
                drift_info = drift_results.get(column, {})
                drift_status = "DRIFT DÉTECTÉ" if drift_info.get('drift_detected', False) else "OK"
                color = 'red' if drift_info.get('drift_detected', False) else 'green'
                
                ax.set_title(f'{column}\n{drift_status} (p={drift_info.get("drift_score", 0):.4f})', 
                           color=color, fontweight='bold')
                ax.set_xlabel('Valeur')
                ax.set_ylabel('Densité')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            # Masquer les axes vides
            for i in range(n_features, n_rows * n_cols):
                row = i // n_cols
                col = i % n_cols
                axes[row, col].set_visible(False)
            
            plt.tight_layout()
            
            # Sauvegarder la figure avec un chemin corrigé
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            viz_path = os.path.join(self.reports_dir, f"drift_visualizations_{timestamp}.png")
            plt.savefig(viz_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Visualisations sauvegardées: {viz_path}")
            return viz_path
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération des visualisations: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def generate_html_report(self, drift_results, viz_path, X_ref, X_new):
        """Génère un rapport HTML complet."""
        try:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Calculer les statistiques globales
            total_features = len(drift_results)
            drift_detected_count = sum(1 for r in drift_results.values() if r.get('drift_detected', False))
            drift_percentage = (drift_detected_count / total_features) * 100 if total_features > 0 else 0
            
            # Déterminer le statut global
            if drift_percentage > 50:
                global_status = "ALARMANT"
                status_color = "#ff4444"
            elif drift_percentage > 20:
                global_status = "ATTENTION"
                status_color = "#ff8800"
            else:
                global_status = "OK"
                status_color = "#44aa44"
            
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Rapport de Surveillance du Data Drift</title>
                <meta charset="UTF-8">
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                    .status {{ font-size: 24px; font-weight: bold; color: {status_color}; }}
                    .summary {{ margin: 20px 0; padding: 15px; background-color: #f9f9f9; border-radius: 5px; }}
                    .feature-table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                    .feature-table th, .feature-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    .feature-table th {{ background-color: #f2f2f2; }}
                    .drift-detected {{ background-color: #ffeeee; }}
                    .drift-ok {{ background-color: #eeffee; }}
                    .visualization {{ text-align: center; margin: 20px 0; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>Rapport de Surveillance du Data Drift</h1>
                    <p><strong>Généré le:</strong> {timestamp}</p>
                    <p class="status">Statut Global: {global_status}</p>
                </div>
                
                <div class="summary">
                    <h2>Résumé</h2>
                    <ul>
                        <li><strong>Nombre total de caractéristiques:</strong> {total_features}</li>
                        <li><strong>Caractéristiques avec drift détecté:</strong> {drift_detected_count}</li>
                        <li><strong>Pourcentage de drift:</strong> {drift_percentage:.1f}%</li>
                        <li><strong>Échantillons de référence:</strong> {len(X_ref)}</li>
                        <li><strong>Nouveaux échantillons:</strong> {len(X_new)}</li>
                        <li><strong>Seuil de détection:</strong> {self.drift_threshold}</li>
                    </ul>
                </div>
                
                <h2>Détails par Caractéristique</h2>
                <table class="feature-table">
                    <tr>
                        <th>Caractéristique</th>
                        <th>Drift Détecté</th>
                        <th>Score de Drift</th>
                        <th>Moyenne Réf.</th>
                        <th>Moyenne Nouvelle</th>
                        <th>Décalage Moyen</th>
                        <th>KS p-value</th>
                        <th>MW p-value</th>
                    </tr>
            """
            
            for feature, results in drift_results.items():
                row_class = "drift-detected" if results.get('drift_detected', False) else "drift-ok"
                drift_status = "OUI" if results.get('drift_detected', False) else "NON"
                
                html_content += f"""
                    <tr class="{row_class}">
                        <td>{feature}</td>
                        <td>{drift_status}</td>
                        <td>{results.get('drift_score', 0):.4f}</td>
                        <td>{results.get('ref_mean', 0):.4f}</td>
                        <td>{results.get('new_mean', 0):.4f}</td>
                        <td>{results.get('mean_shift', 0):.4f}</td>
                        <td>{results.get('ks_p_value', 0):.4f}</td>
                        <td>{results.get('mw_p_value', 0):.4f}</td>
                    </tr>
                """
            
            html_content += """
                </table>
                
                <div class="visualization">
                    <h2>Visualisations</h2>
            """
            
            if viz_path and os.path.exists(viz_path):
                # Encoder l'image en base64 pour l'intégrer dans le HTML
                import base64
                with open(viz_path, 'rb') as img_file:
                    img_data = base64.b64encode(img_file.read()).decode()
                html_content += f'<img src="data:image/png;base64,{img_data}" style="max-width: 100%; height: auto;">'
            else:
                html_content += '<p>Visualisations non disponibles</p>'
            
            html_content += """
                </div>
                
                <div class="summary">
                    <h2>Recommandations</h2>
            """
            
            if global_status == "ALARMANT":
                html_content += """
                    <p style="color: #ff4444;"><strong>Action Immédiate Requise:</strong></p>
                    <ul>
                        <li>Plus de 50% des caractéristiques montrent un drift significatif</li>
                        <li>Considérer un réentraînement du modèle</li>
                        <li>Vérifier la qualité des nouvelles données</li>
                        <li>Analyser les causes du changement de distribution</li>
                    </ul>
                """
            elif global_status == "ATTENTION":
                html_content += """
                    <p style="color: #ff8800;"><strong>Surveillance Renforcée:</strong></p>
                    <ul>
                        <li>20-50% des caractéristiques montrent un drift</li>
                        <li>Surveiller de près l'évolution</li>
                        <li>Préparer un plan de réentraînement</li>
                        <li>Analyser les caractéristiques affectées</li>
                    </ul>
                """
            else:
                html_content += """
                    <p style="color: #44aa44;"><strong>Situation Normale:</strong></p>
                    <ul>
                        <li>Moins de 20% des caractéristiques montrent un drift</li>
                        <li>Continuer la surveillance régulière</li>
                        <li>Le modèle peut continuer à être utilisé</li>
                    </ul>
                """
            
            html_content += """
                </div>
            </body>
            </html>
            """
            
            # Sauvegarder le rapport HTML
            html_path = os.path.join(self.reports_dir, f"drift_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"Rapport HTML généré: {html_path}")
            return html_path
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération du rapport HTML: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def generate_json_report(self, drift_results, global_status, X_ref, X_new):
        """Génère un rapport JSON pour l'intégration avec d'autres systèmes."""
        try:
            timestamp = datetime.now().isoformat()
            
            # Calculer les statistiques globales
            total_features = len(drift_results)
            drift_detected_count = sum(1 for r in drift_results.values() if r.get('drift_detected', False))
            drift_percentage = (drift_detected_count / total_features) * 100 if total_features > 0 else 0
            
            json_report = {
                'timestamp': timestamp,
                'global_status': global_status,
                'summary': {
                    'total_features': total_features,
                    'drift_detected_count': drift_detected_count,
                    'drift_percentage': drift_percentage,
                    'reference_samples': len(X_ref),
                    'new_samples': len(X_new),
                    'drift_threshold': self.drift_threshold
                },
                'feature_results': drift_results,
                'metadata': {
                    'model_config': self.config['model'],
                    'data_config': self.config['data']
                }
            }
            
            # Sauvegarder le rapport JSON
            json_path = os.path.join(self.reports_dir, f"drift_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            with open(json_path, 'w') as f:
                json.dump(json_report, f, indent=2)
            
            # Sauvegarder aussi le dernier rapport pour Streamlit
            latest_path = os.path.join(self.reports_dir, "latest_drift_report.json")
            with open(latest_path, 'w') as f:
                json.dump(json_report, f, indent=2)
            
            logger.info(f"Rapport JSON généré: {json_path}")
            return json_path
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération du rapport JSON: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def monitor_drift(self, new_data_path=None):
        """Fonction principale pour surveiller le drift."""
        try:
            logger.info("Début de la surveillance du data drift")
            
            # Charger les données de référence
            X_ref, y_ref = self.load_reference_data()
            if X_ref is None:
                logger.error("Impossible de charger les données de référence")
                return False
            
            # Charger les nouvelles données
            X_new, y_new = self.load_new_data(new_data_path)
            if X_new is None:
                logger.error("Impossible de charger les nouvelles données")
                return False
            
            # Calculer les métriques de drift
            drift_results = self.calculate_drift_metrics(X_ref, X_new)
            if drift_results is None:
                logger.error("Impossible de calculer les métriques de drift")
                return False
            
            # Déterminer le statut global
            total_features = len(drift_results)
            drift_detected_count = sum(1 for r in drift_results.values() if r.get('drift_detected', False))
            drift_percentage = (drift_detected_count / total_features) * 100 if total_features > 0 else 0
            
            if drift_percentage > 50:
                global_status = "ALARMANT"
            elif drift_percentage > 20:
                global_status = "ATTENTION"
            else:
                global_status = "OK"
            
            logger.info(f"Statut global du drift: {global_status} ({drift_percentage:.1f}% des caractéristiques affectées)")
            
            # Générer les visualisations
            viz_path = self.generate_drift_visualizations(X_ref, X_new, drift_results)
            
            # Générer les rapports
            html_path = self.generate_html_report(drift_results, viz_path, X_ref, X_new)
            json_path = self.generate_json_report(drift_results, global_status, X_ref, X_new)
            
            logger.info(f"Rapports générés:")
            if html_path:
                logger.info(f"  - Rapport HTML: {html_path}")
            if json_path:
                logger.info(f"  - Rapport JSON: {json_path}")
            if viz_path:
                logger.info(f"  - Visualisations: {viz_path}")
            
            # Retourner le statut (True si OK, False si alarmant)
            return global_status in ["OK", "ATTENTION"]
            
        except Exception as e:
            logger.error(f"Erreur lors de la surveillance du drift: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Fonction principale."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Surveillance du Data Drift')
    parser.add_argument('--new-data', type=str, help='Chemin vers les nouvelles données')
    parser.add_argument('--config', type=str, default='models/model_config.yml', help='Chemin vers la configuration')
    
    args = parser.parse_args()
    
    # Créer le moniteur
    monitor = DataDriftMonitor(args.config)
    
    # Exécuter la surveillance
    success = monitor.monitor_drift(args.new_data)
    
    if success:
        logger.info("Surveillance du drift terminée avec succès")
        sys.exit(0)
    else:
        logger.error("Surveillance du drift échouée ou drift alarmant détecté")
        sys.exit(1)

if __name__ == "__main__":
    main()