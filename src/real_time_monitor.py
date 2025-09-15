#!/usr/bin/env python3
"""
Real-time monitoring system for ML models in production.
Monitors model performance, data drift, and system health.
"""

import os
import json
import time
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
import sqlite3
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
from pathlib import Path

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Déterminer le répertoire de base
BASE_DIR = Path(__file__).resolve().parent.parent

@dataclass
class MetricThreshold:
    """Seuils pour les métriques de monitoring."""
    metric_name: str
    warning_threshold: float
    critical_threshold: float
    comparison_type: str  # 'greater', 'less', 'equal'

@dataclass
class Alert:
    """Structure d'une alerte."""
    id: str
    timestamp: datetime
    level: str  # 'INFO', 'WARNING', 'CRITICAL'
    component: str
    message: str
    metric_name: Optional[str] = None
    metric_value: Optional[float] = None
    threshold: Optional[float] = None
    resolved: bool = False
    resolved_at: Optional[datetime] = None

class MetricsCollector:
    """Collecteur de métriques en temps réel."""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or str(BASE_DIR / "monitoring" / "metrics.db")
        self.ensure_db_exists()
        
    def ensure_db_exists(self):
        """Crée la base de données si elle n'existe pas."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    component TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    metadata TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id TEXT PRIMARY KEY,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    level TEXT NOT NULL,
                    component TEXT NOT NULL,
                    message TEXT NOT NULL,
                    metric_name TEXT,
                    metric_value REAL,
                    threshold REAL,
                    resolved BOOLEAN DEFAULT FALSE,
                    resolved_at DATETIME
                )
            """)
    
    def record_metric(self, component: str, metric_name: str, value: float, metadata: Dict = None):
        """Enregistre une métrique."""
        metadata_json = json.dumps(metadata) if metadata else None
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO metrics (component, metric_name, metric_value, metadata) VALUES (?, ?, ?, ?)",
                (component, metric_name, value, metadata_json)
            )
        
        logger.debug(f"Recorded metric: {component}.{metric_name} = {value}")
    
    def get_recent_metrics(self, component: str, metric_name: str, hours: int = 24) -> List[Dict]:
        """Récupère les métriques récentes."""
        since = datetime.now() - timedelta(hours=hours)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM metrics WHERE component = ? AND metric_name = ? AND timestamp > ? ORDER BY timestamp DESC",
                (component, metric_name, since.isoformat())
            )
            return [dict(row) for row in cursor.fetchall()]

class AlertManager:
    """Gestionnaire d'alertes."""
    
    def __init__(self, db_path: str = None, config_path: str = None):
        self.db_path = db_path or str(BASE_DIR / "monitoring" / "metrics.db")
        self.config_path = config_path or str(BASE_DIR / "monitoring" / "alert_config.json")
        self.thresholds = self.load_thresholds()
        self.notification_channels = self.load_notification_config()
        
    def load_thresholds(self) -> Dict[str, MetricThreshold]:
        """Charge les seuils de métriques."""
        default_thresholds = {
            "api_response_time": MetricThreshold("api_response_time", 1.0, 3.0, "greater"),
            "api_error_rate": MetricThreshold("api_error_rate", 0.05, 0.1, "greater"),
            "model_accuracy": MetricThreshold("model_accuracy", 0.85, 0.8, "less"),
            "drift_score": MetricThreshold("drift_score", 0.05, 0.1, "greater"),
            "memory_usage": MetricThreshold("memory_usage", 0.8, 0.9, "greater"),
            "cpu_usage": MetricThreshold("cpu_usage", 0.8, 0.9, "greater")
        }
        
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    for name, threshold_config in config.get('thresholds', {}).items():
                        default_thresholds[name] = MetricThreshold(**threshold_config)
            except Exception as e:
                logger.warning(f"Could not load threshold config: {e}")
        
        return default_thresholds
    
    def load_notification_config(self) -> Dict:
        """Charge la configuration des notifications."""
        default_config = {
            "email": {
                "enabled": False,
                "smtp_server": "smtp.gmail.com",
                "smtp_port": 587,
                "username": "",
                "password": "",
                "recipients": []
            },
            "webhook": {
                "enabled": False,
                "url": "",
                "headers": {}
            }
        }
        
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    return config.get('notifications', default_config)
            except Exception as e:
                logger.warning(f"Could not load notification config: {e}")
        
        return default_config
    
    def check_thresholds(self, component: str, metric_name: str, value: float) -> Optional[Alert]:
        """Vérifie si une métrique dépasse les seuils."""
        threshold_key = f"{component}_{metric_name}" if f"{component}_{metric_name}" in self.thresholds else metric_name
        
        if threshold_key not in self.thresholds:
            return None
        
        threshold = self.thresholds[threshold_key]
        alert_level = None
        threshold_value = None
        
        if threshold.comparison_type == "greater":
            if value > threshold.critical_threshold:
                alert_level = "CRITICAL"
                threshold_value = threshold.critical_threshold
            elif value > threshold.warning_threshold:
                alert_level = "WARNING"
                threshold_value = threshold.warning_threshold
        elif threshold.comparison_type == "less":
            if value < threshold.critical_threshold:
                alert_level = "CRITICAL"
                threshold_value = threshold.critical_threshold
            elif value < threshold.warning_threshold:
                alert_level = "WARNING"
                threshold_value = threshold.warning_threshold
        
        if alert_level:
            alert_id = f"{component}_{metric_name}_{int(time.time())}"
            message = f"{component} {metric_name} is {value:.4f}, threshold: {threshold_value:.4f}"
            
            return Alert(
                id=alert_id,
                timestamp=datetime.now(),
                level=alert_level,
                component=component,
                message=message,
                metric_name=metric_name,
                metric_value=value,
                threshold=threshold_value
            )
        
        return None
    
    def create_alert(self, alert: Alert):
        """Crée une nouvelle alerte."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO alerts (id, timestamp, level, component, message, metric_name, metric_value, threshold)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (alert.id, alert.timestamp.isoformat(), alert.level, alert.component,
                 alert.message, alert.metric_name, alert.metric_value, alert.threshold)
            )
        
        logger.warning(f"Alert created: {alert.level} - {alert.message}")
        self.send_notification(alert)
    
    def send_notification(self, alert: Alert):
        """Envoie une notification pour l'alerte."""
        if self.notification_channels["email"]["enabled"]:
            self.send_email_notification(alert)
        
        if self.notification_channels["webhook"]["enabled"]:
            self.send_webhook_notification(alert)
    
    def send_email_notification(self, alert: Alert):
        """Envoie une notification par email."""
        try:
            config = self.notification_channels["email"]
            
            msg = MIMEMultipart()
            msg['From'] = config["username"]
            msg['To'] = ", ".join(config["recipients"])
            msg['Subject'] = f"ML Pipeline Alert: {alert.level} - {alert.component}"
            
            body = f"""
            Alert Details:
            - Level: {alert.level}
            - Component: {alert.component}
            - Message: {alert.message}
            - Timestamp: {alert.timestamp}
            - Metric: {alert.metric_name} = {alert.metric_value}
            - Threshold: {alert.threshold}
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(config["smtp_server"], config["smtp_port"])
            server.starttls()
            server.login(config["username"], config["password"])
            text = msg.as_string()
            server.sendmail(config["username"], config["recipients"], text)
            server.quit()
            
            logger.info(f"Email notification sent for alert {alert.id}")
        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")
    
    def send_webhook_notification(self, alert: Alert):
        """Envoie une notification via webhook."""
        try:
            config = self.notification_channels["webhook"]
            
            payload = {
                "alert": asdict(alert),
                "timestamp": alert.timestamp.isoformat()
            }
            
            response = requests.post(
                config["url"],
                json=payload,
                headers=config.get("headers", {}),
                timeout=10
            )
            response.raise_for_status()
            
            logger.info(f"Webhook notification sent for alert {alert.id}")
        except Exception as e:
            logger.error(f"Failed to send webhook notification: {e}")

class ModelPerformanceMonitor:
    """Moniteur de performance des modèles."""
    
    def __init__(self, metrics_collector: MetricsCollector, alert_manager: AlertManager):
        self.metrics_collector = metrics_collector
        self.alert_manager = alert_manager
        
    def monitor_api_performance(self, api_url: str = "http://localhost:8000"):
        """Surveille les performances de l'API."""
        try:
            start_time = time.time()
            response = requests.get(f"{api_url}/health", timeout=5)
            response_time = time.time() - start_time
            
            # Enregistrer la métrique
            self.metrics_collector.record_metric("api", "response_time", response_time)
            
            # Vérifier les seuils
            alert = self.alert_manager.check_thresholds("api", "response_time", response_time)
            if alert:
                self.alert_manager.create_alert(alert)
            
            # Surveiller le taux d'erreur
            error_rate = 0 if response.status_code == 200 else 1
            self.metrics_collector.record_metric("api", "error_rate", error_rate)
            
            alert = self.alert_manager.check_thresholds("api", "error_rate", error_rate)
            if alert:
                self.alert_manager.create_alert(alert)
                
        except Exception as e:
            logger.error(f"API monitoring failed: {e}")
            self.metrics_collector.record_metric("api", "error_rate", 1)
            
            alert = self.alert_manager.check_thresholds("api", "error_rate", 1)
            if alert:
                self.alert_manager.create_alert(alert)
    
    def monitor_model_accuracy(self, model_path: str = "models"):
        """Surveille la précision du modèle."""
        try:
            evaluation_path = os.path.join(model_path, "evaluation_report.json")
            if os.path.exists(evaluation_path):
                with open(evaluation_path, 'r') as f:
                    evaluation = json.load(f)
                
                metrics = evaluation.get('metrics', {})
                for metric_name, value in metrics.items():
                    self.metrics_collector.record_metric("model", metric_name, value)
                    
                    alert = self.alert_manager.check_thresholds("model", metric_name, value)
                    if alert:
                        self.alert_manager.create_alert(alert)
                        
        except Exception as e:
            logger.error(f"Model accuracy monitoring failed: {e}")
    
    def monitor_system_resources(self):
        """Surveille les ressources système."""
        try:
            try:
                import psutil
            except ImportError:
                logger.warning("psutil not installed, system monitoring disabled")
                # Enregistrer une métrique indiquant que le monitoring système est désactivé
                self.metrics_collector.record_metric(
                    "system", "monitoring_status", 0,
                    {"message": "psutil not installed, system monitoring disabled"}
                )
                return
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1) / 100.0
            self.metrics_collector.record_metric("system", "cpu_usage", cpu_percent)
            
            alert = self.alert_manager.check_thresholds("system", "cpu_usage", cpu_percent)
            if alert:
                self.alert_manager.create_alert(alert)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent / 100.0
            self.metrics_collector.record_metric("system", "memory_usage", memory_percent)
            
            alert = self.alert_manager.check_thresholds("system", "memory_usage", memory_percent)
            if alert:
                self.alert_manager.create_alert(alert)
                
        except Exception as e:
            logger.error(f"System monitoring failed: {e}")
            self.metrics_collector.record_metric(
                "system", "error", 1,
                {"error": str(e), "component": "system_monitoring"}
            )

class RealTimeMonitor:
    """Moniteur principal en temps réel."""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path or str(BASE_DIR / "monitoring" / "monitor_config.json")
        self.create_default_configs()
        self.config = self.load_config(self.config_path)
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager()
        self.performance_monitor = ModelPerformanceMonitor(
            self.metrics_collector, self.alert_manager
        )
        self.running = False
        self.monitor_thread = None
        
    def create_default_configs(self):
        """Crée les fichiers de configuration par défaut s'ils n'existent pas."""
        config_dir = os.path.dirname(self.config_path)
        os.makedirs(config_dir, exist_ok=True)
        
        # Configuration du moniteur
        monitor_config = {
            "monitoring_interval": 60,
            "api_url": "http://localhost:8000",
            "model_path": str(BASE_DIR / "models"),
            "enabled_monitors": ["api", "model", "system"]
        }
        
        if not os.path.exists(self.config_path):
            with open(self.config_path, 'w') as f:
                json.dump(monitor_config, f, indent=2)
            logger.info(f"Created default monitor config: {self.config_path}")
        
        # Configuration des alertes
        alert_config_path = str(BASE_DIR / "monitoring" / "alert_config.json")
        alert_config = {
            "thresholds": {
                "api_response_time": {
                    "metric_name": "api_response_time",
                    "warning_threshold": 1.0,
                    "critical_threshold": 3.0,
                    "comparison_type": "greater"
                },
                "api_error_rate": {
                    "metric_name": "api_error_rate",
                    "warning_threshold": 0.05,
                    "critical_threshold": 0.1,
                    "comparison_type": "greater"
                },
                "model_accuracy": {
                    "metric_name": "model_accuracy",
                    "warning_threshold": 0.85,
                    "critical_threshold": 0.8,
                    "comparison_type": "less"
                }
            },
            "notifications": {
                "email": {
                    "enabled": False,
                    "smtp_server": "smtp.gmail.com",
                    "smtp_port": 587,
                    "username": "",
                    "password": "",
                    "recipients": []
                },
                "webhook": {
                    "enabled": False,
                    "url": "",
                    "headers": {}
                }
            }
        }
        
        if not os.path.exists(alert_config_path):
            with open(alert_config_path, 'w') as f:
                json.dump(alert_config, f, indent=2)
            logger.info(f"Created default alert config: {alert_config_path}")
        
    def load_config(self, config_path: str) -> Dict:
        """Charge la configuration du moniteur."""
        default_config = {
            "monitoring_interval": 60,  # secondes
            "api_url": "http://localhost:8000",
            "model_path": str(BASE_DIR / "models"),
            "enabled_monitors": ["api", "model", "system"]
        }
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    return {**default_config, **config}
            except Exception as e:
                logger.warning(f"Could not load monitor config: {e}")
        
        return default_config
    
    def start_monitoring(self):
        """Démarre le monitoring en arrière-plan."""
        if self.running:
            logger.warning("Monitoring already running")
            return
        
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Real-time monitoring started")
    
    def stop_monitoring(self):
        """Arrête le monitoring."""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Real-time monitoring stopped")
    
    def _monitoring_loop(self):
        """Boucle principale de monitoring."""
        while self.running:
            try:
                if "api" in self.config["enabled_monitors"]:
                    self.performance_monitor.monitor_api_performance(self.config["api_url"])
                
                if "model" in self.config["enabled_monitors"]:
                    self.performance_monitor.monitor_model_accuracy(self.config["model_path"])
                
                if "system" in self.config["enabled_monitors"]:
                    self.performance_monitor.monitor_system_resources()
                
                time.sleep(self.config["monitoring_interval"])
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(10)  # Wait before retrying

def main():
    """Fonction principale pour lancer le monitoring."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Real-time ML Pipeline Monitor')
    parser.add_argument('--config', type=str, default='../monitoring/monitor_config.json',
                       help='Path to monitor configuration file')
    parser.add_argument('--daemon', action='store_true',
                       help='Run as daemon')
    
    args = parser.parse_args()
    
    monitor = RealTimeMonitor(args.config)
    
    try:
        monitor.start_monitoring()
        
        if args.daemon:
            # Run as daemon
            while True:
                time.sleep(60)
        else:
            # Interactive mode
            print("Monitoring started. Press Ctrl+C to stop.")
            while True:
                time.sleep(1)
                
    except KeyboardInterrupt:
        logger.info("Stopping monitoring...")
        monitor.stop_monitoring()

if __name__ == "__main__":
    main()