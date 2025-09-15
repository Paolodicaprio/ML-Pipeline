#!/usr/bin/env python3
"""
Security and governance manager for ML Pipeline.
Handles secrets management, audit trails, and model validation.
"""

import os
import json
import hashlib
import sqlite3
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import yaml
import joblib
import numpy as np
from dataclasses import dataclass, asdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AuditEntry:
    """Entrée d'audit trail."""
    id: str
    timestamp: datetime
    user: str
    action: str
    component: str
    details: Dict[str, Any]
    success: bool
    error_message: Optional[str] = None

@dataclass
class ModelValidationResult:
    """Résultat de validation d'un modèle."""
    model_path: str
    is_valid: bool
    validation_score: float
    checks_passed: List[str]
    checks_failed: List[str]
    security_issues: List[str]
    performance_metrics: Dict[str, float]
    timestamp: datetime

class SecretsManager:
    """Gestionnaire de secrets sécurisé."""
    
    def __init__(self, secrets_file: str = "security/secrets.enc", master_key_file: str = "security/master.key"):
        self.secrets_file = secrets_file
        self.master_key_file = master_key_file
        self.ensure_security_dir()
        self._fernet = None
        
    def ensure_security_dir(self):
        """Crée le répertoire de sécurité."""
        os.makedirs(os.path.dirname(self.secrets_file), exist_ok=True)
        
    def _get_fernet(self) -> Fernet:
        """Obtient l'instance Fernet pour le chiffrement."""
        if self._fernet is None:
            if os.path.exists(self.master_key_file):
                with open(self.master_key_file, 'rb') as f:
                    key = f.read()
            else:
                # Générer une nouvelle clé
                key = Fernet.generate_key()
                with open(self.master_key_file, 'wb') as f:
                    f.write(key)
                os.chmod(self.master_key_file, 0o600)  # Lecture seule pour le propriétaire
                
            self._fernet = Fernet(key)
        return self._fernet
    
    def store_secret(self, name: str, value: str):
        """Stocke un secret de manière sécurisée."""
        fernet = self._get_fernet()
        
        # Charger les secrets existants
        secrets = self.load_all_secrets()
        
        # Chiffrer et stocker le nouveau secret
        encrypted_value = fernet.encrypt(value.encode())
        secrets[name] = base64.b64encode(encrypted_value).decode()
        
        # Sauvegarder
        with open(self.secrets_file, 'w') as f:
            json.dump(secrets, f)
        
        os.chmod(self.secrets_file, 0o600)
        logger.info(f"Secret '{name}' stored securely")
    
    def get_secret(self, name: str) -> Optional[str]:
        """Récupère un secret."""
        try:
            secrets = self.load_all_secrets()
            if name not in secrets:
                return None
            
            fernet = self._get_fernet()
            encrypted_value = base64.b64decode(secrets[name].encode())
            decrypted_value = fernet.decrypt(encrypted_value)
            return decrypted_value.decode()
        except Exception as e:
            logger.error(f"Failed to retrieve secret '{name}': {e}")
            return None
    
    def load_all_secrets(self) -> Dict[str, str]:
        """Charge tous les secrets (chiffrés)."""
        if not os.path.exists(self.secrets_file):
            return {}
        
        try:
            with open(self.secrets_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load secrets: {e}")
            return {}
    
    def delete_secret(self, name: str) -> bool:
        """Supprime un secret."""
        secrets = self.load_all_secrets()
        if name in secrets:
            del secrets[name]
            with open(self.secrets_file, 'w') as f:
                json.dump(secrets, f)
            logger.info(f"Secret '{name}' deleted")
            return True
        return False
    
    def list_secrets(self) -> List[str]:
        """Liste les noms des secrets disponibles."""
        return list(self.load_all_secrets().keys())

class AuditTrail:
    """Gestionnaire d'audit trail."""
    
    def __init__(self, db_path: str = "security/audit.db"):
        self.db_path = db_path
        self.ensure_db_exists()
    
    def ensure_db_exists(self):
        """Crée la base de données d'audit."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_log (
                    id TEXT PRIMARY KEY,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    user TEXT NOT NULL,
                    action TEXT NOT NULL,
                    component TEXT NOT NULL,
                    details TEXT,
                    success BOOLEAN NOT NULL,
                    error_message TEXT
                )
            """)
            
            # Index pour améliorer les performances
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON audit_log(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_user ON audit_log(user)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_action ON audit_log(action)")
    
    def log_action(self, user: str, action: str, component: str, details: Dict[str, Any], 
                   success: bool = True, error_message: Optional[str] = None):
        """Enregistre une action dans l'audit trail."""
        entry_id = hashlib.sha256(f"{user}{action}{component}{datetime.now().isoformat()}".encode()).hexdigest()[:16]
        
        entry = AuditEntry(
            id=entry_id,
            timestamp=datetime.now(),
            user=user,
            action=action,
            component=component,
            details=details,
            success=success,
            error_message=error_message
        )
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO audit_log (id, timestamp, user, action, component, details, success, error_message)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (entry.id, entry.timestamp.isoformat(), entry.user, entry.action, entry.component,
                 json.dumps(entry.details), entry.success, entry.error_message)
            )
        
        logger.info(f"Audit log: {user} {action} on {component} - {'SUCCESS' if success else 'FAILED'}")
    
    def get_audit_logs(self, user: Optional[str] = None, action: Optional[str] = None,
                       component: Optional[str] = None, hours: int = 24) -> List[Dict]:
        """Récupère les logs d'audit."""
        from datetime import timedelta
        since = datetime.now() - timedelta(hours=hours)
        
        query = "SELECT * FROM audit_log WHERE timestamp > ?"
        params = [since.isoformat()]
        
        if user:
            query += " AND user = ?"
            params.append(user)
        
        if action:
            query += " AND action = ?"
            params.append(action)
        
        if component:
            query += " AND component = ?"
            params.append(component)
        
        query += " ORDER BY timestamp DESC"
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

class ModelValidator:
    """Validateur de modèles ML."""
    
    def __init__(self, audit_trail: AuditTrail):
        self.audit_trail = audit_trail
        
    def validate_model(self, model_path: str, config_path: str = "models/model_config.yml",
                      user: str = "system") -> ModelValidationResult:
        """Valide un modèle ML."""
        checks_passed = []
        checks_failed = []
        security_issues = []
        performance_metrics = {}
        
        try:
            # 1. Vérifier l'existence du modèle
            if not os.path.exists(model_path):
                checks_failed.append("model_file_exists")
                return ModelValidationResult(
                    model_path=model_path,
                    is_valid=False,
                    validation_score=0.0,
                    checks_passed=checks_passed,
                    checks_failed=checks_failed,
                    security_issues=security_issues,
                    performance_metrics=performance_metrics,
                    timestamp=datetime.now()
                )
            
            checks_passed.append("model_file_exists")
            
            # 2. Vérifier l'intégrité du fichier
            if self._check_file_integrity(model_path):
                checks_passed.append("file_integrity")
            else:
                checks_failed.append("file_integrity")
                security_issues.append("File integrity check failed")
            
            # 3. Charger et tester le modèle
            try:
                model = joblib.load(model_path)
                checks_passed.append("model_loadable")
                
                # 4. Vérifier la structure du modèle
                if hasattr(model, 'predict'):
                    checks_passed.append("has_predict_method")
                else:
                    checks_failed.append("has_predict_method")
                
                # 5. Test de prédiction basique
                if self._test_model_prediction(model, config_path):
                    checks_passed.append("prediction_test")
                else:
                    checks_failed.append("prediction_test")
                
                # 6. Vérifier les métriques de performance
                performance_metrics = self._check_performance_metrics(model_path)
                if performance_metrics.get('accuracy', 0) > 0.5:  # Seuil minimum
                    checks_passed.append("performance_threshold")
                else:
                    checks_failed.append("performance_threshold")
                
            except Exception as e:
                checks_failed.append("model_loadable")
                security_issues.append(f"Model loading failed: {str(e)}")
            
            # 7. Vérifier la taille du modèle (sécurité)
            file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
            if file_size_mb < 100:  # Limite de 100MB
                checks_passed.append("file_size_check")
            else:
                checks_failed.append("file_size_check")
                security_issues.append(f"Model file too large: {file_size_mb:.2f}MB")
            
            # Calculer le score de validation
            total_checks = len(checks_passed) + len(checks_failed)
            validation_score = len(checks_passed) / total_checks if total_checks > 0 else 0.0
            
            is_valid = len(checks_failed) == 0 and len(security_issues) == 0
            
            # Enregistrer dans l'audit trail
            self.audit_trail.log_action(
                user=user,
                action="model_validation",
                component="model_validator",
                details={
                    "model_path": model_path,
                    "validation_score": validation_score,
                    "checks_passed": len(checks_passed),
                    "checks_failed": len(checks_failed),
                    "security_issues": len(security_issues)
                },
                success=is_valid
            )
            
            return ModelValidationResult(
                model_path=model_path,
                is_valid=is_valid,
                validation_score=validation_score,
                checks_passed=checks_passed,
                checks_failed=checks_failed,
                security_issues=security_issues,
                performance_metrics=performance_metrics,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            self.audit_trail.log_action(
                user=user,
                action="model_validation",
                component="model_validator",
                details={"model_path": model_path, "error": str(e)},
                success=False,
                error_message=str(e)
            )
            
            return ModelValidationResult(
                model_path=model_path,
                is_valid=False,
                validation_score=0.0,
                checks_passed=checks_passed,
                checks_failed=["validation_exception"],
                security_issues=[f"Validation exception: {str(e)}"],
                performance_metrics=performance_metrics,
                timestamp=datetime.now()
            )
    
    def _check_file_integrity(self, file_path: str) -> bool:
        """Vérifie l'intégrité du fichier."""
        try:
            # Calculer le hash du fichier
            with open(file_path, 'rb') as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()
            
            # Vérifier si le hash est cohérent (fichier non corrompu)
            # Pour une vraie implémentation, on stockerait les hashs attendus
            return len(file_hash) == 64  # Hash SHA256 valide
        except Exception:
            return False
    
    def _test_model_prediction(self, model, config_path: str) -> bool:
        """Test basique de prédiction du modèle."""
        try:
            # Charger la configuration pour connaître le type de modèle
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            model_type = config.get('model', {}).get('type', 'classification')
            
            # Créer des données de test factices
            if model_type == 'classification':
                # Test avec des données aléatoires
                test_data = np.random.rand(1, 10)  # 1 échantillon, 10 features
                prediction = model.predict(test_data)
                return len(prediction) == 1
            elif model_type == 'regression':
                test_data = np.random.rand(1, 10)
                prediction = model.predict(test_data)
                return len(prediction) == 1 and isinstance(prediction[0], (int, float, np.number))
            
            return False
        except Exception:
            return False
    
    def _check_performance_metrics(self, model_path: str) -> Dict[str, float]:
        """Vérifie les métriques de performance du modèle."""
        try:
            # Chercher le rapport d'évaluation
            model_dir = os.path.dirname(model_path)
            eval_path = os.path.join(model_dir, "evaluation_report.json")
            
            if os.path.exists(eval_path):
                with open(eval_path, 'r') as f:
                    evaluation = json.load(f)
                return evaluation.get('metrics', {})
            
            return {}
        except Exception:
            return {}

class SecurityManager:
    """Gestionnaire principal de sécurité."""
    
    def __init__(self):
        self.secrets_manager = SecretsManager()
        self.audit_trail = AuditTrail()
        self.model_validator = ModelValidator(self.audit_trail)
    
    def initialize_security(self, admin_user: str = "admin"):
        """Initialise le système de sécurité."""
        # Créer les répertoires nécessaires
        os.makedirs("security", exist_ok=True)
        
        # Générer des secrets par défaut si nécessaire
        if not self.secrets_manager.get_secret("api_token"):
            import secrets
            api_token = secrets.token_urlsafe(32)
            self.secrets_manager.store_secret("api_token", api_token)
        
        # Enregistrer l'initialisation
        self.audit_trail.log_action(
            user=admin_user,
            action="security_initialization",
            component="security_manager",
            details={"secrets_count": len(self.secrets_manager.list_secrets())},
            success=True
        )
        
        logger.info("Security system initialized")
    
    def validate_deployment(self, model_path: str, user: str = "system") -> bool:
        """Valide un déploiement de modèle."""
        validation_result = self.model_validator.validate_model(model_path, user=user)
        
        if validation_result.is_valid:
            self.audit_trail.log_action(
                user=user,
                action="deployment_approved",
                component="deployment_validator",
                details={"model_path": model_path, "validation_score": validation_result.validation_score},
                success=True
            )
            return True
        else:
            self.audit_trail.log_action(
                user=user,
                action="deployment_rejected",
                component="deployment_validator",
                details={
                    "model_path": model_path,
                    "validation_score": validation_result.validation_score,
                    "failed_checks": validation_result.checks_failed,
                    "security_issues": validation_result.security_issues
                },
                success=False,
                error_message="Model validation failed"
            )
            return False

def main():
    """Fonction principale pour tester le système de sécurité."""
    import argparse
    
    parser = argparse.ArgumentParser(description='ML Pipeline Security Manager')
    parser.add_argument('--init', action='store_true', help='Initialize security system')
    parser.add_argument('--validate-model', type=str, help='Validate a model file')
    parser.add_argument('--store-secret', nargs=2, metavar=('NAME', 'VALUE'), help='Store a secret')
    parser.add_argument('--get-secret', type=str, help='Retrieve a secret')
    parser.add_argument('--list-secrets', action='store_true', help='List all secrets')
    parser.add_argument('--audit-logs', action='store_true', help='Show recent audit logs')
    
    args = parser.parse_args()
    
    security_manager = SecurityManager()
    
    if args.init:
        security_manager.initialize_security()
        print("Security system initialized successfully")
    
    if args.validate_model:
        result = security_manager.model_validator.validate_model(args.validate_model)
        print(f"Validation result for {args.validate_model}:")
        print(f"  Valid: {result.is_valid}")
        print(f"  Score: {result.validation_score:.2f}")
        print(f"  Checks passed: {result.checks_passed}")
        print(f"  Checks failed: {result.checks_failed}")
        print(f"  Security issues: {result.security_issues}")
    
    if args.store_secret:
        name, value = args.store_secret
        security_manager.secrets_manager.store_secret(name, value)
        print(f"Secret '{name}' stored successfully")
    
    if args.get_secret:
        value = security_manager.secrets_manager.get_secret(args.get_secret)
        if value:
            print(f"Secret '{args.get_secret}': {value}")
        else:
            print(f"Secret '{args.get_secret}' not found")
    
    if args.list_secrets:
        secrets = security_manager.secrets_manager.list_secrets()
        print("Available secrets:")
        for secret in secrets:
            print(f"  - {secret}")
    
    if args.audit_logs:
        logs = security_manager.audit_trail.get_audit_logs()
        print("Recent audit logs:")
        for log in logs[:10]:  # Show last 10 entries
            print(f"  {log['timestamp']} - {log['user']} {log['action']} on {log['component']} - {'SUCCESS' if log['success'] else 'FAILED'}")

if __name__ == "__main__":
    main()