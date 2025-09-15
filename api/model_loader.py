"""
Model loading and prediction utilities for the ML Pipeline API.
"""
import os
import json
import pickle
import logging
import threading
from typing import Dict, Any, List, Union, Optional
from datetime import datetime
import numpy as np
import pandas as pd
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelLoader:
    """Handles loading and managing the v_best model."""
    
    def __init__(self, base_path: str = "."):
        self.base_path = Path(base_path)
        self.model = None
        self.metadata = None
        self.v_best_metadata = None
        self.last_loaded = None
        self._lock = threading.Lock()
        
        # Load model on initialization
        self.load_model()
    
    def get_model_path(self) -> Path:
        """Get the path to the v_best model file."""
        return self.base_path / "deploy" / "v_best" / "model.pkl"
    
    def get_metadata_path(self) -> Path:
        """Get the path to the model metadata file."""
        return self.base_path / "deploy" / "v_best" / "model_metadata.json"
    
    def get_v_best_metadata_path(self) -> Path:
        """Get the path to the v_best metadata file."""
        return self.base_path / "deploy" / "v_best" / "v_best_metadata.json"
    
    def load_model(self) -> bool:
        """
        Load the v_best model and its metadata.
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        with self._lock:
            try:
                model_path = self.get_model_path()
                metadata_path = self.get_metadata_path()
                v_best_metadata_path = self.get_v_best_metadata_path()
                
                # Check if files exist
                if not model_path.exists():
                    logger.error(f"Model file not found: {model_path}")
                    return False
                
                if not metadata_path.exists():
                    logger.error(f"Metadata file not found: {metadata_path}")
                    return False
                
                # Load model
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                
                # Load metadata
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                
                # Load v_best metadata if available
                if v_best_metadata_path.exists():
                    with open(v_best_metadata_path, 'r') as f:
                        self.v_best_metadata = json.load(f)
                
                self.last_loaded = datetime.now()
                logger.info(f"Model loaded successfully at {self.last_loaded}")
                logger.info(f"Model type: {self.metadata.get('model_type', 'unknown')}")
                logger.info(f"Model version: {self.metadata.get('model_version', 'unknown')}")
                
                return True
                
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                return False
    
    def is_model_loaded(self) -> bool:
        """Check if model is loaded and ready for predictions."""
        return self.model is not None and self.metadata is not None
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dict containing model information
        """
        if not self.is_model_loaded():
            return {"error": "Model not loaded"}
        
        info = {
            "model_name": self.metadata.get("model_name", "unknown"),
            "model_type": self.metadata.get("model_type", "unknown"),
            "model_version": self.metadata.get("model_version", "unknown"),
            "last_loaded": self.last_loaded.isoformat() if self.last_loaded else None,
            "training_params": self.metadata.get("training_params", {}),
            "validation_metrics": self.metadata.get("validation_metrics", {})
        }
        
        # Add v_best specific information if available
        if self.v_best_metadata:
            info["v_best_info"] = {
                "timestamp": self.v_best_metadata.get("timestamp"),
                "commit_id": self.v_best_metadata.get("commit_id"),
                "metrics": self.v_best_metadata.get("metrics", {}),
                "promoted_from_build": self.v_best_metadata.get("promoted_from_build")
            }
        
        return info
    
    def predict(self, data: Union[List, Dict, pd.DataFrame]) -> Dict[str, Any]:
        """
        Make predictions using the loaded model.
        
        Args:
            data: Input data for prediction (list, dict, or DataFrame)
            
        Returns:
            Dict containing predictions and metadata
        """
        if not self.is_model_loaded():
            raise ValueError("Model not loaded. Please check model files.")
        
        try:
            # Convert input data to DataFrame if needed
            if isinstance(data, dict):
                # Single prediction: convert dict to DataFrame
                df = pd.DataFrame([data])
            elif isinstance(data, list):
                if len(data) == 0:
                    raise ValueError("Empty data list provided")
                
                # Check if it's a list of dicts (multiple predictions)
                if isinstance(data[0], dict):
                    df = pd.DataFrame(data)
                else:
                    # Single prediction: list of values
                    # We need to know the feature names - use numeric indices
                    df = pd.DataFrame([data])
            elif isinstance(data, pd.DataFrame):
                df = data.copy()
            else:
                raise ValueError(f"Unsupported data type: {type(data)}")
            
            # Make predictions
            predictions = self.model.predict(df)
            
            # For classification models, also get prediction probabilities if available
            probabilities = None
            if (self.metadata.get("model_type") == "classification" and 
                hasattr(self.model, "predict_proba")):
                try:
                    probabilities = self.model.predict_proba(df).tolist()
                except Exception as e:
                    logger.warning(f"Could not get prediction probabilities: {e}")
            
            # Prepare response
            response = {
                "predictions": predictions.tolist(),
                "model_info": {
                    "model_name": self.metadata.get("model_name"),
                    "model_type": self.metadata.get("model_type"),
                    "model_version": self.metadata.get("model_version")
                },
                "prediction_timestamp": datetime.now().isoformat(),
                "input_shape": df.shape
            }
            
            if probabilities is not None:
                response["probabilities"] = probabilities
            
            return response
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise ValueError(f"Prediction failed: {str(e)}")
    
    def reload_model(self) -> bool:
        """
        Reload the model from disk.
        
        Returns:
            bool: True if reload successful, False otherwise
        """
        logger.info("Reloading model...")
        return self.load_model()

# Global model loader instance
model_loader = ModelLoader()