"""
FastAPI REST API for serving the v_best ML model.
"""
import os
import logging
from typing import Dict, Any, List, Union, Optional
from datetime import datetime
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from auth import verify_token, verify_token_optional
from model_loader import model_loader

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="ML Pipeline API",
    description="REST API for serving the v_best machine learning model",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response validation
class PredictionRequest(BaseModel):
    """Request model for predictions."""
    data: Union[List[Dict[str, Any]], Dict[str, Any], List[List[float]]]
    
    class Config:
        schema_extra = {
            "example": {
                "data": {
                    "feature1": 1.0,
                    "feature2": 2.0,
                    "feature3": 3.0
                }
            }
        }

class PredictionResponse(BaseModel):
    """Response model for predictions."""
    predictions: List[Union[float, int, str]]
    probabilities: Optional[List[List[float]]] = None
    model_info: Dict[str, Any]
    prediction_timestamp: str
    input_shape: List[int]

class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    timestamp: str
    model_loaded: bool
    model_info: Optional[Dict[str, Any]] = None

class ModelInfoResponse(BaseModel):
    """Response model for model information."""
    model_name: str
    model_type: str
    model_version: str
    last_loaded: Optional[str]
    training_params: Dict[str, Any]
    validation_metrics: Dict[str, Any]
    v_best_info: Optional[Dict[str, Any]] = None

class ReloadResponse(BaseModel):
    """Response model for model reload."""
    success: bool
    message: str
    timestamp: str

# Exception handlers
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"error": "Bad Request", "detail": str(exc)}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal Server Error", "detail": "An unexpected error occurred"}
    )

# API Endpoints

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    Returns the API status and model loading status.
    """
    model_loaded = model_loader.is_model_loaded()
    model_info = None
    
    if model_loaded:
        try:
            model_info = model_loader.get_model_info()
        except Exception as e:
            logger.warning(f"Could not get model info: {e}")
    
    return HealthResponse(
        status="ok",
        timestamp=datetime.now().isoformat(),
        model_loaded=model_loaded,
        model_info=model_info
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    token: str = Depends(verify_token)
):
    """
    Make predictions using the v_best model.
    Requires authentication.
    """
    if not model_loader.is_model_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please check model files or contact administrator."
        )
    
    try:
        result = model_loader.predict(request.data)
        return PredictionResponse(**result)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed due to internal error"
        )

@app.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info(token: Optional[str] = Depends(verify_token_optional)):
    """
    Get information about the currently loaded model.
    Authentication is optional - provides more details when authenticated.
    """
    if not model_loader.is_model_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    try:
        info = model_loader.get_model_info()
        
        # If not authenticated, remove sensitive information
        if not token:
            # Remove detailed training params and v_best info for unauthenticated requests
            info.pop("training_params", None)
            info.pop("v_best_info", None)
        
        return ModelInfoResponse(**info)
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not retrieve model information"
        )

@app.post("/model/reload", response_model=ReloadResponse)
async def reload_model(token: str = Depends(verify_token)):
    """
    Reload the model from disk.
    Requires authentication.
    """
    try:
        success = model_loader.reload_model()
        
        if success:
            message = "Model reloaded successfully"
            logger.info(message)
        else:
            message = "Failed to reload model"
            logger.error(message)
        
        return ReloadResponse(
            success=success,
            message=message,
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Error reloading model: {e}")
        return ReloadResponse(
            success=False,
            message=f"Reload failed: {str(e)}",
            timestamp=datetime.now().isoformat()
        )

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "ML Pipeline API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "predict": "/predict (POST, requires auth)",
            "model_info": "/model/info (GET, optional auth)",
            "model_reload": "/model/reload (POST, requires auth)",
            "health": "/health (GET, public)"
        }
    }

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize the API on startup."""
    logger.info("Starting ML Pipeline API...")
    
    # Verify model is loaded
    if model_loader.is_model_loaded():
        logger.info("Model loaded successfully on startup")
        model_info = model_loader.get_model_info()
        logger.info(f"Loaded model: {model_info.get('model_name')} v{model_info.get('model_version')}")
    else:
        logger.warning("Model not loaded on startup - check model files")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down ML Pipeline API...")

if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8000))
    debug = os.getenv("DEBUG", "True").lower() == "true"
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info"
    )