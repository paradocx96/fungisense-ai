"""
Health routes for the FungiSense AI API
"""
import os
import sys
import time

from fastapi import APIRouter

# Project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from api.models.schemas import (
    EnhancedHealthCheckResponse,
    HealthCheckResponse
)
from api.routes import prediction
from config.config import settings

# Track startup time
STARTUP_TIME = time.time()

router = APIRouter(prefix="/health", tags=["Health"])


# Health check endpoint
@router.get("/", response_model=HealthCheckResponse)
async def health_check():
    """
    Health check endpoint
    """
    return {
        "status": "healthy",
        "version": settings.API_VERSION,
        "model_loaded": prediction.model is not None
    }


# Enhanced health check endpoint
@router.get("/detailed", response_model=EnhancedHealthCheckResponse)
async def enhanced_health_check():
    """
    Enhanced health check with detailed system status

    Returns:
    - API status and version
    - Model loaded status and type
    - Encoder loading status
    - System uptime
    - Total predictions made
    """
    # Calculate uptime
    uptime = time.time() - STARTUP_TIME

    # Check model type
    model_type = None
    if prediction.model is not None:
        # Check if multi-output (has multiple outputs)
        if hasattr(prediction.model, 'outputs'):
            num_outputs = len(prediction.model.outputs)
            model_type = "multi-output" if num_outputs > 1 else "single-output"
        else:
            model_type = "unknown"

    # Check encoder status
    encoders_status = {
        "feature_encoder": "feature_encoder" in prediction.encoders or "onehot" in prediction.encoders,
        "scaler": "scaler" in prediction.encoders,
        "species_encoder": "species_encoder" in prediction.encoders,
        "family_encoder": "family_encoder" in prediction.encoders,
        "habitat_encoder": "habitat_encoder" in prediction.encoders,
        "season_encoder": "season_encoder" in prediction.encoders,
        "feature_info": "feature_info" in prediction.encoders
    }

    return {
        "status": "healthy" if prediction.model is not None else "degraded",
        "version": settings.API_VERSION,
        "model_loaded": prediction.model is not None,
        "model_type": model_type,
        "encoders_loaded": encoders_status,
        "uptime_seconds": uptime,
        "total_predictions": prediction.prediction_count
    }
