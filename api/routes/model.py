"""
Models routes for the FungiSense AI API
"""
import os
import sys

from fastapi import APIRouter, HTTPException

# Project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from api.models.schemas import (
    ModelMetadata
)
from api.routes import prediction
from config.config import settings

router = APIRouter(prefix="/api/v1/model", tags=["Model"])


# Model metadata endpoint
@router.get("/info", response_model=ModelMetadata)
async def get_model_metadata():
    """
    Get model metadata and information

    Returns detailed information about the loaded model including
    - Architecture details
    - Number of classes for each output
    - Model size
    - Input features
    """
    if prediction.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Get model file size
    import os
    model_size_mb = None
    model_files = [
        os.path.join(settings.MODEL_DIR, 'multi_output_model.h5'),
        os.path.join(settings.MODEL_DIR, 'mushroom_classifier.h5')
    ]
    for model_file in model_files:
        if os.path.exists(model_file):
            model_size_mb = os.path.getsize(model_file) / (1024 * 1024)
            break

    # Get the number of classes from encoders
    num_species = len(
        prediction.encoders.get('species_encoder').classes_) if 'species_encoder' in prediction.encoders and hasattr(
        prediction.encoders['species_encoder'], 'classes_') else 0
    num_families = len(
        prediction.encoders.get('family_encoder').classes_) if 'family_encoder' in prediction.encoders and hasattr(
        prediction.encoders['family_encoder'], 'classes_') else 0
    num_habitats = len(
        prediction.encoders.get('habitat_encoder').classes_) if 'habitat_encoder' in prediction.encoders and hasattr(
        prediction.encoders['habitat_encoder'], 'classes_') else 0
    num_seasons = len(
        prediction.encoders.get('season_encoder').classes_) if 'season_encoder' in prediction.encoders and hasattr(
        prediction.encoders['season_encoder'], 'classes_') else 0

    # Get input shape
    input_features = prediction.model.input_shape[1] if hasattr(prediction.model, 'input_shape') else 0

    # Determine model type
    model_type = "multi-output" if hasattr(prediction.model, 'outputs') and len(
        prediction.model.outputs) > 1 else "single-output"

    return {
        "model_version": settings.API_VERSION,
        "model_type": model_type,
        "training_date": None,  # Could be added to a model metadata file
        "num_species": num_species,
        "num_families": num_families,
        "num_habitats": num_habitats,
        "num_seasons": num_seasons,
        "input_features": input_features,
        "model_size_mb": round(model_size_mb, 2) if model_size_mb else None,
        "performance_metrics": None  # Could be loaded from training logs
    }
