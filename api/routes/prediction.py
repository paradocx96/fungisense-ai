"""
Prediction routes for the FungiSense AI API
"""
import os
import pickle
import sys
import time
from datetime import datetime

import numpy as np
from fastapi import APIRouter, HTTPException

# Project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from api.models.schemas import (
    MushroomInput,
    CompletePredictionResponse,
    SpeciesPrediction,
    FamilyPrediction,
    EdibilityPrediction,
    HabitatPrediction,
    SeasonPrediction,
    RiskAssessment,
    PredictionMetadata,
    LookalikeWarning
)
from config.config import settings, LOOKALIKE_DATABASE, get_risk_level

router = APIRouter(prefix="/api/v1/predict", tags=["Predictions"])

# Global variables
model = None
encoders = {}
prediction_count = 0


def load_model_and_encoders():
    """Load the trained model and encoders"""
    global model, encoders

    try:
        # Try to load TensorFlow/Keras model
        import tensorflow as tf
        from tensorflow import keras

        # Try multi-output model first, then fall back to simple classifier
        model_paths = [
            os.path.join(settings.MODEL_DIR, 'multi_output_model.h5'),
            os.path.join(settings.MODEL_DIR, 'mushroom_classifier.h5')
        ]

        model_path = None
        for path in model_paths:
            if os.path.exists(path):
                model_path = path
                break

        if model_path:
            model = keras.models.load_model(model_path)
            print(f"✅ Model loaded from: {model_path}")

            # Load encoders - try new names first, then fall back to old names
            encoder_files = {
                'feature_encoder': 'feature_encoder.pkl',
                'onehot': 'onehot_encoder.pkl',
                'scaler': 'scaler.pkl',
                'feature_info': 'feature_info.pkl',
                'species_encoder': 'species_encoder.pkl',
                'family_encoder': 'family_encoder.pkl',
                'edibility_encoder': 'edibility_encoder.pkl',
                'habitat_encoder': 'habitat_encoder.pkl',
                'season_encoder': 'season_encoder.pkl'
            }

            for key, filename in encoder_files.items():
                encoder_path = os.path.join(settings.MODEL_DIR, filename)
                if os.path.exists(encoder_path):
                    with open(encoder_path, 'rb') as f:
                        encoders[key] = pickle.load(f)
                    print(f"✅ {filename} loaded")

            print("✅ All model components loaded successfully!")
        else:
            print(f"⚠️  Model not found in: {settings.MODEL_DIR}")
            print("   Tried: multi_output_model.h5, mushroom_classifier.h5")
            print("   Please train the model first:")
            print("   Run: python scripts/train_model.py")
            model = None

    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        model = None


# Load model on startup
load_model_and_encoders()


def preprocess_input(mushroom_data: MushroomInput) -> np.ndarray:
    """
    Preprocess input data for model prediction

    Args:
        mushroom_data: Input mushroom features

    Returns:
        Preprocessed numpy array ready for model
    """
    import pandas as pd

    # Get feature info
    feature_info = encoders.get('feature_info', {})
    numerical_cols = feature_info.get('numerical_cols', ['cap_diameter', 'stem_height', 'stem_width'])
    categorical_cols = feature_info.get('categorical_cols', [])

    # Convert input to dictionary and extract enum values
    data_dict = {}
    for field_name, field_value in mushroom_data.model_dump().items():
        # If it's an enum-like object, get the value
        if hasattr(field_value, 'value'):
            data_dict[field_name] = field_value.value
        else:
            data_dict[field_name] = field_value

    # Create DataFrame
    df = pd.DataFrame([data_dict])

    # Separate numerical and categorical
    numerical_data = df[numerical_cols]
    categorical_data = df[categorical_cols]

    # Transform numerical features
    scaler = encoders.get('scaler')
    if scaler:
        numerical_scaled = scaler.transform(numerical_data)
    else:
        numerical_scaled = numerical_data.values

    # Transform categorical features
    # Try both 'onehot' and 'feature_encoder' keys
    feature_encoder = encoders.get('feature_encoder') or encoders.get('onehot')
    if feature_encoder:
        categorical_encoded = feature_encoder.transform(categorical_data)
    else:
        raise ValueError("Feature encoder not loaded")

    # Combine features
    X_processed = np.hstack([numerical_scaled, categorical_encoded])

    return X_processed


def predict_with_model(mushroom_data: MushroomInput) -> dict:
    """
    Make prediction using the trained model

    Args:
        mushroom_data: Input mushroom features

    Returns:
        dict with predictions
    """
    # Preprocess input
    X = preprocess_input(mushroom_data)

    # Make prediction
    predictions = model.predict(X, verbose=0)

    # Check if multi-output model (returns list) or single-output (returns array)
    if isinstance(predictions, list):
        # Multi-output model: [species, family, edibility, habitat, season]
        species_probs, family_probs, edibility_prob, habitat_probs, season_probs = predictions

        # Get edibility prediction
        edibility_value = float(edibility_prob[0][0])
        is_poisonous = edibility_value > 0.5
        edibility_confidence = max(edibility_value, 1 - edibility_value)

        # Get species prediction (argmax of probabilities)
        species_idx = int(np.argmax(species_probs[0]))
        species_confidence = float(species_probs[0][species_idx])

        # Get family prediction
        family_idx = int(np.argmax(family_probs[0]))
        family_confidence = float(family_probs[0][family_idx])

        # Get habitat prediction
        habitat_idx = int(np.argmax(habitat_probs[0]))

        # Get season prediction
        season_idx = int(np.argmax(season_probs[0]))

        # Load encoders to get label names
        species_encoder = encoders.get('species_encoder')
        family_encoder = encoders.get('family_encoder')
        habitat_encoder = encoders.get('habitat_encoder')
        season_encoder = encoders.get('season_encoder')

        # Get species and family names from encoders if available
        if species_encoder and hasattr(species_encoder, 'classes_'):
            species_name = species_encoder.classes_[species_idx]
        else:
            species_name = "Unknown Species"

        if family_encoder and hasattr(family_encoder, 'classes_'):
            family_name = family_encoder.classes_[family_idx]
        else:
            family_name = "Unknown Family"

        if habitat_encoder and hasattr(habitat_encoder, 'classes_'):
            habitat_name = habitat_encoder.classes_[habitat_idx]
        else:
            habitat_name = mushroom_data.habitat.value.capitalize()

        if season_encoder and hasattr(season_encoder, 'classes_'):
            season_name = season_encoder.classes_[season_idx]
        else:
            season_name = mushroom_data.season.value.capitalize()

        # Overall confidence is average of edibility and species
        overall_confidence = (edibility_confidence + species_confidence) / 2

        return {
            "species": species_name,
            "family": family_name,
            "edibility": "Poisonous" if is_poisonous else "Edible",
            "confidence": float(overall_confidence),
            "habitat": habitat_name,
            "season": season_name
        }
    else:
        # Single-output model (edibility only)
        prediction_prob = predictions[0][0]
        is_poisonous = prediction_prob > 0.5
        confidence = max(prediction_prob, 1 - prediction_prob)

        return {
            "species": "Unknown Species",
            "family": "Unknown Family",
            "edibility": "Poisonous" if is_poisonous else "Edible",
            "confidence": float(confidence),
            "habitat": mushroom_data.habitat.value.capitalize(),
            "season": mushroom_data.season.value.capitalize()
        }


def mock_prediction(mushroom_data: MushroomInput) -> dict:
    """
    Mock prediction function for testing API when model is not trained

    Args:
        mushroom_data: Input mushroom features

    Returns:
        dict with mock predictions
    """
    # Simple heuristic for demo: if cap is red/orange and convex, likely Fly Agaric
    is_fly_agaric = (
            mushroom_data.cap_color.value in ["red", "orange"] and
            mushroom_data.cap_shape.value == "convex"
    )

    if is_fly_agaric:
        species = "Fly Agaric"
        family = "Amanita Family"
        edibility = "Poisonous"
        confidence = 0.85
    else:
        species = "Common Field Mushroom"
        family = "Agaricus Family"
        edibility = "Edible"
        confidence = 0.75

    return {
        "species": species,
        "family": family,
        "edibility": edibility,
        "confidence": confidence,
        "habitat": mushroom_data.habitat.value.capitalize(),
        "season": mushroom_data.season.value.capitalize()
    }


@router.post("/complete", response_model=CompletePredictionResponse)
async def predict_complete(mushroom: MushroomInput):
    """
    Complete prediction endpoint - predicts all features

    Returns species, family, edibility, habitat, and season predictions
    along with risk assessment and lookalike warnings.

    **Note:** Currently uses trained model for edibility if available,
    otherwise falls back to mock predictions for demonstration.
    """
    global prediction_count
    start_time = time.time()

    try:
        # Use real model if available, otherwise mock
        if model is not None and encoders:
            prediction = predict_with_model(mushroom)
        else:
            prediction = mock_prediction(mushroom)

        # Increment prediction counter
        prediction_count += 1

        # Build response
        species_pred = SpeciesPrediction(
            name=prediction["species"],
            confidence=prediction["confidence"]
        )

        family_pred = FamilyPrediction(
            name=prediction["family"],
            confidence=prediction["confidence"] * 0.95
        )

        edibility_pred = EdibilityPrediction(
            class_name=prediction["edibility"],
            confidence=prediction["confidence"] * 0.98,
            warning=(
                f"⚠️ POISONOUS - {prediction['species']} can cause serious illness"
                if prediction["edibility"] == "Poisonous"
                else None
            )
        )

        habitat_pred = HabitatPrediction(
            primary=prediction["habitat"],
            confidence=prediction["confidence"] * 0.85,
            also_found_in=["Urban areas"] if prediction["habitat"] == "Woods" else []
        )

        season_pred = SeasonPrediction(
            primary=prediction["season"],
            confidence=prediction["confidence"] * 0.80,
            also_found_in=["Summer"] if prediction["season"] == "Autumn" else []
        )

        # Get lookalikes
        lookalikes = []
        if prediction["species"] in LOOKALIKE_DATABASE:
            for lookalike in LOOKALIKE_DATABASE[prediction["species"]]:
                lookalikes.append(LookalikeWarning(**lookalike))

        # Get risk assessment
        risk_info = get_risk_level(
            prediction["species"],
            prediction["edibility"],
            prediction["confidence"]
        )

        risk_assessment = RiskAssessment(
            level=risk_info["level"],
            message=risk_info["message"],
            lookalikes=lookalikes
        )

        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000  # Convert to ms

        metadata = PredictionMetadata(
            timestamp=datetime.utcnow().isoformat() + "Z",
            model_version=settings.API_VERSION,
            processing_time_ms=round(processing_time, 2)
        )

        return CompletePredictionResponse(
            species=species_pred,
            family=family_pred,
            edibility=edibility_pred,
            habitat=habitat_pred,
            season=season_pred,
            risk_assessment=risk_assessment,
            metadata=metadata
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@router.post("/species", response_model=dict)
async def predict_species_only(mushroom: MushroomInput):
    """
    Predict only the species of the mushroom

    Faster endpoint for species identification only.
    """
    if model is not None and encoders:
        prediction = predict_with_model(mushroom)
    else:
        prediction = mock_prediction(mushroom)

    return {
        "species": prediction["species"],
        "confidence": prediction["confidence"],
        "processing_time_ms": 25.0
    }


@router.post("/safety", response_model=dict)
async def predict_safety(mushroom: MushroomInput):
    """
    Predict edibility and provide safety warnings

    Focus on safety - returns edibility classification and lookalike warnings.
    """
    if model is not None and encoders:
        prediction = predict_with_model(mushroom)
    else:
        prediction = mock_prediction(mushroom)

    # Get lookalikes
    lookalikes = []
    if prediction["species"] in LOOKALIKE_DATABASE:
        lookalikes = LOOKALIKE_DATABASE[prediction["species"]]

    # Get risk assessment
    risk_info = get_risk_level(
        prediction["species"],
        prediction["edibility"],
        prediction["confidence"]
    )

    return {
        "species": prediction["species"],
        "edibility": prediction["edibility"],
        "confidence": prediction["confidence"],
        "risk_level": risk_info["level"],
        "warning": risk_info["message"],
        "lookalikes": lookalikes
    }


@router.post("/foraging", response_model=dict)
async def predict_foraging_info(mushroom: MushroomInput):
    """
    Predict foraging information (habitat, season, edibility)

    Useful for foragers to know when and where to find similar mushrooms.
    """
    if model is not None and encoders:
        prediction = predict_with_model(mushroom)
    else:
        prediction = mock_prediction(mushroom)

    return {
        "species": prediction["species"],
        "edibility": prediction["edibility"],
        "habitat": prediction["habitat"],
        "season": prediction["season"],
        "confidence": prediction["confidence"],
        "foraging_tip": (
            f"Look for {prediction['species']} in {prediction['habitat'].lower()} "
            f"during {prediction['season'].lower()}."
        )
    }


@router.post("/batch", response_model=dict)
async def predict_batch(request: dict):
    """
    Batch prediction endpoint - predict multiple mushrooms at once

    Accepts a list of mushroom inputs and returns predictions for all.
    Maximum 100 mushrooms per request.

    **Request body:**
    ```json
    {
        "mushrooms": [/* list of MushroomInput objects */],
        "include_risk_assessment": true
    }
    ```
    """
    global prediction_count
    from api.models.schemas import MushroomInput

    start_time = time.time()

    # Validate request
    mushrooms = request.get("mushrooms", [])
    include_risk = request.get("include_risk_assessment", True)

    if not mushrooms:
        raise HTTPException(status_code=400, detail="No mushrooms provided")

    if len(mushrooms) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 mushrooms per request")

    # Process each mushroom
    predictions = []
    edibility_counts = {"Edible": 0, "Poisonous": 0}
    species_counts = {}

    for mushroom_data in mushrooms:
        # Convert dict to MushroomInput
        try:
            mushroom = MushroomInput(**mushroom_data)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid mushroom data: {str(e)}")

        # Make prediction (reuse existing logic)
        if model is not None and encoders:
            prediction = predict_with_model(mushroom)
        else:
            prediction = mock_prediction(mushroom)

        # Build response similar to predict_complete
        species_pred = SpeciesPrediction(
            name=prediction["species"],
            confidence=prediction["confidence"]
        )

        family_pred = FamilyPrediction(
            name=prediction["family"],
            confidence=prediction["confidence"] * 0.95
        )

        edibility_pred = EdibilityPrediction(
            class_name=prediction["edibility"],
            confidence=prediction["confidence"] * 0.98,
            warning=(
                f"⚠️ POISONOUS - {prediction['species']} can cause serious illness"
                if prediction["edibility"] == "Poisonous"
                else None
            )
        )

        habitat_pred = HabitatPrediction(
            primary=prediction["habitat"],
            confidence=prediction["confidence"] * 0.85,
            also_found_in=[]
        )

        season_pred = SeasonPrediction(
            primary=prediction["season"],
            confidence=prediction["confidence"] * 0.80,
            also_found_in=[]
        )

        # Get risk assessment if requested
        risk_assessment = None
        if include_risk:
            lookalikes = []
            if prediction["species"] in LOOKALIKE_DATABASE:
                for lookalike in LOOKALIKE_DATABASE[prediction["species"]]:
                    lookalikes.append(LookalikeWarning(**lookalike))

            risk_info = get_risk_level(
                prediction["species"],
                prediction["edibility"],
                prediction["confidence"]
            )

            risk_assessment = RiskAssessment(
                level=risk_info["level"],
                message=risk_info["message"],
                lookalikes=lookalikes
            )

        metadata = PredictionMetadata(
            timestamp=datetime.utcnow().isoformat() + "Z",
            model_version=settings.API_VERSION,
            processing_time_ms=0  # Will be calculated at the end
        )

        pred_response = CompletePredictionResponse(
            species=species_pred,
            family=family_pred,
            edibility=edibility_pred,
            habitat=habitat_pred,
            season=season_pred,
            risk_assessment=risk_assessment,
            metadata=metadata
        )

        predictions.append(pred_response)

        # Update counts for summary
        edibility_counts[prediction["edibility"]] = edibility_counts.get(prediction["edibility"], 0) + 1
        species_counts[prediction["species"]] = species_counts.get(prediction["species"], 0) + 1

    # Update prediction counter
    prediction_count += len(mushrooms)

    # Calculate processing time
    processing_time = (time.time() - start_time) * 1000

    # Create summary
    summary = {
        "total_mushrooms": len(mushrooms),
        "edibility_breakdown": edibility_counts,
        "unique_species": len(species_counts),
        "most_common_species": max(species_counts.items(), key=lambda x: x[1])[0] if species_counts else None,
        "poisonous_count": edibility_counts.get("Poisonous", 0),
        "edible_count": edibility_counts.get("Edible", 0)
    }

    metadata = PredictionMetadata(
        timestamp=datetime.utcnow().isoformat() + "Z",
        model_version=settings.API_VERSION,
        processing_time_ms=round(processing_time, 2)
    )

    return {
        "predictions": predictions,
        "summary": summary,
        "metadata": metadata
    }


@router.post("/top-species", response_model=dict)
async def predict_top_species(mushroom: MushroomInput, top_n: int = 5):
    """
    Predict top-N most likely species

    Returns the top N most probable species based on input features,
    ranked by confidence score.

    **Query parameters:**
    - `top_n`: Number of top predictions to return (default: 5, max: 20)
    """
    global prediction_count

    if top_n < 1 or top_n > 20:
        raise HTTPException(status_code=400, detail="top_n must be between 1 and 20")

    start_time = time.time()

    # Check if model is loaded
    if model is None or not encoders:
        raise HTTPException(status_code=503, detail="Model not loaded - top-N predictions require trained model")

    # Preprocess input
    X = preprocess_input(mushroom)

    # Make prediction
    predictions = model.predict(X, verbose=0)

    # Check if multi-output model
    if not isinstance(predictions, list):
        raise HTTPException(status_code=501, detail="Top-N predictions only available for multi-output models")

    # Extract species probabilities (first output)
    species_probs, family_probs, edibility_prob, habitat_probs, season_probs = predictions

    # Get top-N species indices
    top_indices = np.argsort(species_probs[0])[::-1][:top_n]

    # Get encoders
    species_encoder = encoders.get('species_encoder')
    family_encoder = encoders.get('family_encoder')
    edibility_encoder = encoders.get('edibility_encoder')

    if not species_encoder or not hasattr(species_encoder, 'classes_'):
        raise HTTPException(status_code=503, detail="Species encoder not available")

    # Build top predictions
    top_predictions = []
    for idx in top_indices:
        species_name = species_encoder.classes_[idx]
        species_prob = float(species_probs[0][idx])

        # Get corresponding family (most likely)
        family_idx = int(np.argmax(family_probs[0]))
        family_name = family_encoder.classes_[family_idx] if family_encoder and hasattr(family_encoder,
                                                                                        'classes_') else "Unknown Family"

        # Get edibility
        edibility_value = float(edibility_prob[0][0])
        is_poisonous = edibility_value > 0.5
        edibility = "Poisonous" if is_poisonous else "Edible"
        edibility_confidence = max(edibility_value, 1 - edibility_value)

        # Overall confidence (average of species and edibility)
        overall_confidence = (species_prob + edibility_confidence) / 2

        top_predictions.append({
            "species": species_name,
            "family": family_name,
            "edibility": edibility,
            "probability": species_prob,
            "confidence": overall_confidence
        })

    # Update prediction counter
    prediction_count += 1

    # Calculate processing time
    processing_time = (time.time() - start_time) * 1000

    # Create summary of input features
    input_summary = {
        "cap_diameter": mushroom.cap_diameter,
        "cap_color": mushroom.cap_color.value,
        "cap_shape": mushroom.cap_shape.value,
        "habitat": mushroom.habitat.value,
        "season": mushroom.season.value
    }

    metadata = PredictionMetadata(
        timestamp=datetime.utcnow().isoformat() + "Z",
        model_version=settings.API_VERSION,
        processing_time_ms=round(processing_time, 2)
    )

    return {
        "top_predictions": top_predictions,
        "input_features": input_summary,
        "metadata": metadata
    }
