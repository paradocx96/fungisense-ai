import os

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""

    # API Settings
    API_TITLE: str = "FungiSense AI - Multi-Feature Prediction API"
    API_DESCRIPTION: str = "Advanced mushroom classification system predicting species, family, edibility, habitat, and season"
    API_VERSION: str = "2.0.0"

    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    RELOAD: bool = True

    # Base Directory
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Data Paths
    DATA_DIR: str = "data"
    RAW_DATA_DIR: str = os.path.join(BASE_DIR, DATA_DIR, "raw")
    PROCESSED_DATA_DIR: str = os.path.join(BASE_DIR, DATA_DIR, "processed")

    # Dataset Files
    DATASET_DIR: str = "dataset"
    PRIMARY_DATA_PATH: str = os.path.join(BASE_DIR, DATASET_DIR, "primary_data.csv")
    SECONDARY_DATA_PATH: str = os.path.join(BASE_DIR, DATASET_DIR, "secondary_data.csv")

    # Model Paths (absolute paths from project root)
    MODEL_DIR: str = os.path.join(BASE_DIR, "models_saved")
    MODEL_PATH: str = os.path.join(BASE_DIR, "models_saved", "multi_output_model.h5")

    # Encoder Paths
    FEATURE_ENCODER_PATH: str = os.path.join(BASE_DIR, "models_saved", "feature_encoder.pkl")
    SPECIES_ENCODER_PATH: str = os.path.join(BASE_DIR, "models_saved", "species_encoder.pkl")
    FAMILY_ENCODER_PATH: str = os.path.join(BASE_DIR, "models_saved", "family_encoder.pkl")
    HABITAT_ENCODER_PATH: str = os.path.join(BASE_DIR, "models_saved", "habitat_encoder.pkl")
    SEASON_ENCODER_PATH: str = os.path.join(BASE_DIR, "models_saved", "season_encoder.pkl")

    # Model Training Settings
    RANDOM_SEED: int = 42
    TEST_SIZE: float = 0.15
    VALIDATION_SIZE: float = 0.15
    BATCH_SIZE: int = 32
    EPOCHS: int = 50
    LEARNING_RATE: float = 0.001

    # Prediction Settings
    CONFIDENCE_THRESHOLD: float = 0.8
    LOW_CONFIDENCE_WARNING: bool = True

    # Logging
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()

# Lookalike database (manual curation - can be moved to a separate JSON file)
LOOKALIKE_DATABASE = {
    "Death Cap": [
        {
            "name": "False Death Cap",
            "edible": True,
            "warning": "Easily confused with Death Cap - DO NOT consume unless 100% certain. Death Cap has white gills and smells like honey when young."
        },
        {
            "name": "Paddy Straw Mushroom",
            "edible": True,
            "warning": "Can be confused with young Death Cap. Always check spore print and gill color."
        }
    ],
    "Destroying Angel": [
        {
            "name": "Button Mushroom (young)",
            "edible": True,
            "warning": "Young Destroying Angel can look similar to white button mushrooms. NEVER eat wild white mushrooms."
        }
    ],
    "Fly Agaric": [
        {
            "name": "Caesar's Mushroom",
            "edible": True,
            "warning": "Caesar's Mushroom has yellow gills while Fly Agaric has white gills. Be absolutely certain."
        }
    ],
    "Panther Cap": [
        {
            "name": "The Blusher",
            "edible": True,
            "warning": "Panther Cap and The Blusher look similar. The Blusher bruises reddish, Panther Cap does not."
        }
    ]
}


# Risk assessment rules
def get_risk_level(species_name: str, edibility: str, confidence: float) -> dict:
    """
    Determine risk level based on species, edibility, and confidence

    Args:
        species_name: Predicted species name
        edibility: 'Edible' or 'Poisonous'
        confidence: Prediction confidence (0-1)

    Returns:
        dict with 'level' and 'message'
    """
    deadly_species = [
        "Death Cap", "Destroying Angel", "Funeral Bell",
        "Deadly Webcap", "Angel of Death"
    ]

    if edibility == "Poisonous":
        if species_name in deadly_species:
            return {
                "level": "CRITICAL",
                "message": f"DEADLY POISONOUS - {species_name} is extremely dangerous and can be fatal. DO NOT consume under any circumstances."
            }
        else:
            return {
                "level": "DANGER",
                "message": f"POISONOUS - {species_name} can cause serious illness. DO NOT consume."
            }
    else:  # Edible
        if confidence < settings.CONFIDENCE_THRESHOLD:
            return {
                "level": "CAUTION",
                "message": f"Low confidence prediction ({confidence:.1%}). Do NOT consume without expert verification."
            }
        elif species_name in LOOKALIKE_DATABASE:
            return {
                "level": "CAUTION",
                "message": f"This mushroom has poisonous lookalikes. Only consume if you are 100% certain of identification."
            }
        else:
            return {
                "level": "SAFE",
                "message": f"Predicted as edible with {confidence:.1%} confidence. Still recommended to verify with expert."
            }
