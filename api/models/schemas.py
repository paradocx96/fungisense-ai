from enum import Enum
from typing import Literal, Optional, List

from pydantic import BaseModel, Field, validator


# Enums for categorical values
class CapShape(str, Enum):
    BELL = "bell"
    CONICAL = "conical"
    CONVEX = "convex"
    FLAT = "flat"
    SUNKEN = "sunken"
    SPHERICAL = "spherical"
    OTHERS = "others"


class CapSurface(str, Enum):
    FIBROUS = "fibrous"
    GROOVES = "grooves"
    SCALY = "scaly"
    SMOOTH = "smooth"
    SHINY = "shiny"
    LEATHERY = "leathery"
    SILKY = "silky"
    STICKY = "sticky"
    WRINKLED = "wrinkled"
    FLESHY = "fleshy"


class CapColor(str, Enum):
    BROWN = "brown"
    BUFF = "buff"
    GRAY = "gray"
    GREEN = "green"
    PINK = "pink"
    PURPLE = "purple"
    RED = "red"
    WHITE = "white"
    YELLOW = "yellow"
    BLUE = "blue"
    ORANGE = "orange"
    BLACK = "black"


class GillAttachment(str, Enum):
    ADNATE = "adnate"
    ADNEXED = "adnexed"
    DECURRENT = "decurrent"
    FREE = "free"
    SINUATE = "sinuate"
    PORES = "pores"
    NONE = "none"
    UNKNOWN = "unknown"


class GillSpacing(str, Enum):
    CLOSE = "close"
    DISTANT = "distant"
    NONE = "none"


class StemRoot(str, Enum):
    BULBOUS = "bulbous"
    SWOLLEN = "swollen"
    CLUB = "club"
    CUP = "cup"
    EQUAL = "equal"
    RHIZOMORPHS = "rhizomorphs"
    ROOTED = "rooted"


class VeilType(str, Enum):
    PARTIAL = "partial"
    UNIVERSAL = "universal"


class RingType(str, Enum):
    COBWEBBY = "cobwebby"
    EVANESCENT = "evanescent"
    FLARING = "flaring"
    GROOVED = "grooved"
    LARGE = "large"
    PENDANT = "pendant"
    SHEATHING = "sheathing"
    ZONE = "zone"
    SCALY = "scaly"
    MOVABLE = "movable"
    NONE = "none"
    UNKNOWN = "unknown"


class Habitat(str, Enum):
    GRASSES = "grasses"
    LEAVES = "leaves"
    MEADOWS = "meadows"
    PATHS = "paths"
    HEATHS = "heaths"
    URBAN = "urban"
    WASTE = "waste"
    WOODS = "woods"


class Season(str, Enum):
    SPRING = "spring"
    SUMMER = "summer"
    AUTUMN = "autumn"
    WINTER = "winter"


# Input Model
class MushroomInput(BaseModel):
    """Input features for mushroom prediction"""

    cap_diameter: float = Field(
        ...,
        gt=0,
        lt=50,
        description="Cap diameter in cm",
        example=12.5
    )
    cap_shape: CapShape = Field(..., description="Shape of the mushroom cap")
    cap_surface: CapSurface = Field(..., description="Surface texture of the cap")
    cap_color: CapColor = Field(..., description="Color of the cap")
    does_bruise_or_bleed: Literal["bruises", "no"] = Field(
        ...,
        description="Does the mushroom bruise or bleed when touched"
    )
    gill_attachment: GillAttachment = Field(..., description="How gills attach to stem")
    gill_spacing: GillSpacing = Field(..., description="Spacing between gills")
    gill_color: CapColor = Field(..., description="Color of the gills")
    stem_height: float = Field(
        ...,
        gt=0,
        lt=100,
        description="Stem height in cm",
        example=10.5
    )
    stem_width: float = Field(
        ...,
        gt=0,
        lt=50,
        description="Stem width in mm",
        example=15.0
    )
    stem_root: StemRoot = Field(..., description="Type of stem root")
    stem_surface: CapSurface = Field(..., description="Surface texture of the stem")
    stem_color: CapColor = Field(..., description="Color of the stem")
    veil_type: VeilType = Field(..., description="Type of veil")
    veil_color: CapColor = Field(..., description="Color of the veil")
    has_ring: Literal["ring", "none"] = Field(..., description="Does mushroom have a ring")
    ring_type: RingType = Field(..., description="Type of ring")
    spore_print_color: CapColor = Field(..., description="Color of spore print")
    habitat: Habitat = Field(..., description="Where the mushroom grows")
    season: Season = Field(..., description="When the mushroom appears")

    @validator('cap_diameter')
    def validate_cap_diameter(cls, v):
        if v > 40:
            raise ValueError('Cap diameter seems unrealistically large (>40cm)')
        return v

    @validator('stem_height')
    def validate_stem_height(cls, v):
        if v > 80:
            raise ValueError('Stem height seems unrealistically tall (>80cm)')
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "cap_diameter": 12.5,
                "cap_shape": "convex",
                "cap_surface": "smooth",
                "cap_color": "brown",
                "does_bruise_or_bleed": "no",
                "gill_attachment": "free",
                "gill_spacing": "close",
                "gill_color": "white",
                "stem_height": 10.5,
                "stem_width": 15.0,
                "stem_root": "bulbous",
                "stem_surface": "smooth",
                "stem_color": "white",
                "veil_type": "partial",
                "veil_color": "white",
                "has_ring": "ring",
                "ring_type": "pendant",
                "spore_print_color": "white",
                "habitat": "woods",
                "season": "autumn"
            }
        }


# Output Models
class SpeciesPrediction(BaseModel):
    """Species prediction result"""
    name: str = Field(..., description="Predicted species name")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence (0-1)")


class FamilyPrediction(BaseModel):
    """Family prediction result"""
    name: str = Field(..., description="Predicted family name")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence (0-1)")


class EdibilityPrediction(BaseModel):
    """Edibility prediction result"""
    class_name: Literal["Edible", "Poisonous"] = Field(..., description="Edibility classification")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence (0-1)")
    warning: Optional[str] = Field(None, description="Safety warning if poisonous")


class HabitatPrediction(BaseModel):
    """Habitat prediction result"""
    primary: str = Field(..., description="Primary habitat")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence (0-1)")
    also_found_in: Optional[List[str]] = Field(default=[], description="Other possible habitats")


class SeasonPrediction(BaseModel):
    """Season prediction result"""
    primary: str = Field(..., description="Primary season")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence (0-1)")
    also_found_in: Optional[List[str]] = Field(default=[], description="Other possible seasons")


class LookalikeWarning(BaseModel):
    """Lookalike mushroom warning"""
    name: str = Field(..., description="Name of lookalike species")
    edible: bool = Field(..., description="Is the lookalike edible")
    warning: str = Field(..., description="Warning message")


class RiskAssessment(BaseModel):
    """Risk assessment for the mushroom"""
    level: Literal["SAFE", "CAUTION", "DANGER", "CRITICAL"] = Field(..., description="Risk level")
    message: str = Field(..., description="Risk assessment message")
    lookalikes: Optional[List[LookalikeWarning]] = Field(default=[], description="Similar species warnings")


class PredictionMetadata(BaseModel):
    """Metadata about the prediction"""
    timestamp: str = Field(..., description="Prediction timestamp")
    model_version: str = Field(..., description="Model version used")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


class CompletePredictionResponse(BaseModel):
    """Complete prediction response with all features"""
    species: SpeciesPrediction
    family: FamilyPrediction
    edibility: EdibilityPrediction
    habitat: HabitatPrediction
    season: SeasonPrediction
    risk_assessment: RiskAssessment
    metadata: PredictionMetadata

    class Config:
        json_schema_extra = {
            "example": {
                "species": {
                    "name": "Death Cap",
                    "confidence": 0.92
                },
                "family": {
                    "name": "Amanita Family",
                    "confidence": 0.98
                },
                "edibility": {
                    "class_name": "Poisonous",
                    "confidence": 0.99,
                    "warning": "⚠️ DEADLY POISONOUS - Contains lethal amatoxins"
                },
                "habitat": {
                    "primary": "Woods",
                    "confidence": 0.87,
                    "also_found_in": ["Urban areas"]
                },
                "season": {
                    "primary": "Autumn",
                    "confidence": 0.85,
                    "also_found_in": ["Summer", "Winter"]
                },
                "risk_assessment": {
                    "level": "CRITICAL",
                    "message": "This mushroom is extremely dangerous. DO NOT consume.",
                    "lookalikes": [
                        {
                            "name": "False Death Cap",
                            "edible": True,
                            "warning": "Easily confused with Death Cap - DO NOT consume unless 100% certain"
                        }
                    ]
                },
                "metadata": {
                    "timestamp": "2025-12-16T10:30:45Z",
                    "model_version": "2.0.0",
                    "processing_time_ms": 45.2
                }
            }
        }


class HealthCheckResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="API status")
    version: str = Field(..., description="API version")
    model_loaded: bool = Field(..., description="Is ML model loaded")


class ErrorResponse(BaseModel):
    """Error response"""
    error: bool = Field(default=True, description="Error flag")
    message: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")


# Phase 1 Enhanced Endpoint Schemas

class EnhancedHealthCheckResponse(BaseModel):
    """Enhanced health check response with detailed system status"""
    status: str = Field(..., description="API status (healthy/unhealthy)")
    version: str = Field(..., description="API version")
    model_loaded: bool = Field(..., description="Is ML model loaded")
    model_type: Optional[str] = Field(None, description="Type of model (multi-output/single-output)")
    encoders_loaded: dict = Field(..., description="Status of each encoder")
    uptime_seconds: float = Field(..., description="API uptime in seconds")
    total_predictions: int = Field(default=0, description="Total predictions made since startup")


class ModelMetadata(BaseModel):
    """Model metadata and performance information"""
    model_version: str = Field(..., description="Model version")
    model_type: str = Field(..., description="Model architecture type")
    training_date: Optional[str] = Field(None, description="Model training date")
    num_species: int = Field(..., description="Number of species in model")
    num_families: int = Field(..., description="Number of families in model")
    num_habitats: int = Field(..., description="Number of habitats in model")
    num_seasons: int = Field(..., description="Number of seasons in model")
    input_features: int = Field(..., description="Number of input features")
    model_size_mb: Optional[float] = Field(None, description="Model file size in MB")
    performance_metrics: Optional[dict] = Field(None, description="Model performance metrics")


class SpeciesProbability(BaseModel):
    """Species prediction with probability"""
    species: str = Field(..., description="Species name")
    family: str = Field(..., description="Family name")
    edibility: str = Field(..., description="Edibility classification")
    probability: float = Field(..., ge=0, le=1, description="Prediction probability")
    confidence: float = Field(..., ge=0, le=1, description="Overall confidence")


class TopNSpeciesResponse(BaseModel):
    """Top-N species predictions response"""
    top_predictions: List[SpeciesProbability] = Field(..., description="Top N species predictions")
    input_features: dict = Field(..., description="Input features summary")
    metadata: PredictionMetadata


class BatchPredictionRequest(BaseModel):
    """Batch prediction request"""
    mushrooms: List[MushroomInput] = Field(..., min_length=1, max_length=100, description="List of mushrooms to predict (max 100)")
    include_risk_assessment: bool = Field(default=True, description="Include risk assessment in response")


class BatchPredictionResponse(BaseModel):
    """Batch prediction response"""
    predictions: List[CompletePredictionResponse] = Field(..., description="List of predictions")
    summary: dict = Field(..., description="Summary statistics")
    metadata: PredictionMetadata
