<div align="center">

# 🍄 FungiSense AI

### Multi-Feature Mushroom Classification System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16.1-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Advanced deep learning API for multi-feature mushroom prediction including species, family, edibility, habitat, and seasonal occurrence.**

[Quick Start](#-quick-start) • [API Documentation](#-api-endpoints) • [Installation](#-installation) • [Contributing](#-contributing)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Usage](#-usage)
- [API Endpoints](#-api-endpoints)
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Model Performance](#-model-performance)
- [Development](#-development)
- [Docker Deployment](#-docker-deployment)
- [Testing](#-testing)
- [Safety & Disclaimer](#%EF%B8%8F-safety--disclaimer)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

---

## 🎯 Overview

FungiSense AI is a **production-ready machine learning API** that leverages deep neural networks to perform comprehensive mushroom classification. Unlike traditional binary classifiers, this system simultaneously predicts **five distinct characteristics** from physical features.

### What it Does

| Prediction Type | Output | Classes |
|----------------|---------|---------|
| 🔬 **Species Identification** | Specific mushroom species | 173 species |
| 🧬 **Family Classification** | Taxonomic family | 23 families |
| 🍽️ **Edibility Assessment** | Safe to eat or poisonous | 2 classes (Binary) |
| 🌲 **Habitat Prediction** | Growth environment | 7 habitats |
| 📅 **Season Prediction** | Time of appearance | 4 seasons |

### Use Cases

- 🔍 Educational mycology research and learning
- 📱 Integration into field guide applications
- 🧪 Biodiversity assessment and cataloging
- 📊 Ecological pattern analysis
- ⚠️ Pre-screening for foraging safety (with expert verification)

---

## ✨ Key Features

### Technical Capabilities

- **🧠 Multi-Output Neural Network**: Single unified model with shared feature extraction and specialized output heads
- **⚡ High-Performance API**: FastAPI framework with async support and automatic OpenAPI documentation
- **✅ Type-Safe Validation**: Pydantic schemas ensure data integrity and automatic request/response validation
- **📊 Comprehensive Predictions**: Returns confidence scores, risk assessments, and lookalike warnings
- **🛡️ Safety Systems**: Built-in danger alerts for toxic species and risky lookalikes
- **📈 Scalable Architecture**: Supports batch predictions (up to 100 items per request)
- **🔄 Model Versioning**: Track model versions, metadata, and performance metrics
- **📝 Auto-Generated Docs**: Interactive Swagger UI and ReDoc interfaces

### Dataset

- **61,069 training samples** (353 per species)
- **173 unique species** from diverse mushroom families
- **20 input features**: 3 numerical (measurements) + 17 categorical (characteristics)
- Based on Patrick Hardin's "Mushrooms & Toadstools" reference work

---

## 🚀 Quick Start

Get FungiSense AI running in under 5 minutes!

### Prerequisites

- **Python 3.8+** (Python 3.10 or 3.11 recommended)
- **pip** (Python package manager)
- **Git** (for cloning the repository)
- **10GB+ free disk space** (for dependencies and models)

### 1️⃣ Clone & Setup

```bash
# Clone the repository
git clone https://github.com/paradocx96/fungisense-ai.git
cd fungisense-ai

# Create and activate virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2️⃣ Train the Model

```bash
# Run the training script
python scripts/train_model.py
```

**Expected output:**
- Training time: ~5-10 minutes on CPU, ~2-3 minutes on GPU
- Model files saved to `models_saved/`
- Accuracy: ~98-100% on edibility, 85-95% on species

### 3️⃣ Start the API Server

```bash
# Start FastAPI server
python api/main.py

# Alternative: Use uvicorn directly
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

**Server will be available at:**
- **API**: http://localhost:8000
- **Swagger UI**: http://localhost:8000/docs (Interactive API testing)
- **ReDoc**: http://localhost:8000/redoc (API documentation)

### 4️⃣ Make Your First Prediction

#### Option A: Swagger UI (Easiest)

1. Navigate to http://localhost:8000/docs
2. Find `POST /api/v1/predict/complete`
3. Click **"Try it out"**
4. Use the example request below or modify it
5. Click **"Execute"**

#### Option B: cURL

```bash
curl -X POST "http://localhost:8000/api/v1/predict/complete" \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

#### Option C: Python

```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/predict/complete",
    json={
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
)

print(response.json())
```

**Example Response:**

```json
{
  "species": {
    "prediction": "agaricus_campestris",
    "confidence": 0.89,
    "top_3_predictions": [...]
  },
  "family": {
    "prediction": "agaricaceae",
    "confidence": 0.94
  },
  "edibility": {
    "prediction": "edible",
    "confidence": 0.97,
    "risk_level": "SAFE"
  },
  "habitat": {
    "prediction": "woods",
    "confidence": 0.82
  },
  "season": {
    "prediction": "autumn",
    "confidence": 0.78
  },
  "warnings": [],
  "metadata": {
    "model_version": "2.0.0",
    "prediction_time_ms": 45.2
  }
}
```

---

## 📦 Installation

### Standard Installation

```bash
# Clone repository
git clone https://github.com/paradocx96/fungisense-ai.git
cd fungisense-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt
```

### Development Installation

```bash
# Install with development dependencies
pip install -r requirements.txt

# Install pre-commit hooks (optional)
pip install pre-commit
pre-commit install

# Verify installation
python -c "import tensorflow; import fastapi; print('✅ Installation successful!')"
```

### Docker Installation

```bash
# Build Docker image
docker build -t fungisense-ai:latest .

# Run container
docker run -d -p 8000:8000 fungisense-ai:latest

# Access API at http://localhost:8000
```

### Using Docker Compose

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

---

## 💻 Usage

### Basic Prediction

```python
import requests

# Define mushroom characteristics
mushroom_data = {
    "cap_diameter": 8.5,
    "cap_shape": "flat",
    "cap_surface": "scaly",
    "cap_color": "brown",
    "does_bruise_or_bleed": "yes",
    "gill_attachment": "free",
    "gill_spacing": "close",
    "gill_color": "white",
    "stem_height": 10.0,
    "stem_width": 12.0,
    "stem_root": "equal",
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

# Make prediction
response = requests.post(
    "http://localhost:8000/api/v1/predict/complete",
    json=mushroom_data
)

result = response.json()
print(f"Species: {result['species']['prediction']}")
print(f"Edibility: {result['edibility']['prediction']}")
print(f"Confidence: {result['edibility']['confidence']:.2%}")
```

### Batch Predictions

```python
# Predict multiple mushrooms at once
batch_data = {
    "mushrooms": [mushroom1_data, mushroom2_data, mushroom3_data],
    "include_risk_assessment": True
}

response = requests.post(
    "http://localhost:8000/api/v1/predict/batch",
    json=batch_data
)

predictions = response.json()["predictions"]
for idx, pred in enumerate(predictions, 1):
    print(f"Mushroom {idx}: {pred['species']['prediction']}")
```

### Safety-Focused Prediction

```python
# Get only safety-critical information
response = requests.post(
    "http://localhost:8000/api/v1/predict/safety",
    json=mushroom_data
)

safety = response.json()
if safety["edibility"]["risk_level"] in ["DANGER", "CRITICAL"]:
    print(f"⚠️ WARNING: {safety['warnings']}")
```

### Input Features Reference

| Feature | Type | Valid Values | Example |
|---------|------|--------------|---------|
| `cap_diameter` | float | 0.5 - 50.0 cm | 12.5 |
| `cap_shape` | string | bell, conical, convex, flat, sunken, spherical, others | "convex" |
| `cap_surface` | string | fibrous, grooves, scaly, smooth, shiny, leathery, etc. | "smooth" |
| `cap_color` | string | brown, buff, gray, green, pink, purple, red, white, yellow, etc. | "brown" |
| `does_bruise_or_bleed` | string | yes, no | "no" |
| `gill_attachment` | string | adnate, adnexed, decurrent, free, sinuate, pores, etc. | "free" |
| `gill_spacing` | string | close, distant, none | "close" |
| `gill_color` | string | (same as cap_color) | "white" |
| `stem_height` | float | 0.5 - 40.0 cm | 10.5 |
| `stem_width` | float | 0.5 - 50.0 mm | 15.0 |
| `stem_root` | string | bulbous, swollen, club, cup, equal, rhizomorphs, rooted | "bulbous" |
| `stem_surface` | string | (same as cap_surface) | "smooth" |
| `stem_color` | string | (same as cap_color) | "white" |
| `veil_type` | string | partial, universal | "partial" |
| `veil_color` | string | (same as cap_color) | "white" |
| `has_ring` | string | ring, none | "ring" |
| `ring_type` | string | cobwebby, evanescent, flaring, grooved, large, pendant, etc. | "pendant" |
| `spore_print_color` | string | (same as cap_color + black, olive) | "white" |
| `habitat` | string | woods, grasses, leaves, meadows, paths, urban, waste | "woods" |
| `season` | string | spring, summer, autumn, winter | "autumn" |

---

## 🔗 API Endpoints

### Core Prediction Endpoints

#### 1. Complete Prediction

```http
POST /api/v1/predict/complete
```

Returns all 5 predictions with full metadata.

**Request Body:**
```json
{
  "cap_diameter": 12.5,
  "cap_shape": "convex",
  // ... all 20 features
}
```

**Response:** Complete prediction object with species, family, edibility, habitat, season

---

#### 2. Species Identification

```http
POST /api/v1/predict/species
```

Fast species-only prediction.

**Response:**
```json
{
  "prediction": "agaricus_campestris",
  "confidence": 0.89,
  "top_3_predictions": [
    {"species": "agaricus_campestris", "confidence": 0.89},
    {"species": "agaricus_bisporus", "confidence": 0.07},
    {"species": "lepiota_procera", "confidence": 0.02}
  ]
}
```

---

#### 3. Safety Check

```http
POST /api/v1/predict/safety
```

Edibility prediction with lookalike warnings.

**Response:**
```json
{
  "edibility": {
    "prediction": "poisonous",
    "confidence": 0.95,
    "risk_level": "DANGER"
  },
  "warnings": [
    "Similar to Amanita phalloides (Death Cap) - DEADLY if consumed"
  ]
}
```

---

#### 4. Foraging Information

```http
POST /api/v1/predict/foraging
```

Returns habitat, season, and edibility for foragers.

---

#### 5. Batch Predictions

```http
POST /api/v1/predict/batch
```

Process up to 100 mushrooms in a single request.

**Request:**
```json
{
  "mushrooms": [ /* array of mushroom objects */ ],
  "include_risk_assessment": true
}
```

---

### System Endpoints

#### Health Check

```http
GET /health
GET /health/detailed
```

Returns API health status and system information.

**Response:**
```json
{
  "status": "healthy",
  "version": "2.0.0",
  "model_loaded": true,
  "model_type": "multi-output",
  "uptime_seconds": 1234.5,
  "total_predictions": 150
}
```

---

#### Model Information

```http
GET /api/v1/model/info
```

Returns model metadata and capabilities.

**Response:**
```json
{
  "model_version": "2.0.0",
  "model_type": "multi-output",
  "num_species": 173,
  "num_families": 23,
  "num_habitats": 7,
  "num_seasons": 4,
  "input_features": 111,
  "model_size_mb": 2.45
}
```

---

### Interactive Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

---

## 🏗️ Architecture

### System Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                        CLIENT APPLICATION                        │
│                  (Web/Mobile/CLI via REST API)                   │
└───────────────────────────────┬──────────────────────────────────┘
                                │
                                │ HTTP/JSON
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│                         FASTAPI SERVER                           │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  Request Validation (Pydantic)                             │  │
│  │  • 20 input features validated                             │  │
│  │  • Type checking & constraints                             │  │
│  └────────────────────────────────────────────────────────────┘  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  Preprocessing Pipeline                                    │  │
│  │  • OneHotEncoder (categorical → binary vectors)            │  │
│  │  • StandardScaler (normalize numerical features)           │  │
│  └────────────────────────────────────────────────────────────┘  │
└───────────────────────────────┬──────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────────────┐
│              MULTI-OUTPUT NEURAL NETWORK (TensorFlow)                 │
│                                                                       │
│  Input Layer (111 features after encoding)                            │
│       │                                                               │
│       ├─→ Dense(256, relu) ──────┐                                    │
│       │                          │                                    │
│       └─→ Dense(128, relu) ──────┤  Shared Feature Extraction         │
│                                  │                                    │
│              Dense(64, relu) ────┘                                    │
│                    │                                                  │
│       ┌────────────┼───────────┬───────────┬────────────┐             │
│       │            │           │           │            │             │
│       ▼            ▼           ▼           ▼            ▼             │
│   ┌───────┐  ┌────────┐  ┌──────────┐  ┌────────┐  ┌─────────┐        │
│   │Species│  │Family  │  │Edibility │  │Habitat │  │Season   │        │
│   │ Head  │  │ Head   │  │  Head    │  │ Head   │  │ Head    │        │
│   │(173)  │  │ (23)   │  │   (2)    │  │  (7)   │  │  (4)    │        │
│   └───────┘  └────────┘  └──────────┘  └────────┘  └─────────┘        │
│      │           │            │            │            │             │
│   softmax     softmax      sigmoid      softmax      softmax          │
│                                                                       │
│   Loss Weights: Species=1.0, Family=0.5, Edibility=2.0 (critical)     │
│                 Habitat=0.8, Season=0.8                               │
└───────────────────────────────┬───────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│                    POST-PROCESSING & RESPONSE                    │
│  • Confidence scores for all predictions                         │
│  • Risk assessment (SAFE/CAUTION/DANGER/CRITICAL)                │
│  • Lookalike warnings for dangerous species                      │
│  • Formatted JSON response                                       │
└──────────────────────────────────────────────────────────────────┘
```

### Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **API Framework** | FastAPI 0.104.1 | High-performance async REST API |
| **ML Framework** | TensorFlow 2.16.1 + Keras 3.12.0 | Deep learning model training and inference |
| **Validation** | Pydantic 2.5.0 | Request/response validation with type safety |
| **Data Processing** | Pandas 2.1.4, NumPy 1.26.2 | Data manipulation and numerical operations |
| **Preprocessing** | Scikit-learn 1.3.2 | OneHotEncoder, StandardScaler, train/test split |
| **Server** | Uvicorn (ASGI) | Production-grade async server |
| **API Docs** | OpenAPI/Swagger | Auto-generated interactive documentation |
| **Development** | Jupyter, pytest, black | Notebooks, testing, code formatting |

### Data Flow

1. **Input**: Client sends 20 mushroom features via POST request
2. **Validation**: Pydantic validates data types and constraints
3. **Encoding**: Categorical features → one-hot vectors (17 features → ~111 binary)
4. **Scaling**: Numerical features normalized using StandardScaler
5. **Inference**: Multi-output model predicts all 5 characteristics simultaneously
6. **Decoding**: Class indices converted back to human-readable labels
7. **Risk Analysis**: Safety checks performed on edibility + lookalike database
8. **Response**: Comprehensive JSON with predictions, confidence scores, warnings

---

## 📁 Project Structure

```
fungisense-ai/
│
├── 📂 api/                         # FastAPI application
│   ├── main.py                     # Main FastAPI app with CORS and middleware
│   ├── models/
│   │   └── schemas.py              # Pydantic models for validation
│   ├── routes/
│   │   ├── prediction.py           # Prediction endpoints
│   │   ├── health.py               # Health check endpoints
│   │   ├── model.py                # Model metadata endpoints
│   │   └── system.py               # System information endpoints
│   └── utils/                      # API utilities
│
├── 📂 src/                         # Core Python modules (reusable)
│   ├── data/
│   │   ├── loader.py               # Data loading utilities
│   │   └── preprocessor.py         # MushroomPreprocessor class
│   ├── models/
│   │   └── multi_output_model.py   # Neural network architecture
│   ├── features/                   # Feature engineering (future)
│   └── utils/                      # Shared utilities
│
├── 📂 notebooks/                   # Jupyter notebooks
│   ├── 01_data_exploration.ipynb   # EDA and visualization
│   ├── 02_model_training.ipynb     # Model development
│   └── 03_api_testing.ipynb        # API testing examples
│
├── 📂 scripts/                     # Executable scripts
│   └── train_model.py              # Model training pipeline
│
├── 📂 tests/                       # Unit and integration tests
│   ├── test_endpoints.py           # API endpoint tests
│   ├── test_data.py                # Data processing tests (future)
│   └── test_models.py              # Model tests (future)
│
├── 📂 config/                      # Configuration
│   └── config.py                   # Settings using pydantic-settings
│
├── 📂 data/                        # Data directory (gitignored)
│   ├── raw/                        # Raw datasets
│   └── processed/                  # Processed/cleaned data
│
├── 📂 dataset/                     # Original datasets
│   ├── primary_data.csv            # 173 species metadata
│   ├── primary_data_meta.txt       # Dataset documentation
│   ├── secondary_data.csv          # 61,069 training samples
│   └── secondary_data_meta.txt     # Training data documentation
│
├── 📂 models_saved/                # Trained models and encoders
│   ├── multi_output_model.h5       # Main neural network
│   ├── feature_encoder.pkl         # OneHotEncoder for features
│   ├── scaler.pkl                  # StandardScaler for normalization
│   ├── species_encoder.pkl         # LabelEncoder for species
│   ├── family_encoder.pkl          # LabelEncoder for families
│   ├── habitat_encoder.pkl         # LabelEncoder for habitats
│   ├── season_encoder.pkl          # LabelEncoder for seasons
│   └── feature_info.pkl            # Feature metadata
│
├── 📄 requirements.txt             # Python dependencies
├── 📄 Dockerfile                   # Docker configuration
├── 📄 docker-compose.yml           # Docker Compose configuration
├── 📄 .env.example                 # Environment variables template
├── 📄 .gitignore                   # Git ignore rules
│
├── 📄 README.md                    # This file
├── 📄 CONTRIBUTING.md              # Contribution guidelines
└── 📄 LICENSE                      # MIT License with safety disclaimer
```

### Key Directories Explained

| Directory | Purpose |
|-----------|---------|
| **`api/`** | Production API code using FastAPI framework |
| **`src/`** | Reusable Python modules for data processing and models |
| **`notebooks/`** | Research, exploration, and experimentation |
| **`scripts/`** | Command-line tools for training and utilities |
| **`tests/`** | Test suite for quality assurance |
| **`config/`** | Centralized configuration management |
| **`dataset/`** | Original training datasets (version controlled) |
| **`models_saved/`** | Trained models and preprocessing artifacts |

---

## 📊 Model Performance

### Expected Accuracy

Based on validation set performance with 61,069 training samples:

| Output | Accuracy | Notes |
|--------|----------|-------|
| **Edibility** | 98-100% | Binary classification, safety-critical |
| **Family** | 95-98% | Strong taxonomic patterns |
| **Species** | 85-92% | 173-class classification (complex) |
| **Habitat** | 80-85% | Good feature correlation |
| **Season** | 75-85% | Seasonal pattern recognition |

### Model Metrics

- **Input Features**: 20 original → 111 after one-hot encoding
- **Model Size**: ~2.45 MB (efficient for deployment)
- **Inference Time**: ~40-60ms per prediction on CPU
- **Batch Performance**: ~100 predictions in ~2-3 seconds
- **Training Time**: 5-10 minutes on CPU, 2-3 minutes on GPU

### Loss Function Configuration

The model uses weighted loss to prioritize safety:

```python
loss_weights = {
    'edibility': 2.0,    # Highest priority - safety critical
    'species': 1.0,      # Primary identification task
    'family': 0.5,       # Secondary taxonomic info
    'habitat': 0.8,      # Ecological context
    'season': 0.8        # Temporal context
}
```

---

## 🛠️ Development

### Setting Up Development Environment

```bash
# Clone repository
git clone https://github.com/paradocx96/fungisense-ai.git
cd fungisense-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt

# Install pre-commit hooks (optional)
pip install pre-commit
pre-commit install
```

### Code Quality

This project uses industry-standard tools:

- **Black**: Code formatting (`black .`)
- **Flake8**: Linting (`flake8 api/ src/ scripts/`)
- **pytest**: Testing framework
- **Type hints**: Throughout codebase for IDE support

```bash
# Format code
black api/ src/ scripts/

# Check code quality
flake8 api/ src/ scripts/ --max-line-length=100

# Run tests
pytest tests/ -v
```

### Running Jupyter Notebooks

```bash
# Start Jupyter server
jupyter notebook

# Notebooks are in notebooks/ directory:
# 1. 01_data_exploration.ipynb - Data analysis and visualization
# 2. 02_model_training.ipynb - Model development and tuning
# 3. 03_api_testing.ipynb - API testing examples
```

### Training Custom Models

```bash
# Modify config/config.py for hyperparameters
# Then run training
python scripts/train_model.py

# Monitor training progress
# Models saved to models_saved/
# Logs available in console output
```

### Adding New Endpoints

1. Create route in `api/routes/`
2. Define Pydantic schema in `api/models/schemas.py`
3. Register router in `api/main.py`
4. Add tests in `tests/test_endpoints.py`

Example:

```python
# api/routes/custom.py
from fastapi import APIRouter

router = APIRouter(prefix="/api/v1/custom", tags=["Custom"])

@router.post("/endpoint")
async def custom_endpoint(data: CustomRequest):
    # Implementation
    return {"result": "success"}
```

---

## 🐳 Docker Deployment

### Quick Start with Docker

```bash
# Build image
docker build -t fungisense-ai:latest .

# Run container
docker run -d \
  -p 8000:8000 \
  --name fungisense-api \
  fungisense-ai:latest

# Check logs
docker logs -f fungisense-api

# Stop container
docker stop fungisense-api
```

### Docker Compose (Recommended)

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f api

# Stop services
docker-compose down

# Rebuild after code changes
docker-compose up -d --build
```

### Production Deployment

For production, consider:

- Using environment variables for configuration
- Setting up reverse proxy (nginx)
- Enabling HTTPS with SSL certificates
- Implementing rate limiting
- Adding monitoring (Prometheus + Grafana)
- Using orchestration (Kubernetes, Docker Swarm)

---

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_endpoints.py -v

# Run with coverage report
pytest --cov=api --cov=src tests/

# Generate HTML coverage report
pytest --cov=api --cov=src --cov-report=html tests/
open htmlcov/index.html  # View in browser
```

### Test Structure

Tests follow the Arrange-Act-Assert pattern:

```python
def test_complete_prediction_success():
    """Test successful complete prediction with valid data."""
    # Arrange
    mushroom_data = {...}
    
    # Act
    response = client.post("/api/v1/predict/complete", json=mushroom_data)
    
    # Assert
    assert response.status_code == 200
    assert "species" in response.json()
```

### Manual Testing

Use the interactive Swagger UI at http://localhost:8000/docs to manually test all endpoints with example data.

---

## ⚠️ Safety & Disclaimer

### ⚠️ CRITICAL WARNING

**This system is for EDUCATIONAL and RESEARCH purposes ONLY.**

### DO NOT:
- ❌ **Eat wild mushrooms** based solely on AI predictions
- ❌ Use this as a **field identification guide**
- ❌ Assume predictions are **100% accurate**
- ❌ Risk your health or life based on these predictions

### DO:
- ✅ Use for **learning about mycology and machine learning**
- ✅ **Always consult expert mycologists** before consuming wild mushrooms
- ✅ Understand that **many edible species have deadly lookalikes**
- ✅ Treat all unknown mushrooms as **potentially poisonous**
- ✅ Use as a **supplementary tool** alongside expert knowledge

### Built-in Safety Features

This system includes multiple safety layers:

1. **Risk Assessment**: Every prediction includes risk levels (SAFE, CAUTION, DANGER, CRITICAL)
2. **Lookalike Warnings**: Alerts about dangerous species with similar characteristics
3. **Confidence Scores**: Low-confidence predictions flagged for extra caution
4. **Deadly Species Database**: Known lethal species trigger critical warnings
5. **Multi-Factor Analysis**: Considers multiple characteristics before prediction

### Known Limitations

- Training data is simulated based on real species characteristics
- Model has not been validated against field specimens
- Regional variations in species not accounted for
- Some rare species may not be well-represented
- Visual characteristics (images) not currently used

### Legal Disclaimer

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED. IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY ARISING FROM THE USE OF THIS SOFTWARE. **CONSUMING WILD MUSHROOMS BASED ON THIS SOFTWARE'S PREDICTIONS MAY RESULT IN SERIOUS INJURY OR DEATH.**

---

## 🤝 Contributing

We welcome contributions from the community! See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

### Quick Contribution Guide

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest`)
5. Format code (`black .`)
6. Commit (`git commit -m 'feat: add amazing feature'`)
7. Push to branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

### Areas Needing Contribution

- 🧪 **Testing**: Increase test coverage
- 📝 **Documentation**: Improve guides and examples
- 🎨 **Frontend**: Build web interface
- 📱 **Mobile**: iOS/Android apps
- 🌍 **Internationalization**: Multi-language support
- 📸 **Image Classification**: CNN-based visual identification
- 🗺️ **Geographic Data**: Region-specific species
- 🔬 **Model Improvements**: Better accuracy and performance

---

## 📄 License

This project is released under the **MIT License** with additional safety disclaimers. See [LICENSE](LICENSE) for details.

**⚠️ IMPORTANT**: This software is for educational purposes only. The authors assume **NO LIABILITY** for any harm resulting from the use of predictions made by this system.

---

## 🙏 Acknowledgments

### Data Sources

- **Primary Dataset**: Based on Patrick Hardin's *"Mushrooms & Toadstools"* reference work
- **Secondary Dataset**: Created by Dennis Wagner for the UCI Machine Learning Repository
- Inspired by the classic [UCI Mushroom Dataset](https://archive.ics.uci.edu/ml/datasets/mushroom)
- Secondary dataset [UCI Mushroom Dataset](https://archive.ics.uci.edu/dataset/848/secondary+mushroom+dataset)

### Technology Stack

Built with industry-leading open-source technologies:

- **[FastAPI](https://fastapi.tiangolo.com/)** - Modern, fast Python web framework
- **[TensorFlow](https://www.tensorflow.org/)** - Machine learning platform
- **[Keras](https://keras.io/)** - High-level neural networks API
- **[Scikit-learn](https://scikit-learn.org/)** - Machine learning library
- **[Pydantic](https://pydantic-docs.helpmanual.io/)** - Data validation using Python type hints
- **[Uvicorn](https://www.uvicorn.org/)** - Lightning-fast ASGI server

### Community

Special thanks to:
- The open-source Python and ML communities
- Contributors to scientific mushroom databases
- Early testers and feedback providers
- Everyone who stars and shares this project

---

## 📞 Support & Contact

### Documentation

- **API Docs**: http://localhost:8000/docs (when running)
- **Contributing**: [CONTRIBUTING.md](CONTRIBUTING.md)

### Getting Help

1. 📖 **Read the documentation** above
2. 🔍 **Search existing issues** on GitHub
3. 💬 **Ask questions** by opening a new issue
4. 📧 **Contact maintainers** for specific concerns

### Reporting Issues

When reporting bugs, please include:

- Python version (`python --version`)
- OS and version
- Full error message and stack trace
- Minimal code to reproduce the issue
- Expected vs actual behavior

---

## 🌟 Star History

If you find this project useful, please consider giving it a ⭐️ on GitHub!

[![Star History Chart](https://api.star-history.com/svg?repos=paradocx96/fungisense-ai&type=Date)](https://star-history.com/#paradocx96/fungisense-ai&Date)

---

## 📊 Project Status

**Current Version**: 2.0.0 (Multi-Output Production Release)

**Status**: ✅ Active Development

**Last Updated**: December 2025

### Roadmap

- ✅ **Phase 1**: Multi-output model with five predictions (COMPLETE)
- ✅ **Phase 2**: REST API with FastAPI (COMPLETE)
- 🚧 **Phase 3**: Image-based classification (IN PLANNING)
- 🚧 **Phase 4**: Web frontend interface (IN PLANNING)
- 🚧 **Phase 5**: Mobile application (IN PLANNING)

---

<div align="center">

*Remember: Always consult expert mycologists before consuming wild mushrooms. This tool is for education, not identification of edible species.*

**Built with ❤️ by Machine Learning Practitioner [paradocx96](https://github.com/paradocx96)**

[⬆ Back to Top](#-fungisense-ai)

---

</div>

