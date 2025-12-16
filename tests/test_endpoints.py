"""
Test Enhanced Endpoints

This script tests all 4 endpoints:
1. Enhanced health check
2. Model metadata
3. Batch predictions
4. Top-N species predictions
"""
import requests

API_BASE_URL = "http://localhost:8000"

# Sample mushroom data
sample_mushroom = {
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


def test_enhanced_health_check():
    """Test enhanced health check endpoint"""
    print("\n" + "=" * 80)
    print("1. ENHANCED HEALTH CHECK")
    print("=" * 80)

    response = requests.get(f"{API_BASE_URL}/health/detailed")
    print(f"Status Code: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"✅ API Status: {data['status']}")
        print(f"✅ Model Loaded: {data['model_loaded']}")
        print(f"✅ Model Type: {data.get('model_type')}")
        print(f"✅ Uptime: {data['uptime_seconds']:.2f} seconds")
        print(f"✅ Total Predictions: {data['total_predictions']}")
        print(f"\nEncoder Status:")
        for encoder, loaded in data['encoders_loaded'].items():
            status = "✅" if loaded else "❌"
            print(f"  {status} {encoder}: {loaded}")
    else:
        print(f"❌ Error: {response.text}")


def test_model_metadata():
    """Test model metadata endpoint"""
    print("\n" + "=" * 80)
    print("2. MODEL METADATA")
    print("=" * 80)

    response = requests.get(f"{API_BASE_URL}/api/v1/model/info")
    print(f"Status Code: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"✅ Model Version: {data['model_version']}")
        print(f"✅ Model Type: {data['model_type']}")
        print(f"✅ Number of Species: {data['num_species']}")
        print(f"✅ Number of Families: {data['num_families']}")
        print(f"✅ Number of Habitats: {data['num_habitats']}")
        print(f"✅ Number of Seasons: {data['num_seasons']}")
        print(f"✅ Input Features: {data['input_features']}")
        print(f"✅ Model Size: {data.get('model_size_mb', 'N/A')} MB")
    else:
        print(f"❌ Error: {response.text}")


def test_batch_predictions():
    """Test batch prediction endpoint"""
    print("\n" + "=" * 80)
    print("3. BATCH PREDICTIONS")
    print("=" * 80)

    # Create a batch of 3 mushrooms with slight variations
    batch_request = {
        "mushrooms": [
            sample_mushroom,
            {**sample_mushroom, "cap_color": "red", "cap_shape": "convex"},
            {**sample_mushroom, "cap_color": "white", "habitat": "grasses"}
        ],
        "include_risk_assessment": True
    }

    response = requests.post(
        f"{API_BASE_URL}/api/v1/predict/batch",
        json=batch_request
    )
    print(f"Status Code: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        summary = data['summary']

        print(f"\n✅ Batch Prediction Summary:")
        print(f"  Total Mushrooms: {summary['total_mushrooms']}")
        print(f"  Unique Species: {summary['unique_species']}")
        print(f"  Most Common: {summary['most_common_species']}")
        print(f"  Edible: {summary['edible_count']}")
        print(f"  Poisonous: {summary['poisonous_count']}")
        print(f"  Processing Time: {data['metadata']['processing_time_ms']:.2f}ms")

        print(f"\n✅ Individual Predictions:")
        for i, pred in enumerate(data['predictions'], 1):
            print(f"  {i}. {pred['species']['name']} - {pred['edibility']['class_name']} "
                  f"(Confidence: {pred['species']['confidence']:.2%})")
    else:
        print(f"❌ Error: {response.text}")


def test_top_n_species():
    """Test top-N species prediction endpoint"""
    print("\n" + "=" * 80)
    print("4. TOP-N SPECIES PREDICTIONS")
    print("=" * 80)

    response = requests.post(
        f"{API_BASE_URL}/api/v1/predict/top-species?top_n=5",
        json=sample_mushroom
    )
    print(f"Status Code: {response.status_code}")

    if response.status_code == 200:
        data = response.json()

        print(f"\n✅ Top 5 Most Likely Species:")
        for i, pred in enumerate(data['top_predictions'], 1):
            print(f"  {i}. {pred['species']}")
            print(f"     Family: {pred['family']}")
            print(f"     Edibility: {pred['edibility']}")
            print(f"     Probability: {pred['probability']:.2%}")
            print(f"     Confidence: {pred['confidence']:.2%}")
            print()

        print(f"Processing Time: {data['metadata']['processing_time_ms']:.2f}ms")
    else:
        print(f"❌ Error: {response.text}")


if __name__ == "__main__":
    print("\n" + "🍄" * 40)
    print("TESTING PHASE 1 ENHANCED ENDPOINTS")
    print("🍄" * 40)

    try:
        # Test all endpoints
        test_enhanced_health_check()
        test_model_metadata()
        test_batch_predictions()
        test_top_n_species()

        print("\n" + "=" * 80)
        print("✅ ALL TESTS COMPLETED")
        print("=" * 80)

    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Could not connect to API")
        print("   Make sure the API is running: python api/main.py")
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
