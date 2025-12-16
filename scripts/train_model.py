"""
Training script for the multioutput mushroom classification model

This script:
1. Loads and preprocesses the data
2. Creates the multi-output model
3. Trains the model
4. Evaluates performance
5. Saves the model and encoders
"""
import os
import sys
from pathlib import Path

import numpy as np

# Project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loader import (
    load_secondary_data,
    load_primary_data,
    clean_column_names,
    split_features_targets,
    map_species_to_secondary
)
from src.data.preprocessor import MushroomPreprocessor, create_train_val_test_split
from src.models.multi_output_model import (
    create_multi_output_model,
    compile_model,
    print_model_summary
)
from config.config import settings
import tensorflow as tf

# Set random seeds for reproducibility
np.random.seed(settings.RANDOM_SEED)
tf.random.set_seed(settings.RANDOM_SEED)

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():
    """
    Main training pipeline
    """
    print("=" * 80)
    print(" FungiSense AI - MULTI-OUTPUT MODEL TRAINING")
    print("=" * 80)

    # Step 1: Load Data
    print("\n STEP 1: Loading Data...")
    print("-" * 80)

    # Load primary data (species information)
    primary_df = load_primary_data()
    primary_df = clean_column_names(primary_df)

    # Load secondary data (training samples)
    secondary_df = load_secondary_data()
    secondary_df = clean_column_names(secondary_df)

    # Map species and family to secondary data
    df = map_species_to_secondary(primary_df, secondary_df)

    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # Step 2: Prepare Features and Targets
    print("\n STEP 2: Splitting Features and Targets...")
    print("-" * 80)
    X, y_dict = split_features_targets(df)

    # Add all target columns (species and family)
    if 'species' in df.columns:
        y_dict['species'] = df['species'].copy()
    if 'family' in df.columns:
        y_dict['family'] = df['family'].copy()
    if 'habitat' in df.columns:
        y_dict['habitat'] = df['habitat'].copy()
    if 'season' in df.columns:
        y_dict['season'] = df['season'].copy()

    print(f"Features shape: {X.shape}")
    print(f"Targets available: {list(y_dict.keys())}")
    for target_name, target_data in y_dict.items():
        print(f"  - {target_name}: {target_data.nunique()} unique classes")

    # Step 3: Preprocess Data
    print("\n STEP 3: Preprocessing Data...")
    print("-" * 80)
    preprocessor = MushroomPreprocessor()
    preprocessor.fit(X, y_dict)

    # Transform features
    X_transformed = preprocessor.transform_features(X)
    print(f"Original features: {X.shape}")
    print(f"Transformed features: {X_transformed.shape}")

    # Transform targets
    y_transformed = preprocessor.transform_targets(y_dict)

    # Step 4: Create the Train / Val / Test Split
    print("\n STEP 4: Creating Train / Val / Test Split...")
    print("-" * 80)
    X_train, X_val, X_test, y_train_dict, y_val_dict, y_test_dict = create_train_val_test_split(
        X_transformed,
        y_transformed,
        test_size=settings.TEST_SIZE,
        val_size=settings.VALIDATION_SIZE,
        random_state=settings.RANDOM_SEED
    )

    # Step 5: Create Model
    print("\n STEP 5: Creating Multi-Output Model...")
    print("-" * 80)

    # Get input shape
    input_shape = X_train.shape[1]
    print(f"Input shape: {input_shape}")

    # Get a number of classes for each output
    num_species = len(preprocessor.species_encoder.classes_) if hasattr(preprocessor.species_encoder, 'classes_') else 2
    num_families = len(preprocessor.family_encoder.classes_) if hasattr(preprocessor.family_encoder, 'classes_') else 2
    num_habitats = len(preprocessor.habitat_encoder.classes_) if hasattr(preprocessor.habitat_encoder, 'classes_') else 7
    num_seasons = len(preprocessor.season_encoder.classes_) if hasattr(preprocessor.season_encoder, 'classes_') else 4

    print(f"Output classes:")
    print(f"  - Species: {num_species}")
    print(f"  - Families: {num_families}")
    print(f"  - Habitats: {num_habitats}")
    print(f"  - Seasons: {num_seasons}")

    # Check required all targets for a full multi-output model
    has_all_targets = all(key in y_train_dict for key in ['species', 'family', 'edibility', 'habitat', 'season'])

    if has_all_targets:
        print("\n Creating full multi-output model with all 5 targets")
        model = create_multi_output_model(
            input_shape=input_shape,
            num_species=num_species,
            num_families=num_families,
            num_habitats=num_habitats,
            num_seasons=num_seasons
        )
        # Compile the multi-output model
        model = compile_model(model, learning_rate=settings.LEARNING_RATE)
    else:
        # Fallback to a simplified edibility-only model
        print("\n Creating simplified edibility-only model")
        from tensorflow import keras
        from keras import layers

        model = keras.Sequential([
            layers.Input(shape=(input_shape,)),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(1, activation='sigmoid', name='edibility_output')
        ], name='mushroom_edibility_model')

        model.compile(
            optimizer=keras.optimizers.Nadam(learning_rate=settings.LEARNING_RATE),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )

    print_model_summary(model)

    # Step 6: Train Model
    print("\n STEP 6: Training Model...")
    print("-" * 80)

    # Create a model directory (if it doesn't exist)
    model_dir = PROJECT_ROOT / settings.MODEL_DIR
    os.makedirs(model_dir, exist_ok=True)

    # Callbacks
    from tensorflow import keras
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]

    # Train based on the model type
    if has_all_targets:
        # Multi-output training
        print("Training multi-output model...")
        history = model.fit(
            X_train,
            {
                'species_output': y_train_dict['species'],
                'family_output': y_train_dict['family'],
                'edibility_output': y_train_dict['edibility'],
                'habitat_output': y_train_dict['habitat'],
                'season_output': y_train_dict['season']
            },
            validation_data=(
                X_val,
                {
                    'species_output': y_val_dict['species'],
                    'family_output': y_val_dict['family'],
                    'edibility_output': y_val_dict['edibility'],
                    'habitat_output': y_val_dict['habitat'],
                    'season_output': y_val_dict['season']
                }
            ),
            epochs=settings.EPOCHS,
            batch_size=settings.BATCH_SIZE,
            callbacks=callbacks,
            verbose=1
        )
    else:
        # Single-output training (edibility only)
        print("Training edibility-only model...")
        history = model.fit(
            X_train,
            y_train_dict['edibility'],
            validation_data=(X_val, y_val_dict['edibility']),
            epochs=settings.EPOCHS,
            batch_size=settings.BATCH_SIZE,
            callbacks=callbacks,
            verbose=1
        )

    # Step 7: Evaluate Model
    print("\n STEP 7: Evaluating Model...")
    print("-" * 80)

    # Evaluate on the test set
    if has_all_targets:
        # Multi-output evaluation
        test_results = model.evaluate(
            X_test,
            {
                'species_output': y_test_dict['species'],
                'family_output': y_test_dict['family'],
                'edibility_output': y_test_dict['edibility'],
                'habitat_output': y_test_dict['habitat'],
                'season_output': y_test_dict['season']
            },
            verbose=0
        )

        print("Multi-output test results:")
        print(f"Overall Loss: {test_results[0]:.4f}")
        # Additional metrics for each output would be in test_results[1:]
        print("(See training logs for detailed metrics per output)")
    else:
        # Single-output evaluation
        test_results = model.evaluate(X_test, y_test_dict['edibility'], verbose=0)

        print(f"Test Loss: {test_results[0]:.4f}")
        print(f"Test Accuracy: {test_results[1]:.4f}")
        print(f"Test Precision: {test_results[2]:.4f}")
        print(f"Test Recall: {test_results[3]:.4f}")

    # Step 8: Save Model and Preprocessors
    print("\n STEP 8: Saving Model and Preprocessors...")
    print("-" * 80)

    # Save the model
    if has_all_targets:
        model_filename = 'multi_output_model.h5'
    else:
        # For API compatibility
        model_filename = 'mushroom_classifier.h5'

    model_path = model_dir / model_filename
    model.save(str(model_path))
    print(f"✅ Model saved to: {model_path}")

    # Save preprocessor
    preprocessor.save(str(model_dir))
    print(f"✅ Preprocessors saved to: {model_dir}")

    # Final Summary
    print("\n" + "=" * 80)
    print(" TRAINING COMPLETE!")
    print("=" * 80)
    if has_all_targets:
        print(f"\n✅ Multi-output model trained successfully!")
        print(f"✅ Model predicts: Species, Family, Edibility, Habitat, Season")
        print(f"✅ Overall Test Loss: {test_results[0]:.4f}")
    else:
        print(f"\n✅ Edibility model trained successfully!")
        print(f"✅ Test Accuracy: {test_results[1]:.2%}")
    print("=" * 80)


if __name__ == "__main__":
    main()
