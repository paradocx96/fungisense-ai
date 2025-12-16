"""
Multi-output neural network model for mushroom classification
"""
import os
import sys

import tensorflow as tf
from keras import layers, models
from tensorflow import keras

# Project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.config import settings


def create_multi_output_model(
        input_shape: int,
        num_species: int,
        num_families: int,
        num_habitats: int,
        num_seasons: int
) -> keras.Model:
    """
    Create a multi-output neural network model

    The model has:
    - Shared layers for feature extraction
    - 5 output heads for different predictions:
      1. Species (173 classes)
      2. Family (20-30 classes)
      3. Edibility (2 classes - binary)
      4. Habitat (7 classes)
      5. Season (4 classes)

    Args:
        input_shape: Number of input features
        num_species: Number of species classes
        num_families: Number of family classes
        num_habitats: Number of habitat classes
        num_seasons: Number of season classes

    Returns:
        Compiled Keras model
    """
    # Input layer
    input_layer = layers.Input(shape=(input_shape,), name='input')

    # Shared layers (feature extraction)
    x = layers.Dense(256, activation='relu', kernel_initializer='he_normal', name='shared_dense_1')(input_layer)
    x = layers.BatchNormalization(name='shared_bn_1')(x)
    x = layers.Dropout(0.3, name='shared_dropout_1')(x)

    x = layers.Dense(128, activation='relu', kernel_initializer='he_normal', name='shared_dense_2')(x)
    x = layers.BatchNormalization(name='shared_bn_2')(x)
    x = layers.Dropout(0.3, name='shared_dropout_2')(x)

    x = layers.Dense(64, activation='relu', kernel_initializer='he_normal', name='shared_dense_3')(x)
    x = layers.BatchNormalization(name='shared_bn_3')(x)
    x = layers.Dropout(0.2, name='shared_dropout_3')(x)

    # Output head 1: Species (most complex - 173 classes)
    species_branch = layers.Dense(128, activation='relu', name='species_dense_1')(x)
    species_branch = layers.Dropout(0.3, name='species_dropout')(species_branch)
    species_output = layers.Dense(
        num_species,
        activation='softmax',
        name='species_output'
    )(species_branch)

    # Output head 2: Family (20-30 classes)
    family_branch = layers.Dense(64, activation='relu', name='family_dense_1')(x)
    family_branch = layers.Dropout(0.2, name='family_dropout')(family_branch)
    family_output = layers.Dense(
        num_families,
        activation='softmax',
        name='family_output'
    )(family_branch)

    # Output head 3: Edibility (binary classification - most important!)
    edibility_branch = layers.Dense(32, activation='relu', name='edibility_dense_1')(x)
    edibility_branch = layers.Dropout(0.2, name='edibility_dropout')(edibility_branch)
    edibility_output = layers.Dense(
        1,
        activation='sigmoid',
        name='edibility_output'
    )(edibility_branch)

    # Output head 4: Habitat (7 classes)
    habitat_branch = layers.Dense(32, activation='relu', name='habitat_dense_1')(x)
    habitat_branch = layers.Dropout(0.2, name='habitat_dropout')(habitat_branch)
    habitat_output = layers.Dense(
        num_habitats,
        activation='softmax',
        name='habitat_output'
    )(habitat_branch)

    # Output head 5: Season (4 classes)
    season_branch = layers.Dense(16, activation='relu', name='season_dense_1')(x)
    season_branch = layers.Dropout(0.2, name='season_dropout')(season_branch)
    season_output = layers.Dense(
        num_seasons,
        activation='softmax',
        name='season_output'
    )(season_branch)

    # Create the model with multiple outputs
    model = models.Model(
        inputs=input_layer,
        outputs=[
            species_output,
            family_output,
            edibility_output,
            habitat_output,
            season_output
        ],
        name='mushroom_multi_output_model'
    )

    return model


def compile_model(model: keras.Model, learning_rate: float = 0.001) -> keras.Model:
    """
    Compile the model with appropriate loss functions and metrics

    Args:
        model: Keras model to compile
        learning_rate: Learning rate for optimizer

    Returns:
        Compiled model
    """
    # Define loss functions for each output
    losses = {
        'species_output': 'sparse_categorical_crossentropy',
        'family_output': 'sparse_categorical_crossentropy',
        'edibility_output': 'binary_crossentropy',
        'habitat_output': 'sparse_categorical_crossentropy',
        'season_output': 'sparse_categorical_crossentropy'
    }

    # Loss weights (edibility is most important for safety!)
    loss_weights = {
        'species_output': 1.0,
        'family_output': 0.8,
        'edibility_output': 2.0,  # Highest weight - safety critical
        'habitat_output': 0.5,
        'season_output': 0.5
    }

    # Metrics for each output
    metrics = {
        'species_output': ['accuracy'],
        'family_output': ['accuracy'],
        'edibility_output': ['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
        'habitat_output': ['accuracy'],
        'season_output': ['accuracy']
    }

    # Optimizer
    optimizer = keras.optimizers.Nadam(learning_rate=learning_rate)

    # Compile model
    model.compile(
        optimizer=optimizer,
        loss=losses,
        loss_weights=loss_weights,
        metrics=metrics
    )

    return model


def get_callbacks(patience_early_stopping: int = 10, patience_lr_reduction: int = 5) -> list:
    """
    Get callbacks for model training

    Args:
        patience_early_stopping: Patience for early stopping
        patience_lr_reduction: Patience for learning rate reduction

    Returns:
        List of callbacks
    """
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience_early_stopping,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=patience_lr_reduction,
            min_lr=1e-6,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(settings.MODEL_DIR, 'best_model.h5'),
            monitor='val_edibility_output_accuracy',  # Focus on edibility accuracy
            save_best_only=True,
            verbose=1
        )
    ]

    return callbacks


def print_model_summary(model: keras.Model):
    """
    Print detailed model summary

    Args:
        model: Keras model
    """
    print("=" * 80)
    print("MODEL ARCHITECTURE")
    print("=" * 80)
    model.summary()

    print("\n" + "=" * 80)
    print("MODEL OUTPUT SHAPES")
    print("=" * 80)
    for output in model.outputs:
        print(f"{output.name}: {output.shape}")

    print("\n" + "=" * 80)
    print("TOTAL PARAMETERS")
    print("=" * 80)
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    non_trainable_params = sum([tf.size(w).numpy() for w in model.non_trainable_weights])
    total_params = trainable_params + non_trainable_params

    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {non_trainable_params:,}")
    print(f"Total parameters: {total_params:,}")
    print("=" * 80)


if __name__ == "__main__":
    """
    Test model creation
    """
    print("=" * 80)
    print("Testing Model Creation")
    print("=" * 80)

    # Create the model with example parameters
    model = create_multi_output_model(
        input_shape=100,  # Example: 100 features after encoding
        num_species=173,  # 173 species
        num_families=30,  # ~30 families
        num_habitats=7,  # 7 habitat types
        num_seasons=4  # 4 seasons
    )

    # Compile model
    model = compile_model(model, learning_rate=0.001)

    # Print summary
    print_model_summary(model)

    print("\n Model creation test completed!")
