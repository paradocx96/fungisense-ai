"""
Data preprocessing utilities for mushroom dataset
"""
import os
import pickle
import sys
from typing import Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

# Project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class MushroomPreprocessor:
    """
    Preprocessing pipeline for mushroom data
    """

    def __init__(self):
        self.feature_encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
        self.scaler = StandardScaler()

        # Label encoders for each target
        self.edibility_encoder = LabelEncoder()
        self.species_encoder = LabelEncoder()
        self.family_encoder = LabelEncoder()
        self.habitat_encoder = LabelEncoder()
        self.season_encoder = LabelEncoder()

        self.numerical_features = ['cap_diameter', 'stem_height', 'stem_width']
        self.categorical_features = None
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y_dict: Dict[str, pd.Series] = None):
        """
        Fit preprocessors on training data

        Args:
            X: Features DataFrame
            y_dict: Dictionary of target Series
        """
        print(" Fitting preprocessors...")

        # Identify categorical features
        self.categorical_features = [
            col for col in X.columns if col not in self.numerical_features
        ]

        # Fit feature encoder on categorical features
        if self.categorical_features:
            self.feature_encoder.fit(X[self.categorical_features])

        # Fit scaler on numerical features
        if self.numerical_features:
            numerical_data = X[self.numerical_features]
            self.scaler.fit(numerical_data)

        # Fit label encoders on targets
        if y_dict:
            if 'edibility' in y_dict:
                self.edibility_encoder.fit(y_dict['edibility'])

            if 'species' in y_dict:
                self.species_encoder.fit(y_dict['species'])

            if 'family' in y_dict:
                self.family_encoder.fit(y_dict['family'])

            if 'habitat' in y_dict:
                self.habitat_encoder.fit(y_dict['habitat'])

            if 'season' in y_dict:
                self.season_encoder.fit(y_dict['season'])

        self.is_fitted = True
        print(" Preprocessors fitted successfully!")

    def transform_features(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform features using fitted preprocessors

        Args:
            X: Features DataFrame

        Returns:
            Transformed features as a numpy array
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted. Call fit() first.")

        transformed_parts = []

        # Transform numerical features
        if self.numerical_features:
            numerical_data = X[self.numerical_features]
            numerical_scaled = self.scaler.transform(numerical_data)
            transformed_parts.append(numerical_scaled)

        # Transform categorical features
        if self.categorical_features:
            categorical_data = X[self.categorical_features]
            categorical_encoded = self.feature_encoder.transform(categorical_data)
            transformed_parts.append(categorical_encoded)

        # Concatenate all features
        if len(transformed_parts) > 0:
            X_transformed = np.hstack(transformed_parts)
        else:
            X_transformed = np.array([])

        return X_transformed

    def transform_targets(self, y_dict: Dict[str, pd.Series]) -> Dict[str, np.ndarray]:
        """
        Transform target variables using fitted label encoders

        Args:
            y_dict: Dictionary of target Series

        Returns:
            Dictionary of transformed target arrays
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted. Call fit() first.")

        y_transformed = {}

        if 'edibility' in y_dict:
            y_transformed['edibility'] = self.edibility_encoder.transform(y_dict['edibility'])

        if 'species' in y_dict:
            y_transformed['species'] = self.species_encoder.transform(y_dict['species'])

        if 'family' in y_dict:
            y_transformed['family'] = self.family_encoder.transform(y_dict['family'])

        if 'habitat' in y_dict:
            y_transformed['habitat'] = self.habitat_encoder.transform(y_dict['habitat'])

        if 'season' in y_dict:
            y_transformed['season'] = self.season_encoder.transform(y_dict['season'])

        return y_transformed

    def inverse_transform_targets(self, y_pred_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Convert predictions back to original labels

        Args:
            y_pred_dict: Dictionary of prediction arrays (class indices)

        Returns:
            Dictionary of original label arrays
        """
        y_original = {}

        if 'edibility' in y_pred_dict:
            y_original['edibility'] = self.edibility_encoder.inverse_transform(
                y_pred_dict['edibility']
            )

        if 'species' in y_pred_dict:
            y_original['species'] = self.species_encoder.inverse_transform(
                y_pred_dict['species']
            )

        if 'family' in y_pred_dict:
            y_original['family'] = self.family_encoder.inverse_transform(
                y_pred_dict['family']
            )

        if 'habitat' in y_pred_dict:
            y_original['habitat'] = self.habitat_encoder.inverse_transform(
                y_pred_dict['habitat']
            )

        if 'season' in y_pred_dict:
            y_original['season'] = self.season_encoder.inverse_transform(
                y_pred_dict['season']
            )

        return y_original

    def save(self, directory: str):
        """
        Save all preprocessors to disk

        Args:
            directory: Directory to save preprocessors
        """
        os.makedirs(directory, exist_ok=True)

        # Save encoders
        with open(os.path.join(directory, 'feature_encoder.pkl'), 'wb') as f:
            pickle.dump(self.feature_encoder, f)

        with open(os.path.join(directory, 'scaler.pkl'), 'wb') as f:
            pickle.dump(self.scaler, f)

        with open(os.path.join(directory, 'edibility_encoder.pkl'), 'wb') as f:
            pickle.dump(self.edibility_encoder, f)

        with open(os.path.join(directory, 'species_encoder.pkl'), 'wb') as f:
            pickle.dump(self.species_encoder, f)

        with open(os.path.join(directory, 'family_encoder.pkl'), 'wb') as f:
            pickle.dump(self.family_encoder, f)

        with open(os.path.join(directory, 'habitat_encoder.pkl'), 'wb') as f:
            pickle.dump(self.habitat_encoder, f)

        with open(os.path.join(directory, 'season_encoder.pkl'), 'wb') as f:
            pickle.dump(self.season_encoder, f)

        # Save feature info (column names for API)
        feature_info = {
            'numerical_cols': self.numerical_features,
            'categorical_cols': self.categorical_features
        }
        with open(os.path.join(directory, 'feature_info.pkl'), 'wb') as f:
            pickle.dump(feature_info, f)

        print(f" Preprocessors saved to {directory}")

    @classmethod
    def load(cls, directory: str):
        """
        Load preprocessors from disk

        Args:
            directory: Directory containing saved preprocessors

        Returns:
            Loaded MushroomPreprocessor instance
        """
        preprocessor = cls()

        # Load encoders
        with open(os.path.join(directory, 'feature_encoder.pkl'), 'rb') as f:
            preprocessor.feature_encoder = pickle.load(f)

        with open(os.path.join(directory, 'scaler.pkl'), 'rb') as f:
            preprocessor.scaler = pickle.load(f)

        with open(os.path.join(directory, 'edibility_encoder.pkl'), 'rb') as f:
            preprocessor.edibility_encoder = pickle.load(f)

        with open(os.path.join(directory, 'species_encoder.pkl'), 'rb') as f:
            preprocessor.species_encoder = pickle.load(f)

        with open(os.path.join(directory, 'family_encoder.pkl'), 'rb') as f:
            preprocessor.family_encoder = pickle.load(f)

        with open(os.path.join(directory, 'habitat_encoder.pkl'), 'rb') as f:
            preprocessor.habitat_encoder = pickle.load(f)

        with open(os.path.join(directory, 'season_encoder.pkl'), 'rb') as f:
            preprocessor.season_encoder = pickle.load(f)

        preprocessor.is_fitted = True
        print(f" Preprocessors loaded from {directory}")

        return preprocessor


def create_train_val_test_split(
        X: np.ndarray,
        y_dict: Dict[str, np.ndarray],
        test_size: float = 0.15,
        val_size: float = 0.15,
        random_state: int = 42
) -> Tuple:
    """
    Split data into train, validation, and test sets

    Args:
        X: Features array
        y_dict: Dictionary of target arrays
        test_size: Proportion of data for the test set
        val_size: Proportion of training data for the validation set
        random_state: Random seed

    Returns:
        Tuple of (X_train, X_val, X_test, y_train_dict, y_val_dict, y_test_dict)
    """
    # First split: separate test set
    X_temp, X_test, y_temp_dict, y_test_dict = {}, {}, {}, {}

    # Get one target for stratification (edibility is most important)
    stratify_target = y_dict.get('edibility', None)

    # Split features
    X_temp, X_test = train_test_split(
        X,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_target
    )

    # Split all targets
    for key, y in y_dict.items():
        y_temp, y_test_dict[key] = train_test_split(
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_target
        )
        y_temp_dict[key] = y_temp

    # Second split: separate validation from training
    stratify_temp = y_temp_dict.get('edibility', None)

    X_train, X_val = train_test_split(
        X_temp,
        test_size=val_size / (1 - test_size),
        random_state=random_state,
        stratify=stratify_temp
    )

    y_train_dict, y_val_dict = {}, {}
    for key, y_temp in y_temp_dict.items():
        y_train_dict[key], y_val_dict[key] = train_test_split(
            y_temp,
            test_size=val_size / (1 - test_size),
            random_state=random_state,
            stratify=stratify_temp
        )

    print(f" Data split:")
    print(f"   Train: {X_train.shape[0]} samples")
    print(f"   Validation: {X_val.shape[0]} samples")
    print(f"   Test: {X_test.shape[0]} samples")

    return X_train, X_val, X_test, y_train_dict, y_val_dict, y_test_dict


if __name__ == "__main__":
    """
    Test preprocessing
    """
    print("=" * 60)
    print("Testing Preprocessing")
    print("=" * 60)

    from src.data.loader import load_secondary_data, clean_column_names, split_features_targets

    # Load data
    df = load_secondary_data()
    df = clean_column_names(df)
    X, y_dict = split_features_targets(df)

    # Create preprocessor
    print("\n Creating preprocessor...")
    preprocessor = MushroomPreprocessor()
    preprocessor.fit(X, y_dict)

    # Transform features
    print("\n Transforming features...")
    X_transformed = preprocessor.transform_features(X)
    print(f"Original shape: {X.shape}")
    print(f"Transformed shape: {X_transformed.shape}")

    # Transform targets
    print("\n Transforming targets...")
    y_transformed = preprocessor.transform_targets(y_dict)
    for key, y in y_transformed.items():
        print(f"{key}: {y.shape}, unique classes: {len(np.unique(y))}")

    print("\n Preprocessing tests completed!")
