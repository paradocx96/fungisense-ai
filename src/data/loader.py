"""
Data loading utilities for mushroom dataset
"""
import os
import sys
from typing import Tuple

import pandas as pd

# Project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.config import settings


def load_primary_data() -> pd.DataFrame:
    """
    Load primary mushroom data

    Returns:
        DataFrame with primary mushroom data
    """
    try:
        df = pd.read_csv(settings.PRIMARY_DATA_PATH, delimiter=';')
        print(f"Loaded primary data: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {settings.PRIMARY_DATA_PATH}")
        raise
    except Exception as e:
        print(f"Error loading primary data: {str(e)}")
        raise


def load_secondary_data() -> pd.DataFrame:
    """
    Load secondary mushroom data

    This is the main dataset for training the model.

    Returns:
        DataFrame with secondary mushroom data
    """
    try:
        df = pd.read_csv(settings.SECONDARY_DATA_PATH, delimiter=';')
        print(f"Loaded secondary data: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {settings.SECONDARY_DATA_PATH}")
        raise
    except Exception as e:
        print(f"Error loading secondary data: {str(e)}")
        raise


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean column names to lowercase and replace hyphens with underscores

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with cleaned column names
    """
    df.columns = df.columns.str.lower().str.replace('-', '_')
    return df


def get_data_info(df: pd.DataFrame) -> dict:
    """
    Get basic information about the dataset

    Args:
        df: Input DataFrame

    Returns:
        Dictionary with dataset statistics
    """
    info = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "unique_counts": {col: df[col].nunique() for col in df.columns},
        "class_distribution": df['class'].value_counts().to_dict() if 'class' in df.columns else None
    }
    return info


def split_features_targets(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Split the dataframe into features (X) and multiple targets (y)

    Args:
        df: Input DataFrame with all columns

    Returns:
        Tuple of (X, y_dict) where y_dict contains multiple target variables
    """
    # Features (excluding target columns)
    feature_cols = [
        'cap_diameter', 'cap_shape', 'cap_surface', 'cap_color',
        'does_bruise_or_bleed', 'gill_attachment', 'gill_spacing', 'gill_color',
        'stem_height', 'stem_width', 'stem_root', 'stem_surface', 'stem_color',
        'veil_type', 'veil_color', 'has_ring', 'ring_type',
        'spore_print_color', 'habitat', 'season'
    ]

    # Check if all feature columns exist
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        print(f"    Warning: Missing columns: {missing_cols}")

    available_feature_cols = [col for col in feature_cols if col in df.columns]

    X = df[available_feature_cols].copy()

    # Multiple targets for multi-output model
    y_dict = {}

    # For secondary data, we only have 'class' (edibility)
    # For training multi-output model, we need to join with primary data
    if 'class' in df.columns:
        y_dict['edibility'] = df['class'].copy()

    # These will be populated from primary data during preprocessing
    # y_dict['species'] = ...
    # y_dict['family'] = ...
    # y_dict['habitat'] = ...
    # y_dict['season'] = ...

    return X, y_dict


def map_species_to_secondary(primary_df: pd.DataFrame, secondary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Map species and family information from primary to secondary data

    The secondary dataset contains 353 samples per species in the same order
    as the primary dataset (173 species * 353 samples = 61,069 total).

    Args:
        primary_df: Primary data with species information
        secondary_df: Secondary data with generated samples

    Returns:
        Secondary dataframe with added species/family columns
    """
    samples_per_species = 353
    num_species = len(primary_df)
    expected_samples = num_species * samples_per_species

    if len(secondary_df) != expected_samples:
        print(f"    Warning: Expected {expected_samples} samples but got {len(secondary_df)}")
        print(f"    Species mapping may be incorrect.")

    # Create lists to hold species and family labels
    species_labels = []
    family_labels = []

    # Assign species and family to each batch of 353 samples
    for idx, row in primary_df.iterrows():
        species_name = row['name']
        family_name = row['family']

        # Repeat species/family labels 353 times
        species_labels.extend([species_name] * samples_per_species)
        family_labels.extend([family_name] * samples_per_species)

    # Trim to actual secondary data length (in case of mismatch)
    species_labels = species_labels[:len(secondary_df)]
    family_labels = family_labels[:len(secondary_df)]

    # Add columns to secondary dataframe
    secondary_df = secondary_df.copy()
    secondary_df['species'] = species_labels
    secondary_df['family'] = family_labels

    print(f"Mapped {num_species} species to {len(secondary_df)} samples")
    print(f"    Unique species: {secondary_df['species'].nunique()}")
    print(f"    Unique families: {secondary_df['family'].nunique()}")

    return secondary_df


def get_feature_types(df: pd.DataFrame) -> Tuple[list, list]:
    """
    Separate numerical and categorical features

    Args:
        df: Input DataFrame

    Returns:
        Tuple of (numerical_features, categorical_features)
    """
    numerical_features = ['cap_diameter', 'stem_height', 'stem_width']
    categorical_features = [
        col for col in df.columns
        if col not in numerical_features and col not in ['class', 'family', 'name']
    ]

    return numerical_features, categorical_features


if __name__ == "__main__":
    """
    Test data loading
    """
    print("=" * 60)
    print("Testing Data Loading")
    print("=" * 60)

    # Load primary data
    print("\n Loading primary data...")
    primary = load_primary_data()
    print(primary.head())

    # Load secondary data
    print("\n Loading secondary data...")
    secondary = load_secondary_data()
    secondary = clean_column_names(secondary)
    print(secondary.head())

    # Get data info
    print("\n Secondary Data Info:")
    info = get_data_info(secondary)
    print(f"Shape: {info['shape']}")
    print(f"Columns: {info['columns']}")
    print(f"Class distribution: {info['class_distribution']}")

    # Split features and targets
    print("\n Splitting features and targets...")
    X, y_dict = split_features_targets(secondary)
    print(f"Features shape: {X.shape}")
    print(f"Targets: {list(y_dict.keys())}")

    print("\nData loading tests completed!")
