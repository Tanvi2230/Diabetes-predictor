from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.utils import DATA_PATH, FEATURE_COLUMNS, TARGET_COLUMN, ZERO_AS_MISSING_COLUMNS


def load_dataset(dataset_path: Path | str = DATA_PATH) -> pd.DataFrame:
    """Load the diabetes dataset from CSV."""
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. Generate or place diabetes.csv first."
        )
    return pd.read_csv(dataset_path)


def clean_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Replace impossible zero values with missing values and drop duplicates."""
    cleaned = dataframe.copy()
    cleaned[ZERO_AS_MISSING_COLUMNS] = cleaned[ZERO_AS_MISSING_COLUMNS].replace(0, np.nan)
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    return cleaned


def split_features_target(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate predictors and target column."""
    return dataframe[FEATURE_COLUMNS], dataframe[TARGET_COLUMN]


def create_preprocessor(k_features: int = "all") -> Pipeline:
    """Build a preprocessing pipeline with imputation, scaling, and feature selection."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("selector", SelectKBest(score_func=mutual_info_classif, k=k_features)),
        ]
    )


def prepare_train_test_data(
    dataframe: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Produce a reproducible train-test split."""
    features, target = split_features_target(dataframe)
    return train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )
