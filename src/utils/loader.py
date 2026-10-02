import pickle
import json
from pathlib import Path

import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split


def load_model(model_path: str | Path):
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
    return model


def load_features(features_path: str | Path):
    with open(features_path, 'r') as features_file:
        features = json.load(features_file)
    return features


def load_demographics(demographics_path: str | Path) -> dict:
    """Return {zipcode: {demographic column: value}} for constant-time lookup."""
    demographics = pd.read_csv(demographics_path, dtype={'zipcode': str})
    return demographics.set_index('zipcode').to_dict(orient='index')


def fit_imputer(sales_path: str | Path, house_features: list) -> KNNImputer:
    """Fit the KNN imputer selected in notebooks/imputation_experiment.ipynb.

    Same configuration as the notebook (n_neighbors=5, weights="distance", no
    scaling). Reference rows are the house attributes of the model's training
    split, reproduced with the same train_test_split call as create_model.py,
    so test rows never act as neighbors. Price, ids and dates are not loaded.
    """
    houses = pd.read_csv(sales_path, usecols=house_features)[house_features]
    train_houses, _ = train_test_split(houses, random_state=42)
    return KNNImputer(n_neighbors=5, weights="distance").fit(train_houses)
