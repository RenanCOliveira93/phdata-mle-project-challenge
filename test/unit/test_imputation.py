"""A. Imputation functionality: KNN configuration, reference data and request handling."""
import logging
import math

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from api import endpoints
from api.endpoints import HOUSE_FEATURES
from conftest import DATA_DIR


@pytest.fixture
def imputer_calls(monkeypatch):
    """Record every DataFrame passed to the fitted imputer's transform()."""
    calls = []
    original_transform = endpoints.imputer.transform

    def spy(X):
        calls.append(X.copy())
        return original_transform(X)

    monkeypatch.setattr(endpoints.imputer, "transform", spy)
    return calls


def house_data():
    return pd.read_csv(DATA_DIR / "kc_house_data.csv", usecols=HOUSE_FEATURES)[HOUSE_FEATURES]


def test_imputer_matches_notebook_configuration():
    params = endpoints.imputer.get_params()
    assert params["n_neighbors"] == 5
    assert params["weights"] == "distance"
    assert params["metric"] == "nan_euclidean"
    # Only the 7 house attributes: no price, ids, zipcode or demographics
    assert list(endpoints.imputer.feature_names_in_) == HOUSE_FEATURES


def test_imputer_reference_data_is_model_training_split():
    """With every field missing KNNImputer returns reference means, which identify the reference rows."""
    houses = house_data()
    train, _ = train_test_split(houses, random_state=42)  # same split as create_model.py
    all_missing = pd.DataFrame([[np.nan] * len(HOUSE_FEATURES)], columns=HOUSE_FEATURES)

    imputed = endpoints.imputer.transform(all_missing)[0]

    np.testing.assert_allclose(imputed, train.mean().to_numpy())
    assert not np.allclose(imputed, houses.mean().to_numpy())  # not the full dataset


def test_imputer_leaves_complete_rows_unchanged(future_unseen_rows):
    complete = pd.DataFrame(future_unseen_rows)[HOUSE_FEATURES].astype(float)
    np.testing.assert_array_equal(endpoints.imputer.transform(complete), complete.to_numpy())


def test_imputer_fills_only_missing_cells_within_reference_range(future_unseen_rows):
    complete = pd.DataFrame(future_unseen_rows)[HOUSE_FEATURES].astype(float)
    mask = np.random.default_rng(0).random(complete.shape) < 0.3
    imputed = endpoints.imputer.transform(complete.mask(mask))

    assert not np.isnan(imputed).any()
    np.testing.assert_array_equal(imputed[~mask], complete.to_numpy()[~mask])
    # A weighted average of neighbors cannot leave the reference data's range
    houses = house_data()
    for col, feature in enumerate(HOUSE_FEATURES):
        values = imputed[mask[:, col], col]
        assert (values >= houses[feature].min()).all() and (values <= houses[feature].max()).all()


def test_complete_request_skips_imputer(test_client, sample_home_features, imputer_calls):
    assert test_client.post("/predict", json=sample_home_features).status_code == 200
    assert imputer_calls == []


@pytest.mark.parametrize(
    "missing_fields, send_as_null",
    [
        (["bathrooms"], True),
        (["bathrooms", "sqft_lot"], True),
        (["sqft_living", "floors", "sqft_basement"], False),
    ],
    ids=["one-null", "multiple-null", "multiple-omitted"],
)
def test_predict_with_missing_fields(test_client, sample_home_features, model_inputs,
                                     imputer_calls, missing_fields, send_as_null):
    payload = dict(sample_home_features)
    for field in missing_fields:
        if send_as_null:
            payload[field] = None
        else:
            del payload[field]

    response = test_client.post("/predict", json=payload)

    assert response.status_code == 200
    predicted = response.json()["predicted_price"]
    assert math.isfinite(predicted) and predicted > 0
    assert len(imputer_calls) == 1

    (model_input,) = model_inputs
    assert list(model_input.columns) == endpoints.model_features
    assert not model_input.isna().to_numpy().any()
    # Provided values reach the model unchanged; only missing ones are filled
    for field, value in payload.items():
        if field != "zipcode" and value is not None:
            assert model_input[field].iloc[0] == value


def test_predict_with_all_house_fields_missing(test_client, model_inputs, caplog):
    """Accepted: house fields fall back to reference means, and a warning is logged."""
    with caplog.at_level(logging.WARNING, logger="api.endpoints"):
        response = test_client.post("/predict", json={"zipcode": "98042"})

    assert response.status_code == 200
    assert math.isfinite(response.json()["predicted_price"])
    (model_input,) = model_inputs
    assert not model_input.isna().to_numpy().any()
    train, _ = train_test_split(house_data(), random_state=42)
    np.testing.assert_allclose(model_input[HOUSE_FEATURES].iloc[0], train.mean())
    assert any("Imputing" in r.message for r in caplog.records if r.levelno == logging.WARNING)
