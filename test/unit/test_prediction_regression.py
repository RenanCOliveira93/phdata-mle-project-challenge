"""B. Prediction non-regression: parity with the original API, lookup integrity, determinism.

future_unseen_examples.csv has no price column, so these tests check consistency
with the original API, not prediction accuracy.
"""
import math

import pandas as pd
import pytest

from api import endpoints
from api.endpoints import HOUSE_FEATURES
from conftest import DATA_DIR


def test_complete_requests_match_original_api(test_client, future_unseen_rows,
                                              baseline_predictions, model_inputs):
    """All 100 future_unseen rows give exactly the original API's predictions."""
    predictions = [
        test_client.post("/predict", json=row).json()["predicted_price"]
        for row in future_unseen_rows
    ]

    assert predictions == baseline_predictions
    assert all(math.isfinite(p) for p in predictions)
    for model_input in model_inputs:
        assert list(model_input.columns) == endpoints.model_features
        assert not model_input.isna().to_numpy().any()


def test_demographics_lookup_matches_csv():
    csv = pd.read_csv(DATA_DIR / "zipcode_demographics.csv", dtype={"zipcode": str})
    assert csv["zipcode"].is_unique
    assert set(endpoints.demographics) == set(csv["zipcode"])
    for row in csv.to_dict(orient="records"):
        zipcode = row.pop("zipcode")
        assert endpoints.demographics[zipcode] == row
    # Every model feature is either a house attribute or a demographic column
    assert endpoints.model_features == HOUSE_FEATURES + [c for c in csv.columns if c != "zipcode"]


def test_every_dataset_zipcode_has_demographics():
    for name in ["kc_house_data.csv", "future_unseen_examples.csv"]:
        zipcodes = pd.read_csv(DATA_DIR / name, usecols=["zipcode"], dtype={"zipcode": str})["zipcode"]
        assert set(zipcodes) <= set(endpoints.demographics), name


def test_model_receives_demographics_of_requested_zipcode(test_client, sample_home_features,
                                                          model_inputs):
    test_client.post("/predict", json=sample_home_features)
    (model_input,) = model_inputs
    expected = endpoints.demographics[sample_home_features["zipcode"]]
    assert model_input[list(expected)].iloc[0].to_dict() == expected


@pytest.mark.parametrize("missing", [[], ["bathrooms", "sqft_lot"]], ids=["complete", "incomplete"])
def test_predictions_are_deterministic(test_client, sample_home_features, missing):
    payload = dict(sample_home_features, **{field: None for field in missing})
    predictions = {test_client.post("/predict", json=payload).json()["predicted_price"]
                   for _ in range(3)}
    assert len(predictions) == 1
