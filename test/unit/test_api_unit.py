"""C. API robustness: health, request validation and clear errors."""
import json

import pandas as pd
import pytest

from conftest import DATA_DIR


def post_raw(test_client, body: str):
    return test_client.post("/predict", content=body,
                            headers={"Content-Type": "application/json"})


def test_health_endpoint(test_client):
    """Test the /health endpoint returns correct status."""
    response = test_client.get("/health")
    assert response.status_code == 200
    response_data = response.json()
    assert "status" in response_data
    assert response_data["status"] == "healthy"


def test_predict_endpoint_valid_input(test_client, sample_home_features):
    """Test the /predict endpoint with valid input."""
    response = test_client.post("/predict", json=sample_home_features)
    assert response.status_code == 200
    response_data = response.json()
    assert "predicted_price" in response_data
    assert isinstance(response_data["predicted_price"], float)


@pytest.mark.parametrize("zipcode_action", ["omit", "null"])
def test_predict_requires_zipcode(test_client, sample_home_features, zipcode_action):
    payload = dict(sample_home_features)
    if zipcode_action == "omit":
        del payload["zipcode"]
    else:
        payload["zipcode"] = None
    response = test_client.post("/predict", json=payload)
    assert response.status_code == 422


@pytest.mark.parametrize("zipcode", [98042, "", "abc", "9804", "980421"])
def test_predict_rejects_malformed_zipcode(test_client, sample_home_features, zipcode):
    response = test_client.post("/predict", json=dict(sample_home_features, zipcode=zipcode))
    assert response.status_code == 422


def test_predict_unknown_zipcode_returns_422(test_client, sample_home_features):
    payload = dict(sample_home_features, zipcode="00000")
    response = test_client.post("/predict", json=payload)
    assert response.status_code == 422
    assert response.json()["detail"] == "Unknown zipcode: 00000"


@pytest.mark.parametrize(
    "field, value",
    [
        ("sqft_living", -100.0),
        ("sqft_lot", 0.0),
        ("bedrooms", -1),
        ("floors", 0.0),
        ("bedrooms", 2.5),
        ("bathrooms", "two"),
    ],
)
def test_predict_rejects_invalid_numbers(test_client, sample_home_features, field, value):
    payload = dict(sample_home_features, **{field: value})
    response = test_client.post("/predict", json=payload)
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", field]


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
def test_predict_rejects_non_finite_numbers(test_client, sample_home_features, literal):
    """Non-standard JSON numbers get a clear 422 (not a 500); missing means null/omitted."""
    body = json.dumps(dict(sample_home_features, bathrooms="__X__")).replace('"__X__"', literal)
    response = post_raw(test_client, body)
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "bathrooms"]


def test_validation_errors_keep_default_format(test_client, sample_home_features):
    payload = dict(sample_home_features, sqft_living=-1)
    error = test_client.post("/predict", json=payload).json()["detail"][0]
    assert error["loc"] == ["body", "sqft_living"]
    assert error["input"] == -1


@pytest.mark.parametrize("body", ["", "{not json", "[]"])
def test_predict_rejects_malformed_body(test_client, body):
    assert post_raw(test_client, body).status_code == 422


def test_predict_ignores_extra_listing_columns(test_client):
    """Clients may send full listings (all 18 columns of future_unseen_examples.csv)."""
    examples = pd.read_csv(DATA_DIR / "future_unseen_examples.csv", dtype={"zipcode": str})
    full_listing = examples.iloc[0].to_dict()
    response = test_client.post("/predict", json=full_listing)
    assert response.status_code == 200
