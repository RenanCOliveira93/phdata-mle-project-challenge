import pytest


@pytest.mark.integration
def test_predict_endpoint_integration(http_client, sample_home_features):
    """Test the /predict endpoint via HTTP using httpx client."""
    response = http_client.post("/predict", json=sample_home_features)
    assert response.status_code == 200
    response_data = response.json()
    assert "predicted_price" in response_data
    assert isinstance(response_data["predicted_price"], float)
    assert response_data["predicted_price"] > 0


@pytest.mark.integration
def test_health_endpoint_integration(http_client):
    """Test the /health endpoint via HTTP using httpx client."""
    response = http_client.get("/health")
    assert response.status_code == 200
    response_data = response.json()
    assert "status" in response_data
    assert response_data["status"] == "healthy"


@pytest.mark.integration
def test_predict_with_missing_fields_integration(http_client, sample_home_features):
    """Test /predict imputes null house attributes in the running container."""
    payload = dict(sample_home_features, bathrooms=None, sqft_lot=None)
    response = http_client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["predicted_price"] > 0


@pytest.mark.integration
def test_complete_requests_match_original_api_integration(http_client, future_unseen_rows,
                                                          baseline_predictions):
    """The Docker container reproduces the original API's predictions exactly."""
    predictions = [
        http_client.post("/predict", json=row).json()["predicted_price"]
        for row in future_unseen_rows
    ]
    assert predictions == baseline_predictions
