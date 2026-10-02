import json
import os
from pathlib import Path

import httpx
import pandas as pd
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def test_client():
    """Fixture for unit tests using FastAPI TestClient."""
    from src.main import app
    return TestClient(app)


@pytest.fixture
def api_base_url():
    """Fixture providing API base URL for integration tests."""
    return os.getenv("API_BASE_URL", "http://localhost:8000")


@pytest.fixture
def http_client(api_base_url):
    """Fixture for integration tests using httpx."""
    return httpx.Client(base_url=api_base_url, timeout=10.0)


@pytest.fixture
def sample_home_features():
    """Fixture providing sample input data for tests."""
    return {
        "bedrooms": 3,
        "bathrooms": 2.0,
        "sqft_living": 1500.0,
        "sqft_lot": 5000.0,
        "floors": 1.0,
        "sqft_above": 1200.0,
        "sqft_basement": 300.0,
        "zipcode": "98042"
    }


API_FIELDS = [
    "bedrooms", "bathrooms", "sqft_living", "sqft_lot",
    "floors", "sqft_above", "sqft_basement", "zipcode",
]
DATA_DIR = Path(__file__).resolve().parent.parent / "src" / "data"
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def future_unseen_rows():
    """The 8 API fields of every row in future_unseen_examples.csv, in file order."""
    examples = pd.read_csv(DATA_DIR / "future_unseen_examples.csv", dtype={"zipcode": str})
    return examples[API_FIELDS].to_dict(orient="records")


@pytest.fixture
def baseline_predictions():
    """Predictions of the original (pre-change) API for future_unseen_rows."""
    with open(FIXTURES_DIR / "baseline_predictions.json") as f:
        return json.load(f)["predictions"]


@pytest.fixture
def model_inputs(monkeypatch):
    """Record every DataFrame passed to the model's predict()."""
    from sklearn.pipeline import Pipeline

    calls = []
    original_predict = Pipeline.predict

    def spy(self, X, **kwargs):
        calls.append(X.copy())
        return original_predict(self, X, **kwargs)

    monkeypatch.setattr(Pipeline, "predict", spy)
    return calls
