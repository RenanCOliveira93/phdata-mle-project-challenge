"""D. API performance: static resources are loaded once, never per request.

Latency itself is measured with benchmarks/benchmark.py against the Docker
container; timing thresholds are not asserted here because they depend on the host.
"""
import json
import pickle

import pandas as pd
import pytest
from sklearn.impute import KNNImputer

from api import endpoints


@pytest.fixture
def forbid_resource_loading(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("static resource loaded during a request")

    for target, name in [
        (pickle, "load"), (json, "load"), (pd, "read_csv"), (KNNImputer, "fit"),
        (endpoints, "load_model"), (endpoints, "load_features"),
        (endpoints, "load_demographics"), (endpoints, "fit_imputer"),
    ]:
        monkeypatch.setattr(target, name, fail)


@pytest.mark.parametrize("missing", [[], ["bathrooms", "sqft_lot"]], ids=["complete", "incomplete"])
def test_predict_does_not_load_resources_per_request(test_client, sample_home_features,
                                                     forbid_resource_loading, missing):
    payload = dict(sample_home_features, **{field: None for field in missing})
    assert test_client.post("/predict", json=payload).status_code == 200
