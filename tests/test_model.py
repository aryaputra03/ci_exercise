import pytest
import numpy as np
import tempfile
from pathlib import Path
from src.model import MLModel

@pytest.fixture
def sample_data():
    from sklearn.datasets import make_classification
    X, y =make_classification(n_samples=100, n_features=10, random_state=42)
    return X, y

def test_model_initialization():
    model = MLModel(n_estimators=50, random_state=42)

    assert model.model.n_estimators == 50
    assert model.is_fitted is False

def test_model_fit(sample_data):
    X,y = sample_data
    model = MLModel()

    model.fit(X,y)
    assert model.is_fitted is True

def test_model_predict_before_fit(sample_data):
    X, y = sample_data
    model = MLModel()

    with pytest.raises(ValueError, match="Model must be fitted"):
        model.predict(X)
        