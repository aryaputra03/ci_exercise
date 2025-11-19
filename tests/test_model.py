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

    with pytest.raises(ValueError, match="Model harus di fit terlebih dahulu"):
        model.predict(X)


def test_model_predict(sample_data):
    X,y = sample_data
    model = MLModel()
    model.fit(X,y)

    prediction = model.predict(X)

    assert prediction.shape[0] == X.shape[0]
    assert set(np.unique(prediction)).issubset({0, 1})

def test_model_predict_proba(sample_data):
    X, y = sample_data
    model = MLModel()
    model.fit(X,y)
    proba = model.predict_proba(X)

    assert proba.shape[0] == X.shape[0]
    assert proba.shape[1] == 2
    assert np.allclose(proba.sum(axis=1), 1.0)

def test_model_save_load(sample_data):
    X, y = sample_data
    model = MLModel()
    model.fit(X,y)

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir)/"test_model.pkl"
        model.save(str(model_path))

        loaded_model = MLModel.load(str(model_path))

        assert loaded_model.is_fitted is True
        np.testing.assert_array_equal(
            model.predict(X),
            loaded_model.predict(X)
        )

def test_model_accuracy(sample_data):
    X,y = sample_data
    model = MLModel(n_estimators=50)
    model.fit(X,y)

    prediction = model.predict(X)
    accuracy = (prediction == y).mean()

    assert accuracy > 0.8


