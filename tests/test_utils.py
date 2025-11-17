"""Test for Utility Function"""
import pytest
import numpy as np
from src.utils import load_data, split_data, calculate_metrics

def test_load_data_synthetic():
    X, y = load_data()

    assert X.shape[0] == 1000, "Harus mempunyai 1000 sample"
    assert X.shape[1] == 20, "Harus punya 20 fitur"
    assert y.shape[0] == 1000, "Harus ada 1000 baris target"
    assert len(np.unique(y)) == 2, "harus klasifikasi biner"

def test_split_data():
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)

    assert X_train.shape[0] == 800
    assert X_test.shape[0] == 200
    assert y_train.shape[0] == 800
    assert y_test.shape[0] == 200

def test_split_data_reproducibility():
    X, y = load_data()
    X_train1, X_test1, y_train1, y_test1 = split_data(X, y, random_state=42)
    X_train2, X_test2, y_train2, y_test2 = split_data(X, y, random_state=42)

    np.testing.assert_array_equal(X_train1, X_train2)
    np.testing.assert_array_equal(y_train1, y_train2)

def test_calculate_metrics():
    y_true = np.array([0,1,0,1,0])
    y_pred = np.array([0,1,0,0,0])

    metrics = calculate_metrics(y_true, y_pred)

    assert 'accuracy' in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics
    assert 0 <= metrics["accuarcy"] <=1
    assert metrics["accuarcy"] == 0.8

def test_calculate_metrics_perfect():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])

    metrics = calculate_metrics(y_true, y_pred)

    assert metrics['accuracy'] == 1.0
    assert metrics['precision'] == 1.0
    assert metrics['recall'] == 1.0
    assert metrics['f1_score'] == 1.0