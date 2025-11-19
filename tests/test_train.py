import tempfile
from pathlib import Path
from src.train import train_model

def test_train_model_basic():
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = str(Path(tmpdir)/"model.pkl")
        metrics_path = str(Path(tmpdir)/"metrics.json")

        metrics = train_model(
            model_path=model_path,
            metrics_path=metrics_path,
            n_estimators=10
        )

        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert Path(model_path).exists()
        assert Path(metrics_path).exists()

def test_train_model_metrics_range():
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = str(Path(tmpdir)/"model.pkl")
        metrics_path = str(Path(tmpdir)/"metrics.json")

        metrics = train_model(
            model_path=model_path,
            metrics_path=metrics_path,
            n_estimators=10
        )

        for metric_name, metric_value in metrics.items():
            assert 0 <= metric_value <= 1, f"{metric_name} should be between 0 and 1"

def test_train_model_different_test_sizes():
    with tempfile.TemporaryDirectory() as tmpdir:
        for test_size in [0.1,0.2,0.3]:
            model_path = str(Path(tmpdir)/f"model_{test_size}.pkl")
            metrics_path = str(Path(tmpdir)/f"metrics_{test_size}.json")

            metrics = train_model(
                model_path = model_path,
                metrics_path=metrics_path,
                test_size=test_size,
                n_estimators=10
            )

            assert metrics["accuracy"] > 0.5