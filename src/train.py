"""Training Script For ML Model"""
import argparse
from pathlib import Path
import json
from src.model import MLModel
from src.utils import load_data, split_data, calculate_metrics

def train_model(
        data_path: str = None,
        model_path: str = "models/model.pkl",
        metrics_path: str = "models/metrics.json",
        n_estimators: int = 100,
        test_size: float = 0.2
) -> dict:
    print("Loading data...")
    X, y = load_data(data_path)
    
    print("Splitting data...")
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=test_size)
    
    print(f"Training model with {n_estimators} estimators...")
    model = MLModel(n_estimators=n_estimators)
    model.fit(X_train, y_train)

    print("Evaluating Model...")
    y_pred = model.predict(X_test)
    metrics = calculate_metrics(y_test, y_pred)

    print("Metrics:")
    for key, values in metrics.items():
        print(f"    {key}: {values:.4f}")

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    print(f"Model save to {model_path}")

    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"Metrics Saved to: {metrics_path}")
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Train ML Model")
    parser.add_argument("--data", type=str, default=None, help="Path to data file")
    parser.add_argument("--model", type=str, default="models/model.pkl", help="Output model path")
    parser.add_argument("--metrics", type=str, default="models/metrics.json", help="Output metrics path")
    parser.add_argument("--n-estimators", type=int, default=100, help="Number of estimators")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set size")

    args = parser.parse_args()

    train_model(
        data_path=args.data,
        model_path=args.model,
        metrics_path=args.metrics,
        n_estimators=args.n_estimators,
        test_size=args.test_size
    )

if __name__ == "__main__":
    main()    