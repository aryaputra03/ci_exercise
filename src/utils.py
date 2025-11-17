"""Utils Functon for Pipeline"""
import numpy as np
import pandas as pd
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split

def load_data(file_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset (synthetic or from file)
    
    Args:
        file_path: Path to CSV file (optional)
        
    Returns:
        X: Features array
        y: Target array
    """
    if file_path:
        df = pd.read_csv(file_path)
        X = df.iloc[:,:-1].values
        y = df.iloc[:,-1].values
    
    else:
        from sklearn.datasets import make_classification
        X, y = make_classification(
            n_samples=1000,
            n_features=20,
            n_informative=15,
            n_redundant=5,
            random_state=42
        )
        return X, y

def split_data(X: np.ndarray,
               y: np.ndarray,
               test_size: float = 0.2,
               random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return train_test_split(X,y, test_size=test_size, random_state=random_state)

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    return{
        "accuracy": accuracy_score(y_true, y_pred),
        "precission": precision_score(y_true, y_pred, average='weighted', zero_division=0),
        "recall": recall_score(y_true, y_pred, average='weighted', zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }
