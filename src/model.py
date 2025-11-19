"""Machine Leanring Model Definition"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

class MLModel:
    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state
        )
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MLModel':
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model harus di fit terlebih dahulu")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model harus di fit terlebih dahulu")
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def save(self, file_path: str) -> None:
        joblib.dump(self, file_path)
    
    @staticmethod
    def load(file_path: str) -> 'MLModel':
        return joblib.load(file_path)