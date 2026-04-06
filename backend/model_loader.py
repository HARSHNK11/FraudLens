"""
Model and artifact loading utilities for FraudLens backend.
Loads best_model.pkl, scaler.pkl, and threshold.json at startup.
"""

import os
import json
import joblib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(ROOT, "artifacts")


class ModelRegistry:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.threshold = 0.5
        self.model_name = "Unknown"
        self.metrics = {}
        self.features = []

    def load(self):
        model_path = os.path.join(ARTIFACTS_DIR, "best_model.pkl")
        scaler_path = os.path.join(ARTIFACTS_DIR, "scaler.pkl")
        threshold_path = os.path.join(ARTIFACTS_DIR, "threshold.json")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                "Please run `python ml/train.py` first."
            )

        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

        if os.path.exists(threshold_path):
            with open(threshold_path) as f:
                info = json.load(f)
            self.threshold = info.get("threshold", 0.5)
            self.model_name = info.get("model_name", "Unknown")
            self.metrics = info.get("metrics", {})

        features_path = os.path.join(ARTIFACTS_DIR, "features.json")
        if os.path.exists(features_path):
            with open(features_path) as f:
                self.features = json.load(f)

        print(f"✅  Loaded model: {self.model_name}")
        print(f"    Threshold: {self.threshold:.4f}")
        print(f"    Features : {len(self.features)}")


# Global singleton
registry = ModelRegistry()
