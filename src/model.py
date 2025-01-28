import joblib
import pandas as pd

def load_model(model_path: str):
    """Load the trained model from a file."""
    return joblib.load(model_path)

def predict(model, input_data):
    """Make predictions using the loaded model."""
    return model.predict(input_data)