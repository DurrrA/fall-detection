import joblib
from src.ml.model import FallDetectionModel

def export_model(model, model_path):
    joblib.dump(model, model_path)
    print(f"Model exported successfully to {model_path}")

if __name__ == "__main__":
    model = FallDetectionModel()  # Load or create your model here
    model_path = "models/fall_detection_model.pkl"  # Specify the path to save the model
    export_model(model, model_path)