from fastapi import FastAPI
from src.schemas import TransactionData
from src.model import load_model, predict
from src.preprocess import preprocess_input

app = FastAPI()

# Load the trained model
model = load_model("model/best_rf_model.pkl")

@app.post("/predict/")
def make_prediction(data: TransactionData):
    # Preprocess the input data
    input_data_processed = preprocess_input(data)

    # Make predictions
    prediction = predict(model, input_data_processed)

    return {"prediction": int(prediction[0])}