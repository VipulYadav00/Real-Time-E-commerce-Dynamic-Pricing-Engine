from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()
model = joblib.load('pricing_model.pkl')

@app.get("/")
def home():
    return {"message": "Dynamic Pricing API is Live"}

@app.post("/predict")
def predict_price(data: dict):
    # Convert input dict to DataFrame
    input_df = pd.DataFrame([data])
    prediction = model.predict(input_df)[0]
    return {"predicted_demand": float(prediction)}

# To run this, you will need to install uvicorn: pip install uvicorn
# Then run: uvicorn app:app --reload