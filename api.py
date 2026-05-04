import os
import pickle

import pandas as pd
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ──────────────────────────────────────────────
# 1. App Initialization
# ──────────────────────────────────────────────
app = FastAPI(
    title="E-Commerce Purchase Intention API",
    description="XGBoost-powered prediction API for Chrome Extension integration.",
    version="1.0.0",
)

# ──────────────────────────────────────────────
# 2. CORS Configuration
# ──────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────
# 3. Model Loading
# ──────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "xgboost_model.pkl")

with open(MODEL_PATH, "rb") as f:
    model_dict = pickle.load(f)

model = model_dict["model"]
le_month = model_dict["le_month"]
le_visitor = model_dict["le_visitor"]
feature_columns = model_dict["feature_columns"]

# ──────────────────────────────────────────────
# 4. Pydantic Schema
# ──────────────────────────────────────────────

class UserSessionData(BaseModel):
    Administrative: int = 0
    Administrative_Duration: float = 0.0
    Informational: int = 0
    Informational_Duration: float = 0.0
    ProductRelated: int = 1
    ProductRelated_Duration: float = 0.0
    BounceRates: float = 0.0
    ExitRates: float = 0.0
    PageValues: float = 0.0
    SpecialDay: float = 0.0
    Month: str = "Nov"
    OperatingSystems: int = 1
    Browser: int = 1
    Region: int = 1
    TrafficType: int = 1
    VisitorType: str = "New_Visitor"
    Weekend: bool = False


# ──────────────────────────────────────────────
# 5. Health Check
# ──────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "Online", "message": "Purchase Intent API is running"}


# ──────────────────────────────────────────────
# 6. Prediction Endpoint
# ──────────────────────────────────────────────

@app.post("/predict")
def predict(data: UserSessionData):
    row = data.model_dump()

    # Preprocess: Weekend bool → int
    row["Weekend"] = int(row["Weekend"])

    # Encode Month (graceful fallback for unseen labels)
    try:
        row["Month"] = le_month.transform([row["Month"]])[0]
    except ValueError:
        row["Month"] = le_month.transform([le_month.classes_[0]])[0]

    # Encode VisitorType (graceful fallback for unseen labels)
    try:
        row["VisitorType"] = le_visitor.transform([row["VisitorType"]])[0]
    except ValueError:
        row["VisitorType"] = le_visitor.transform([le_visitor.classes_[0]])[0]

    # Build single-row DataFrame with correct column order
    df = pd.DataFrame([row])[feature_columns]

    # Predict
    proba = float(model.predict_proba(df)[0][1])

    return {
        "purchase_probability": round(proba, 4),
        "high_intent": proba > 0.5,
    }


# ──────────────────────────────────────────────
# 6. Execution
# ──────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
