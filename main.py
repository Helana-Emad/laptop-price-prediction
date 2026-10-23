import __main__
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from model_classes import LaptopFeatureEngineer, CompatibleCatBoostRegressor
from fastapi.middleware.cors import CORSMiddleware

# الكلاسات اتحفظت على __main__ في النوتبوك، فلازم نرجّعها هناك قبل joblib.load
__main__.LaptopFeatureEngineer = LaptopFeatureEngineer
__main__.CompatibleCatBoostRegressor = CompatibleCatBoostRegressor

MODEL_PATH = Path(__file__).parent / "laptop_price_pipeline.joblib"
model = joblib.load(MODEL_PATH)

app = FastAPI(title="Laptop Price API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "https://helana-emad.github.io",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# نفس ترتيب الأعمدة اللي الـ pipeline اتدرب بيها (Laptop بيتستخدم لاستخراج CPU/GPU Detail)
INPUT_COLUMNS = [
    "Laptop", "Status", "Brand", "Model", "CPU", "RAM", "Storage",
    "Storage type", "GPU", "Screen", "Touch",
]


class Laptop(BaseModel):
    Laptop: str
    Status: str
    Brand: str
    Model: str
    CPU: str
    RAM: Optional[float] = None
    Storage: Optional[float] = None
    Storage_type: Optional[str] = None
    GPU: Optional[str] = None
    Screen: Optional[float] = None
    Touch: str

    model_config = {
        "json_schema_extra": {
            "example": {
                "Laptop": 'MSI Modern 15 A11SBU-620XES Intel Core i7-1165G7/32GB/1TB SSD/MX450/15.6"',
                "Status": "Refurbished",
                "Brand": "MSI",
                "Model": "Modern",
                "CPU": "Intel Core i7",
                "RAM": 32,
                "Storage": 1000,
                "Storage_type": "SSD",
                "GPU": "MX 450",
                "Screen": 15.6,
                "Touch": "No",
            }
        }
    }


@app.get("/")
def home():
    return {"message": "Laptop Price API is running"}


@app.post("/predict")
def predict(laptop: Laptop):
    data = laptop.model_dump()
    data["Storage type"] = data.pop("Storage_type")  # الاسم الأصلي للعمود فيه مسافة

    df = pd.DataFrame([data])[INPUT_COLUMNS]

    try:
        price = model.predict(df)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    return {"predicted_price": round(float(price), 2)}
