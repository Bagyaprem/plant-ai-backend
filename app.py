from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from PIL import Image

import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn

import io
import numpy as np
import pandas as pd

# =========================================
# FASTAPI APP
# =========================================

app = FastAPI()

# =========================================
# CORS
# =========================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================
# LOAD CLASSES
# =========================================

with open("classes.txt", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# =========================================
# LOAD MODEL
# =========================================

model = models.mobilenet_v2(weights=None)

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    len(classes)
)

model.load_state_dict(
    torch.load(
        "model/plant_disease_model.pth",
        map_location="cpu"
    )
)

model.eval()

# =========================================
# IMAGE TRANSFORM
# =========================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# =========================================
# LOAD LEAF WETNESS DATASET
# =========================================

df = pd.read_csv(
    "datasets/leaf_wetness_fungal_risk_data.csv"
)

# =========================================
# SENSOR DATA MODEL
# =========================================

class SensorData(BaseModel):
    temperature: float
    humidity: float
    rainfall: float

# =========================================
# HOME ROUTE
# =========================================

@app.get("/")
def home():

    return {
        "message": "Plant Disease API Running"
    }

# =========================================
# PLANT DISEASE PREDICTION
# =========================================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    try:

        # Read image
        image_bytes = await file.read()

        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

        # Convert image to numpy
        img_np = np.array(image)

        # Brightness check
        brightness = img_np.mean()

        # Reject dark image
        if brightness < 30:

            return {
                "prediction": "Image too dark or invalid",
                "confidence": "0%"
            }

        # Transform image
        image_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():

            outputs = model(image_tensor)

            probabilities = torch.nn.functional.softmax(
                outputs[0],
                dim=0
            )

            confidence, predicted = torch.max(
                probabilities,
                0
            )

        confidence_percent = confidence.item() * 100

        result = classes[predicted.item()]

        # Reject invalid image
        if confidence_percent < 85:

            return {
                "prediction": "Not a valid plant leaf image",
                "confidence": f"{confidence_percent:.2f}%"
            }

        return {
            "prediction": result,
            "confidence": f"{confidence_percent:.2f}%"
        }

    except Exception as e:

        return {
            "error": str(e)
        }

# =========================================
# FUNGAL RISK PREDICTION
# =========================================

@app.post("/fungal-risk")
def fungal_risk(data: SensorData):

    filtered = df[
        (df["Temperature (°C)"] >= data.temperature - 2) &
        (df["Temperature (°C)"] <= data.temperature + 2) &
        (df["Humidity (%)"] >= data.humidity - 5) &
        (df["Humidity (%)"] <= data.humidity + 5)
    ]

    # No matching dataset row
    if filtered.empty:

        if data.humidity > 85 and data.rainfall > 2:
            risk = "HIGH"

        elif data.humidity > 70:
            risk = "MEDIUM"

        else:
            risk = "LOW"

    else:

        # Most common fungal risk
        risk = filtered["Fungal Risk"].mode()[0]

    # Virtual leaf wetness
    leaf_wetness = (
        0.7 * data.humidity
        +
        0.3 * (data.rainfall * 10)
    )

    return {
        "temperature": data.temperature,
        "humidity": data.humidity,
        "rainfall": data.rainfall,
        "virtual_leaf_wetness": round(
            leaf_wetness,
            2
        ),
        "fungal_risk": risk
    }

# =========================================
# RUN SERVER
# =========================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        port=8000
    )