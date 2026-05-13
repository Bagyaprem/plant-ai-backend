from fastapi import FastAPI, File, UploadFile
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import io
import numpy as np

app = FastAPI()

# Load classes
with open("classes.txt", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Load MobileNetV2
model = models.mobilenet_v2(weights=None)

# Final layer
model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    len(classes)
)

# Load weights
model.load_state_dict(
    torch.load(
        "model/plant_disease_model.pth",
        map_location="cpu"
    )
)

model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

@app.get("/")
def home():
    return {
        "message": "Plant Disease API Running"
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    try:
        # Read image
        image_bytes = await file.read()

        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

        # Convert to numpy
        img_np = np.array(image)

        # Basic validation
        brightness = img_np.mean()

        # Reject dark or invalid images
        if brightness < 30:
            return {
                "prediction": "Image too dark or invalid",
                "confidence": "0%"
            }

        # Transform
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

        # Strong rejection threshold
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