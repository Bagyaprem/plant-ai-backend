from fastapi import FastAPI, File, UploadFile
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import io

app = FastAPI()

# Load classes
with open("classes.txt", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Load MobileNetV2 model
model = models.mobilenet_v2(weights=None)

# Modify final layer
model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    len(classes)
)

# Load trained weights
model.load_state_dict(
    torch.load(
        "model/plant_disease_model.pth",
        map_location="cpu"
    )
)

model.eval()

# Image transform
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
    image_bytes = await file.read()

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)

        probabilities = torch.nn.functional.softmax(
            outputs[0],
            dim=0
        )

        confidence, predicted = torch.max(
            probabilities,
            0
        )

    result = classes[predicted.item()]

    return {
        "prediction": result,
        "confidence": f"{confidence.item()*100:.2f}%"
    }