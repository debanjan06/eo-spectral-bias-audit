from fastapi import FastAPI, UploadFile, File
import torch
import numpy as np
import io
from PIL import Image

app = FastAPI(title="AgriGuard Global Inference API")

# Load your California weights
device = torch.device("cpu") # Docker usually runs on CPU for inference
model = MultiModalCNN(num_classes=3)
model.load_state_dict(torch.load("models/agriguard_california_final.pth", map_location=device))
model.eval()

@app.post("/predict")
async def predict(ndvi: float, temp: float, humidity: float):
    # In a real app, you'd process an uploaded .npy patch
    # For now, we simulate the spatial input to show the pipeline works
    spatial_tensor = torch.randn(1, 4, 32, 32) 
    spectral_tensor = torch.tensor([[ndvi, ndvi*0.8, ndvi*0.9, 720.0]]).float()
    weather_tensor = torch.tensor([[temp, temp-10, humidity, 0.0]]).float()

    with torch.no_grad():
        logits = model(spatial_tensor, spectral_tensor, weather_tensor)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs).item()

    classes = ['Diseased', 'Healthy', 'Stressed']
    return {
        "prediction": classes[pred],
        "confidence": float(probs[0][pred]),
        "deployment_status": "Global (Dockerized)"
    }