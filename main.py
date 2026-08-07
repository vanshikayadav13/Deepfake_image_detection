import os
import io
import base64
import warnings
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from facenet_pytorch import MTCNN, InceptionResnetV1
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

warnings.filterwarnings("ignore")

app = FastAPI(title="Deepfake Detection API", version="1.0.0")

# Enable CORS for the Vercel frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Set to specific Vercel domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Device & Paths ────────────────────────────────────────────────────────────
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_PATH = os.path.join(BASE_DIR, "checkpoint.pth")

# ── Models Initialization ─────────────────────────────────────────────────────
print(f"Loading models on device: {DEVICE}...")

# Face detector
mtcnn = MTCNN(
    select_largest=False,
    post_process=False,
    device=DEVICE
).to(DEVICE).eval()

# Classifier model - pretrained=None avoids downloading vggface2 weights (107MB)
# since we immediately load our own checkpoint anyway
model = InceptionResnetV1(
    pretrained=None,
    classify=True,
    num_classes=1,
    device=DEVICE,
)

if os.path.exists(CHECKPOINT_PATH):
    print(f"Loading checkpoint from: {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    print(f"WARNING: Checkpoint file not found at {CHECKPOINT_PATH}. Using pretrained face backbone only.")

model.to(DEVICE)
model.eval()

# GradCAM target layer reference
TARGET_LAYERS = [model.block8.branch1[-1]]

@app.get("/")
def read_root():
    return {"status": "online", "device": DEVICE, "model": "InceptionResnetV1"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 1. Validate file extension
    filename = file.filename.lower()
    if not (filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.webp')):
        raise HTTPException(status_code=400, detail="Only PNG, JPG, JPEG, and WEBP images are supported.")

    # 2. Read image
    try:
        contents = await file.read()
        input_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

    # 3. Detect Face
    face = mtcnn(input_image)
    if face is None:
        raise HTTPException(status_code=422, detail="No face detected in the image. Please upload a clear photo.")

    # 4. Preprocess face tensor
    face = face.unsqueeze(0)  # (1, C, H, W)
    face = F.interpolate(face, size=(256, 256), mode="bilinear", align_corners=False)

    # Convert to uint8 format for blending
    prev_face = face.squeeze(0).permute(1, 2, 0).cpu().detach().numpy().astype("uint8")

    # Normalize to [0, 1] for model input & GradCAM
    face = face.to(DEVICE).to(torch.float32) / 255.0
    face_float = face.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()

    # 5. GradCAM explainability
    try:
        cam = GradCAM(model=model, target_layers=TARGET_LAYERS)
        targets = [ClassifierOutputTarget(0)]
        grayscale_cam = cam(input_tensor=face, targets=targets, eigen_smooth=True)
        grayscale_cam = grayscale_cam[0]

        visualization = show_cam_on_image(face_float, grayscale_cam, use_rgb=True)
        face_with_mask = cv2.addWeighted(prev_face, 1, visualization, 0.5, 0)
        
        # Convert output image to Base64
        buffered = io.BytesIO()
        Image.fromarray(face_with_mask).save(buffered, format="JPEG")
        encoded_img = base64.b64encode(buffered.getvalue()).decode("utf-8")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explainability visual generation failed: {str(e)}")

    # 6. Model Prediction
    with torch.no_grad():
        output = torch.sigmoid(model(face).squeeze(0))

    fake_prob = float(output.item())
    real_prob = float(1.0 - fake_prob)

    return {
        "prediction": "fake" if fake_prob >= 0.5 else "real",
        "confidences": {
            "real": real_prob,
            "fake": fake_prob
        },
        "explainability_image": f"data:image/jpeg;base64,{encoded_img}"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
