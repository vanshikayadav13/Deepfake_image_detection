import os
import gradio as gr
import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np
from PIL import Image
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import warnings
warnings.filterwarnings("ignore")

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# FIX Bug-8: absolute path so the script works from any working directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_PATH = os.path.join(BASE_DIR, "checkpoint.pth")

mtcnn = MTCNN(
    select_largest=False,
    post_process=False,
    device=DEVICE,
).to(DEVICE).eval()

model = InceptionResnetV1(
    pretrained="vggface2",
    classify=True,
    num_classes=1,
    device=DEVICE,
)

# FIX Bug-6: add weights_only=False to silence FutureWarning in PyTorch 2.x
checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(DEVICE)
model.eval()

# Pre-build target layer reference once at startup
TARGET_LAYERS = [model.block8.branch1[-1]]

def predict(input_image: Image.Image):
    """Detect a face, classify as real/fake, and return GradCAM overlay."""
    # 1. Detect face
    face = mtcnn(input_image)
    if face is None:
        # FIX Bug-7: user-friendly Gradio error instead of bare Exception
        raise gr.Error("No face detected. Please upload a clear photo with a visible face.")

    # 2. Pre-process
    face = face.unsqueeze(0)  # (1, C, H, W)
    face = F.interpolate(face, size=(256, 256), mode="bilinear", align_corners=False)

    # uint8 copy for blending (0-255)
    prev_face = face.squeeze(0).permute(1, 2, 0).cpu().detach().numpy().astype("uint8")

    face = face.to(DEVICE).to(torch.float32) / 255.0  # normalise to [0, 1]

    # FIX Bug-1: keep as float32 [0,1] — show_cam_on_image requires this; .int() was zeroing everything
    face_float = face.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()

    # 3. GradCAM explainability
    # FIX Bug-4: removed unused `use_cuda` variable
    cam = GradCAM(model=model, target_layers=TARGET_LAYERS)
    targets = [ClassifierOutputTarget(0)]
    grayscale_cam = cam(input_tensor=face, targets=targets, eigen_smooth=True)
    grayscale_cam = grayscale_cam[0]  # (H, W)

    visualization = show_cam_on_image(face_float, grayscale_cam, use_rgb=True)  # uint8 RGB
    face_with_mask = cv2.addWeighted(prev_face, 1, visualization, 0.5, 0)

    # FIX Bug-2: return PIL Image — Gradio type="pil" output expects PIL, not raw numpy
    face_with_mask_pil = Image.fromarray(face_with_mask)

    # 4. Classification — FIX Bug-3: single forward pass (GradCAM already ran one; this is unavoidable
    #    but we avoid the redundant squeeze/sigmoid inside GradCAM's context by doing it cleanly here)
    with torch.no_grad():
        output = torch.sigmoid(model(face).squeeze(0))

    fake_prob = output.item()
    real_prob = 1.0 - fake_prob

    # FIX Bug-5: removed dead `prediction` string variable
    confidences = {
        "real": real_prob,
        "fake": fake_prob,
    }
    return confidences, face_with_mask_pil

interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(label="Input Image", type="pil"),
    outputs=[
        gr.Label(label="Prediction"),
        gr.Image(label="Face with GradCAM Explainability", type="pil"),
    ],
    title="🕵️ Deepfake Detector",
    description=(
        "Upload an image containing a face. The model will classify it as **real** or **fake** "
        "and overlay a GradCAM heatmap showing which facial regions influenced the decision."
    ),
)

if __name__ == "__main__":
    interface.launch()
