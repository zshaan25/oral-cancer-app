import os
import threading
from pathlib import Path

import torch
from flask import Flask, request, jsonify, render_template
from PIL import Image
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModelForImageClassification

BASE_DIR = Path(__file__).resolve().parent

# Folder that Colab saved with model.save_pretrained(...) / processor.save_pretrained(...)
# Download "swin_oral_model" from Google Drive and put it next to this app.py
MODEL_DIR = os.environ.get("MODEL_DIR", str(BASE_DIR / "swin_oral_model"))

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ---- load model, processor, device (these were notebook variables in Colab) ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not Path(MODEL_DIR).is_dir():
    raise SystemExit(
        f"Model folder not found: {MODEL_DIR}\n"
        "Download the 'swin_oral_model' folder from Google Drive and place it next to app.py."
    )

processor = AutoImageProcessor.from_pretrained(MODEL_DIR)
model = AutoModelForImageClassification.from_pretrained(MODEL_DIR).to(device)
model.eval()
model_lock = threading.Lock()

# class names come from the saved config (index order matches training)
class_names = [model.config.id2label[i] for i in range(len(model.config.id2label))]

# same preprocessing as test_transform in the notebook
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(processor.image_mean, processor.image_std),
])


# aik image ki prediction (label, confidence aur dono classes ki probability %)
def predict_image(image):
    tensor = test_transform(image.convert("RGB")).unsqueeze(0).to(device)
    with model_lock, torch.no_grad():
        outputs = model(pixel_values=tensor)
    probs = torch.softmax(outputs.logits, dim=1)[0].cpu().numpy()
    idx = int(probs.argmax())
    return {
        "label": class_names[idx],
        "confidence": round(float(probs[idx]) * 100, 2),
        "probabilities": {name: round(float(p) * 100, 2) for name, p in zip(class_names, probs)},
    }


app = Flask(__name__, template_folder=str(BASE_DIR / "templates"))
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB upload limit


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("image")
    if file is None or file.filename == "":
        return jsonify(error="No image was uploaded."), 400
    if Path(file.filename).suffix.lower() not in IMG_EXT:
        return jsonify(error="Unsupported file type. Upload a JPG, PNG, BMP or WEBP image."), 400
    try:
        image = Image.open(file.stream)
        image.load()
    except Exception:
        return jsonify(error="The file could not be read as an image."), 400
    return jsonify(predict_image(image))


@app.errorhandler(413)
def too_large(_):
    return jsonify(error="The file is larger than 10 MB."), 413


if __name__ == "__main__":
    print("Classes:", class_names, "| Device:", device)
    app.run(host="127.0.0.1", port=5000, debug=False)
