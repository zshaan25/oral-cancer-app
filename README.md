# Oral Cancer Classifier (Swin Transformer)

A Flask web application that classifies oral cavity images as **CANCER** or **NON CANCER** using a fine-tuned Swin Transformer. Upload an image, and the app returns the predicted class, the model's confidence, and the probability for each class.

> **Disclaimer:** This project is for demonstration and learning purposes only. It is not a medical device and must not be used for diagnosis or treatment decisions. Always consult a qualified healthcare professional.

## Features

- Fine-tuned Swin Transformer image classifier (PyTorch + Hugging Face Transformers)
- Simple web interface with drag-and-drop image upload and image preview
- Prediction label, confidence score, and per-class probability bars
- Low-confidence warning for uncertain predictions
- JSON API endpoint for programmatic use
- Input validation for file type (JPG, PNG, BMP, WEBP) and size (up to 10 MB)
- Runs on CPU, and uses a GPU automatically if one is available

## Project Structure

```
oral-cancer-app/
├── app.py                  # Flask server and prediction logic
├── requirements.txt        # Python dependencies
├── templates/
│   └── index.html          # Web interface
├── swin_oral_model/        # Trained model (stored with Git LFS)
│   ├── config.json
│   ├── preprocessor_config.json
│   └── model.safetensors
└── README.md
```

## Installation

**Requirements:** Python 3.9+ and Git with [Git LFS](https://git-lfs.com) installed.

```bash
# 1. Install Git LFS once (needed to download the model weights)
git lfs install

# 2. Clone the repository
git clone https://github.com/YOUR-USERNAME/oral-cancer-app.git
cd oral-cancer-app

# 3. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

# 4. Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
python app.py
```

Open http://127.0.0.1:5000 in your browser, choose an image, and click **Analyze**.

To load the model from a different location, set the `MODEL_DIR` environment variable:

```bash
set MODEL_DIR=C:\path\to\swin_oral_model       # Windows
# export MODEL_DIR=/path/to/swin_oral_model    # macOS / Linux
```

## API

**POST** `/predict` with form-data field `image`.

```bash
curl -X POST -F "image=@sample.jpg" http://127.0.0.1:5000/predict
```

Example response:

```json
{
  "label": "NON CANCER",
  "confidence": 93.42,
  "probabilities": { "CANCER": 6.58, "NON CANCER": 93.42 }
}
```

Errors return a JSON object such as `{ "error": "Unsupported file type." }` with HTTP status 400 (or 413 if the file is too large).

## Model Details

| Item | Details |
|---|---|
| Architecture | Swin Transformer (Hugging Face `AutoModelForImageClassification`) |
| Task | Binary image classification |
| Classes | `CANCER`, `NON CANCER` |
| Input size | 224 x 224 RGB |
| Preprocessing | Resize, tensor conversion, normalisation with the model's mean and std |
| Training | Fine-tuned in Google Colab (see the notebook) |

## Model Performance

Evaluated on the held-out test set:

| Metric | Cancer class | Macro average |
|---|---|---|
| Accuracy | 0.9219 | 0.9219 |
| Precision | 0.9574 | 0.9226 |
| Recall | 0.8911 | 0.9236 |
| F1-score | 0.9231 | 0.9219 |

## Limitations

- Trained on a limited dataset, so results may not generalise to different cameras, lighting, or patient populations.
- Not clinically validated.
- Predictions with low confidence should not be trusted.

## Tech Stack

Python, PyTorch, torchvision, Hugging Face Transformers, Flask, Pillow, HTML/CSS/JavaScript

## Author

**Muhammad Zeeshan** ([@zshaan25](https://github.com/zshaan25))
Developed as part of the NETSOL NIAI AI/ML program (Week 8 assignment).
