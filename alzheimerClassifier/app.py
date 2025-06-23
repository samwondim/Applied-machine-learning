from flask import Flask, request, jsonify, send_file
import torch
from torchvision import models, transforms
from PIL import Image
import os
import json

import glob
import re
from flask_cors import CORS
import time

# Define the Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})


# Load the trained model
def load_model(model_path):
    model = models.resnet18(pretrained=False)
    model.conv1 = torch.nn.Conv2d(
        1, 64, kernel_size=7, stride=2, padding=3, bias=False
    )  # 1-channel input
    model.fc = torch.nn.Linear(model.fc.in_features, 4)  # 4 classes
    try:
        model.load_state_dict(
            torch.load(model_path, map_location=torch.device("cpu")), strict=True
        )
    except Exception as e:
        print(f"Error loading state dict: {e}")
        raise
    model.eval()
    return model


# Load the model (replace 'model.pth' with your actual model file)
model = load_model("model.pth")


def preprocess_image(image_path):
    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),  # Keep 1 channel
            transforms.Resize((128, 128)),  # Match training size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),  # Grayscale normalization
        ]
    )
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)
    return image


# Prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():
    class_names = ["Mild", "Moderate", "Non", "Very Mild"]
    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),  # Keep 1 channel
            transforms.Resize((128, 128)),  # Match training size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),  # Grayscale normalization
        ]
    )
    print(request.files)
    if "file" not in request.files:
        return jsonify({"error": "No files uploaded"}), 400

    files = request.files.getlist("file")
    print("files", files)
    results = []
    for file in files:
        start_time = time.time()
        try:
            image = Image.open(file.stream)
            image = transform(image).unsqueeze(0)
            with torch.no_grad():
                output = model(image)
                probabilities = torch.nn.functional.softmax(output, dim=1)[0]
                top_probs, top_indices = probabilities.topk(4)
                print(probabilities.tolist())
                predictions = [
                    {"label": class_names[idx], "confidence": prob.item()}
                    for idx, prob in zip(top_indices, top_probs)
                ]
            print("predictions", predictions)
            results.append(
                {
                    "filename": file.filename,
                    "predictions": predictions,
                    "processingTime": time.time() - start_time,
                }
            )
        except Exception as e:
            results.append({"filename": file.filename, "error": str(e)})
    return jsonify({"results": results})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"})


@app.route("/confusion_matrix", methods=["GET"])
def get_confusion_matrix():
    return jsonify({"status": "healthy"})


@app.route("/latest_confusion_matrix")
def latest_confusion_matrix():
    # List all confusion matrix images in the static directory
    files = glob.glob(os.path.join(app.static_folder, "confusion_matrix_*.png"))
    print("FILES", files)

    # If no files are found, return a 404 error
    if not files:
        return "No confusion matrix found", 404

    # Find the file with the latest timestamp
    latest_file = max(
        files,
        key=lambda f: float(
            re.search(r"confusion_matrix\_(.+)\.png", os.path.basename(f)).group(1)
        ),
    )

    # Serve the latest image
    return send_file(latest_file, mimetype="image/jpeg")


@app.route("/metrics", methods=["GET"])
def get_metrics():
    try:
        with open("metrics.json", "r") as f:
            metrics = json.load(f)
        return jsonify(metrics)
    except FileNotFoundError:
        return jsonify({"error": "Metrics file not found"}), 404


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
