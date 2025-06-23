import time
import json
import pandas as pd
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from flask import Flask, jsonify
import json

app = Flask(__name__)
from torchvision import models, transforms

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

train = "Data/train.parquet"
test = "Data/test.parquet"


def load_training_data(train_path):
    df_train = pd.read_parquet(train_path, engine="pyarrow")
    df_train["img_arr"] = df_train["image"].apply(dict_to_image)
    df_train.drop("image", axis=1, inplace=True)

    label_mapping = {
        0: "Mild_Demented",
        1: "Moderate_Demented",
        2: "Non_Demented",
        3: "Very_Mild_Demented",
    }
    df_train["class_name"] = df_train["label"].map(label_mapping)
    train_df, val_df = train_test_split(
        df_train, test_size=0.2, stratify=df_train["class_name"], random_state=42
    )

    # Scale to [0, 1] and ensure 128x128 size
    def preprocess_image(img):
        if img.shape != (128, 128):
            img = cv2.resize(img, (128, 128))
        return img / 255.0  # Scale to [0, 1]

    train_df["img_arr"] = train_df["img_arr"].apply(preprocess_image)
    val_df["img_arr"] = val_df["img_arr"].apply(preprocess_image)

    return train_df, val_df


def prepare_data(train_df, val_df):
    # Stack images (already scaled to [0, 1] and resized)
    x_train = (
        np.stack(train_df["img_arr"].values).reshape(-1, 128, 128, 1).astype(np.float32)
    )
    x_val = (
        np.stack(val_df["img_arr"].values).reshape(-1, 128, 128, 1).astype(np.float32)
    )

    y_train = train_df["label"].values
    y_val = val_df["label"].values

    x_train_tensor = torch.tensor(x_train).permute(0, 3, 1, 2)  # (N, 1, 128, 128)
    x_val_tensor = torch.tensor(x_val).permute(0, 3, 1, 2)

    # Apply normalization to match prediction
    mean = 0.5
    std = 0.5
    x_train_tensor = (x_train_tensor - mean) / std  # Normalize to [-1, 1]
    x_val_tensor = (x_val_tensor - mean) / std

    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)

    return (x_train_tensor, x_val_tensor, y_train_tensor, y_val_tensor, y_train, y_val)


def load_dataset_and_data(
    X_train_tensor, X_val_tensor, y_train_tensor, y_val_tensor, batch_size=32
):
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False)

    return (train_dataset, val_dataset, train_loader, val_loader)


def train_model(train_df, val_df, num_epochs=10, batch_size=32, learning_rate=0.001):
    x_train_tensor, x_val_tensor, y_train_tensor, y_val_tensor, y_train, y_val = (
        prepare_data(train_df, val_df)
    )

    train_dataset, val_dataset, train_loader, val_loader = load_dataset_and_data(
        x_train_tensor, x_val_tensor, y_train_tensor, y_val_tensor
    )

    model = models.resnet18(pretrained=True)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, 4)  # 4-class

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.unique(y_train), y=y_train
    )
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    patience = 7
    best_val_loss = float("inf")
    early_stop_counter = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        val_loss = 0.0
        y_pred, y_true = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, pred = torch.max(outputs, 1)
                y_pred.extend(pred.cpu().numpy())
                y_true.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        acc = accuracy_score(y_true, y_pred)
        print(
            f"Epoch {epoch + 1}, Train Loss: {running_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {acc:.4f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print("Early stopping triggered")
                model.load_state_dict(best_model_state)
                break

    torch.save(model.state_dict(), "model.pth")

    return model, device, val_loader


def dict_to_image(image_dict):
    if isinstance(image_dict, dict) and "bytes" in image_dict:
        byte_string = image_dict["bytes"]
        nparr = np.frombuffer(byte_string, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        return img
    else:
        raise TypeError(f"Expected dictionary with 'bytes' key, got {type(image_dict)}")


def confusionMatrix(model, val_loader, device):
    model.eval()
    y_pred, y_true = [], []
    inference_times = []  # Optional: to measure inference time
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            start_time = time.time()  # Optional: measure inference time
            outputs = model(inputs)
            end_time = time.time()  # Optional
            _, preds = torch.max(outputs, 1)
            y_pred.extend(preds.cpu().numpy())
            y_true.extend(labels.cpu().numpy())
            inference_times.append(end_time - start_time)

    # Optional: Compute average inference time per image
    batch_size = val_loader.batch_size
    avg_inference_time = (
        sum(inference_times) / len(inference_times) / batch_size
        if inference_times
        else 0
    )

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    class_names = ["Mild", "Moderate", "Non", "Very Mild"]

    plt.figure(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, values_format="d")
    plt.title("Confusion Matrix (Validation Set)")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.grid(False)
    plt.tight_layout()
    t = time.time()
    plt.savefig(f"static/confusion_matrix_{t}.png")
    plt.close()  # Close the figure to free memory

    # Get the classification report dictionary
    report = precision_recall_f1(y_true, y_pred)

    # Extract overall metrics
    metrics = {
        "accuracy": report["accuracy"],
        "precision": report["macro avg"]["precision"],
        "recall": report["macro avg"]["recall"],
        "f1_score": report["macro avg"]["f1-score"],
        "inference_time": avg_inference_time,  # Optional
    }

    # Save metrics to a JSON file
    with open("metrics.json", "w") as f:
        json.dump(metrics, f)


def precision_recall_f1(y_true, y_pred):
    class_names = ["Mild", "Moderate", "Non", "Very Mild"]

    report = classification_report(
        y_true, y_pred, target_names=class_names, digits=4, output_dict=True
    )
    print("\nClassification Report:\n")
    print(report)
    return report


def predict_on_test_data(test, model, device):
    df_test = pd.read_parquet(test, engine="pyarrow")
    df_test["img_arr"] = df_test["image"].apply(dict_to_image)

    # Preprocess test images
    def preprocess_image(img):
        if img.shape != (128, 128):
            img = cv2.resize(img, (128, 128))
        return img / 255.0

    df_test["img_arr"] = df_test["img_arr"].apply(preprocess_image)
    X_test = (
        np.stack(df_test["img_arr"].values).reshape(-1, 128, 128, 1).astype(np.float32)
    )
    X_test_tensor = torch.tensor(X_test).permute(0, 3, 1, 2)

    # Normalize test data
    mean = 0.5
    std = 0.5
    X_test_tensor = (X_test_tensor - mean) / std

    test_dataset = torch.utils.data.TensorDataset(X_test_tensor)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, shuffle=False
    )

    model.eval()
    predictions = []
    with torch.no_grad():
        for (inputs,) in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())

    df_test["predicted_label"] = predictions
    label_mapping = {
        0: "Mild_Demented",
        1: "Moderate_Demented",
        2: "Non_Demented",
        3: "Very_Mild_Demented",
    }
    df_test["predicted_class"] = df_test["predicted_label"].map(label_mapping)
    print(df_test[["predicted_label", "predicted_class"]].head())


train_df, val_df = load_training_data(train)
model, device, val_loader = train_model(train_df, val_df, num_epochs=5)
confusionMatrix(model, val_loader, device)
# predict_on_test_data(test, model,device)
