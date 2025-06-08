import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

# === 1. Load the model ===
model_path = r"C:\Users\GABNAbeni\Desktop\kaggle\Alz web\alzheimer_mobilenet_model.h5"
model = load_model(model_path)

# === 2. Class labels ===
class_names = [
    "Mild_Demented",
    "Moderate_Demented",
    "Non_Demented",
    "Very_Mild_Demented",
]

# === 3. Load and preprocess the image ===
img_path = r"C:\Users\GABNAbeni\Downloads\Telegram Desktop\UoG exit courses\Imaging\943d9bdb-7c4a-4bdb-8dfa-4d8f8fe42ba9.jpg"  # Change as needed
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# === 4. Predict ===
predictions = model.predict(img_array)[0]
predicted_index = np.argmax(predictions)
predicted_label = class_names[predicted_index]
confidence = predictions[predicted_index] * 100

# === 5. Display the image with black background and prediction below ===
fig, ax = plt.subplots(figsize=(6, 6))
fig.patch.set_facecolor("black")  # Set figure (outside) background
ax.set_facecolor("black")  # Set axes (inside) background
ax.imshow(img)
ax.axis("off")

# Display prediction below the image
plt.figtext(
    0.5,
    0.02,
    f"Detection: {predicted_label} - {confidence:.2f}%",
    ha="center",
    fontsize=14,
    color="green",
)

plt.tight_layout()
plt.show()
