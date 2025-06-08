import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Path to your trained model
model_path = r"C:\Users\GABNAbeni\Desktop\kaggle\Alz web\alzheimer_mobilenet_model.h5"
model = load_model(model_path)

# Class names (must match your training order)
class_names = [
    "Mild_Demented",
    "Moderate_Demented",
    "Non_Demented",
    "Very_Mild_Demented",
]

# Path to the image you want to classify
img_path = r"C:\Users\GABNAbeni\Desktop\test\NonDemented\a39e79b7-498d-448f-930a-83c521e4428a.jpg"  # Replace with your actual image path

# Load and preprocess image
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
predictions = model.predict(img_array)[0]
predicted_index = np.argmax(predictions)
predicted_label = class_names[predicted_index]
confidence = predictions[predicted_index]

# Display the image with prediction
plt.imshow(image.load_img(img_path))
plt.title(f"Prediction: {predicted_label}\nConfidence: {confidence:.2f}")
plt.axis("off")
plt.show()
