import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import sys
import os

# Load the trained binary classification model
model = tf.keras.models.load_model("butterfly_binary_model.h5")

# Prediction function
def predict(img_path):
    if not os.path.exists(img_path):
        print(f"❌ File not found: {img_path}")
        return

    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 224, 224, 3)

    # Predict
    prediction = model.predict(img_array)
    prob = prediction[0][0]

    print(f"\n🧠 Raw prediction score: {prob:.4f}")

    # Threshold logic (assuming 1 = butterfly, 0 = not a butterfly)
    if prob > 0.5:
        label = "🦋 Butterfly"
        confidence = prob * 100
    else:
        label = "❌ Not a Butterfly"
        confidence = (1 - prob) * 100

    print(f"🔍 Predicted: {label}")
    print(f"📊 Confidence: {confidence:.2f}%\n")

# Example usage:
predict("sample_test.jpg")  # Replace with your test image path
