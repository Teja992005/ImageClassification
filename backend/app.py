import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
from flask import Flask, request, jsonify
from flask_cors import CORS
# from io import BytesIO
import json
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Ensure model exists before loading
model_path = "model/model.h5"
labels_path = "model/labels.json"

if not os.path.exists(model_path):
    raise FileNotFoundError("❌ Model file not found at 'model/model.h5'. Train the model first!")

if not os.path.exists(labels_path):
    raise FileNotFoundError("❌ Labels file not found at 'model/labels.json'. Train the model first!")

# Load the trained model
model = load_model(model_path)

# Load class labels dynamically
with open(labels_path, "r") as f:
    class_labels = json.load(f)
labels = {str(i): name for i, name in enumerate(class_labels)}

@app.route("/predict", methods=["POST"])
def predict():
    try:
        print("📥 Received a request at /predict")

        if "file" not in request.files:
            print("⚠️ No file found in request")
            return jsonify({"error": "No file part"}), 400

        file = request.files["file"]
        if file.filename == "":
            print("⚠️ No selected file")
            return jsonify({"error": "No selected file"}), 400

        print("📂 File received:", file.filename)

        from io import BytesIO

        # Load and preprocess image
        img = image.load_img(BytesIO(file.read()), target_size=(32, 32))
        img_array = image.img_to_array(img) / 255.0  # Normalize pixel values
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        print("✅ Image successfully preprocessed with shape:", img_array.shape)

        # Make prediction
        prediction = model.predict(img_array)
        print("🎯 Prediction output:", prediction)

        predicted_class = labels[str(np.argmax(prediction))]
        confidence = float(np.max(prediction))

        return jsonify({"class": predicted_class, "confidence": confidence})

    except Exception as e:
        print("❌ Error in prediction:", str(e))  # Log the actual error
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
