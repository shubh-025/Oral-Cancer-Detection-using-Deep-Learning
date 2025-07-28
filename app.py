# app.py

import os
from flask import Flask, render_template, request, jsonify
from model_utils import predict_image

from flask_cors import CORS
app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    1) Read name, age, gender from form
    2) Read image file bytes
    3) Call model_utils.predict_image(...) to get label + confidence
    4) Return JSON with all relevant fields
    """
    # 1) Get form fields
    name = request.form.get("name", "").strip()
    age_str = request.form.get("age", "").strip()
    gender = request.form.get("gender", "").strip()

    # Validate inputs
    if not name or not age_str or not gender:
        return jsonify({"error": "Name, age, and gender are required."}), 400

    try:
        age = int(age_str)
        if age <= 0:
            raise ValueError()
    except ValueError:
        return jsonify({"error": "Age must be a positive integer."}), 400

    # 2) Get the uploaded image
    if "image" not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    image_file = request.files["image"]
    if image_file.filename == "":
        return jsonify({"error": "No image selected."}), 400

    image_bytes = image_file.read()

    # 3) Call the ML model
    try:
        predicted_label, confidence = predict_image(image_bytes)
    except Exception as e:
        return jsonify({"error": f"Model prediction failed: {e}"}), 500

    # 4) Return JSON
    return jsonify({
        "name": name,
        "age": age,
        "gender": gender,
        "predicted_label": predicted_label,
        "confidence": confidence,
    })


if __name__ == "__main__":
    # For local development only; in production use gunicorn or similar
    app.run(host="0.0.0.0", port=8000, debug=True)
