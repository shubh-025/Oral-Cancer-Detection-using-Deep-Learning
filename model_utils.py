# model_utils.py

import numpy as np
from PIL import Image
import io
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
# Path to your .h5 model (relative to this file)
MODEL_PATH = "model/resnet50_oral_cancer_model.h5"

# Module‐level model reference so we only load it once
_model = None

def load_model():
    global _model
    if _model is None:
        _model = tf.keras.models.load_model(MODEL_PATH)
    return _model

def preprocess_image(image_bytes):
    """
    - Opens the raw bytes with PIL, resizes to (224,224), scales to [0,1].
    - Returns a (1,224,224,3) numpy array.
    """
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    pil_img = pil_img.resize((224, 224))
    arr = image.img_to_array(pil_img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)

    return arr

def predict_image(image_bytes):
    """
    - Preprocesses the bytes, runs model.predict(...)
    - Returns (label_str, confidence_float).
      For a sigmoid‐output binary model: label is "Cancer Detected" or "No Cancer Detected",
      confidence is the probability (0.0–1.0).
    """
    model = load_model()
    img_arr = preprocess_image(image_bytes)
    prob = float(model.predict(img_arr)[0][0])
    
    if prob > 0.5:
      label = "No Cancer Detected" 
    else:
      label = "Cancer Detected"     
      prob = 1 - prob
    return label, prob
