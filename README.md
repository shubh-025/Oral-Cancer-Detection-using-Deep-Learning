# Early Stage Oral Cancer Detection using CNN

This project helps detect early-stage oral cancer using deep learning. A CNN based on ResNet50 classifies clinical oral cavity images as cancerous or non-cancerous. A Flask-based web app allows image upload and returns predictions with confidence.

## Features

- Upload oral cavity images
- Predict cancerous or non-cancerous
- Show confidence score
- Web interface using Flask

## Tech Stack

- Python, Flask
- TensorFlow / Keras
- ResNet50 CNN
- HTML, CSS, JavaScript

## Model Info

- **Architecture**: ResNet50
- **Input**: 224x224 RGB
- **Output**: Binary (Cancer / No Cancer)
- **Preprocessing**: ResNet-specific `preprocess_input`
