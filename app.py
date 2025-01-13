# Required Imports
from flask import Flask, render_template
import os
from waitress import serve

# Initialize Flask App
app = Flask(__name__)

# Basic route
@app.route('/')
def index():
    return render_template('index.html')

# Health check for Cloud Run
@app.route('/healthz')
def health_check():
    return 'OK', 200

# Test route
@app.route('/test')
def test():
    return "Flask app is running!", 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    serve(app, host="0.0.0.0", port=port)

"""
# Commented out functionality - uncomment as needed
--------------------------------------------------
import cv2
import numpy as np
from flask import request, jsonify
import pytesseract
from tensorflow.keras.models import load_model
from flask_cors import CORS

# Enable CORS if needed
# CORS(app)

# Debug mode
# app.debug = True

# Model loading
# MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'digit_recognizer.h5')
# model = load_model(MODEL_PATH)

# Invalid cells directory
# INVALID_CELLS_DIR = "invalid_cells"
# os.makedirs(INVALID_CELLS_DIR, exist_ok=True)

# OCR configuration
# OCR_CONFIDENCE_THRESHOLD = 0.8

# Image processing functions
# def preprocess_image(image):
#     [Your preprocessing code here]

# def detect_grid(binary):
#     [Your grid detection code here]

# def extract_cells(image, bounding_box):
#     [Your cell extraction code here]

# def recognize_digits(cells, model):
#     [Your digit recognition code here]

# def solve_sudoku(board):
#     [Your sudoku solver code here]

# def prepare_board(image, model):
#     [Your board preparation code here]

# def convert_to_native_types(obj):
#     [Your type conversion code here]

# Upload route
# @app.route('/upload', methods=['POST'])
# def upload_image():
#     [Your upload handling code here]
"""