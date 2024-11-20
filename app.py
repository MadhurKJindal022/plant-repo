from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import pandas as pd
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load model and label dictionary
model = load_model('model3.h5')
labels = {
    0: 'Apple', 1: 'Blueberry', 2: 'Cherry', 3: 'Corn', 4: 'Grape', 5: 'Orange',
    6: 'Peach', 7: 'Pepper', 8: 'Potato', 9: 'Raspberry', 10: 'Soybean', 11: 'Squash',
    12: 'Strawberry', 13: 'Tomato'
}

# Define the function to preprocess images
def preprocess_image(image, target_size=(225, 225)):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(target_size)
    img_array = img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0
    return img_array

@app.route('/')
def index():
    return "Welcome to the Plant Classification API!"

# Endpoint for leaf classification
@app.route('/classify', methods=['POST'])
def classify_leaf():
    if 'image' not in request.files:
        return jsonify({"error": "No image file in request"}), 400
    
    file = request.files['image']
    image = Image.open(file)
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    predicted_class = np.argmax(predictions[0])
    predicted_label = labels[predicted_class]
    return jsonify({"predicted_label": predicted_label})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
