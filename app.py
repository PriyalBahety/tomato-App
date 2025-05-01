from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
MODEL_PATH = "tomato_disease_model.h5"
model = load_model(MODEL_PATH)

# Home route
@app.route('/')
def index():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded"

    file = request.files['file']
    if file.filename == '':
        return "No file selected"

    filepath = os.path.join("static", file.filename)
    file.save(filepath)

    img = image.load_img(filepath, target_size=(224, 224))  # adjust size to your model input
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    result = np.argmax(prediction)

    return render_template('result.html', prediction=result)

# Run Flask on Render-compatible host and port
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
