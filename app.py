from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
import os

app = Flask(__name__)

# Load your model (update the path if needed)
model = tf.keras.models.load_model('your_model.h5')  # Replace with your actual model file

# Home route (if using templates)
@app.route('/')
def home():
    return "Flask + TensorFlow app is running!"

# Example prediction route (change logic as needed)
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Example: get JSON input like {"data": [1.0, 2.0, 3.0]}
        input_data = request.json['data']
        input_array = np.array([input_data])
        prediction = model.predict(input_array)
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Required for Render to detect the PORT
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Use Render's PORT or fallback to 5000
    app.run(host='0.0.0.0', port=port)


