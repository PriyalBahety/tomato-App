from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load model
model = load_model("tomato_disease_model.h5")
class_names = [
    'Tomato__Bacterial_spot', 'Tomato_Early_blight', 'Tomato__Late_blight',
    'Tomato__Leaf_Mold', 'Tomato__Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato__Target_Spot', 'Tomato__Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato__Tomato_mosaic_virus', 'Tomato__healthy'
]

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    file_path = None

    if request.method == 'POST':
        file = request.files['image']
        if file:
            file_path = os.path.join("static", file.filename)
            file.save(file_path)

            # Preprocess image
            img = image.load_img(file_path, target_size=(256, 256))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0

            # Make prediction
            result = model.predict(img_array, verbose=0)
            predicted_class = class_names[np.argmax(result)]
            prediction = f"Predicted Disease: {predicted_class}"

    return render_template("index.html", prediction=prediction, img_path=file_path)

if __name__ == '__main__':
    app.run(debug=True)
    import os

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Render sets the PORT env variable
    app.run(host='0.0.0.0', port=port)

