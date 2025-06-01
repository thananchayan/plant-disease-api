from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import io
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# 🔁 Lazy-load model to reduce memory footprint
model = None

# 🔤 Update your class names here as per your training labels
class_names = ['Early Blight', 'Late Blight', 'Healthy']

@app.route('/')
def home():
    return "Plant Disease Detection API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    global model

    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # 🔁 Load model only once
        if model is None:
            model = load_model("plant_disease_model.h5")

        # ✅ Read and preprocess image
        img = image.load_img(io.BytesIO(file.read()), target_size=(128, 128))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = float(np.max(prediction))

        return jsonify({
            'class': predicted_class,
            'confidence': confidence
        })

    except Exception as e:
        print(f"[ERROR] {str(e)}")
        return jsonify({'error': 'Internal server error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
