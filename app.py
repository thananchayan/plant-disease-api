import io
import time
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

app = Flask(__name__)
CORS(app)

model = None  # Lazy load to avoid crashing on boot

# Class labels (ensure this matches your training label order)
class_labels = [
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

def preprocess_image(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image = image.resize((128, 128))  # Match training size
        image_array = img_to_array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        return image_array
    except Exception as e:
        raise ValueError(f"Failed to preprocess image: {str(e)}")

@app.route('/')
def home():
    return "🌿 Plant Disease Detection API is live!"

@app.route('/predict', methods=['POST'])
def predict():
    global model

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    try:
        # Lazy-load the model if not already loaded
        if model is None:
            print("[INFO] Loading model...")
            model = load_model('plant_disease_model.h5')
            print("[INFO] Model loaded")

        image_bytes = file.read()
        processed_image = preprocess_image(image_bytes)

        start = time.time()
        predictions = model.predict(processed_image)
        duration = time.time() - start
        print(f"[INFO] Prediction took {duration:.2f} seconds")

        predicted_index = int(np.argmax(predictions))
        predicted_class = class_labels[predicted_index]
        confidence = float(np.max(predictions))

        return jsonify({
            'prediction': predicted_class,
            'confidence': f"{confidence:.2f}"
        })

    except Exception as e:
        print(f"[ERROR] {str(e)}")
        return jsonify({'error': 'Prediction failed', 'details': str(e)}), 500

if __name__ == '__main__':
    # Render uses PORT env variable, but locally we set it manually
    app.run(debug=False, host='0.0.0.0', port=10000)
