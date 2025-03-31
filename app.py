from flask import Flask, request, jsonify
from flask_cors import CORS
import gdown
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from PIL import Image
import numpy as np
import io
import os

app = Flask(__name__)
CORS(app,origins='https://detectfire.vercel.app/')

MODEL_PATH = "fire_detection_model.h5"

GDRIVE_FILE_ID = "1YRuxxmOZ7B_8QX81WgRanVKqeoPqn9mA" 

def download_model():
     if not os.path.exists(MODEL_PATH):
        print("Downloading model from Google Drive...")

        gdrive_url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
        try:
            gdown.download(gdrive_url, MODEL_PATH, quiet=False)
            print("Model downloaded successfully!")
        except Exception as e:
            print(f"Failed to download model: {e}")
            exit(1)

download_model()

os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

model = tf.keras.models.load_model(MODEL_PATH, compile=False)
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
print("Model loaded and compiled successfully!")

def preprocess_image(image):
    img = image.resize((224, 224)) 
    img = np.asarray(img) / 255.0
    img = np.expand_dims(img, axis=0) 
    return img

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    image = Image.open(io.BytesIO(file.read()))
    img = preprocess_image(image)

    prediction = model.predict(img)[0]

    result = {
        "fire_probability": float(prediction[0]),
        "no_fire_probability": float(prediction[1]),
        "is_fire_detected": bool(prediction[0] > prediction[1])
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True, port=10000)