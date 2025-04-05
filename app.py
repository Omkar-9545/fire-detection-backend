from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from PIL import Image
import numpy as np
import io
import os

app = Flask(__name__)
CORS(app,origins='*')

MODEL_PATH = "fire_detection_model.tflite"

os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Model loaded and compiled successfully!")

def preprocess_image(image):
    img = image.resize((224, 224)) 
    img = (np.asarray(img) / 255.0).astype(np.float32)
    img = np.expand_dims(img, axis=0) 
    return img

@app.before_request
def log_request_info():
    print('Headers: %s', request.headers)

@app.route('/',methods=['POST','GET'])
def greet():
    print("Your server is running successfully......")
    tp = {"status":"All well"}
    return jsonify(tp)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    image = Image.open(io.BytesIO(file.read()))
    img = preprocess_image(image)

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])[0]

    result = {
        "fire_probability": float(prediction[0]),
        "no_fire_probability": float(prediction[1]),
        "is_fire_detected": bool(prediction[0] > prediction[1])
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)

