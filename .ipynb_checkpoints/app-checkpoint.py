from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load the trained model
autoencoder = tf.keras.models.load_model('model/autoencoder.keras')

@app.route('/predict', methods=['POST'])
def predict():
    # Load image from POST request
    file = request.files['image']
    img = Image.open(file).convert('RGB')
    img = np.array(img) / 255.0  # Normalize to [0, 1]
    
    # Prepare image for model
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = np.resize(img, (1, 32, 32, 3))  # Resize if needed
    
    # Make prediction
    prediction = autoencoder.predict(img)
    
    # Convert prediction to image
    pred_img = prediction[0] * 255.0
    pred_img = Image.fromarray(pred_img.astype(np.uint8))
    
    # Save or return the image
    img_byte_arr = io.BytesIO()
    pred_img.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    
    return img_byte_arr, 200, {'Content-Type': 'image/png'}

if __name__ == '__main__':
    app.run(debug=True)
