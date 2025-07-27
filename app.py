from flask import Flask, request, jsonify
from Stage1.spatial import extract_features_from_frame
import cv2
import numpy as np
from tensorflow import keras
from keras.models import load_model

app = Flask(__name__)

# Load CNN model yang sudah dilatih
model = load_model('models/cnn_model.h5')

# Simpan buffer kalimat hasil prediksi
current_sentence = []

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    img_array = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # === Stage 1: Ekstraksi fitur dari frame ===
    features = extract_features_from_frame(frame)

    if features is None:
        return jsonify({'message': 'No hand or pose detected'}), 200

    # === Stage 2: Prediksi CNN ===
    prediction = model.predict(np.expand_dims(features, axis=0))[0]
    predicted_label = np.argmax(prediction)
    confidence = float(np.max(prediction))

    # Dummy label (diganti sesuai label asli)
    label_map = {0: "Halo", 1: "Terima Kasih", 2: "Apa Kabar"}
    word = label_map.get(predicted_label, "Unknown")

    # Simpan ke kalimat
    if confidence > 0.7:
        current_sentence.append(word)

    return jsonify({
        'detected_word': word,
        'confidence': confidence,
        'current_sentence': ' '.join(current_sentence)
    })

if __name__ == '__main__':
    app.run(debug=True)
