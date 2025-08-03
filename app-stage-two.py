from flask import Flask, request, jsonify
from Stage1.spatial import extract_features_from_frame
from Stage2.symspell_gemini import correct_spelling, rephrase_sentence
import cv2
import numpy as np

app = Flask(__name__)
current_sentence = []

# ======= Dummy CNN model prediction (kamu bisa aktifkan CNN aslinya kalau tersedia) ========
def dummy_cnn_predict(features):
    # Simulasi output CNN
    return {"label": 0, "confidence": 0.85}


# ======== Label mapping (label CNN) ========
label_map = {0: "Haloo", 1: "Trima Kasih", 2: "Apa kabarr"}

@app.route('/', methods=['GET'])
def index():
    return "API Aktif - Kirim gambar ke POST /detect"

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    img_array = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

# ========== Stage 1: Ekstraksi fitur dari frame ==========

    features = extract_features_from_frame(frame)

    if features is None:
        return jsonify({'message': 'No hand or pose detected'}), 200


# ========== Stage 2: Prediksi CNN ===========no
    pred = dummy_cnn_predict(features)  # Ganti dengan model.predict jika model CNN tersedia
    predicted_label = pred["label"]
    confidence = pred["confidence"]

    word = label_map.get(predicted_label, "Unknown")

    if confidence > 0.7:
        current_sentence.append(word)

    # Stage 3: Koreksi Ejaan & Tata Bahasa
    original_sentence = " ".join(current_sentence)
    corrected = correct_spelling(original_sentence)
    rephrased = rephrase_sentence(corrected)

    return jsonify({
        "detected_word": word,
        "confidence": confidence,
        "original_sentence": original_sentence,
        "corrected_spelling": corrected,
        "final_output": rephrased
    })

if __name__ == '__main__':
    app.run(debug=True)
