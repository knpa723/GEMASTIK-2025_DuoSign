import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
import time
import threading
from Stage1.spatial import extract_features_from_frame
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
from tensorflow import keras
from keras.models import load_model
import mediapipe as mp

from Stage2.symspell_gemini import correct_spelling, rephrase_sentence


app = Flask(__name__)

model = load_model('best_model.h5')
label_map = np.load("label_map.npy", allow_pickle=True).item()
rev_label_map = {v: k for k, v in label_map.items()}

# Global variabel
cap = None
detecting = False
frame_lock = threading.Lock()
current_frame = None
stream_thread = None
detection_buffer = []
last_detect_time = None
result_lock = threading.Lock()
latest_result = {"label": "", "confidence": 0.0}
mp_holistic = mp.solutions.holistic

final_result = {"original": "", "corrected": "", "rephrased": ""}
final_result_lock = threading.Lock()  

def classify_and_stream():
    global cap, detecting, current_frame, detection_buffer, last_detect_time, latest_result, mp_holistic

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
    )
    while detecting and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        current_time = time.time()
        
        # ✅ Sudah sekaligus menggambar dan memprediksi
        label_text, confidence, frame = extract_features_from_frame(frame, model, rev_label_map, holistic)

        # Just show frame without running inference
        frame = cv2.resize(frame, (1920, 1080)) 
        with frame_lock:
            current_frame = frame.copy()

        with result_lock:
            latest_result["label"] = label_text
            latest_result["confidence"] = round(confidence, 2)

        # Lanjutkan buffer logika
        if confidence > 0.7:
            detection_buffer.append(label_text)
            last_detect_time = current_time
            if len(detection_buffer) >= 5:
                send_to_stage2(detection_buffer.copy())
                detection_buffer.clear()
        else:
            # Untuk buffer logika saja, tidak menyentuh latest_result
            label_text = "Tidak yakin"

        # ✅ Kirim otomatis setelah 2 detik
        if last_detect_time and (current_time - last_detect_time > 2) and detection_buffer:
            send_to_stage2(detection_buffer.copy())
            detection_buffer.clear()
            last_detect_time = None

    if cap:
        cap.release()
        cap = None
    print("Kamera dimatikan")




def send_to_stage2(sequence):  # 📌 Ganti isi fungsi
    def process_stage2(seq):
        print(f"[Stage2] Sequence dikirim: {seq}")
        sentence = " ".join(seq)
        corrected = correct_spelling(sentence)
        rephrased = rephrase_sentence(corrected)
        print(f"[Stage2] Original   : {sentence}")
        print(f"[Stage2] Corrected  : {corrected}")
        print(f"[Stage2] Rephrased  : {rephrased}")

        with final_result_lock:  # 📌 Simpan hasil untuk bisa ditampilkan
            final_result["rephrased"] = rephrased

    threading.Thread(target=process_stage2, args=(sequence,), daemon=True).start()


@app.route('/')
def index():
    return render_template('main.html')


@app.route('/start', methods=['POST'])
def start():
    global detecting, stream_thread
    if not detecting:
        detecting = True
        stream_thread = threading.Thread(target=classify_and_stream, daemon=True)
        stream_thread.start()
    return '', 204


@app.route('/stop', methods=['POST'])
def stop():
    global detecting, cap, current_frame, detection_buffer
    detecting = False
    detection_buffer.clear()
    current_frame = None
    time.sleep(0.01)
    if cap:
        cap.release()
        cap = None
    return '', 204


@app.route('/detect')
def detect():
    def generate():
        while True:
            if not detecting:
                time.sleep(0.1)
                continue

            with frame_lock:
                if current_frame is None:
                    continue
                frame = current_frame.copy()

            _, jpeg = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

            time.sleep(0.05)

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')



@app.route('/status')
def status():
    global latest_result
    with result_lock:
        print("DEBUG:", latest_result)  # Debug log ke console
        return jsonify(latest_result)
    

@app.route('/final_result')  
def final_result_api():      
    with final_result_lock:
        return jsonify(final_result)


if __name__ == '__main__':
    app.run(debug=True)
