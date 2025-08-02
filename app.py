from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
import time
import threading
from Stage1.spatial import extract_features_from_frame
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
from tensorflow import keras
from keras.models import load_model




app = Flask(__name__)

model = load_model('best_model.h5')
label_map = np.load("label_map.npy", allow_pickle=True).item()

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


def classify_and_stream():
    global cap, detecting, current_frame, detection_buffer, last_detect_time, latest_result

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while detecting and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()

        # ✅ Sudah sekaligus menggambar dan memprediksi
        label_text, confidence, frame = extract_features_from_frame(frame, model, label_map)

        frame = cv2.resize(frame, (1280, 720)) 

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




def send_to_stage2(sequence):
    print(f"[Stage2] Sequence dikirim: {sequence}")
    # requests.post('http://stage2/endpoint', json={"sequence": sequence})


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
    time.sleep(0.5)
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
    with result_lock:
        print("DEBUG:", latest_result)  # Debug log ke console
        return jsonify(latest_result)


if __name__ == '__main__':
    app.run(debug=True)
