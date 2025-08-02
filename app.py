from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
import time
from Stage1.spatial import extract_features_from_frame
from tensorflow import keras
from keras.models import load_model
import threading

app = Flask(__name__)

# Load model dan label
model = load_model('best_model.h5')

label_map = {}
label_map = np.load("label_map.npy", allow_pickle=True).item()


# Video capture global
cap = None
detecting = False

# Buffer hasil deteksi
detection_buffer = []
last_detect_time = None
result_lock = threading.Lock()

latest_result = {"label": "", "confidence": 0.0}

def classify_and_stream():
    global cap, detecting, detection_buffer, last_detect_time, latest_result

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while detecting and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        features = extract_features_from_frame(frame)
        current_time = time.time()

        if features is not None and features.shape[0] == 320:
            input_data = np.expand_dims(features, axis=0)
            prediction = model.predict(input_data, verbose=0)[0]
            pred_label = np.argmax(prediction)
            confidence = float(np.max(prediction))
            label_text = label_map.get(pred_label, "Unknown")

            with result_lock:
                latest_result["label"] = label_text
                latest_result["confidence"] = round(confidence, 2)

            if confidence > 0.7:
                detection_buffer.append(label_text)
                last_detect_time = current_time

                if len(detection_buffer) >= 5:
                    send_to_stage2(detection_buffer.copy())
                    detection_buffer.clear()
            else:
                label_text = "Tidak yakin"

        else:
            label_text = "Landmark tidak terdeteksi"

        # Gap lebih dari 2 detik tanpa deteksi
        if last_detect_time and (current_time - last_detect_time > 2) and detection_buffer:
            send_to_stage2(detection_buffer.copy())
            detection_buffer.clear()
            last_detect_time = None

        # Untuk debug stream (jika mau live feed di html)
        _, jpeg = cv2.imencode('.jpg', frame)
        frame_bytes = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    if cap:
        cap.release()

def send_to_stage2(sequence):
    print(f"Sequence dikirim ke Stage2: {sequence}")
    # Simulasi, bisa diganti dengan request ke endpoint Stage2
    # requests.post('http://stage2/endpoint', json={"sequence": sequence})

@app.route('/')
def index():
    return render_template('main.html')

@app.route('/start', methods=['POST'])
def start():
    global detecting
    if not detecting:
        detecting = True
        threading.Thread(target=classify_and_stream, daemon=True).start()
    return render_template('main.html')

@app.route('/stop', methods=['POST'])
def stop():
    global detecting
    detecting = False
    return render_template('main.html')

@app.route('/detect')
def detect():
    return Response(classify_and_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    with result_lock:
        return jsonify(latest_result)

if __name__ == '__main__':
    app.run(debug=True)
