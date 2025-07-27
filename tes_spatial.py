import cv2
import numpy as np
from Stage1.spatial import extract_features_from_frame
from keras.models import load_model

# Load model gesture
model = load_model('Stage1/mobilenet_gesture_model.h5')
label_map = {
    0: 'Buka',
    1: 'Bukti',
    2: 'Cacat',
    3: 'Cara',
    4: 'Cari',
    5: 'Cepat',
    6: 'Daftar'
             }

# Buka webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Gagal mengambil frame.")
        break

    features = extract_features_from_frame(frame)

    if features is not None and features.shape[0] == 258:
        input_data = np.expand_dims(features, axis=0)  # (1, 258)
        prediction = model.predict(input_data, verbose=0)[0]
        pred_label = np.argmax(prediction)
        confidence = float(np.max(prediction))
        word = label_map.get(pred_label, "Unknown")

        print(f"Prediksi: {word} (confidence: {confidence:.2f})")

        if confidence > 0.7:
            cv2.putText(frame, word, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        else:
            cv2.putText(frame, "Tidak yakin", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

    else:
        cv2.putText(frame, "Landmark tidak terdeteksi", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Sign Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
