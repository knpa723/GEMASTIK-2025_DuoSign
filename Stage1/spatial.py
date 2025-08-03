import numpy as np
import cv2
from collections import deque
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2

# Init MediaPipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
pose_indices = range(11, 17)
MAX_SEQ_LENGTH = 30
sequence = deque(maxlen=MAX_SEQ_LENGTH)

# === Ekstraksi fitur dari hasil landmark ===
def extract_features(results):
    feat = []

    # Pose subset
    if results.pose_landmarks:
        full = results.pose_landmarks.landmark
        for i in pose_indices:
            lm = full[i]
            feat.extend([lm.x, lm.y, lm.z])
    else:
        feat.extend([0.0] * len(pose_indices) * 3)

    # Left hand
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            feat.extend([lm.x, lm.y, lm.z])
    else:
        feat.extend([0.0] * 21 * 3)

    # Right hand
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            feat.extend([lm.x, lm.y, lm.z])
    else:
        feat.extend([0.0] * 21 * 3)

    return np.array(feat)


# === Gambar landmark tangan dan sebagian pose (tanpa wajah) ===
def draw_landmarks_only_pose_and_hands(image, results):
    annotated = image.copy()
    MODIFIED_POSE_CONNECTION = sorted(list(mp_holistic.POSE_CONNECTIONS))[10:]

    if results.pose_landmarks:
        full = results.pose_landmarks.landmark
        indices = range(11, 17)
        subset = landmark_pb2.NormalizedLandmarkList(
            landmark=[full[i] for i in indices]
        )
        orig_to_subset = {orig_idx: new_i for new_i, orig_idx in enumerate(indices)}
        filtered = [
            (orig_to_subset[a], orig_to_subset[b])
            for (a, b) in MODIFIED_POSE_CONNECTION
            if a in orig_to_subset and b in orig_to_subset
        ]
        mp_drawing.draw_landmarks(
            annotated,
            subset,
            filtered,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
        )

    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            annotated,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2)
        )

    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            annotated,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2)
        )

    return annotated


# === Fungsi utama untuk dipanggil dari app.py ===
def extract_features_from_frame(frame_bgr, model, label_list, holistic):
    global sequence
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    results = holistic.process(rgb)
    rgb.flags.writeable = True

    frame_with_skeleton = draw_landmarks_only_pose_and_hands(frame_bgr, results)
    frame_with_skeleton = cv2.flip(frame_with_skeleton, 1)
    # Check keberadaan landmark untuk proses prediksi
    if (
        results.pose_landmarks or
        (results.left_hand_landmarks or
        results.right_hand_landmarks)
    ):
        feature_vector = extract_features(results)
        sequence.append(feature_vector)
        if len(sequence) == MAX_SEQ_LENGTH:
            input_data = np.expand_dims(np.array(sequence), axis=0)
            pred = model.predict(input_data, verbose=0)[0]
            class_id = int(np.argmax(pred))
            confidence = float(np.max(pred))

            if confidence > 0.9:
                label = label_list[class_id]
                
                # Tambahkan teks ke frame (pojok kiri atas)
                text = f"{label} ({confidence:.2f})"
                cv2.putText(
                    frame_with_skeleton,
                    text,
                    (10, 40),  # posisi teks
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,            # ukuran font
                    (0, 255, 0),    # warna hijau terang
                    3,              # ketebalan
                    cv2.LINE_AA
                )

                return label, confidence, frame_with_skeleton


    # Jika tidak terdeteksi
    #sequence.clear()
    return None, 0.0, frame_with_skeleton
