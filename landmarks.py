import cv2
import numpy as np
import tensorflow as tf
from collections import deque
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2

# Load model & label map
model = tf.keras.models.load_model("best_model.h5")
label_map = np.load("label_map.npy", allow_pickle=True).item()
rev_label_map = {v: k for k, v in label_map.items()}

# Setup
MAX_SEQ_LENGTH = 30
pose_indices = range(11, 17)
sequence = deque(maxlen=MAX_SEQ_LENGTH)

# MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# Feature extractor
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

# Drawing function (same as yours)
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

# Webcam loop
cap = cv2.VideoCapture(1)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = holistic.process(rgb)
        rgb.flags.writeable = True

        #frame = cv2.resize(frame, (480, 270))
         # Draw landmarks
        frame = draw_landmarks_only_pose_and_hands(frame, results)
        frame = cv2.flip(frame, 1)
        # Check if both hands and arm landmarks are present
        if (
            results.pose_landmarks or
            (results.left_hand_landmarks or
            results.right_hand_landmarks)
        ):
            # Extract & append features
            feature_vector = extract_features(results)
            sequence.append(feature_vector)

            # Predict only if sequence is full
            if len(sequence) == MAX_SEQ_LENGTH:
                input_data = np.expand_dims(np.array(sequence), axis=0)
                pred = model.predict(input_data, verbose=0)[0]
                class_id = np.argmax(pred)
                confidence = np.max(pred)

                if confidence > 0.99:
                    label = rev_label_map[class_id]
                    cv2.putText(
                        frame, f"{label} ({confidence:.2f})", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2, cv2.LINE_AA
                    )
        else:
            # Clear sequence if incomplete body parts
            sequence.clear()


        cv2.imshow('Gesture Detection', frame)

        if cv2.waitKey(5) & 0xFF == 27:  # ESC to quit
            break

cap.release()
cv2.destroyAllWindows()
