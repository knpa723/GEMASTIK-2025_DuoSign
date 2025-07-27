import cv2
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Inisialisasi MediaPipe pose dan hands
pose = mp_pose.Pose(static_image_mode=False,
                    model_complexity=1,
                    enable_segmentation=False,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)

hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

# Urutan pose landmark upper-body yang digunakan
UPPER_BODY_LANDMARKS = [
    mp_pose.PoseLandmark.NOSE,
    mp_pose.PoseLandmark.LEFT_EYE,
    mp_pose.PoseLandmark.RIGHT_EYE,
    mp_pose.PoseLandmark.LEFT_EAR,
    mp_pose.PoseLandmark.RIGHT_EAR,
    mp_pose.PoseLandmark.LEFT_SHOULDER,
    mp_pose.PoseLandmark.RIGHT_SHOULDER,
    mp_pose.PoseLandmark.LEFT_ELBOW,
    mp_pose.PoseLandmark.RIGHT_ELBOW,
    mp_pose.PoseLandmark.LEFT_WRIST,
    mp_pose.PoseLandmark.RIGHT_WRIST,
]

# Fungsi untuk ekstraksi fitur dari frame
def extract_features_from_frame(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Pose detection
    pose_result = pose.process(rgb)
    hand_result = hands.process(rgb)

    feature_vector = []

    # Ekstrak pose upper-body
    if pose_result.pose_landmarks:
        landmarks = pose_result.pose_landmarks.landmark
        for idx in UPPER_BODY_LANDMARKS:
            lm = landmarks[idx]
            feature_vector.extend([lm.x, lm.y, lm.z, lm.visibility])
    else:
        # Jika tidak terdeteksi, isi dengan nol
        feature_vector.extend([0] * 4 * len(UPPER_BODY_LANDMARKS))

    # Ekstrak hand landmark (maks 2 tangan)
    for hand_label in ['left', 'right']:
        hand_landmarks = None
        if hand_result.multi_handedness and hand_result.multi_hand_landmarks:
            for i, handedness in enumerate(hand_result.multi_handedness):
                label = handedness.classification[0].label.lower()
                if label == hand_label:
                    hand_landmarks = hand_result.multi_hand_landmarks[i]
                    break

        if hand_landmarks:
            for lm in hand_landmarks.landmark:
                feature_vector.extend([lm.x, lm.y, lm.z])
        else:
            # Jika tangan tidak terdeteksi, isi nol
            feature_vector.extend([0] * 21 * 3)  # 21 titik, 3 dimensi

    return np.array(feature_vector, dtype=np.float32)
