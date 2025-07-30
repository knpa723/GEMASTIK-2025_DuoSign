import cv2
import numpy as np
import mediapipe as mp

# Inisialisasi MediaPipe
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

pose = mp_pose.Pose(static_image_mode=False,
                    model_complexity=1,
                    enable_segmentation=False,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)

hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

# Landmark upper-body yang digunakan
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

# Fungsi ekstraksi fitur
def extract_features_from_frame(frame_bgr):
    frame_resized = cv2.resize(frame_bgr, (320, 320))  # Warna tetap (BGR)
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

    pose_result = pose.process(frame_rgb)
    hand_result = hands.process(frame_rgb)

    feature_vector = []

    # Pose
    if pose_result.pose_landmarks:
        for idx in UPPER_BODY_LANDMARKS:
            lm = pose_result.pose_landmarks.landmark[idx]
            feature_vector.extend([lm.x, lm.y, lm.z, lm.visibility])
    else:
        feature_vector.extend([0] * 4 * len(UPPER_BODY_LANDMARKS))

    # Hand
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
            feature_vector.extend([0] * 21 * 3)

    return np.array(feature_vector, dtype=np.float32)
