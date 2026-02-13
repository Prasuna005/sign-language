import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle

MODEL_PATH = "models/number_keypoint_model.h5"
LABELS_PATH = "models/number_labels.pkl"

model = tf.keras.models.load_model(MODEL_PATH)
with open(LABELS_PATH, "rb") as f:
    labels = pickle.load(f)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

def extract_keypoints(hand_landmarks):
    return np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()

cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            keypoints = extract_keypoints(hand_landmarks).reshape(1, -1)
            preds = model.predict(keypoints, verbose=0)
            idx = np.argmax(preds)
            number = labels[idx]
            confidence = np.max(preds)

            cv2.putText(frame, f"Number: {number} ({confidence*100:.1f}%)",
                        (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                        (0,255,0), 3)

        else:
            cv2.putText(frame, "Show Hand",
                        (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0,0,255), 2)

        cv2.imshow("ASL Number Skeleton Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()