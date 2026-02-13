import cv2
import mediapipe as mp
import numpy as np
import os
import time

# ---------------- CONFIG ----------------
WORDS = ["GO", "HELLO", "HELP", "HOME", "NO", "PLEASE", "STOP", "THANKYOU", "WAIT", "YES"]
DATASET_DIR = "dataset_word_keypoints"
SAMPLES_PER_WORD = 30
SEQUENCE_LENGTH = 30
# ---------------------------------------

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def normalize_landmarks(lms):
    arr = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32)
    arr[:, :2] -= arr[0, :2]
    maxd = np.max(np.linalg.norm(arr[:, :2], axis=1))
    if maxd > 0:
        arr[:, :2] /= maxd
    return arr.flatten()

cap = cv2.VideoCapture(0)

with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
) as hands:

    for word in WORDS:
        print(f"\nCollecting data for word: {word}")
        os.makedirs(os.path.join(DATASET_DIR, word), exist_ok=True)

        sample_count = 0

        while sample_count < SAMPLES_PER_WORD:
            sequence = []

            # ---- READY SCREEN ----
            start_time = time.time()
            while time.time() - start_time < 2:
                ret, frame = cap.read()
                frame = cv2.flip(frame, 1)

                cv2.putText(frame, f"WORD: {word}", (50, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)
                cv2.putText(frame, "GET READY...", (50, 160),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

                cv2.imshow("COLLECT WORD DATA", frame)
                cv2.waitKey(1)

            # ---- RECORDING ----
            while len(sequence) < SEQUENCE_LENGTH:
                ret, frame = cap.read()
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)

                if results.multi_hand_landmarks:
                    hand = results.multi_hand_landmarks[0]
                    mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

                    keypoints = normalize_landmarks(hand.landmark)
                    sequence.append(keypoints)

                # UI
                cv2.putText(frame, f"WORD: {word}", (50, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
                cv2.putText(frame, f"SAMPLE: {sample_count+1}/{SAMPLES_PER_WORD}", (50, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 0), 3)
                cv2.putText(frame, f"FRAMES: {len(sequence)}/{SEQUENCE_LENGTH}", (50, 210),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)

                cv2.imshow("COLLECT WORD DATA", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    exit()

            # ---- SAVE ----
            np.save(
                os.path.join(DATASET_DIR, word, f"{sample_count}.npy"),
                np.array(sequence)
            )
            sample_count += 1
            print(f"{word} → Sample {sample_count}/{SAMPLES_PER_WORD} saved")

cap.release()
cv2.destroyAllWindows()