import cv2
import mediapipe as mp
import numpy as np
import os
import time

# =========================
# SETTINGS
# =========================
DATASET_PATH = "dataset_word_keypoints"
WORDS = ["GO", "HELLO", "HELP", "HOME", "NO", "PLEASE", "STOP", "THANKYOU", "WAIT", "YES"]
SEQUENCES_PER_WORD = 30     # how many samples per word
FRAMES_PER_SEQUENCE = 30   # frames per sample

# =========================
# MEDIAPIPE
# =========================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def extract_keypoints(results):
    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        return np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark]).flatten()
    else:
        return np.zeros(21 * 3)

# =========================
# CREATE BASE FOLDER
# =========================
os.makedirs(DATASET_PATH, exist_ok=True)

cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
) as hands:

    for word in WORDS:
        word_path = os.path.join(DATASET_PATH, word)
        os.makedirs(word_path, exist_ok=True)

        existing_samples = len(os.listdir(word_path))
        if existing_samples >= SEQUENCES_PER_WORD:
            print(f"✅ {word} already collected. Skipping...")
            continue

        print(f"\n📌 Collecting data for: {word}")

        for sample in range(existing_samples, SEQUENCES_PER_WORD):
            sequence = []

            print(f"{word} → Sample {sample + 1}/{SEQUENCES_PER_WORD}")
            time.sleep(1.5)

            for frame_num in range(FRAMES_PER_SEQUENCE):
                ret, frame = cap.read()
                if not ret:
                    continue

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)

                keypoints = extract_keypoints(results)
                sequence.append(keypoints)

                if results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        results.multi_hand_landmarks[0],
                        mp_hands.HAND_CONNECTIONS
                    )

                cv2.putText(frame, f"WORD: {word}", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                cv2.putText(frame, f"Sample: {sample+1}/{SEQUENCES_PER_WORD}", (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
                cv2.putText(frame, f"Frame: {frame_num+1}/{FRAMES_PER_SEQUENCE}", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

                cv2.imshow("Collecting Word Data", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    exit()

            np.save(os.path.join(word_path, f"{sample}.npy"), np.array(sequence))
            print(f"✔ Saved {word} sample {sample}")

cap.release()
cv2.destroyAllWindows()
print("\n🎉 WORD DATA COLLECTION COMPLETED")