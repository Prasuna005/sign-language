import cv2
import mediapipe as mp
import numpy as np
import os

DATASET_DIR = "dataset/word_videos"
OUTPUT_DIR = "dataset_word_keypoints"

os.makedirs(OUTPUT_DIR, exist_ok=True)

mp_hands = mp.solutions.hands

def normalize_landmarks(lms):
    arr = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32)
    arr[:, :2] -= arr[0, :2]
    maxd = np.max(np.linalg.norm(arr[:, :2], axis=1))
    if maxd > 0:
        arr[:, :2] /= maxd
    return arr.flatten()  # 63 values

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
) as hands:

    for word in os.listdir(DATASET_DIR):
        word_path = os.path.join(DATASET_DIR, word)
        if not os.path.isdir(word_path):
            continue

        print(f"Processing word: {word}")
        word_out = os.path.join(OUTPUT_DIR, word)
        os.makedirs(word_out, exist_ok=True)

        for video_name in os.listdir(word_path):
            video_path = os.path.join(word_path, video_name)
            cap = cv2.VideoCapture(video_path)

            frames_keypoints = []

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)

                if results.multi_hand_landmarks:
                    lm = results.multi_hand_landmarks[0]
                    vec = normalize_landmarks(lm.landmark)
                    frames_keypoints.append(vec)

            cap.release()

            if len(frames_keypoints) > 0:
                save_name = video_name.replace(".mp4", ".npy")
                np.save(os.path.join(word_out, save_name),
                        np.array(frames_keypoints))
