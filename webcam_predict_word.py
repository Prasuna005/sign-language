import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle
import time

# ---------------- CONFIG ----------------
MODEL_PATH = "models/word_lstm_model.h5"
LABELS_PATH = "models/word_labels.pkl"

SEQUENCE_LENGTH = 30
CONF_THRESHOLD = 0.80
PREDICTION_COOLDOWN = 2.0
# ---------------------------------------

# Load model and labels
model = tf.keras.models.load_model(MODEL_PATH)
with open(LABELS_PATH, "rb") as f:
    class_names = pickle.load(f)

print("Loaded classes:", class_names)

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

sequence = []
sentence = ""
cursor_index = 0

last_prediction = ""
last_prediction_time = 0
last_hand_state = None  # to avoid CMD spam

cursor_visible = True
cursor_timer = time.time()
cursor_blink_interval = 0.5

with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        display_text = "Show hand"
        display_color = (0, 0, 255)

        if results.multi_hand_landmarks:
            if last_hand_state != "HAND":
                print("HAND DETECTED")
                last_hand_state = "HAND"

            hand = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            kp = normalize_landmarks(hand.landmark)
            sequence.append(kp)

            if len(sequence) > SEQUENCE_LENGTH:
                sequence.pop(0)

            if len(sequence) == SEQUENCE_LENGTH:
                input_data = np.expand_dims(sequence, axis=0)
                preds = model.predict(input_data, verbose=0)[0]

                idx = np.argmax(preds)
                conf = preds[idx]
                word = class_names[idx]

                if conf >= CONF_THRESHOLD:
                    now = time.time()
                    if word != last_prediction or (now - last_prediction_time) > PREDICTION_COOLDOWN:
                        sentence = sentence[:cursor_index] + word + " " + sentence[cursor_index:]
                        cursor_index += len(word) + 1
                        last_prediction = word
                        last_prediction_time = now
                        sequence.clear()

                        display_text = f"{word} ({conf:.2f})"
                        display_color = (0, 255, 0)

        else:
            if last_hand_state != "NO_HAND":
                print("NO HAND")
                last_hand_state = "NO_HAND"

            sequence.clear()

        # ---------------- UI ----------------
        h, w, _ = frame.shape
        board_w = 420
        board = np.zeros((h, board_w, 3), dtype=np.uint8)

        cv2.putText(board, "Predicted Words:", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
        cv2.rectangle(board, (15, 60), (board_w-15, h-80), (255,255,255), 2)

        # Cursor blink
        if time.time() - cursor_timer > cursor_blink_interval:
            cursor_visible = not cursor_visible
            cursor_timer = time.time()

        display_sentence = sentence
        if cursor_visible:
            display_sentence = sentence[:cursor_index] + "|" + sentence[cursor_index:]

        # Wrap text
        y = 100
        max_chars = 22
        for i in range(0, len(display_sentence), max_chars):
            line = display_sentence[i:i+max_chars]
            cv2.putText(board, line, (30, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
            y += 35

        # Controls hint
        cv2.putText(board, "[D] Delete  [C] Clear", (20, h-40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)

        combined = np.hstack((frame, board))

        cv2.putText(combined, display_text, (30, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, display_color, 3)

        cv2.imshow("WORD SIGN DETECTOR", combined)

        # -------- KEY CONTROLS --------
        key = cv2.waitKeyEx(1)

        if key == ord('q'):
            break
        elif key == ord('c'):  # clear
            sentence = ""
            cursor_index = 0
        elif key == ord('d'):  # delete one word
            if cursor_index > 0:
                sentence = sentence[:cursor_index-1] + sentence[cursor_index:]
                cursor_index -= 1
        elif key == 2424832:  # left arrow
            cursor_index = max(0, cursor_index - 1)
        elif key == 2555904:  # right arrow
            cursor_index = min(len(sentence), cursor_index + 1)

cap.release()
cv2.destroyAllWindows()