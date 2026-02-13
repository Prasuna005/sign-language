# app.py
import os
import time
import base64
import io
from threading import Lock

from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from PIL import Image
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import pickle

# ----- Config -----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "asl_keypoint_model.h5")
LABELS_PATH = os.path.join(BASE_DIR, "models", "labels.pkl")

CONF_THRESHOLD = 0.75      # min confidence
STABLE_TIME_SEC = 1.5      # 1.5 sec stable gesture
# ------------------

app = Flask(__name__, static_folder="static", template_folder="templates")
socketio = SocketIO(app, cors_allowed_origins="*")
thread_lock = Lock()

# load model + labels
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
if not os.path.exists(LABELS_PATH):
    raise FileNotFoundError(f"Labels not found: {LABELS_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)
with open(LABELS_PATH, "rb") as f:
    class_names = pickle.load(f)

# mediapipe hands
mp_hands = mp.solutions.hands
hands_proc = mp_hands.Hands(
    static_image_mode=False, 
    max_num_hands=1,
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5
)

# stability state (per client: simple single-client)
client_state = {
    "last_label": None,
    "last_time": 0.0,
    "sentence": "",
}

def normalize_landmarks(lms):
    arr = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32)
    arr[:, :2] -= arr[0, :2]
    maxd = np.max(np.linalg.norm(arr[:, :2], axis=1))
    if maxd > 0:
        arr[:, :2] /= maxd
    return arr.flatten()

@app.route("/")
def index():
    return render_template("index.html")

@socketio.on("connect")
def connect():
    emit("init", {"sentence": client_state["sentence"]})

@socketio.on("frame")  
def handle_frame(data):
    try:
        b64 = data.get("image", "")
        header, encoded = b64.split(",", 1)
        frame_bytes = base64.b64decode(encoded)
        img = Image.open(io.BytesIO(frame_bytes)).convert("RGB")
        arr = np.array(img)[:, :, ::-1]  

        # mediapipe expects RGB
        rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        results = hands_proc.process(rgb)

        predicted = None
        conf = 0.0

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            vec = normalize_landmarks(hand.landmark).reshape(1, -1)

            preds = model.predict(vec, verbose=0)
            idx = int(np.argmax(preds))
            predicted = class_names[idx] if idx < len(class_names) else str(idx)
            conf = float(np.max(preds))

            now = time.time()

            if conf >= CONF_THRESHOLD:
                if predicted == client_state["last_label"]:
                    # stable check
                    if (now - client_state["last_time"]) >= STABLE_TIME_SEC:
                        if len(client_state["sentence"]) == 0 or client_state["sentence"][-1] != predicted:
                            if predicted.lower() == "space":
                                client_state["sentence"] += " "
                            elif predicted.lower() == "del":
                                client_state["sentence"] = client_state["sentence"][:-1]
                            else:
                                client_state["sentence"] += predicted
                        # reset timer
                        client_state["last_time"] = now
                else:
                    # new label starts tracking
                    client_state["last_label"] = predicted
                    client_state["last_time"] = now
        else:
            client_state["last_label"] = None
            client_state["last_time"] = 0.0

        emit("prediction", {
            "label": predicted or "",
            "confidence": conf,
            "sentence": client_state["sentence"]
        })

    except Exception as e:
        emit("error", {"error": str(e)})

@socketio.on("control")  
def handle_control(msg):
    a = msg.get("action", "")
    if a == "clear":
        client_state["sentence"] = ""
    elif a == "space":
        client_state["sentence"] += " "
    elif a == "del":
        client_state["sentence"] = client_state["sentence"][:-1]
    emit("update_sentence", {"sentence": client_state["sentence"]})

if __name__ == "__main__":
    print("Using model:", MODEL_PATH)
    print("Server running at http://127.0.0.1:5000")
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)


