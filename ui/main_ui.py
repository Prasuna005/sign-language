import tkinter as tk
import cv2
import PIL.Image, PIL.ImageTk
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle
import time
import pyttsx3
import threading

# ================= CONFIG =================
WINDOW_W = 1200
WINDOW_H = 720
CAM_W = 800
CAM_H = 450

MAX_CHARS_PER_LINE = 24

LETTER_CONF = 0.7
NUMBER_CONF = 0.60   # reduced for better number detection
WORD_CONF = 0.8

LETTER_DELAY = 1.4
NUMBER_DELAY = 1.8
WORD_DELAY = 4.0

CURSOR_BLINK = 0.5
WORD_SEQ_LEN = 30
# =========================================


class SignLanguageApp:
    def __init__(self, root):

        self.root = root
        self.root.title("Next-Gen Multimodal Platform for Inclusive Digital Connectivity")
        self.root.geometry(f"{WINDOW_W}x{WINDOW_H}")
        self.root.configure(bg="#2b2e33")

        # -------- STATE --------
        self.cap = None
        self.running = False
        self.paused = False
        self.after_id = None
        self.mode = None

        self.text = ""
        self.cursor_visible = True
        self.cursor_timer = time.time()

        self.last_pred_time = 0
        self.last_pred_value = ""
        self.word_sequence = []

        # -------- TTS --------
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", 150)

        # -------- MODELS --------
        self.letters_model = tf.keras.models.load_model("models/asl_keypoint_model.h5")
        with open("models/labels.pkl", "rb") as f:
            self.letters_labels = pickle.load(f)

        self.words_model = tf.keras.models.load_model("models/word_lstm_model.h5")
        with open("models/word_labels.pkl", "rb") as f:
            self.word_labels = pickle.load(f)

        self.numbers_model = tf.keras.models.load_model("models/number_keypoint_model.h5")
        with open("models/number_labels.pkl", "rb") as f:
            self.number_labels = pickle.load(f)

        # -------- MEDIAPIPE --------
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils

        self.build_ui()
        self.show_black_frame()

        # -------- KEY BINDS --------
        self.root.bind("d", self.delete_char)
        self.root.bind("c", self.clear_text)
        self.root.bind("<Return>", self.new_line)

    # ---------------- UI ----------------
    def build_ui(self):

        tk.Label(
            self.root,
            text="Next-Gen Multimodal Platform for Inclusive Digital Connectivity",
            font=("Segoe UI", 20, "bold"),
            fg="white", bg="#2b2e33"
        ).pack(pady=12)

        main = tk.Frame(self.root, bg="#2b2e33")
        main.pack(expand=True, fill="both")

        # Camera
        self.camera_frame = tk.Label(main, bg="black", width=CAM_W, height=CAM_H)
        self.camera_frame.pack(side="left", padx=20)

        # Right panel
        right = tk.Frame(main, bg="#1f2227", width=360)
        right.pack(side="right", padx=20, fill="y")

        tk.Label(right, text="Predicted Text",
                 font=("Segoe UI", 14, "bold"),
                 fg="white", bg="#1f2227").pack(pady=10)

        self.pred_label = tk.Label(
            right, font=("Consolas", 15),
            bg="black", fg="lime",
            anchor="nw", justify="left",
            width=28, height=18
        )
        self.pred_label.pack(padx=10, pady=10)

        self.mode_label = tk.Label(
            right, text="Mode: NONE",
            font=("Segoe UI", 12, "bold"),
            fg="cyan", bg="#1f2227"
        )
        self.mode_label.pack(pady=5)

        # Controls
        controls = tk.Frame(self.root, bg="#2b2e33")
        controls.pack(pady=10)

        tk.Button(controls, text="Start", width=10, command=self.start_camera).pack(side="left", padx=4)
        tk.Button(controls, text="Pause", width=10, command=self.pause_camera).pack(side="left", padx=4)
        tk.Button(controls, text="Stop", width=10, command=self.stop_camera).pack(side="left", padx=4)

        tk.Button(controls, text="Letters", width=10,
                  command=lambda: self.set_mode("LETTERS")).pack(side="left", padx=10)

        tk.Button(controls, text="Numbers", width=10,
                  command=lambda: self.set_mode("NUMBERS")).pack(side="left", padx=4)

        tk.Button(controls, text="Words", width=10,
                  command=lambda: self.set_mode("WORDS")).pack(side="left", padx=4)

        tk.Button(controls, text="Speak", width=10,
                  command=self.speak_text).pack(side="left", padx=10)

        tk.Button(controls, text="Exit", width=10,
                  command=self.exit_app).pack(side="left", padx=4)

    # ---------------- CAMERA ----------------
    def show_black_frame(self):
        img = PIL.Image.new("RGB", (CAM_W, CAM_H), (0, 0, 0))
        imgtk = PIL.ImageTk.PhotoImage(img)
        self.camera_frame.configure(image=imgtk)
        self.camera_frame.imgtk = imgtk

    def start_camera(self):
        if self.running:
            return
        self.cap = cv2.VideoCapture(0)
        self.running = True
        self.paused = False
        self.last_pred_time = 0
        self.last_pred_value = ""
        self.loop()

    def pause_camera(self):
        self.paused = not self.paused

    def stop_camera(self):
        self.running = False
        self.paused = False
        if self.after_id:
            self.root.after_cancel(self.after_id)
            self.after_id = None
        if self.cap:
            self.cap.release()
            self.cap = None
        self.show_black_frame()

    def exit_app(self):
        self.stop_camera()
        self.root.destroy()

    # ---------------- MODE ----------------
    def set_mode(self, mode):
        self.mode = mode
        self.text = ""
        self.word_sequence.clear()
        self.last_pred_time = 0
        self.last_pred_value = ""
        self.mode_label.config(text=f"Mode: {mode}")

    # ---------------- TEXT ----------------
    def delete_char(self, e=None):
        self.text = self.text[:-1]

    def clear_text(self, e=None):
        self.text = ""

    def new_line(self, e=None):
        self.text += "\n"

    def auto_newline(self):
        lines = self.text.split("\n")
        if len(lines[-1]) >= MAX_CHARS_PER_LINE:
            self.text += "\n"

    # ---------------- NORMALIZATION ----------------
    def normalize_letters(self, hand):
        arr = np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark], dtype=np.float32)
        arr[:, :2] -= arr[0, :2]
        scale = np.max(np.linalg.norm(arr[:, :2], axis=1))
        if scale > 0:
            arr[:, :2] /= scale
        return arr.flatten()

    # ---------------- SPEECH ----------------
    def speak_text(self):
        if not self.text.strip():
            return
        threading.Thread(
            target=lambda: (self.engine.say(self.text), self.engine.runAndWait()),
            daemon=True
        ).start()

    # ---------------- MAIN LOOP ----------------
    def loop(self):

        if not self.running:
            return

        if self.paused:
            self.after_id = self.root.after(15, self.loop)
            return

        ret, frame = self.cap.read()
        if not ret:
            self.after_id = self.root.after(15, self.loop)
            return

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        now = time.time()

        if results.multi_hand_landmarks and self.mode is not None:

            hand = results.multi_hand_landmarks[0]
            self.mp_draw.draw_landmarks(frame, hand, self.mp_hands.HAND_CONNECTIONS)

            # LETTERS
            if self.mode == "LETTERS":
                kp = self.normalize_letters(hand)
                preds = self.letters_model.predict(kp.reshape(1, -1), verbose=0)[0]
                idx = np.argmax(preds)
                value = self.letters_labels[idx]

                if preds[idx] > LETTER_CONF and value != self.last_pred_value and now - self.last_pred_time > LETTER_DELAY:
                    self.text += value
                    self.last_pred_value = value
                    self.last_pred_time = now
                    self.auto_newline()

            # NUMBERS (EXACT STANDALONE LOGIC)
            elif self.mode == "NUMBERS":
                kp = np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark]).flatten().reshape(1, -1)
                preds = self.numbers_model.predict(kp, verbose=0)
                idx = np.argmax(preds)
                confidence=np.max(preds)
                value = str(self.number_labels[idx])
                print("predicted:", value, "confidence:", confidence)

                if confidence > NUMBER_CONF and value != self.last_pred_value and now - self.last_pred_time > NUMBER_DELAY:
                    self.text += value
                    self.last_pred_value = value
                    self.last_pred_time = now
                    self.auto_newline()

            # WORDS
            elif self.mode == "WORDS":
                kp = self.normalize_letters(hand)
                self.word_sequence.append(kp)

                if len(self.word_sequence) > WORD_SEQ_LEN:
                    self.word_sequence.pop(0)

                if len(self.word_sequence) == WORD_SEQ_LEN:
                    preds = self.words_model.predict(np.expand_dims(self.word_sequence, axis=0), verbose=0)[0]
                    idx = np.argmax(preds)
                    value = self.word_labels[idx]

                    if preds[idx] > WORD_CONF and value != self.last_pred_value and now - self.last_pred_time > WORD_DELAY:
                        self.text += value + " "
                        self.last_pred_value = value
                        self.last_pred_time = now
                        self.word_sequence.clear()
                        self.auto_newline()

        # Cursor blink
        if time.time() - self.cursor_timer > CURSOR_BLINK:
            self.cursor_visible = not self.cursor_visible
            self.cursor_timer = time.time()

        self.pred_label.config(text=self.text + ("|" if self.cursor_visible else ""))

        frame = cv2.resize(frame, (CAM_W, CAM_H))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = PIL.ImageTk.PhotoImage(PIL.Image.fromarray(frame))
        self.camera_frame.configure(image=img)
        self.camera_frame.imgtk = img

        self.after_id = self.root.after(15, self.loop)


# -------- RUN --------
root = tk.Tk()
app = SignLanguageApp(root)
root.mainloop()