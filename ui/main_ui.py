# ================= FINAL UI + CAMERA + PREDICTION =================
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
import os

WINDOW_W = 1500
WINDOW_H = 850
CAM_W = 700
CAM_H = 400

LETTER_CONF = 0.6
NUMBER_CONF = 0.5
WORD_CONF = 0.80

LETTER_DELAY = 1.2
NUMBER_DELAY = 1.0
WORD_DELAY = 2.0
SEQUENCE_LENGTH = 30

BLINK_DELAY = 500

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LETTER_IMAGES_DIR = os.path.join(BASE_DIR, "images", "letters")
NUMBER_IMAGES_DIR = os.path.join(BASE_DIR, "images", "numbers")
WORD_IMAGES_DIR = os.path.join(BASE_DIR, "images", "words")


class SignLanguageApp:

    def __init__(self, root):

        self.root = root
        self.root.title("Next-Gen Multimodal Platform for Inclusive Digital Connectivity")
        self.root.geometry(f"{WINDOW_W}x{WINDOW_H}")
        self.root.configure(bg="white")

        self.cap = None
        self.running = False
        self.paused = False
        self.mode = None
        self.text = ""
        self.last_pred_time = 0
        self.cursor_visible = True
        self.cursor_pos = 0

        # ✅ NEW (auto space tracking)
        self.last_hand_seen = time.time()

        self.engine = pyttsx3.init()
        rate = self.engine.getProperty("rate")
        self.engine.setProperty("rate", rate - 60)

        self.letters_model = tf.keras.models.load_model("models/asl_keypoint_model.h5")
        self.numbers_model = tf.keras.models.load_model("models/number_keypoint_model.h5")

        self.letters_labels = pickle.load(open("models/labels.pkl", "rb"))
        self.number_labels = pickle.load(open("models/number_labels.pkl", "rb"))

        self.words_model = None
        self.word_labels = None
        self.word_sequence = []

        if os.path.exists("models/word_lstm_model.h5") and os.path.exists("models/word_labels.pkl"):
            self.words_model = tf.keras.models.load_model("models/word_lstm_model.h5")
            self.word_labels = pickle.load(open("models/word_labels.pkl", "rb"))

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1)
        self.mp_draw = mp.solutions.drawing_utils

        self.build_ui()
        self.show_black()
        self.blink_cursor()

        root.bind("<Key>", self.key_handler)
        root.bind("<Up>", self.cursor_up)
        root.bind("<Down>", self.cursor_down)
        root.bind("<Left>", self.cursor_left)
        root.bind("<Right>", self.cursor_right)

    # ================= UI =================
    def build_ui(self):
        header = tk.Frame(self.root, bg="#ff4fa3", height=70)
        header.pack(fill="x")

        tk.Label(header,
            text="✨ Next-Gen Multimodal Platform for Inclusive Digital Connectivity ✨",
            bg="#ff4fa3", fg="white",
            font=("Segoe UI", 18, "bold")
        ).pack(pady=15)

        body = tk.Frame(self.root, bg="white")
        body.pack(expand=True, fill="both", padx=10, pady=10)

        left = tk.Frame(body, bg="#fff2cc", width=200)
        left.pack(side="left", fill="y", padx=10)

        tk.Label(left, text="Reference Image Gallery",
                 bg="#fff2cc", font=("Segoe UI", 12, "bold")).pack(pady=10)

        tk.Button(left, text="Letters", bg="#ff4fa3", fg="white",
                  width=15, height=2,
                  command=lambda: self.open_gallery("LETTERS")).pack(pady=5)

        tk.Button(left, text="Numbers", bg="#4fd1c5", fg="white",
                  width=15, height=2,
                  command=lambda: self.open_gallery("NUMBERS")).pack(pady=5)

        tk.Button(left, text="Words", bg="#9be7e7", fg="white",
                  width=15, height=2,
                  command=lambda: self.open_gallery("WORDS")).pack(pady=5)

        tk.Label(left, text="Prediction Mode",
                 bg="#fff2cc", font=("Segoe UI", 12, "bold")).pack(pady=15)

        tk.Button(left, text="LETTERS", bg="#ec407a", fg="white",
                  width=15, command=lambda: self.set_mode("LETTERS")).pack(pady=4)

        tk.Button(left, text="NUMBERS", bg="#26c6da", fg="white",
                  width=15, command=lambda: self.set_mode("NUMBERS")).pack(pady=4)

        tk.Button(left, text="WORDS", bg="#66bb6a", fg="white",
                  width=15, command=lambda: self.set_mode("WORDS")).pack(pady=4)

        center = tk.Frame(body, bg="white")
        center.pack(side="left", padx=20)

        tk.Label(center, text="Live Camera Feed",
                 font=("Segoe UI", 12, "bold")).pack()

        self.camera = tk.Label(center, bg="black",
                               width=CAM_W, height=CAM_H)
        self.camera.pack(pady=10)

        controls = tk.Frame(center, bg="white")
        controls.pack(pady=10)

        tk.Button(controls, text="START", bg="#00c853", fg="white",
                  width=10, command=self.start).grid(row=0, column=0, padx=5)

        tk.Button(controls, text="PAUSE", bg="#ffab00", fg="white",
                  width=10, command=self.pause).grid(row=0, column=1, padx=5)

        tk.Button(controls, text="STOP", bg="#d50000", fg="white",
                  width=10, command=self.stop).grid(row=0, column=2, padx=5)

        tk.Button(controls, text="EXIT", bg="#37474f", fg="white",
                  width=10, command=self.root.destroy).grid(row=0, column=3, padx=5)

        right = tk.Frame(body, bg="white")
        right.pack(side="left", fill="both", expand=True)

        tk.Label(right, text="Predicted Text",
                 font=("Segoe UI", 12, "bold")).pack()

        self.textbox = tk.Label(
            right,
            bg="white",
            fg="#003366",
            font=("Consolas", 14, "bold"),
            anchor="nw",
            justify="left",
            relief="solid",
            bd=1,
            wraplength=400
        )
        self.textbox.pack(expand=True, fill="both", padx=10, pady=10)

        self.mode_label = tk.Label(
            right, text="Mode: NONE",
            font=("Segoe UI", 11, "bold"),
            fg="#009688"
        )
        self.mode_label.pack()

        tk.Button(
            right,
            text="🔊 Speak",
            bg="#8e24aa",
            fg="white",
            width=15,
            command=self.speak
        ).pack(pady=5)

    # ================= SPEAK =================
    def speak(self):
        if not self.text.strip():
            return

        speak_text = self.text.strip()
        if speak_text.isdigit():
            speak_text = " ".join(list(speak_text))

        def speak_thread():
            try:
                engine = pyttsx3.init()
                rate = engine.getProperty("rate")
                engine.setProperty("rate", rate - 60)
                engine.say(speak_text)
                engine.runAndWait()
            except Exception as e:
                print(f"Error speaking: {e}")

        threading.Thread(target=speak_thread, daemon=True).start()

    # ================= CAMERA =================
    def show_black(self):
        img = PIL.Image.new("RGB", (CAM_W, CAM_H), (0, 0, 0))
        imgtk = PIL.ImageTk.PhotoImage(img)
        self.camera.configure(image=imgtk)
        self.camera.imgtk = imgtk

    def start(self):
        if self.running:
            return
        self.cap = cv2.VideoCapture(0)
        self.running = True
        self.paused = False
        self.loop()

    def pause(self):
        self.paused = not self.paused

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.show_black()

    def set_mode(self, m):
        self.mode = m
        self.text = ""
        self.cursor_pos = 0
        self.word_sequence.clear()
        self.mode_label.config(text=f"Mode: {m}")

    def key_handler(self, e):
        if e.keysym == "Return":
            self.text = self.text[:self.cursor_pos] + "\n" + self.text[self.cursor_pos:]
            self.cursor_pos += 1
        elif e.char == "d":
            if self.cursor_pos > 0:
                self.text = self.text[:self.cursor_pos-1] + self.text[self.cursor_pos:]
                self.cursor_pos -= 1
        elif e.char == "c":
            self.text = ""
            self.cursor_pos = 0

    def cursor_up(self, e=None):
        if self.cursor_pos > 0:
            self.cursor_pos -= 1

    def cursor_down(self, e=None):
        if self.cursor_pos < len(self.text):
            self.cursor_pos += 1

    def cursor_left(self, e=None):
        if self.cursor_pos > 0:
            self.cursor_pos -= 1

    def cursor_right(self, e=None):
        if self.cursor_pos < len(self.text):
            self.cursor_pos += 1

    def blink_cursor(self):
        self.cursor_visible = not self.cursor_visible
        self.root.after(BLINK_DELAY, self.blink_cursor)

    def normalize(self, hand):
        arr = np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark])
        arr[:, :2] -= arr[0, :2]
        scale = np.max(np.linalg.norm(arr[:, :2], axis=1))
        if scale > 0:
            arr[:, :2] /= scale
        return arr.flatten()

    # ================= LOOP =================
    def loop(self):

        if not self.running:
            return

        if self.paused:
            self.root.after(30, self.loop)
            return

        ret, frame = self.cap.read()
        if not ret:
            self.root.after(30, self.loop)
            return

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.hands.process(rgb)
        now = time.time()

        if res.multi_hand_landmarks and self.mode:

            # ✅ update hand seen time
            self.last_hand_seen = now

            hand = res.multi_hand_landmarks[0]
            self.mp_draw.draw_landmarks(frame, hand, self.mp_hands.HAND_CONNECTIONS)

            kp = self.normalize(hand)
            raw_kp = np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark]).flatten()

            if self.mode == "NUMBERS":

                preds = self.numbers_model.predict(raw_kp.reshape(1, -1), verbose=0)[0]
                idx = np.argmax(preds)

                if preds[idx] > NUMBER_CONF and now - self.last_pred_time > NUMBER_DELAY:
                    num = str(self.number_labels[idx])
                    self.text = self.text[:self.cursor_pos] + num + self.text[self.cursor_pos:]
                    self.cursor_pos += len(num)
                    self.last_pred_time = now

            if self.mode == "LETTERS":

                preds = self.letters_model.predict(kp.reshape(1, -1), verbose=0)[0]
                idx = np.argmax(preds)

                if preds[idx] > LETTER_CONF and now - self.last_pred_time > LETTER_DELAY:
                    ch = self.letters_labels[idx]
                    self.text = self.text[:self.cursor_pos] + ch + self.text[self.cursor_pos:]
                    self.cursor_pos += 1
                    self.last_pred_time = now

            if self.mode == "WORDS" and self.words_model:

                self.word_sequence.append(kp)

                if len(self.word_sequence) > SEQUENCE_LENGTH:
                    self.word_sequence.pop(0)

                if len(self.word_sequence) == SEQUENCE_LENGTH:

                    input_data = np.expand_dims(self.word_sequence, axis=0)
                    preds = self.words_model.predict(input_data, verbose=0)[0]

                    idx = np.argmax(preds)
                    conf = preds[idx]

                    if conf > WORD_CONF and now - self.last_pred_time > WORD_DELAY:

                        word = self.word_labels[idx] + " "
                        self.text = self.text[:self.cursor_pos] + word + self.text[self.cursor_pos:]
                        self.cursor_pos += len(word)
                        self.last_pred_time = now
                        self.word_sequence.clear()

        else:
            # ✅ auto space when hand not detected
            if now - self.last_hand_seen > 1.5:
                if self.text and (self.cursor_pos == 0 or self.text[self.cursor_pos-1] != " "):
                    self.text = self.text[:self.cursor_pos] + " " + self.text[self.cursor_pos:]
                    self.cursor_pos += 1
                self.last_hand_seen = now

        display_text = self.text[:self.cursor_pos] + ("|" if self.cursor_visible else " ") + self.text[self.cursor_pos:]
        self.textbox.config(text=display_text)

        frame = cv2.resize(frame, (CAM_W, CAM_H))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        img = PIL.ImageTk.PhotoImage(PIL.Image.fromarray(frame))
        self.camera.configure(image=img)
        self.camera.imgtk = img

        self.root.after(30, self.loop)

    # ================= GALLERY =================
    def open_gallery(self, t):

        win = tk.Toplevel(self.root)
        win.title(f"{t} Reference")
        win.geometry("1000x600")

        tk.Label(
            win,
            text=f"{t} Sign Language Reference Images",
            bg="#ff4fa3",
            fg="white",
            font=("Segoe UI", 16, "bold")
        ).pack(fill="x")

        grid = tk.Frame(win)
        grid.pack(pady=20)

        if t == "LETTERS":
            items = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
            path = LETTER_IMAGES_DIR
            cols = 10
            img_size = 90

        elif t == "NUMBERS":
            items = [str(i) for i in range(10)]
            path = NUMBER_IMAGES_DIR
            cols = 5
            img_size = 90

        else:
            items = [
                "HELLO","GO","THANKYOU","HELP","STOP",
                "YES","NO","PLEASE","WAIT","HOME"
            ]
            path = WORD_IMAGES_DIR
            cols = 5
            img_size = 150

        r = 0
        c = 0

        for it in items:

            frame = tk.Frame(grid, bd=1, relief="solid")
            frame.grid(row=r, column=c, padx=10, pady=10)

            tk.Label(frame, text=it,
                     font=("Segoe UI", 12, "bold")).pack()

            img_path = None

            for ext in [".png",".jpg",".jpeg",".PNG",".JPG"]:
                p = os.path.join(path, it + ext)

                if os.path.exists(p):
                    img_path = p
                    break

            if img_path:

                img = PIL.Image.open(img_path).resize((img_size,img_size))
                photo = PIL.ImageTk.PhotoImage(img)

                lbl = tk.Label(frame, image=photo)
                lbl.image = photo
                lbl.pack()

            else:

                tk.Label(frame, text="No Image").pack(padx=20, pady=20)

            c += 1

            if c == cols:
                c = 0
                r += 1


root = tk.Tk()
app = SignLanguageApp(root)
root.mainloop()