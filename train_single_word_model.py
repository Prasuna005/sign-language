# train_single_word_model.py
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
import pickle

DATA_DIR = "dataset_word_keypoints"
WORD = "HOME"
SEQ_LEN = 30

def fix_length(seq, max_len=30):
    if seq.shape[0] > max_len:
        return seq[:max_len]
    elif seq.shape[0] < max_len:
        pad = np.zeros((max_len - seq.shape[0], 63))
        return np.vstack([seq, pad])
    return seq

sequences = []
labels = []

word_dir = os.path.join(DATA_DIR, WORD)

for file in os.listdir(word_dir):
    data = np.load(os.path.join(word_dir, file))
    data = fix_length(data, SEQ_LEN)
    sequences.append(data)
    labels.append(0)  # HOME

X = np.array(sequences)
y = np.array(labels)

print("X shape:", X.shape)
print("y shape:", y.shape)

model = Sequential([
    Input(shape=(SEQ_LEN, 63)),
    LSTM(64, return_sequences=True),
    LSTM(64),
    Dense(32, activation="relu"),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

model.fit(X, y, epochs=30, batch_size=2)

os.makedirs("models", exist_ok=True)
model.save("models/word_lstm_model.h5")

with open("models/word_labels.pkl", "wb") as f:
    pickle.dump({0: "HOME"}, f)

print("✅ HOME word model trained successfully")
