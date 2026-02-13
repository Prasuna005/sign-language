import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
import pickle

DATA_PATH = "dataset_word_keypoints"

labels = sorted(os.listdir(DATA_PATH))
label_map = {label: i for i, label in enumerate(labels)}

X, y = [], []

for label in labels:
    for file in os.listdir(os.path.join(DATA_PATH, label)):
        data = np.load(os.path.join(DATA_PATH, label, file))
        X.append(data)
        y.append(label_map[label])

X = np.array(X)
y = to_categorical(y, num_classes=len(labels))

print("Labels:", label_map)
print("X shape:", X.shape)

model = Sequential([
    Input(shape=(30, 63)),
    LSTM(128, return_sequences=True),
    Dropout(0.3),
    LSTM(64),
    Dense(64, activation='relu'),
    Dense(len(labels), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(X, y, epochs=40, batch_size=8, shuffle=True)

os.makedirs("models", exist_ok=True)
model.save("models/word_lstm_model.h5")

with open("models/word_labels.pkl", "wb") as f:
    pickle.dump(labels, f)

print("✅ Word model trained successfully")