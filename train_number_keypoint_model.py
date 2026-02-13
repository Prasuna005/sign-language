import numpy as np
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import pickle

DATA_PATH = "dataset_numbers_keypoints"
NUMBERS = [str(i) for i in range(10)]

X, y = [], []

label_map = {num: idx for idx, num in enumerate(NUMBERS)}

for number in NUMBERS:
    folder = os.path.join(DATA_PATH, number)
    for file in os.listdir(folder):
        if file.endswith(".npy"):
            data = np.load(os.path.join(folder, file))
            X.append(data)
            y.append(label_map[number])

X = np.array(X)
y = to_categorical(y, num_classes=len(NUMBERS))

print("X shape:", X.shape)
print("y shape:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(63,)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=30,
    batch_size=16
)

os.makedirs("models", exist_ok=True)
model.save("models/number_keypoint_model.h5")

with open("models/number_labels.pkl", "wb") as f:
    pickle.dump(NUMBERS, f)

print("✅ Number skeleton model trained & saved")