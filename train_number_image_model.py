import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import os

# ---------------- CONFIG ----------------
IMG_SIZE = 64
BATCH_SIZE = 16
EPOCHS = 10

TRAIN_DIR = "dataset/number_images/train"
MODEL_PATH = "models/number_image_model.h5"

# ---------------- DATA GENERATOR ----------------
train_gen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=10,
    zoom_range=0.1,
    horizontal_flip=True
)

train_data = train_gen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)

# ---------------- MODEL ----------------
model = Sequential([
    tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),

    Conv2D(32, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(10, activation="softmax")   # 0–9
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ---------------- TRAIN ----------------
model.fit(
    train_data,
    epochs=EPOCHS
)

# ---------------- SAVE MODEL ----------------
os.makedirs("models", exist_ok=True)
model.save(MODEL_PATH)

print("✅ Number image model saved at:", MODEL_PATH)
