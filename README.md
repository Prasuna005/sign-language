# ✨ Next-Gen Multimodal Sign Language Recognition Platform 🤟

This project presents a real-time AI-powered Sign Language Recognition system that translates hand gestures into readable text and speech using deep learning and computer vision techniques.

The platform is designed to bridge the communication gap between hearing-impaired individuals and the general public by enabling real-time gesture-to-text and text-to-speech conversion.

---

## 📌 Problem Statement

Communication between hearing-impaired individuals and non-sign language users can be challenging. Most people are not trained in sign language, which creates communication barriers in education, healthcare, workplaces, and daily life.

An intelligent system is required to:
- Detect hand gestures in real time
- Convert gestures into readable text
- Convert text into audible speech
- Ensure accurate and stable predictions

---

## 🎯 Objectives

- Implement real-time sign language detection using a webcam
- Develop separate AI models for Letters (A–Z), Numbers (0–9), and Words
- Improve prediction stability using confidence thresholds and delay control
- Convert predicted text into speech
- Provide a simple, user-friendly graphical interface

---

## 🧠 Methodology

The system follows these steps:

1. Capture live webcam video using OpenCV
2. Detect hand landmarks using MediaPipe
3. Extract 21 keypoints from the detected hand
4. Normalize keypoints (for letters and words)
5. Pass keypoints into trained deep learning models
6. Apply confidence threshold filtering
7. Display predicted text in the GUI
8. Convert predicted text into speech using pyttsx3

---

## 🤖 Models Used

### 🔤 Letters Model
- 26 Classes (A–Z)
- Keypoint normalization applied
- Dense Neural Network
- Confidence threshold-based filtering

### 🔢 Numbers Model
- 10 Classes (0–9)
- Raw landmark coordinates used (as per training format)
- Dense Neural Network
- Delay mechanism for stable prediction

### 📝 Words Model
- Sequence-based prediction (30 frames)
- LSTM Sequential Model
- Used for dynamic gesture recognition

---

## ⚙️ Features

- 🔤 Alphabet Recognition (A–Z)
- 🔢 Number Recognition (0–9)
- 📝 Word Recognition
- 🎥 Real-time Webcam Detection
- 🔊 Text-to-Speech Conversion
- ⏯ Start / Pause / Stop Camera Controls
- 🎯 Mode Selection (Letters / Numbers / Words)
- ✏ Delete / Clear / New Line Support
- 🖥 User-Friendly GUI using Tkinter

---

## 📊 System Configuration

- Window Size: 1200 x 720
- Camera Resolution: 800 x 450
- Letter Confidence Threshold: 0.7
- Number Confidence Threshold: 0.75
- Word Confidence Threshold: 0.8
- Word Sequence Length: 30 Frames

---

## 🛠 Technologies Used

- Python
- OpenCV
- MediaPipe
- TensorFlow / Keras
- NumPy
- Tkinter
- pyttsx3

---

## 🚀 Applications

- Assistive communication systems
- Smart education platforms
- Inclusive AI solutions
- Healthcare support systems
- Real-time gesture-based interaction systems

---

## 📌 Future Improvements

- Improve number prediction accuracy
- Add sentence-level grammar correction
- Deploy as a web application
- Add multilingual speech output
- Improve dataset size for better model generalization

---

⭐ This project demonstrates the practical implementation of Deep Learning and Computer Vision for building inclusive and accessible AI systems.
