✨ Next-Gen Multimodal Sign Language Recognition Platform 🤟
This project presents a real-time AI-powered sign language recognition system that translates hand gestures into text and speech using Computer Vision and Deep Learning techniques.
The platform promotes inclusive digital communication by enabling interaction between hearing-impaired individuals and the general public.
📌 Problem Statement
Hearing-impaired individuals communicate using sign language, but most people do not understand it. This creates communication barriers in education, healthcare, and daily life.
There is a need for an intelligent system that:
Detects sign language gestures in real time
Converts gestures into readable text
Converts text into audible speech
Maintains high accuracy and smooth user interaction
🎯 Objectives
Implement real-time sign language detection using webcam input
Develop separate deep learning models for Letters, Numbers, and Words
Ensure accurate and stable predictions with delay control
Convert predicted text into speech
Design a user-friendly graphical interface
🧠 Methodology
The workflow includes:
Webcam video capture using OpenCV
Hand landmark detection using MediaPipe
Feature extraction from 21 hand keypoints
Model prediction using trained deep learning models
Confidence filtering and delay control
Text display and speech conversion
Models Used
🔤 Letters Model – Dense Neural Network
🔢 Numbers Model – Dense Neural Network
📝 Words Model – LSTM Sequential Model
📂 Dataset
Since real sign language datasets were not directly used in deployment, custom datasets were created.
Letters Dataset
26 classes (A–Z)
Keypoint-based normalized dataset
Numbers Dataset
10 classes (0–9)
Raw landmark coordinate format
Words Dataset
30-frame sequence-based dataset
Used for LSTM training
⚙️ System Configuration
Window Size: 1200 × 720
Camera Resolution: 800 × 450
Maximum Characters per Line: 24
Confidence Thresholds
Letters: 0.7
Numbers: 0.75
Words: 0.8
🚀 Features
🔤 Alphabet Recognition (A–Z)
🔢 Number Recognition (0–9)
📝 Word-Level Recognition
🔊 Text-to-Speech Conversion
⏯ Start / Pause / Stop Camera
✏ Delete / Clear / New Line Support
📏 Automatic Line Wrapping
🎯 Mode Selection
🖥️ Technologies Used
Python
OpenCV
MediaPipe
TensorFlow / Keras
NumPy
Tkinter
pyttsx3
🌍 Applications
Assistive communication systems
Smart education platforms
Public service kiosks
Accessibility AI solutions


