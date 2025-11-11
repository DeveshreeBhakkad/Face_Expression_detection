 # Face Expression Detection

> "The face is a window to the emotions within."

A Python project that detects human facial expressions in real-time using your webcam...

---

## Features

- Real-time face detection using OpenCV.
- Emotion recognition using TensorFlow/Keras models.
- Works with live webcam feed.
- Displays detected emotion labels dynamically on the video.

---

## Installation

1. Clone this repository:

```bash
git clone https://github.com/DeveshreeBhakkad/Face_Expression_detection
```
Navigate to the project folder:
```bash
cd Face_Expression_detection
```

Install required Python packages:
```bash
pip install -r requirements.txt
```
Make sure you have Python 3.11 installed.

Usage
Run the main program:
```bash
python main.py
```
- Your webcam will open automatically.
- The program will detect faces and show the predicted emotion in real-time.
- Press q (if implemented) to close the webcam window.

Project Structure

Face_Expression_detection/
│
├── main.py            # Main script to run the program
├── README.md          # This file
├── requirements.txt   # Required Python packages
├── .gitignore         # Git ignore file
├── model/             # Optional: saved trained models
└── dataset/           # Optional: sample dataset

Notes
- TensorFlow and OpenCV may show some informational warnings during runtime. These do not affect the program functionality.
- If using custom models, place them in the model/ folder.
- The dataset/ folder is optional and can be used to test the program with images.

Technologies Used
Python 3.11
penCV for real-time video and face detection
ensorFlow/Keras for emotion recognition
DeepFace and RetinaFace libraries for advanced facial feature extraction


