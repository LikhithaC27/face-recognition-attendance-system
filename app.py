from flask import Flask, render_template
import cv2
import os
import numpy as np
import sqlite3

app = Flask(__name__)

# Ensure folders exist
if not os.path.exists("dataset"):
    os.makedirs("dataset")

# Home page
@app.route("/")
def index():
    return render_template("index.html")


# REGISTER FACE
@app.route("/register")
def register():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        return "Camera not accessible"

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        return "Failed to capture image"

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return "No face detected. Try again."

    (x, y, w, h) = faces[0]
    face_img = gray[y:y+h, x:x+w]

    cv2.imwrite("dataset/user_1.jpg", face_img)

    return "Face Registered Successfully"


# MARK ATTENDANCE
@app.route("/attendance")
def attendance():
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    faces = []
    labels = []

    img_path = "dataset/user_1.jpg"
    if not os.path.exists(img_path):
        return "No registered face found"

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    faces.append(img)
    labels.append(1)

    recognizer.train(faces, np.array(labels))
    recognizer.save("trainer.yml")

    return "Attendance Marked Successfully"


if __name__ == "__main__":
    app.run(debug=True)
