import face_recognition
import cv2
import numpy as np
import os

known_faces = []
known_names = []

def load_known_faces(folder="images"):
    for file in os.listdir(folder):
        img = face_recognition.load_image_file(f"{folder}/{file}")
        encoding = face_recognition.face_encodings(img)
        if encoding:
            known_faces.append(encoding[0])
            known_names.append(file.split(".")[0])

def recognize_faces(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_recognition.face_locations(rgb)
    encodings = face_recognition.face_encodings(rgb, faces)

    results = []

    for encoding, (top, right, bottom, left) in zip(encodings, faces):
        matches = face_recognition.compare_faces(known_faces, encoding)
        name = "Unknown"

        if True in matches:
            name = known_names[matches.index(True)]

        results.append((name, top, right, bottom, left))

    return results