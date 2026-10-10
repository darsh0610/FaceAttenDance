"""
2_enroll_faces.py
------------------
Enrollment step for face RECOGNITION (not just detection).

Put one clear, front-facing photo of each person you want the system to
recognize into the `known_faces/` folder, named after them:

    known_faces/
        Darsh.jpg
        Priya.png
        Rahul.jpg

Then run this script once to build a face-encoding cache (encodings.pkl).
The main attendance system loads this cache instead of re-scanning images
every time, which is much faster.

Run:
    python 2_enroll_faces.py
"""

import os
import pickle
import numpy as np

try:
    import face_recognition
    HAS_FACE_RECOGNITION = True
except ImportError:
    import cv2
    HAS_FACE_RECOGNITION = False

KNOWN_FACES_DIR = "known_faces"
ENCODINGS_FILE = "encodings.pkl"


def get_face_encoding(image_path):
    """Extract face encoding using face_recognition, or OpenCV fallback if unavailable."""
    if HAS_FACE_RECOGNITION:
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        return encodings
    else:
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            return []
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        if len(faces) == 0:
            return []
        x, y, w, h = faces[0]
        face_roi = cv2.resize(gray[y:y+h, x:x+w], (64, 64))
        hist = cv2.calcHist([face_roi], [0], None, [128], [0, 256]).flatten()
        norm = np.linalg.norm(hist)
        if norm > 0:
            hist = hist / norm
        return [hist]


def build_encodings():
    known_encodings = []
    known_names = []

    if not HAS_FACE_RECOGNITION:
        print("[INFO] 'face_recognition' (dlib) not found. Using OpenCV native descriptor fallback.")

    if not os.path.isdir(KNOWN_FACES_DIR):
        print(f"Error: '{KNOWN_FACES_DIR}' folder not found. Create it and add photos.")
        return

    image_files = [
        f for f in os.listdir(KNOWN_FACES_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if not image_files:
        print(f"No images found in '{KNOWN_FACES_DIR}'. Add a photo per person and re-run.")
        return

    for filename in image_files:
        path = os.path.join(KNOWN_FACES_DIR, filename)
        name = os.path.splitext(filename)[0]  # filename (minus extension) = person's name

        print(f"Processing {filename} -> name: '{name}'")
        encodings = get_face_encoding(path)

        if len(encodings) == 0:
            print(f"  Warning: no face found in {filename}, skipping.")
            continue
        if len(encodings) > 1:
            print(f"  Warning: multiple faces found in {filename}, using the first one.")

        known_encodings.append(encodings[0])
        known_names.append(name)

    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump({"encodings": known_encodings, "names": known_names}, f)

    print(f"\nDone. Enrolled {len(known_names)} face(s): {known_names}")
    print(f"Saved to {ENCODINGS_FILE}")


if __name__ == "__main__":
    build_encodings()
