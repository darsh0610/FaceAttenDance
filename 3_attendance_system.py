"""
3_attendance_system.py
------------------------
Advanced-tier: real-time face RECOGNITION + automatic attendance logging.

Prerequisite: run 2_enroll_faces.py first to generate encodings.pkl from
photos in known_faces/.

What it does:
- Opens the webcam
- Detects faces each frame and compares them against known encodings
- Draws a box + name label around recognized faces ("Unknown" if no match)
- Logs each person's first appearance of the day to attendance.csv
  (won't spam duplicate rows if they stay in frame)

Run:
    python 3_attendance_system.py

Press 'q' to quit.
"""

import os
import csv
import pickle
from datetime import datetime
import numpy as np
import cv2

try:
    import face_recognition
    HAS_FACE_RECOGNITION = True
except ImportError:
    HAS_FACE_RECOGNITION = False

ENCODINGS_FILE = "encodings.pkl"
ATTENDANCE_FILE = "attendance.csv"
TOLERANCE = 0.6          # lower = stricter match (0.6 is face_recognition's default)
FRAME_RESIZE_SCALE = 0.25  # shrink frame for faster processing, then scale boxes back up


def detect_and_encode(frame_bgr):
    """Returns face locations (top, right, bottom, left) and encodings."""
    if HAS_FACE_RECOGNITION:
        small_frame = cv2.resize(frame_bgr, (0, 0), fx=FRAME_RESIZE_SCALE, fy=FRAME_RESIZE_SCALE)
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(rgb_small)
        encodings = face_recognition.face_encodings(rgb_small, locations)
        return locations, encodings
    else:
        cascade_path = os.path.join(os.path.dirname(__file__), "haarcascade_frontalface_default.xml")
        if os.path.exists(cascade_path):
            face_cascade = cv2.CascadeClassifier(cascade_path)
        elif hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        else:
            face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
        locations = []
        encodings = []
        for (x, y, w, h) in faces:
            # Scale coordinates into small_frame coordinates so scaling logic below stays identical
            top = int(y * FRAME_RESIZE_SCALE)
            bottom = int((y + h) * FRAME_RESIZE_SCALE)
            left = int(x * FRAME_RESIZE_SCALE)
            right = int((x + w) * FRAME_RESIZE_SCALE)
            locations.append((top, right, bottom, left))

            face_roi = cv2.resize(gray[y:y+h, x:x+w], (64, 64))
            hist = cv2.calcHist([face_roi], [0], None, [128], [0, 256]).flatten()
            norm = np.linalg.norm(hist)
            if norm > 0:
                hist = hist / norm
            encodings.append(hist)
        return locations, encodings


def match_face(known_encodings, candidate_encoding):
    """Compares candidate against known encodings and returns best match name or 'Unknown'."""
    if len(known_encodings) == 0:
        return "Unknown"
    if HAS_FACE_RECOGNITION:
        matches = face_recognition.compare_faces(known_encodings, candidate_encoding, tolerance=TOLERANCE)
        if True in matches:
            distances = face_recognition.face_distance(known_encodings, candidate_encoding)
            best_idx = distances.argmin()
            if matches[best_idx]:
                return best_idx
        return None
    else:
        best_dist = 1.0
        best_idx = None
        for i, known in enumerate(known_encodings):
            similarity = np.dot(known, candidate_encoding) / (
                np.linalg.norm(known) * np.linalg.norm(candidate_encoding) + 1e-7
            )
            dist = 1.0 - similarity
            if dist < 0.25 and dist < best_dist:
                best_dist = dist
                best_idx = i
        return best_idx


def load_encodings():
    if not os.path.exists(ENCODINGS_FILE):
        raise FileNotFoundError(
            f"'{ENCODINGS_FILE}' not found. Run 2_enroll_faces.py first."
        )
    with open(ENCODINGS_FILE, "rb") as f:
        data = pickle.load(f)
    return data["encodings"], data["names"]


def ensure_attendance_file():
    if not os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "Date", "Time"])


def already_marked_today(name):
    """Check if this person already has an attendance row for today's date."""
    today = datetime.now().strftime("%Y-%m-%d")
    if not os.path.exists(ATTENDANCE_FILE):
        return False
    with open(ATTENDANCE_FILE, "r") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for row in reader:
            if len(row) >= 2 and row[0] == name and row[1] == today:
                return True
    return False


def mark_attendance(name):
    if name == "Unknown":
        return
    if already_marked_today(name):
        return
    now = datetime.now()
    with open(ATTENDANCE_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([name, now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")])
    print(f"Attendance marked: {name} at {now.strftime('%H:%M:%S')}")


def main():
    known_encodings, known_names = load_encodings()
    ensure_attendance_file()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting attendance system. Press 'q' to quit.")
    marked_this_session = set()  # avoid re-checking CSV every single frame for the same person

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        face_locations, face_encodings = detect_and_encode(frame)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            best_match_index = match_face(known_encodings, face_encoding)
            name = known_names[best_match_index] if best_match_index is not None else "Unknown"

            # Scale face box back up to original frame size
            top = int(top / FRAME_RESIZE_SCALE)
            right = int(right / FRAME_RESIZE_SCALE)
            bottom = int(bottom / FRAME_RESIZE_SCALE)
            left = int(left / FRAME_RESIZE_SCALE)

            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, bottom - 25), (right, bottom), color, cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            if name != "Unknown" and name not in marked_this_session:
                mark_attendance(name)
                marked_this_session.add(name)

        cv2.imshow("Attendance System - Press 'q' to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nSession ended. Attendance log: {ATTENDANCE_FILE}")


if __name__ == "__main__":
    main()
