"""
1_basic_face_detection.py
--------------------------
Beginner-tier: real-time face detection using OpenCV's built-in Haar Cascade
classifier. This just DETECTS faces (draws boxes) — it does not know WHO
the person is. Good starting point / sanity check that your webcam + OpenCV
setup works before moving to recognition.

Run:
    python 1_basic_face_detection.py

Press 'q' to quit the webcam window.
"""

import os
import cv2

def get_face_cascade():
    cascade_path = os.path.join(os.path.dirname(__file__), "haarcascade_frontalface_default.xml")
    if os.path.exists(cascade_path):
        return cv2.CascadeClassifier(cascade_path)
    if hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
        return cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    return cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def main():
    face_cascade = get_face_cascade()

    # 0 = default webcam. Change to 1, 2... if you have multiple cameras.
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam. Check camera permissions/index.")
        return

    print("Starting face detection. Press 'q' in the window to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        # Haar cascades work on grayscale images — convert first.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detectMultiScale returns a list of (x, y, w, h) boxes for each face.
        # scaleFactor: how much the image size is reduced at each scale step.
        # minNeighbors: higher = fewer false positives, but may miss real faces.
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame, "Face", (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )

        cv2.putText(
            frame, f"Faces detected: {len(faces)}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )

        cv2.imshow("Face Detection - Press 'q' to quit", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
