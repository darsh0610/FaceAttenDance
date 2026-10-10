# Face Detection & Attendance System

A three-stage project: basic face detection → face recognition enrollment →
full real-time attendance system with CSV logging.

## Project Structure

```
face_attendance/
├── requirements.txt
├── 1_basic_face_detection.py   # Stage 1: just detect faces (Haar Cascade)
├── 2_enroll_faces.py           # Stage 2: register known people
├── 3_attendance_system.py      # Stage 3: recognize + log attendance
├── known_faces/                # Put one photo per person here
└── attendance.csv              # Auto-generated log (created on first run)
```

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

   Note: `face_recognition` depends on `dlib`, which needs CMake and a C++
   compiler to build on some systems.
   - **Windows**: install "Visual Studio Build Tools" (C++ workload) first,
     or use `pip install dlib-binary` before installing face_recognition.
   - **Linux**: `sudo apt install cmake build-essential` first.
   - If dlib install fails, search the exact error — it's almost always a
     missing CMake/compiler, well documented for your OS.

2. (Recommended) Use a virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate      # Windows
   source venv/bin/activate   # macOS/Linux
   pip install -r requirements.txt
   ```

## Usage

### Stage 1 — Just try detection first (sanity check)
```
python 1_basic_face_detection.py
```
Confirms your webcam + OpenCV setup works. Draws boxes around any face,
no identity involved.

### Stage 2 — Enroll people you want recognized
Add one clear, front-facing photo per person into `known_faces/`, named
after them (e.g. `known_faces/Darsh.jpg`). Then run:
```
python 2_enroll_faces.py
```
This creates `encodings.pkl`, a cache of face encodings so the main
system doesn't have to reprocess images every run.

### Stage 3 — Run the attendance system
```
python 3_attendance_system.py
```
- Recognized faces get a green box + name label, and an attendance row
  is written to `attendance.csv` (name, date, time) the first time they're
  seen each day.
- Unrecognized faces get a red box labeled "Unknown" — no log entry.
- Press `q` to quit.

## Browser UI (recommended for testing/demo)

Instead of running each script from the terminal, launch the Streamlit UI:
```
streamlit run app.py
```
This opens a local web page (usually `http://localhost:8501`) with four tabs:
- **Detect Only** — snapshot from your webcam, shows face boxes (Stage 1)
- **Enroll a Face** — capture or upload a photo, type a name, saves it and rebuilds `encodings.pkl`
- **Recognize & Mark** — snapshot, matches against enrolled faces, logs attendance
- **Attendance Log** — view and download `attendance.csv`

This is the easiest way to demo the project live (e.g. to a professor) since
everything is point-and-click in the browser rather than separate terminal
commands.

## How it works (for your lab report / viva)

- **Detection**: Stage 1 uses OpenCV's Haar Cascade classifier, a
  fast, classical (pre-deep-learning) object detection method trained to
  recognize face-like patterns of light/dark regions.
- **Recognition**: Stages 2–3 use the `face_recognition` library, built on
  dlib's deep learning face recognition model. It converts each face into
  a 128-dimensional numerical vector (an "encoding"). Two faces are
  considered a match if the Euclidean distance between their encodings is
  below a threshold (`TOLERANCE = 0.6` here).
- **Attendance logging**: A simple CSV write-once-per-day-per-person guard
  prevents duplicate rows while someone lingers in frame.

## Ideas to extend (if you want to go further for submission)

- Swap CSV storage for SQLite so it scales to a real class roster
- Add a Streamlit front-end so it's demo-able without a terminal
- Add email/SMS notification when someone is marked present
- Add liveness detection (blink check) to prevent someone holding up a photo
- Deploy as a small Flask API + web dashboard showing daily attendance

## Troubleshooting

- **Webcam won't open**: try changing `cv2.VideoCapture(0)` to `1` or `2`
- **No face detected in enrollment photo**: use a well-lit, front-facing,
  single-person photo
- **Low accuracy / false matches**: lower `TOLERANCE` in
  `3_attendance_system.py` (e.g. to 0.5) for stricter matching
