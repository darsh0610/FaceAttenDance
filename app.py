"""
app.py
------
Streamlit Web Interface for the Face Detection & Attendance System.
Provides 4 intuitive tabs:
1. Detect Only (Stage 1: Haar Cascade detection)
2. Enroll a Face (Stage 2: Save photo & compute encodings)
3. Recognize & Mark (Stage 3: Match face & log attendance)
4. Attendance Log (View & download CSV records)
"""

import os
import csv
import pickle
from datetime import datetime
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import cv2

# Optional face_recognition import with fallback
try:
    import face_recognition
    HAS_FACE_RECOGNITION = True
except ImportError:
    HAS_FACE_RECOGNITION = False

# Paths & Constants
KNOWN_FACES_DIR = "known_faces"
ENCODINGS_FILE = "encodings.pkl"
ATTENDANCE_FILE = "attendance.csv"
TOLERANCE = 0.6

os.makedirs(KNOWN_FACES_DIR, exist_ok=True)


# --- Helper Functions ---

def ensure_attendance_file():
    if not os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "Date", "Time"])


def already_marked_today(name):
    today = datetime.now().strftime("%Y-%m-%d")
    if not os.path.exists(ATTENDANCE_FILE):
        return False
    with open(ATTENDANCE_FILE, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) >= 2 and row[0] == name and row[1] == today:
                return True
    return False


def mark_attendance(name):
    if name == "Unknown":
        return False, "Unknown face - not recorded"
    if already_marked_today(name):
        return False, f"Already marked for {name} today"
    now = datetime.now()
    with open(ATTENDANCE_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([name, now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")])
    return True, f"Attendance marked for {name} at {now.strftime('%H:%M:%S')}"


# --- Face Representation (Hybrid: face_recognition or OpenCV fallback) ---

def extract_face_vector(image_np):
    """
    Returns (locations, encodings) for an RGB numpy image.
    If face_recognition is available, uses dlib embeddings.
    Otherwise uses OpenCV Haar Cascade face crop + normalized color/spatial signature.
    """
    if HAS_FACE_RECOGNITION:
        locations = face_recognition.face_locations(image_np)
        encodings = face_recognition.face_encodings(image_np, locations)
        return locations, encodings
    else:
        # Fallback: Haar Cascade + normalized 128-bin multi-channel representation
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        locations = []
        encodings = []
        for (x, y, w, h) in faces:
            # Format like face_recognition: (top, right, bottom, left)
            top, right, bottom, left = y, x + w, y + h, x
            locations.append((top, right, bottom, left))

            face_roi = cv2.resize(gray[y:y+h, x:x+w], (64, 64))
            hist = cv2.calcHist([face_roi], [0], None, [128], [0, 256]).flatten()
            norm = np.linalg.norm(hist)
            if norm > 0:
                hist = hist / norm
            encodings.append(hist)
        return locations, encodings


def compare_encodings(known_encodings, candidate_encoding, tolerance=TOLERANCE):
    if len(known_encodings) == 0:
        return []
    if HAS_FACE_RECOGNITION:
        return face_recognition.compare_faces(known_encodings, candidate_encoding, tolerance=tolerance)
    else:
        # Cosine distance fallback
        matches = []
        for known in known_encodings:
            similarity = np.dot(known, candidate_encoding) / (
                np.linalg.norm(known) * np.linalg.norm(candidate_encoding) + 1e-7
            )
            dist = 1.0 - similarity
            matches.append(dist < 0.25)
        return matches


def compute_distances(known_encodings, candidate_encoding):
    if len(known_encodings) == 0:
        return np.array([])
    if HAS_FACE_RECOGNITION:
        return face_recognition.face_distance(known_encodings, candidate_encoding)
    else:
        distances = []
        for known in known_encodings:
            similarity = np.dot(known, candidate_encoding) / (
                np.linalg.norm(known) * np.linalg.norm(candidate_encoding) + 1e-7
            )
            distances.append(1.0 - similarity)
        return np.array(distances)


def rebuild_encodings():
    known_encodings = []
    known_names = []

    image_files = [
        f for f in os.listdir(KNOWN_FACES_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    for filename in image_files:
        path = os.path.join(KNOWN_FACES_DIR, filename)
        name = os.path.splitext(filename)[0]

        img_bgr = cv2.imread(path)
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        _, encodings = extract_face_vector(img_rgb)

        if len(encodings) > 0:
            known_encodings.append(encodings[0])
            known_names.append(name)

    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump({"encodings": known_encodings, "names": known_names}, f)

    return known_names


def load_encodings():
    if not os.path.exists(ENCODINGS_FILE):
        return [], []
    try:
        with open(ENCODINGS_FILE, "rb") as f:
            data = pickle.load(f)
        return data.get("encodings", []), data.get("names", [])
    except Exception:
        return [], []


# --- Streamlit UI Setup ---

st.set_page_config(
    page_title="Face Detection & Attendance System",
    page_icon="🎓",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #64748B;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .status-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
    }
    .badge-active {
        background-color: #DCFCE7;
        color: #166534;
    }
    .badge-warn {
        background-color: #FEF9C3;
        color: #854D0E;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">🎓 Face Detection & Attendance System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Automated real-time face detection, enrollment, and attendance logging</div>', unsafe_allow_html=True)

ensure_attendance_file()

# Engine notification
if HAS_FACE_RECOGNITION:
    st.sidebar.success("Engine: dlib / face_recognition (Active)")
else:
    st.sidebar.info("Engine: OpenCV Native Backend (Active)\n\n(Install dlib & face_recognition for 128-d deep model)")

tabs = st.tabs(["🔍 Detect Only", "👤 Enroll a Face", "✅ Recognize & Mark", "📋 Attendance Log"])


# --- Tab 1: Detect Only ---
with tabs[0]:
    st.header("Stage 1: Basic Face Detection")
    st.write("Sanity check to detect faces using OpenCV's Haar Cascade classifier.")

    col1, col2 = st.columns([1, 1])

    with col1:
        detect_source = st.radio("Choose input method:", ["Camera Snapshot", "Upload Image"], horizontal=True, key="detect_radio")
        image_to_detect = None

        if detect_source == "Camera Snapshot":
            cam_img = st.camera_input("Capture frame for detection", key="detect_cam")
            if cam_img:
                image_to_detect = Image.open(cam_img)
        else:
            up_file = st.file_uploader("Upload an image for face detection", type=["jpg", "jpeg", "png"], key="detect_upload")
            if up_file:
                image_to_detect = Image.open(up_file)

    with col2:
        if image_to_detect is not None:
            img_np = np.array(image_to_detect.convert("RGB"))
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

            display_img = img_np.copy()
            for (x, y, w, h) in faces:
                cv2.rectangle(display_img, (x, y), (x + w, y + h), (34, 197, 94), 3)
                cv2.putText(display_img, "Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (34, 197, 94), 2)

            st.image(display_img, caption=f"Detection Result: {len(faces)} face(s) found", use_container_width=True)
            st.metric("Total Faces Detected", len(faces))
        else:
            st.info("Take a snapshot or upload an image to run face detection.")


# --- Tab 2: Enroll a Face ---
with tabs[1]:
    st.header("Stage 2: Enroll Faces")
    st.write("Register a new person into `known_faces/` and generate encodings.")

    col_enroll_left, col_enroll_right = st.columns([1, 1])

    with col_enroll_left:
        enroll_name = st.text_input("Person Name (e.g., Darsh, Priya, Alex):", placeholder="Enter name...")
        enroll_source = st.radio("Provide photo via:", ["Camera Snapshot", "Upload Image"], horizontal=True, key="enroll_radio")
        photo_to_enroll = None

        if enroll_source == "Camera Snapshot":
            enroll_cam = st.camera_input("Take photo for enrollment", key="enroll_cam")
            if enroll_cam:
                photo_to_enroll = Image.open(enroll_cam)
        else:
            enroll_up = st.file_uploader("Upload photo for enrollment", type=["jpg", "jpeg", "png"], key="enroll_up")
            if enroll_up:
                photo_to_enroll = Image.open(enroll_up)

        if st.button("Save & Enroll Face", type="primary", use_container_width=True):
            clean_name = enroll_name.strip()
            if not clean_name:
                st.error("Please enter a valid name before enrolling.")
            elif photo_to_enroll is None:
                st.error("Please capture or upload a photo.")
            else:
                save_path = os.path.join(KNOWN_FACES_DIR, f"{clean_name}.jpg")
                photo_to_enroll.convert("RGB").save(save_path, "JPEG")
                st.success(f"Saved photo to `{save_path}`")

                with st.spinner("Building face encodings..."):
                    enrolled = rebuild_encodings()
                st.success(f"Enrolled successfully! Registered persons: {', '.join(enrolled)}")
                st.rerun()

    with col_enroll_right:
        st.subheader("Enrolled Persons")
        known_files = [f for f in os.listdir(KNOWN_FACES_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if not known_files:
            st.info("No enrolled faces yet. Add one on the left.")
        else:
            for kf in known_files:
                name = os.path.splitext(kf)[0]
                img_path = os.path.join(KNOWN_FACES_DIR, kf)
                c_img, c_info = st.columns([1, 2])
                with c_img:
                    st.image(img_path, width=90)
                with c_info:
                    st.write(f"**{name}**")
                    if st.button(f"Delete {name}", key=f"del_{kf}"):
                        os.remove(img_path)
                        rebuild_encodings()
                        st.success(f"Removed {name}")
                        st.rerun()


# --- Tab 3: Recognize & Mark ---
with tabs[2]:
    st.header("Stage 3: Recognize & Mark Attendance")
    st.write("Scan a face, match against enrolled profiles, and auto-record attendance.")

    known_encodings, known_names = load_encodings()

    if len(known_names) == 0:
        st.warning("No face encodings found! Please enroll at least one person in the 'Enroll a Face' tab first.")
    else:
        st.caption(f"Currently recognizing against {len(known_names)} enrolled person(s): {', '.join(known_names)}")

    rec_col1, rec_col2 = st.columns([1, 1])

    with rec_col1:
        rec_source = st.radio("Input method:", ["Camera Snapshot", "Upload Image"], horizontal=True, key="rec_radio")
        rec_image = None
        if rec_source == "Camera Snapshot":
            rec_cam = st.camera_input("Capture frame for attendance", key="rec_cam")
            if rec_cam:
                rec_image = Image.open(rec_cam)
        else:
            rec_up = st.file_uploader("Upload image for recognition", type=["jpg", "jpeg", "png"], key="rec_up")
            if rec_up:
                rec_image = Image.open(rec_up)

    with rec_col2:
        if rec_image is not None and len(known_names) > 0:
            img_np = np.array(rec_image.convert("RGB"))
            locations, encodings = extract_face_vector(img_np)

            annotated_img = img_np.copy()
            detected_names = []

            for (top, right, bottom, left), face_enc in zip(locations, encodings):
                matches = compare_encodings(known_encodings, face_enc, tolerance=TOLERANCE)
                name = "Unknown"

                if True in matches:
                    distances = compute_distances(known_encodings, face_enc)
                    best_match_idx = distances.argmin()
                    if matches[best_match_idx]:
                        name = known_names[best_match_idx]

                detected_names.append(name)

                # Draw bounding box
                color = (34, 197, 94) if name != "Unknown" else (239, 68, 68)
                cv2.rectangle(annotated_img, (left, top), (right, bottom), color, 3)
                cv2.rectangle(annotated_img, (left, bottom - 30), (right, bottom), color, cv2.FILLED)
                cv2.putText(
                    annotated_img,
                    name,
                    (left + 6, bottom - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

            st.image(annotated_img, caption="Recognition Result", use_container_width=True)

            if not detected_names:
                st.warning("No faces detected in the image.")
            else:
                for d_name in detected_names:
                    if d_name != "Unknown":
                        marked, msg = mark_attendance(d_name)
                        if marked:
                            st.success(f"🎉 {msg}")
                        else:
                            st.info(f"ℹ️ {msg}")
                    else:
                        st.error("❌ Unknown person detected. Attendance not marked.")


# --- Tab 4: Attendance Log ---
with tabs[3]:
    st.header("Stage 4: Attendance Log")
    st.write("View, filter, and export the logged attendance data.")

    ensure_attendance_file()
    df = pd.read_csv(ATTENDANCE_FILE)

    today_str = datetime.now().strftime("%Y-%m-%d")
    today_df = df[df["Date"] == today_str] if "Date" in df.columns else pd.DataFrame()

    m1, m2, m3 = st.columns(3)
    m1.metric("Total Records", len(df))
    m2.metric("Today's Attendees", len(today_df))
    m3.metric("Unique Attendees Today", today_df["Name"].nunique() if not today_df.empty else 0)

    st.subheader("Attendance Records")
    if df.empty:
        st.info("No attendance records recorded yet.")
    else:
        st.dataframe(df, use_container_width=True)

        col_dl, col_clr = st.columns([1, 1])
        with col_dl:
            csv_data = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Attendance CSV",
                data=csv_data,
                file_name="attendance.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with col_clr:
            if st.button("🗑️ Clear Attendance Log", use_container_width=True):
                with open(ATTENDANCE_FILE, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["Name", "Date", "Time"])
                st.success("Attendance log cleared.")
                st.rerun()
