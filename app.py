
# ============================================================
# Real-Time Face Emotion Analytics
# Phase 2 – FINAL Website-Style Layout
# ============================================================

import cv2
import streamlit as st
from deepface import DeepFace
from collections import deque, Counter
import pandas as pd

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Real-Time Face Emotion Analytics",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
body {
    background-color: #0e1117;
    color: #eaeaea;
}

/* ===== TOP BAR ===== */
.top-bar {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 72px;
    background-color: #0e1117;
    display: flex;
    align-items: center;
    z-index: 1000;
    border-bottom: 1px solid #222;
    padding: 0 24px;
}

/* Left controls */
.top-left {
    display: flex;
    flex-direction: column;
    gap: 6px;
}

/* Center title */
.top-center {
    flex: 1;
    text-align: center;
}
.top-title {
    font-size: 28px;
    font-weight: 700;
    line-height: 1.1;
}
.top-subtitle {
    font-size: 12px;
    color: #9aa0a6;
}

/* Buttons */
.top-right button {
    padding: 6px 14px !important;
    font-size: 13px !important;
    border-radius: 6px !important;
}

/* ===== SPACER (push content below fixed bar) ===== */
.spacer {
    height: 80px;
}

/* ===== CARDS ===== */
.card {
    background-color: #161b22;
    padding: 14px;
    border-radius: 10px;
    margin-bottom: 14px;
}

/* ===== METRICS ===== */
.metric-label {
    font-size: 15px;
    font-weight: 600;
}
.metric-value {
    font-size: 26px;
    font-weight: 700;
}

/* ===== PROGRESS BAR ===== */
.progress-bg {
    width: 100%;
    height: 10px;
    background-color: #ffffff;
    border-radius: 6px;
    overflow: hidden;
    margin-top: 6px;
}
.progress-fill {
    height: 100%;
    font-size: 10px;
    font-weight: 700;
    color: #ffffff;
    text-align: right;
    padding-right: 6px;
    line-height: 10px;
}
.green { background-color: #2ea043; }
.blue { background-color: #1f6feb; }
.orange { background-color: #d29922; }

/* ===== SESSION SECTION ===== */
.session-title {
    font-size: 20px;
    font-weight: 600;
    margin-top: 32px;
    margin-bottom: 10px;
}

/* ===== FOOTER ===== */
.footer {
    position: fixed;
    bottom: 0;
    left: 0;
    width: 100%;
    background-color: #0e1117;
    text-align: center;
    font-size: 12px;
    color: #9aa0a6;
    padding: 6px;
    border-top: 1px solid #222;
}
</style>
""", unsafe_allow_html=True)

# ---------------- SESSION STATE ----------------
if "camera_running" not in st.session_state:
    st.session_state.camera_running = False
if "emotion_history" not in st.session_state:
    st.session_state.emotion_history = deque(maxlen=10)
if "session_counts" not in st.session_state:
    st.session_state.session_counts = Counter()
if "frame_count" not in st.session_state:
    st.session_state.frame_count = 0

# ---------------- TOP BAR ----------------
st.markdown("<div class='top-bar'>", unsafe_allow_html=True)

left, center, right = st.columns([1, 4, 1])

with left:
    if st.button("🟢 Start"):
        st.session_state.camera_running = True
    if st.button("🔴 Stop"):
        st.session_state.camera_running = False

with center:
    st.markdown("""
    <div class="top-center">
        <div class="top-title">Real-Time Face Emotion Analytics</div>
        <div class="top-subtitle">📊 Live facial emotion recognition with session insights</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# Spacer so content starts just below top bar
st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)

# ---------------- FACE DETECTOR ----------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def analyze_emotion(face_img):
    try:
        result = DeepFace.analyze(
            face_img,
            actions=["emotion"],
            enforce_detection=False
        )
        if isinstance(result, list):
            result = result[0]
        return result["dominant_emotion"]
    except:
        return "Unknown"

def metric_card(label, value, percent, color):
    st.markdown(f"""
    <div class="card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="progress-bg">
            <div class="progress-fill {color}" style="width:{percent}%;">
                {int(percent)}%
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ---------------- MAIN DASHBOARD ----------------
left_col, right_col = st.columns([1, 2], gap="large")

with left_col:
    faces = sum(st.session_state.session_counts.values())
    frames = st.session_state.frame_count
    dominant = (
        st.session_state.session_counts.most_common(1)[0][0]
        if st.session_state.session_counts else "N/A"
    )
    max_ref = max(faces, frames, 1)

    metric_card("Faces Analyzed", faces, (faces/max_ref)*100, "green")
    metric_card("Frames Processed", frames, (frames/max_ref)*100, "blue")
    metric_card("Overall Mood", dominant.capitalize(), 100, "orange")

with right_col:
    video_box = st.empty()

# ---------------- CAMERA LOOP ----------------
if st.session_state.camera_running:
    cap = cv2.VideoCapture(0)
    while cap.isOpened() and st.session_state.camera_running:
        ret, frame = cap.read()
        if not ret:
            break

        st.session_state.frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            emotion = analyze_emotion(face_img)

            st.session_state.emotion_history.append(emotion)
            stable = Counter(st.session_state.emotion_history).most_common(1)[0][0]
            st.session_state.session_counts[stable] += 1

            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,180,0), 2)
            cv2.putText(frame, stable, (x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        video_box.image(frame, channels="BGR")

    cap.release()

# ---------------- SESSION SUMMARY ----------------
st.markdown("<div class='session-title'>📈 Session Summary</div>", unsafe_allow_html=True)

if st.session_state.session_counts:
    df = pd.DataFrame.from_dict(
        st.session_state.session_counts,
        orient="index",
        columns=["Count"]
    )
    st.bar_chart(df)

# ---------------- FOOTER ----------------
st.markdown("""
<div class="footer">
⚠️ Emotion recognition is probabilistic and may vary due to lighting, camera quality,
facial angle, and individual differences.
</div>
""", unsafe_allow_html=True)
