
# Real-Time Emotion Analytics System


import cv2
from deepface import DeepFace
from collections import deque, Counter

# -------------------- FACE DETECTOR --------------------

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# -------------------- EMOTION BUFFERS --------------------

emotion_history = deque(maxlen=10)          # For smoothing
session_emotion_counts = Counter()          # For analytics

# -------------------- FUNCTIONS --------------------

def detect_faces(gray_frame):
    return face_cascade.detectMultiScale(
        gray_frame,
        scaleFactor=1.1,
        minNeighbors=4
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

        emotion = result["dominant_emotion"]
        confidence = int(result["emotion"][emotion])

        return emotion, confidence

    except Exception:
        return "Unknown", 0

def draw_results(frame, faces, emotions):
    for (x, y, w, h), (emotion, confidence) in zip(faces, emotions):
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"{emotion} ({confidence}%)"
        cv2.putText(
            frame,
            label,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

# -------------------- WEBCAM --------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise IOError("Cannot open webcam")

# -------------------- MAIN LOOP --------------------

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detect_faces(gray)
    emotions = []

    for (x, y, w, h) in faces:
        face_img = frame[y:y + h, x:x + w]

        emotion, confidence = analyze_emotion(face_img)
        emotion_history.append(emotion)

        stable_emotion = Counter(emotion_history).most_common(1)[0][0]
        emotions.append((stable_emotion, confidence))

        # Track emotion over entire session
        session_emotion_counts[stable_emotion] += 1

    draw_results(frame, faces, emotions)

    cv2.imshow("Emotion Analytics – Day 2", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# -------------------- SESSION SUMMARY --------------------

print("\nSession Emotion Summary")
print("-" * 30)

total = sum(session_emotion_counts.values())

for emotion, count in session_emotion_counts.items():
    percentage = (count / total) * 100 if total > 0 else 0
    print(f"{emotion:<8}: {percentage:.2f}%")

# -------------------- CLEANUP --------------------

cap.release()
cv2.destroyAllWindows()
