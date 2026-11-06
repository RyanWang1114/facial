import cv2
import math
import numpy as np
import os

# -------------------- Load Cascades with Fallback --------------------
def load_cascade(filename):
    """Try to load cascade from OpenCV data folder; return None if fails."""
    full_path = cv2.data.haarcascades + filename
    if os.path.exists(full_path):
        cascade = cv2.CascadeClassifier(full_path)
        if not cascade.empty():
            return cascade
    return None

face_cascade = load_cascade('haarcascade_frontalface_default.xml')
mouth_cascade = load_cascade('haarcascade_mcs_mouth.xml')
if mouth_cascade is None:
    # Try smile cascade as a proxy for mouth
    mouth_cascade = load_cascade('haarcascade_smile.xml')

if face_cascade is None:
    print("❌ Could not load face cascade. Please check OpenCV installation.")
    exit()

print("✅ Cascades loaded successfully.")
print(f"   Mouth cascade: {'haarcascade_mcs_mouth.xml' if mouth_cascade else 'haarcascade_smile.xml (fallback)' if mouth_cascade else 'None (skipping mouth detection)'}")

# -------------------- Metrics Extraction --------------------
def extract_metrics(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Detect face
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
    if len(faces) == 0:
        return None, None, None, None

    # Largest face
    faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
    (fx, fy, fw, fh) = faces[0]

    # Expand face ROI slightly
    margin_y = int(0.2 * fh)
    margin_x = int(0.1 * fw)
    fx = max(0, fx - margin_x)
    fy = max(0, fy - margin_y)
    fw = min(w - fx, fw + 2 * margin_x)
    fh = min(h - fy, fh + 2 * margin_y)
    face_roi = gray[fy:fy+fh, fx:fx+fw]

    # --- 1. Jaw drop (face aspect ratio) ---
    face_aspect = fh / fw if fw > 0 else 1.0
    jaw_drop = np.clip((face_aspect - 1.1) * 2.0, 0.0, 1.0)

    # --- 2. Mouth openness using cascade if available ---
    mouth_open = 0.0
    smile_intensity = 0.0

    if mouth_cascade is not None:
        # Detect mouth within face ROI
        mouths = mouth_cascade.detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=15, minSize=(30, 15))
        if len(mouths) > 0:
            mouths = sorted(mouths, key=lambda x: x[2]*x[3], reverse=True)
            (mx, my, mw, mh) = mouths[0]
            # Mouth Aspect Ratio (MAR)
            mar = mh / mw if mw > 0 else 0.1
            mouth_open = np.clip((mar - 0.15) * 3.0, 0.0, 1.0)
            # Smile: wide mouth + not too much jaw drop
            mouth_width_ratio = mw / fw
            smile_width_score = np.clip((mouth_width_ratio - 0.2) * 5.0, 0.0, 1.0)
            smile_intensity = smile_width_score * (1.0 - jaw_drop * 0.5)

    else:
        # No mouth cascade: use a simple brightness threshold in the lower half
        lower_face = face_roi[int(fh*0.6):, :]
        if lower_face.size > 0:
            # Mouth is typically darker; compute the darkest 20% average
            flat = lower_face.flatten()
            sorted_vals = np.sort(flat)
            dark_thresh = sorted_vals[int(0.2 * len(sorted_vals))]
            # Count pixels below threshold (dark region)
            dark_pixels = np.sum(lower_face < dark_thresh)
            total_pixels = lower_face.size
            dark_ratio = dark_pixels / total_pixels
            # Map to mouth openness (dark ratio between 0.02 and 0.3)
            mouth_open = np.clip((dark_ratio - 0.02) * 5.0, 0.0, 1.0)
            # Smile intensity: use jaw_drop as proxy (if jaw drops, it's more surprise than smile)
            smile_intensity = (1.0 - jaw_drop) * 0.8

    # --- 3. Eye openness (proxy from jaw_drop and smile) ---
    eye_open = 0.6 + 0.4 * jaw_drop - 0.3 * smile_intensity
    eye_open = np.clip(eye_open, 0.2, 1.0)

    # --- 4. Brow raise (proxy from jaw_drop) ---
    brow_raise = jaw_drop * 0.8 + (1.0 - smile_intensity) * 0.2
    brow_raise = np.clip(brow_raise, 0.0, 1.0)

    return mouth_open, smile_intensity, brow_raise, eye_open

# -------------------- Robot Face Drawing (same as before) --------------------
def draw_robot_face(frame, metrics, smooth_state):
    alpha = 0.6
    smooth_state['mouth'] = alpha * smooth_state['mouth'] + (1 - alpha) * metrics[0]
    smooth_state['smile'] = alpha * smooth_state['smile'] + (1 - alpha) * metrics[1]
    smooth_state['brow'] = alpha * smooth_state['brow'] + (1 - alpha) * metrics[2]
    smooth_state['eye'] = alpha * smooth_state['eye'] + (1 - alpha) * metrics[3]

    m_open = smooth_state['mouth']
    s_int = smooth_state['smile']
    b_raise = smooth_state['brow']
    e_open = smooth_state['eye']

    h, w, _ = frame.shape
    cx = w - 180
    cy = int(h / 2)
    radius = 100

    cv2.circle(frame, (cx, cy), radius, (200, 200, 200), 3)

    # Eyes
    eye_off_x = 30
    eye_off_y = -15
    eye_r = int(10 + e_open * 20)
    for sign in [-1, 1]:
        ex = cx + sign * eye_off_x
        ey = cy + eye_off_y
        cv2.circle(frame, (ex, ey), eye_r, (255, 255, 255), -1)
        cv2.circle(frame, (ex, ey), eye_r, (0, 0, 0), 2)
        cv2.circle(frame, (ex, ey), int(eye_r * 0.4), (0, 0, 0), -1)

    # Eyebrows
    brow_off = int(5 + b_raise * 25)
    for sign in [-1, 1]:
        ex1 = cx + sign * 40
        ex2 = cx + sign * 20
        ey = cy + eye_off_y - 25 - brow_off
        cv2.line(frame, (ex1, ey), (ex2, ey + 5), (50, 50, 50), 4)

    # Mouth
    mouth_y = cy + 35 + int(m_open * 15)
    mouth_w = int(25 + s_int * 20)
    mouth_h = int(5 + m_open * 20)

    if s_int > 0.3:
        cv2.ellipse(frame, (cx, mouth_y), (mouth_w, mouth_h), 0, 180 - int(s_int * 80), int(s_int * 80), (0, 0, 255), 3)
    else:
        cv2.ellipse(frame, (cx, mouth_y), (mouth_w, mouth_h), 0, 180 + int((1 - s_int) * 60), 360 - int((1 - s_int) * 60), (0, 0, 255), 3)

    if m_open > 0.4:
        cv2.ellipse(frame, (cx, mouth_y + 5), (mouth_w - 5, mouth_h - 3), 0, 0, 360, (80, 80, 80), -1)

    cv2.putText(frame, "Robot Face", (cx - 50, cy - radius - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(frame, f"Smile:{s_int:.2f}", (cx - 60, cy + radius + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 2)
    cv2.putText(frame, f"Jaw:{m_open:.2f}", (cx - 60, cy + radius + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 2)

    return smooth_state

# -------------------- Main Loop --------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Cannot open camera")
    exit()

print("🎥 Running with OpenCV Geometry Controller (no MediaPipe needed).")
print("🧠 Using face aspect ratio and optional mouth cascade.")
print("📤 Press 'q' to quit.")

smooth = {'mouth': 0.0, 'smile': 0.0, 'brow': 0.0, 'eye': 0.7}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    mouth_open, smile, brow, eye = extract_metrics(frame)

    if mouth_open is not None:
        # Draw face bounding box
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))
        if len(faces) > 0:
            (x, y, w, h) = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        smooth = draw_robot_face(frame, (mouth_open, smile, brow, eye), smooth)

        # Display metrics
        cv2.putText(frame, f"Mouth Open: {mouth_open:.2f}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"Smile: {smile:.2f}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"Brow: {brow:.2f}", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"Eye: {eye:.2f}", (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        print(f"\r📤 [SERIAL] Smile:{smile:.2f} Jaw:{mouth_open:.2f} Brow:{brow:.2f}", end="")
    else:
        cv2.putText(frame, "No Face Detected", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.putText(frame, "OpenCV Geometry Control", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow('OpenCV Robot Control', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("\n👋 Exited.")
