import cv2
import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F

# ==================== 1. 模型定义 ====================
class EmotionCNN_LSTM(nn.Module):
    def __init__(self, seq_len=10):
        super(EmotionCNN_LSTM, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout_cnn = nn.Dropout2d(0.25)
        self.lstm = nn.LSTM(input_size=256 * 6 * 6, hidden_size=256, num_layers=2, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(256, 7)

    def forward(self, x):
        batch_size, seq_len, C, H, W = x.size()
        x = x.view(batch_size * seq_len, C, H, W)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout_cnn(x)
        x = x.view(batch_size * seq_len, -1)
        x = x.view(batch_size, seq_len, -1)
        lstm_out, _ = self.lstm(x)
        x = torch.mean(lstm_out, dim=1)
        x = self.fc(x)
        return x

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# ==================== 2. 加载模型 ====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️ 使用设备: {device}")

model = EmotionCNN_LSTM(seq_len=10).to(device)
weights_path = os.environ.get('FER2013_LSTM_WEIGHTS', os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'fer2013_lstm_model.pth'))
model.load_state_dict(torch.load(weights_path, map_location=device))
model.eval()
print("✅ 模型加载成功")

# ==================== 3. 预处理 ====================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# ==================== 4. 【自适应人脸检测器】 ====================
# 尝试导入 MediaPipe，如果失败则使用 OpenCV Haar
use_mediapipe = False
face_detector = None

try:
    import mediapipe as mp
    mp_face_detection = mp.solutions.face_detection
    face_detector = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
    use_mediapipe = True
    print("✅ 使用 MediaPipe 深度学习人脸检测（精准）")
except (ImportError, AttributeError) as e:
    print("⚠️ MediaPipe 不可用，降级使用 OpenCV Haar 检测器")
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        print("❌ 无法加载 Haar 级联分类器，请检查 OpenCV 安装")
        exit()
    use_mediapipe = False

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ 无法打开摄像头")
    exit()

frame_buffer = []
print("🎥 摄像头已开启，按 'q' 键退出")
display_text = "Collecting frames (0/10)..."

while True:
    ret, frame = cap.read()
    if not ret:
        break

    face_roi = None
    x = y = box_w = box_h = 0

    # ---------- 人脸检测 ----------
    if use_mediapipe:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detector.process(rgb_frame)
        if results.detections:
            detection = results.detections[0]
            bboxC = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x = int(bboxC.xmin * w)
            y = int(bboxC.ymin * h)
            box_w = int(bboxC.width * w)
            box_h = int(bboxC.height * h)
            # 扩大 15%
            margin_x = int(0.15 * box_w)
            margin_y = int(0.15 * box_h)
            x = max(0, x - margin_x)
            y = max(0, y - margin_y)
            box_w = min(w - x, box_w + 2 * margin_x)
            box_h = min(h - y, box_h + 2 * margin_y)
            face_roi = frame[y:y+box_h, x:x+box_w]
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(60, 60))
        if len(faces) > 0:
            (x, y, box_w, box_h) = faces[0]
            # 扩大 20%
            margin_x = int(0.2 * box_w)
            margin_y = int(0.2 * box_h)
            x = max(0, x - margin_x)
            y = max(0, y - margin_y)
            box_w = min(frame.shape[1] - x, box_w + 2 * margin_x)
            box_h = min(frame.shape[0] - y, box_h + 2 * margin_y)
            face_roi = frame[y:y+box_h, x:x+box_w]

    # ---------- 表情识别 ----------
    emotion_text = ""
    confidence_text = ""

    if face_roi is not None and face_roi.size > 0:
        gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        resized_face = cv2.resize(gray_face, (48, 48), interpolation=cv2.INTER_AREA)
        pil_img = Image.fromarray(resized_face)
        input_tensor = transform(pil_img)

        frame_buffer.append(input_tensor)
        if len(frame_buffer) > 10:
            frame_buffer.pop(0)

        display_text = f"Collecting frames ({len(frame_buffer)}/10)..."

        if len(frame_buffer) == 10:
            batch_tensor = torch.stack(frame_buffer, dim=0).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(batch_tensor)
                probs = torch.softmax(outputs, dim=1)
                max_prob, pred_idx = torch.max(probs, dim=1)
                confidence = max_prob.item() * 100
                pred_label = EMOTIONS[pred_idx.item()]

                if confidence > 55.0:
                    emotion_text = pred_label
                    confidence_text = f"{confidence:.1f}%"
                    display_text = f"{emotion_text} ({confidence_text})"
                else:
                    emotion_text = "Uncertain"
                    confidence_text = f"{confidence:.1f}%"
                    display_text = f"{emotion_text} ({confidence_text})"

        # 画框和标签
        cv2.rectangle(frame, (x, y), (x+box_w, y+box_h), (0, 255, 0), 2)
        if emotion_text:
            cv2.putText(frame, f"{emotion_text} {confidence_text}", (x, y-15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    else:
        display_text = "No Face Detected"
        if len(frame_buffer) > 0:
            frame_buffer.pop(0)

    # 左上角显示
    cv2.putText(frame, f"Emo: {display_text}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    cv2.imshow('Real-time Emotion Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("👋 已退出")
