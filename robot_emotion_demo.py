import cv2
import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import math
import time

# ==================== 1. 模型定义（复制你的训练模型） ====================
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
        x = torch.mean(lstm_out, dim=1)  # 时序平均
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

# ==================== 4. 人脸检测（自适应，优先 MediaPipe） ====================
use_mediapipe = False
face_detector = None

try:
    import mediapipe as mp
    mp_face_detection = mp.solutions.face_detection
    face_detector = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
    use_mediapipe = True
    print("✅ 使用 MediaPipe 人脸检测")
except (ImportError, AttributeError):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    print("⚠️ 使用 OpenCV Haar 人脸检测")

# ==================== 5. 【核心】舵机映射表：情绪 -> {嘴巴角度, 眉毛高度, 眼睛睁大度} ====================
# 模拟舵机角度（0-180度），用于控制虚拟机器人脸
SERVO_MAP = {
    'angry':   {'mouth': 30,  'eyebrow': 70,  'eye_open': 40},
    'disgust': {'mouth': 20,  'eyebrow': 80,  'eye_open': 30},
    'fear':    {'mouth': 80,  'eyebrow': 140, 'eye_open': 100},
    'happy':   {'mouth': 120, 'eyebrow': 60,  'eye_open': 80},
    'sad':     {'mouth': 10,  'eyebrow': 30,  'eye_open': 60},
    'surprise':{'mouth': 160, 'eyebrow': 150, 'eye_open': 120},
    'neutral': {'mouth': 90,  'eyebrow': 90,  'eye_open': 70}
}

# 平滑滤波（防止舵机抖动）
smooth_servo = {'mouth': 90, 'eyebrow': 90, 'eye_open': 70}

# ==================== 6. 绘图函数：绘制机器人脸 ====================
def draw_robot_face(frame, servo_data):
    h, w, _ = frame.shape
    center_x = w - 200  # 将机器人脸放在画面右侧
    center_y = int(h / 2)
    face_radius = 120

    # 获取舵机值（映射到绘制参数）
    mouth_angle = servo_data['mouth']  # 0-180 -> 控制嘴巴张开弧度
    eyebrow_angle = servo_data['eyebrow']  # 控制眉毛高度
    eye_open = servo_data['eye_open']  # 控制眼睛大小

    # --- 画脸部轮廓（圆形） ---
    cv2.circle(frame, (center_x, center_y), face_radius, (200, 200, 200), 3)

    # --- 画眼睛（根据 eye_open 调整大小） ---
    eye_offset_x = 40
    eye_offset_y = -20
    eye_radius = int(15 + (eye_open / 180) * 20)  # 15~35

    # 左眼
    cv2.circle(frame, (center_x - eye_offset_x, center_y + eye_offset_y), eye_radius, (255, 255, 255), -1)
    cv2.circle(frame, (center_x - eye_offset_x, center_y + eye_offset_y), eye_radius, (0, 0, 0), 2)
    cv2.circle(frame, (center_x - eye_offset_x, center_y + eye_offset_y), int(eye_radius*0.4), (0, 0, 0), -1)  # 瞳孔

    # 右眼
    cv2.circle(frame, (center_x + eye_offset_x, center_y + eye_offset_y), eye_radius, (255, 255, 255), -1)
    cv2.circle(frame, (center_x + eye_offset_x, center_y + eye_offset_y), eye_radius, (0, 0, 0), 2)
    cv2.circle(frame, (center_x + eye_offset_x, center_y + eye_offset_y), int(eye_radius*0.4), (0, 0, 0), -1)

    # --- 画眉毛（根据 eyebrow_angle 控制倾斜和高度） ---
    # 眉毛高度偏移：90度时在眼睛上方20px，0度时下垂，180度时上挑
    brow_offset = int(10 + (eyebrow_angle / 180) * 30)
    # 左眉（左高右低 或 左低右高，模拟愤怒/惊讶）
    tilt = (eyebrow_angle - 90) * 0.3  # -27 ~ +27 度倾斜

    # 左眉
    p1 = (center_x - 55, center_y + eye_offset_y - 40 + brow_offset - int(tilt))
    p2 = (center_x - 25, center_y + eye_offset_y - 40 + brow_offset + int(tilt))
    cv2.line(frame, p1, p2, (50, 50, 50), 5)

    # 右眉
    p1 = (center_x + 25, center_y + eye_offset_y - 40 + brow_offset + int(tilt))
    p2 = (center_x + 55, center_y + eye_offset_y - 40 + brow_offset - int(tilt))
    cv2.line(frame, p1, p2, (50, 50, 50), 5)

    # --- 画嘴巴（根据 mouth_angle 控制弧度） ---
    # 嘴巴中心
    mouth_y = center_y + 50
    # 弧度映射：0度=倒U（悲伤），90度=直线（中性），180度=正U（开心）
    # 使用椭圆弧线
    if mouth_angle < 90:
        # 悲伤：向下弯曲 (倒U)
        start_angle = 180 + (90 - mouth_angle) * 1.5  # 180 -> 315
        end_angle = 360 - (90 - mouth_angle) * 1.5   # 360 -> 45
        axes = (40, 25)
    else:
        # 开心：向上弯曲 (正U)
        start_angle = 180 - (mouth_angle - 90) * 1.5  # 180 -> 45
        end_angle = (mouth_angle - 90) * 1.5          # 0 -> 135
        axes = (40, 25)

    cv2.ellipse(frame, (center_x, mouth_y), axes, 0, start_angle, end_angle, (0, 0, 255), 4)

    # 如果嘴巴张开特别大（惊讶），画一个内部椭圆表示张嘴
    if mouth_angle > 130:
        cv2.ellipse(frame, (center_x, mouth_y), (35, 20), 0, 0, 360, (100, 100, 100), -1)

    # --- 显示舵机数值 ---
    cv2.putText(frame, f"Robot Face", (center_x - 60, center_y - face_radius - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.putText(frame, f"Mouth:{mouth_angle:.0f} Eyebrow:{eyebrow_angle:.0f}",
                (center_x - 80, center_y + face_radius + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

# ==================== 7. 主循环 ====================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ 无法打开摄像头")
    exit()

frame_buffer = []
print("🎥 虚拟机器人控制已启动，按 'q' 退出")
print("📤 模拟串口指令将打印在终端...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ---- 人脸检测 ----
    face_roi = None
    x = y = box_w = box_h = 0

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
            margin_x = int(0.2 * box_w)
            margin_y = int(0.2 * box_h)
            x = max(0, x - margin_x)
            y = max(0, y - margin_y)
            box_w = min(frame.shape[1] - x, box_w + 2 * margin_x)
            box_h = min(frame.shape[0] - y, box_h + 2 * margin_y)
            face_roi = frame[y:y+box_h, x:x+box_w]

    # ---- 推理与舵机控制 ----
    emotion_text = ""
    confidence_text = ""
    current_servo = {'mouth': 90, 'eyebrow': 90, 'eye_open': 70}

    if face_roi is not None and face_roi.size > 0:
        gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        resized_face = cv2.resize(gray_face, (48, 48), interpolation=cv2.INTER_AREA)
        pil_img = Image.fromarray(resized_face)
        input_tensor = transform(pil_img)

        frame_buffer.append(input_tensor)
        if len(frame_buffer) > 10:
            frame_buffer.pop(0)

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
                    # 【核心】根据情绪查找舵机目标值
                    target = SERVO_MAP.get(pred_label, SERVO_MAP['neutral'])
                    current_servo = target.copy()

                    # 【模拟串口指令】在终端打印舵机控制指令
                    print(f"\r📤 [SERIAL CMD] Emotion: {pred_label} | Mouth: {target['mouth']}° | Eyebrow: {target['eyebrow']}° | Eye: {target['eye_open']}°", end="")

                else:
                    emotion_text = "Uncertain"
                    confidence_text = f"{confidence:.1f}%"
                    # 没看清时就保持中性
                    current_servo = SERVO_MAP['neutral']

        # 画人脸框
        cv2.rectangle(frame, (x, y), (x+box_w, y+box_h), (0, 255, 0), 2)
        cv2.putText(frame, f"{emotion_text} {confidence_text}", (x, y-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    else:
        if len(frame_buffer) > 0:
            frame_buffer.pop(0)

    # ---- 绘制虚拟机器人脸（右侧） ----
    draw_robot_face(frame, current_servo)

    # 左上角状态
    cv2.putText(frame, f"Real-time Robot Controller", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.imshow('Emotion -> Robot Control Demo', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("\n👋 已退出")
