import json
import os
import socket
import sys
import time
import urllib.request

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

import cv2
import numpy as np
import mediapipe as mp

from emotion_utils import classify_emotion
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    RunningMode,
)

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "face_landmarker.task")

TOP_N_DISPLAY = 10

# UDP target for streaming blendshape coefficients to Blender (see facial/blender_receiver.py)
UDP_IP = "127.0.0.1"
UDP_PORT = 5006

# While interactive_agent.py is talking, it creates this file to claim control of the
# avatar's mouth/expression (its speech + emotion signal shouldn't fight with the raw
# webcam mirroring every frame). We just skip sending while it exists.
AGENT_SPEAKING_FLAG = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".agent_speaking")

# Turn the live "avatar mimics my face over UDP" behavior on/off. Left False: the avatar
# should hold its own idle motion (see blender_receiver_vrm.py's blink/gaze/head-sway)
# instead of copying the user's real face while nobody's talking to it. Emotion capture
# (STATE_PATH below) is independent of this and keeps working either way, since
# interactive_agent.py still needs to know "what does the user's face look like right now".
ENABLE_LIVE_MIRROR = False

# Latest blendshapes + a coarse emotion label, refreshed a few times a second so
# interactive_agent.py can sample "what does the user's face look like right now"
# while it's recording, without needing its own separate camera/UDP feed.
STATE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "latest_state.json")


def ensure_model():
    if os.path.exists(MODEL_PATH):
        return
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    print(f"下载 FaceLandmarker 模型到 {MODEL_PATH} ...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("下载完成。")


def build_landmarker():
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
    )
    return FaceLandmarker.create_from_options(options)


def draw_landmarks(frame, face_landmarks):
    h, w, _ = frame.shape
    for lm in face_landmarks:
        cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 1, (0, 255, 0), -1)


def draw_blendshape_panel(frame, blendshapes):
    top = sorted(blendshapes, key=lambda c: c.score, reverse=True)[:TOP_N_DISPLAY]
    x0, y0 = 10, 30
    for i, cat in enumerate(top):
        y = y0 + i * 22
        bar_len = int(cat.score * 150)
        cv2.putText(frame, f"{cat.category_name:<22}{cat.score:.2f}", (x0, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.rectangle(frame, (x0 + 210, y - 10), (x0 + 210 + bar_len, y - 2), (0, 200, 255), -1)


def get_blendshape_dict(result):
    if not result.face_blendshapes:
        return {}
    return {c.category_name: c.score for c in result.face_blendshapes[0]}


def get_rotation_matrix(result):
    """Top-left 3x3 (rotation-only) part of MediaPipe's head transformation matrix,
    as a numpy array so calibration (see main()) can do matrix math on it before it's
    sent. None if no face this frame."""
    if not result.facial_transformation_matrixes:
        return None
    m = result.facial_transformation_matrixes[0]
    return np.array(m[:3, :3], dtype=float)


def open_camera(max_index=4):
    backends = [(cv2.CAP_DSHOW, "DSHOW"), (cv2.CAP_MSMF, "MSMF"), (cv2.CAP_ANY, "ANY")] \
        if os.name == "nt" else [(cv2.CAP_ANY, "ANY")]
    for backend, backend_name in backends:
        for index in range(max_index):
            cap = cv2.VideoCapture(index, backend)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    print(f"✅ 摄像头已打开 (index={index}, backend={backend_name})")
                    return cap
            cap.release()
    return None


def main():
    ensure_model()
    landmarker = build_landmarker()

    cap = open_camera()
    if cap is None:
        print("❌ 无法打开摄像头，尝试了 DSHOW / MSMF / ANY 后端，索引 0-3 均失败。")
        print("   请检查：")
        print("   1. Windows 设置 > 隐私和安全性 > 相机 > 是否开启了「桌面应用访问相机」")
        print("   2. 是否有其他程序（Zoom/Teams/浏览器标签页/其他 Python 脚本）正占用摄像头")
        print("   3. 设备管理器里摄像头驱动是否正常（无黄色感叹号）")
        return

    print("🎥 摄像头已开启，按 'q' 退出，按 'c' 校准中性姿态基线")
    print(f"📡 blendshape 数据将通过 UDP 发送到 {UDP_IP}:{UDP_PORT}（Blender 接收脚本需运行才能收到）")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    start_time = time.time()
    frame_count = 0
    fps = 0.0

    # Calibration baseline: captured on 'c' keypress from whatever the face is doing at
    # that moment. Subtracting it out means "your neutral face" maps to a neutral/rest
    # target model, instead of whatever small resting activations MediaPipe happens to
    # read off your particular face geometry.
    baseline_bs = None
    baseline_rot_inv = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int((time.time() - start_time) * 1000)

        result = landmarker.detect_for_video(mp_image, timestamp_ms)
        bs_dict = get_blendshape_dict(result)
        rot = get_rotation_matrix(result)

        if result.face_landmarks:
            draw_landmarks(frame, result.face_landmarks[0])
        if result.face_blendshapes:
            draw_blendshape_panel(frame, result.face_blendshapes[0])

            adjusted_bs = bs_dict
            if baseline_bs is not None:
                adjusted_bs = {name: max(0.0, min(1.0, score - baseline_bs.get(name, 0.0)))
                               for name, score in bs_dict.items()}

            if ENABLE_LIVE_MIRROR:
                # While interactive_agent.py is speaking, it owns the mouth/expression (see
                # AGENT_SPEAKING_FLAG above) - so we withhold the whole payload, including
                # rotation: the avatar's head should hold still and act on its own rather
                # than mirroring whatever the user's real head happens to be doing while
                # they're just sitting there listening.
                speaking = os.path.exists(AGENT_SPEAKING_FLAG)
                payload = {}
                if not speaking:
                    payload["blendshapes"] = adjusted_bs
                    if rot is not None:
                        rot_to_send = rot @ baseline_rot_inv if baseline_rot_inv is not None else rot
                        payload["rotation_matrix"] = rot_to_send.tolist()

                if payload:
                    try:
                        sock.sendto(json.dumps(payload).encode("utf-8"), (UDP_IP, UDP_PORT))
                    except OSError:
                        pass

            if frame_count % 3 == 0:
                try:
                    with open(STATE_PATH, "w", encoding="utf-8") as f:
                        json.dump({"blendshapes": adjusted_bs, "emotion": classify_emotion(adjusted_bs)}, f)
                except OSError:
                    pass
        else:
            cv2.putText(frame, "No Face Detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        status = "Calibrated" if baseline_bs is not None else "Press 'c' to calibrate"
        cv2.putText(frame, status, (10, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if baseline_bs is not None else (0, 165, 255), 1)

        frame_count += 1
        if frame_count % 10 == 0:
            fps = frame_count / (time.time() - start_time)
        cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 140, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Face Landmarker - ARKit Blendshapes", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            if bs_dict:
                baseline_bs = dict(bs_dict)
            if rot is not None:
                baseline_rot_inv = rot.T.copy()
            print("✅ 已校准中性姿态基线 (neutral baseline calibrated)")

    cap.release()
    sock.close()
    cv2.destroyAllWindows()
    landmarker.close()
    print("👋 已退出")


if __name__ == "__main__":
    main()
