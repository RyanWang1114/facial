"""共享的粗粒度表情分类：把 MediaPipe 的 ARKit blendshape 字典简化成一个中文情绪标签。
face_capture.py 用它给 latest_state.json 打标签；interactive_agent.py 读这个标签，
让 GPT 的回复能"看到"用户当前的表情。规则是固定阈值，不追求精确，够用即可。
"""


def classify_emotion(bs: dict) -> str:
    def avg(*names):
        vals = [bs.get(n, 0.0) for n in names]
        return sum(vals) / len(vals) if vals else 0.0

    smile = avg("mouthSmileLeft", "mouthSmileRight")
    frown = avg("mouthFrownLeft", "mouthFrownRight")
    brow_down = avg("browDownLeft", "browDownRight")
    eye_squint = avg("eyeSquintLeft", "eyeSquintRight")
    brow_up = bs.get("browInnerUp", 0.0)
    jaw_open = bs.get("jawOpen", 0.0)
    eye_wide = avg("eyeWideLeft", "eyeWideRight")

    if smile > 0.4:
        return "开心"
    if brow_up > 0.4 and (jaw_open > 0.25 or eye_wide > 0.3):
        return "惊讶"
    if brow_down > 0.35 and eye_squint > 0.25:
        return "生气"
    if frown > 0.3 or (brow_up > 0.3 and frown > 0.15):
        return "难过"
    return "平静"
