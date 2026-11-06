"""
语音对话智能体 - 在 face_capture.py + blender_receiver.py 的实时表情镜像基础上，
加一层"对话"：一个小的置顶窗口，回车说话，识别成文字 -> 发给 GPT 得到回复文字+情绪+强度 ->
合成语音播放，播放期间把说话时的嘴部动作、眨眼、情绪表情通过 UDP 发给 Blender（复用
face_capture.py 用的同一个端口/协议），让虚拟形象一边说话一边做出对应表情。3D 脸本身还是
显示在 Blender 那个窗口里，这个小窗口只管文字对话记录 + 触发说话，两个窗口摆在一起用，
体验上像是一个"小应用"（详见 ChatWindow 类）。

需要环境变量 OPENAI_API_KEY。

运行方式：
1. 先在 Blender 里跑 blender_receiver_vrm.py（不用动）
2. 跑 face_capture.py（继续提供摄像头实时表情镜像 + 把当前检测到的表情写进 latest_state.json）
3. 再跑这个脚本：python interactive_agent.py，会弹出一个小对话框
每次在对话框里按回车开始录音，说完停顿一下（约 0.9 秒没声音）会自动停止（见 VAD_RMS_THRESHOLD
等几个常量），不用等固定的秒数。录音期间会顺便采样 face_capture.py 写的 latest_state.json，
猜一个"你录音时的表情"，一起发给 GPT 当上下文。说完等它识别+回复+播放语音，文字记录显示在
对话框里。关掉对话框窗口即可退出。

和 face_capture.py 的协作方式：这个脚本说话时，会在 facial 目录下创建一个
.agent_speaking 标记文件；face_capture.py 每帧检查这个文件是否存在，存在就跳过发送
UDP（不发送摄像头表情），把嘴部/表情控制权让给这个脚本，播放完自动删除标记文件、
摄像头镜像恢复正常。
"""
import io
import json
import os
import queue
import random
import socket
import sys
import threading
import time
import tkinter as tk
from collections import Counter

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import sounddevice as sd
import soundfile as sf
from openai import OpenAI

UDP_IP = "127.0.0.1"
UDP_PORT = 5006

_HERE = os.path.dirname(os.path.abspath(__file__))
FLAG_PATH = os.path.join(_HERE, ".agent_speaking")
STATE_PATH = os.path.join(_HERE, "latest_state.json")

RECORD_SAMPLERATE = 16000
EMOTION_SAMPLE_INTERVAL = 0.5  # 录音时每隔多久看一眼 latest_state.json

# 录音自动止音：不再死等固定秒数，检测到你说完话停顿一下就自动停止录音，反应速度主要就是
# 靠这个提上来的（以前不管说多短的话都要陪录满 5 秒才开始识别）。
VAD_BLOCK_SECONDS = 0.1     # 每块多长时间检测一次音量，判断"现在是不是还在说话"
VAD_RMS_THRESHOLD = 0.01    # 音量超过这个值就算"在说话"
RECORD_MIN_SECONDS = 0.6    # 至少录这么久再考虑判定"说完了"，太短的话静音判断不稳
SILENCE_TAIL_SECONDS = 0.9  # 连续这么久没检测到声音，就认为一句话说完了，自动停止录音
RECORD_MAX_SECONDS = 12     # 硬上限，防止一直不停顿导致录音卡住不结束

# 录到的音量太小（基本是静音/背景噪声）就不送去 Whisper 识别 - 否则 Whisper 有个已知的
# 毛病：给它几乎没声音的音频，它会瞎编一句训练数据里混进去的字幕致谢词（比如
# "字幕由 Amara.org 社区提供"之类），而不是老实说"没听到"。
SILENCE_RMS_THRESHOLD = 0.008

# Whisper 偶尔即使有点声音也会吐出这类已知的训练数据残留字符串，直接过滤掉当成没听清。
KNOWN_HALLUCINATIONS = ("amara.org", "字幕由", "subtitles by")

BLINK_DURATION = 0.18
BLINK_INTERVAL_RANGE = (2.0, 4.0)  # 说话时眨眼的随机间隔（秒）
FIRST_BLINK_RANGE = (0.3, 1.2)  # 第一次眨眼提前一点，不然回复太短的话可能整段都不会眨一次

# 说话时头部一个很轻微的、独立于摄像头的低频摆动（不是跟踪你的真实头部动作，是自己生成的
# "呼吸感"晃动），几个不同频率的低频正弦波叠加，幅度控制在几度以内，避免看起来在乱晃。
HEAD_SWAY_YAW_DEG = 4.0
HEAD_SWAY_PITCH_DEG = 2.5
HEAD_SWAY_ROLL_DEG = 1.5

# 挑几个常见情绪，对应到 ARKit 表情名（和 blender_receiver.py 驱动的 shape key 名字一致）。
# 这里的数值是"这个情绪拉满是什么样子"，实际强度由 GPT 给的 intensity 缩放（见 speak_with_expression
# 里的 INTENSITY_FLOOR，防止 intensity 给得低时表情几乎看不出来）。
EMOTION_BLENDSHAPES = {
    "happy": {"mouthSmileLeft": 0.9, "mouthSmileRight": 0.9, "cheekSquintLeft": 0.5, "cheekSquintRight": 0.5,
              "eyeSquintLeft": 0.25, "eyeSquintRight": 0.25},
    "laughing": {"mouthSmileLeft": 1.0, "mouthSmileRight": 1.0, "cheekSquintLeft": 0.7, "cheekSquintRight": 0.7,
                 "eyeSquintLeft": 0.6, "eyeSquintRight": 0.6, "jawOpen": 0.3},
    "shy": {"mouthSmileLeft": 0.5, "mouthSmileRight": 0.5, "eyeSquintLeft": 0.4, "eyeSquintRight": 0.4,
            "eyeBlinkLeft": 0.2, "eyeBlinkRight": 0.2, "browDownLeft": 0.15, "browDownRight": 0.15},
    "sad": {"mouthFrownLeft": 0.8, "mouthFrownRight": 0.8, "browInnerUp": 0.7,
            "eyeSquintLeft": 0.2, "eyeSquintRight": 0.2},
    "surprised": {"browInnerUp": 0.9, "browOuterUpLeft": 0.8, "browOuterUpRight": 0.8,
                  "eyeWideLeft": 0.7, "eyeWideRight": 0.7, "jawOpen": 0.25},
    "angry": {"browDownLeft": 0.9, "browDownRight": 0.9, "noseSneerLeft": 0.6, "noseSneerRight": 0.6,
              "mouthPressLeft": 0.4, "mouthPressRight": 0.4},
    "disgusted": {"noseSneerLeft": 0.8, "noseSneerRight": 0.8, "mouthUpperUpLeft": 0.6, "mouthUpperUpRight": 0.6,
                  "browDownLeft": 0.4, "browDownRight": 0.4},
    "scared": {"eyeWideLeft": 0.8, "eyeWideRight": 0.8, "browInnerUp": 0.8,
               "mouthStretchLeft": 0.4, "mouthStretchRight": 0.4, "jawOpen": 0.2},
    "confused": {"browDownLeft": 0.5, "browOuterUpRight": 0.6, "mouthLeft": 0.3, "mouthPucker": 0.2},
    "neutral": {},
}

SYSTEM_PROMPT = """你是一个通过语音和用户对话的虚拟角色，重点：你选的 emotion 不是打比方，
你的虚拟形象会真的实时做出这个表情，用户能亲眼看到。

可选 emotion（必须从这些里选一个）：
happy 开心 / laughing 大笑 / shy 害羞 / sad 难过 / surprised 惊讶 /
angry 生气 / disgusted 厌恶 / scared 害怕 / confused 疑惑 / neutral 平静

规则：
- 如果用户明确要求你做出某个表情（比如"笑一个"、"生气的表情"、"你能惊讶一下吗"），必须选对应的
  emotion，intensity 给到 0.85~1.0，回复里用第一人称肯定语气确认（比如"来，笑一个给你看！"），
  不要说"想象"、"假装"、"脑补"这类话——因为你的脸真的会做出来，不是比喻。
- 没有明确要求时，就按这句话本身的情绪自然选一个，intensity 给 0.5~0.7。
- intensity 表示表情该有多强烈，0.0很轻微，1.0很夸张。
- 用户可能用中文或英文跟你说话（可以随时切换，不会提前通知你）。reply 必须跟用户这句话
  用的是同一种语言：用户说英文就整段用英文回复，说中文就整段用中文回复，不要中英混杂，
  也不要固定用某一种语言。

用户的消息里有时会带一行「[摄像头检测到用户当前表情：xxx]」，这是摄像头实时看到的用户表情，
你可以据此调整语气、或者自然地提一下（不用每次都提）。

每次都必须只输出一个 JSON 对象，不要有任何其他文字，格式：
{"reply": "你要说的话", "emotion": "happy|laughing|shy|sad|surprised|angry|disgusted|scared|confused|neutral", "intensity": 0.0到1.0}
reply 要口语化、简短自然（1-2句话），适合语音播报。"""


class ChatWindow:
    """一个小的、总在最前面的对话框：回车开始说话，实时显示状态（录音中/识别中/思考中/
    说话中）和对话记录（你说了什么、她回复了什么）。3D 脸还是 Blender 那边显示，这个窗口
    只负责文字这部分 + 触发说话，两个窗口摆在一起用。

    录音/识别/对话/合成语音都是阻塞调用，不能直接放主线程（会把窗口卡死），所以每次按回车
    起一个后台线程去跑；线程不能直接改 tkinter 控件（线程不安全），所以线程只管往一个
    queue 里塞消息，主线程用 root.after() 定时轮询 queue 来更新界面。
    """

    def __init__(self, on_turn):
        self.on_turn = on_turn
        self.busy = False
        self._queue = queue.Queue()

        self.root = tk.Tk()
        self.root.title("对话")
        self.root.geometry("380x360+60+60")
        self.root.attributes("-topmost", True)

        self.status_var = tk.StringVar(value="按回车开始说话")
        tk.Label(self.root, textvariable=self.status_var, font=("Microsoft YaHei UI", 11),
                 pady=6).pack(fill="x")

        self.log = tk.Text(self.root, wrap="word", state="disabled",
                            font=("Microsoft YaHei UI", 10))
        self.log.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        # 用 bind_all 而不是 bind：bind 只在 root 这个窗口本身持有键盘焦点时才生效，
        # 一旦焦点落到下面的 Text 日志框上（比如鼠标点了一下），root 就再也收不到回车事件，
        # 表现为"按一次有效，之后再按没反应"。bind_all 不管焦点在哪个子控件上都能收到。
        self.root.bind_all("<Return>", self._on_enter)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(50, self._poll_queue)

    def _on_enter(self, _event):
        if self.busy:
            return
        self.busy = True
        threading.Thread(target=self.on_turn, args=(self,), daemon=True).start()

    def _on_close(self):
        if os.path.exists(FLAG_PATH):
            os.remove(FLAG_PATH)
        self.root.destroy()

    def set_status(self, text):
        self._queue.put(("status", text))

    def append_log(self, text):
        self._queue.put(("log", text))

    def turn_done(self):
        self._queue.put(("done", None))

    def _poll_queue(self):
        try:
            while True:
                kind, payload = self._queue.get_nowait()
                if kind == "status":
                    self.status_var.set(payload)
                elif kind == "log":
                    self.log.config(state="normal")
                    self.log.insert("end", payload + "\n\n")
                    self.log.see("end")
                    self.log.config(state="disabled")
                elif kind == "done":
                    self.busy = False
                    self.status_var.set("按回车开始说话")
        except queue.Empty:
            pass
        self.root.after(50, self._poll_queue)

    def run(self):
        self.root.mainloop()


def main():
    client = OpenAI()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    history = []

    def do_turn(win):
        try:
            win.set_status("🎙️ 录音中...（说完停顿一下会自动结束）")
            audio, user_emotion = record_audio_with_emotion(RECORD_SAMPLERATE)

            rms = float(np.sqrt(np.mean(audio ** 2))) if len(audio) else 0.0
            if rms < SILENCE_RMS_THRESHOLD:
                win.append_log("（没录到声音，再试一次 - 检查麦克风或说话声音大一点）")
                return

            win.set_status("🧠 识别中...")
            user_text = transcribe(client, audio, RECORD_SAMPLERATE)
            if any(h in user_text.lower() for h in KNOWN_HALLUCINATIONS):
                user_text = ""
            if not user_text.strip():
                win.append_log("（没听清，再试一次）")
                return
            win.append_log(f"你: {user_text}")

            win.set_status("💭 思考中...")
            reply, emotion, intensity = chat(client, user_text, user_emotion, history)
            win.append_log(f"她: {reply}")

            win.set_status("🗣️ 说话中...")
            speak_with_expression(client, sock, reply, emotion, intensity)
        except Exception as e:
            win.append_log(f"（出错了：{e}）")
        finally:
            win.turn_done()

    window = ChatWindow(do_turn)
    try:
        window.run()
    finally:
        if os.path.exists(FLAG_PATH):
            os.remove(FLAG_PATH)


def read_user_emotion():
    """读 face_capture.py 写的 latest_state.json，取里面的粗粒度情绪标签。文件不存在/
    读取失败（比如正好撞上 face_capture.py 在写）都当作"不知道"处理，不影响主流程。"""
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("emotion")
    except (OSError, ValueError):
        return None


def record_audio_with_emotion(samplerate):
    """录音，说话声音一停（连续 SILENCE_TAIL_SECONDS 秒检测不到声音）就自动结束，不用像
    以前那样死等固定秒数。同时每隔 EMOTION_SAMPLE_INTERVAL 秒采样一次用户的表情，录完取
    出现次数最多的标签，作为"你说这句话时的表情"（这部分逻辑不变）。"""
    frames = []

    def callback(indata, frame_count, time_info, status):
        frames.append(indata.copy())

    emotion_samples = []
    blocksize = max(1, int(samplerate * VAD_BLOCK_SECONDS))
    stream = sd.InputStream(samplerate=samplerate, channels=1, dtype="float32",
                             blocksize=blocksize, callback=callback)
    with stream:
        elapsed = 0.0
        since_emotion_sample = 0.0
        has_speech = False
        silence_duration = 0.0

        while True:
            time.sleep(VAD_BLOCK_SECONDS)
            elapsed += VAD_BLOCK_SECONDS

            recent = frames[-1] if frames else None
            rms = float(np.sqrt(np.mean(recent ** 2))) if recent is not None and len(recent) else 0.0
            if rms >= VAD_RMS_THRESHOLD:
                has_speech = True
                silence_duration = 0.0
            else:
                silence_duration += VAD_BLOCK_SECONDS

            since_emotion_sample += VAD_BLOCK_SECONDS
            if since_emotion_sample >= EMOTION_SAMPLE_INTERVAL:
                since_emotion_sample = 0.0
                e = read_user_emotion()
                if e:
                    emotion_samples.append(e)

            if elapsed >= RECORD_MAX_SECONDS:
                break
            if elapsed >= RECORD_MIN_SECONDS and has_speech and silence_duration >= SILENCE_TAIL_SECONDS:
                break

    audio = np.concatenate(frames, axis=0).reshape(-1) if frames else np.zeros(0, dtype="float32")
    user_emotion = Counter(emotion_samples).most_common(1)[0][0] if emotion_samples else "未知"
    return audio, user_emotion


def transcribe(client, audio, samplerate):
    # 不传 language，让 Whisper 自动识别是中文还是英文——之前写死 language="zh"，用户说
    # 英文也会被强行按中文识别，没法随时中英切换。自动识别对短句偶尔会猜错语言，但这是
    # "随时切换、不用管开关"和"识别更准"之间的取舍，用户明确要的是前者。
    buf = io.BytesIO()
    sf.write(buf, audio, samplerate, format="WAV")
    buf.seek(0)
    buf.name = "audio.wav"
    result = client.audio.transcriptions.create(model="whisper-1", file=buf)
    return result.text


MAX_HISTORY_MESSAGES = 12  # 一问一答算 2 条，12 条约等于最近 6 轮 —— 聊得越久历史越长，
                            # 每次请求要处理的输入也越长，"思考"会跟着变慢，只留最近几轮
CHAT_MAX_TOKENS = 200       # 回复本来就要求 1-2 句话，加个上限防止偶尔啰嗦拖慢生成速度
                            # （GPT 是逐 token 生成的，输出越长耗时越久）


def chat(client, user_text, user_emotion, history):
    augmented = user_text
    if user_emotion and user_emotion != "未知":
        augmented += f"\n\n[摄像头检测到用户当前表情：{user_emotion}]"
    history.append({"role": "user", "content": augmented})

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history[-MAX_HISTORY_MESSAGES:]
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        response_format={"type": "json_object"},
        max_tokens=CHAT_MAX_TOKENS,
    )
    content = resp.choices[0].message.content
    history.append({"role": "assistant", "content": content})
    del history[:-MAX_HISTORY_MESSAGES]

    try:
        data = json.loads(content)
    except ValueError:
        data = {}

    reply = data.get("reply", content)
    emotion = data.get("emotion", "neutral")
    if emotion not in EMOTION_BLENDSHAPES:
        emotion = "neutral"

    try:
        intensity = max(0.0, min(1.0, float(data.get("intensity", 0.7))))
    except (TypeError, ValueError):
        intensity = 0.7

    return reply, emotion, intensity


# intensity=0 情绪还是要能被看出来一点，不然 GPT 给个偏低的 intensity 表情就几乎消失了；
# 用这个下限把 [0,1] 映射到 [0.35,1.0] 再去乘表情目标值。
INTENSITY_FLOOR = 0.35


# OpenAI 官方声音库里没有专门的"萝莉/幼齿"音色（13 个都是成年人声线：alloy/ash/ballad/
# cedar/coral/echo/fable/marin/nova/onyx/sage/shimmer/verse），这里退而求其次：换成
# gpt-4o-mini-tts 这个新模型，它支持用自然语言"风格引导"（TTS_INSTRUCTIONS，tts-1/
# tts-1-hd 不支持这个参数），配上偏活泼的 coral 音色，尽量往"音调高、卡哇伊、俏皮"上靠。
# 效果达不到专业萝莉配音的程度，真要那种效果得换 ElevenLabs 这类有现成动漫声线库的第三方
# TTS（需要另开账号/API key）。另外 gpt-4o-mini-tts 是新模型，生成速度可能比专门为实时
# 场景优化过的 tts-1 略慢一点，如果觉得说话前等待变长了，把下面两行改回
# TTS_MODEL="tts-1", TTS_VOICE="shimmer"，并把 speak_with_expression 里的 instructions
# 参数删掉即可。
TTS_MODEL = "gpt-4o-mini-tts"
TTS_VOICE = "coral"
TTS_INSTRUCTIONS = "说话声音偏稚嫩、音调偏高，语气活泼俏皮、卡哇伊，像动漫里的萝莉角色，语速可以稍微快一点点。"


def speak_with_expression(client, sock, text, emotion, intensity):
    resp = client.audio.speech.create(
        model=TTS_MODEL, voice=TTS_VOICE, input=text, instructions=TTS_INSTRUCTIONS,
    )
    data, samplerate = sf.read(io.BytesIO(resp.content), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)

    scale = INTENSITY_FLOOR + (1.0 - INTENSITY_FLOOR) * intensity
    base_shapes = {name: min(1.0, value * scale) for name, value in EMOTION_BLENDSHAPES.get(emotion, {}).items()}

    open(FLAG_PATH, "w").close()
    try:
        _play_with_mouth_sync(sock, data, samplerate, base_shapes)
    finally:
        if os.path.exists(FLAG_PATH):
            os.remove(FLAG_PATH)


def _head_sway_matrix(t):
    """一个缓慢、幅度很小的头部摆动，跟真实摄像头动作无关——纯粹是几个低频（<0.2Hz）
    正弦波叠加出来的"人说话时头不会完全定住"的效果。返回的矩阵是 MediaPipe 那套
    X-right/Y-up/Z-toward-viewer 约定下的旋转矩阵，跟 face_capture.py 发的格式一致，
    blender_receiver.py 那边会用同一个 AXIS_REMAP 去转换，不用额外处理。
    """
    yaw = np.radians(HEAD_SWAY_YAW_DEG) * np.sin(2 * np.pi * 0.15 * t + 0.3)
    pitch = np.radians(HEAD_SWAY_PITCH_DEG) * np.sin(2 * np.pi * 0.11 * t)
    roll = np.radians(HEAD_SWAY_ROLL_DEG) * np.sin(2 * np.pi * 0.07 * t + 1.1)

    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)

    r_yaw = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    r_pitch = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    r_roll = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])

    return (r_yaw @ r_pitch @ r_roll).tolist()


def _play_with_mouth_sync(sock, data, samplerate, base_shapes):
    """播放音频的同时，按 ~30fps 用每一小段音频的音量(RMS)去驱动 jawOpen，做一个粗糙但
    够用的"张嘴说话"效果（不是真正的口型/音素同步）。另外叠加几样东西让说话看起来更自然：
    - 固定随机间隔的眨眼（不然说话时会一直死盯着，很假）
    - 音量突然变大时（相对最近一小段的平均音量）稍微挑一下眉毛，模拟说话重音/强调
    - 一个很轻微的头部摆动（见 _head_sway_matrix），face_capture.py 说话时不再发送真实头部
      姿态（用户要求头别跟着摄像头动），所以这里补一个独立的、幅度很小的摆动，不然头会
      整段话都完全定住不动
    """
    pos = 0
    t0 = time.time()
    next_blink_at = random.uniform(*FIRST_BLINK_RANGE)
    blink_start = None
    rms_history = []

    def callback(outdata, frames, time_info, status):
        nonlocal pos, next_blink_at, blink_start

        chunk = data[pos:pos + frames]
        finished = len(chunk) < frames
        if finished:
            outdata[:len(chunk), 0] = chunk
            outdata[len(chunk):, 0] = 0
        else:
            outdata[:, 0] = chunk
        pos += frames

        t = time.time() - t0
        rms = float(np.sqrt(np.mean(chunk ** 2))) if len(chunk) else 0.0
        rms_history.append(rms)
        if len(rms_history) > 15:
            rms_history.pop(0)
        avg_rms = sum(rms_history) / len(rms_history) if rms_history else 0.0

        # 音量包络（多大声）决定嘴巴整体动多大；具体张成什么形状则用几个频率不成整数倍
        # 关系的正弦波叠加去"编"出一个连续变化但不重复的口型组合，比单一张合维度更有变化，
        # 相当于假的音素/口型同步 - 没有真的分析你说的是什么音，但看起来不会太死板。
        # base_shapes 里如果情绪本身也带了这几个嘴部分量（比如 laughing 的 jawOpen），
        # 跟说话的动态口型叠加而不是直接覆盖掉，不然静态表情的嘴型一说话就被抹平了。
        envelope = min(1.0, rms * 6.0)
        shapes = dict(base_shapes)
        shapes["jawOpen"] = min(1.0, base_shapes.get("jawOpen", 0.0)
                                 + envelope * (0.85 + 0.15 * np.sin(2 * np.pi * 8 * t)))
        shapes["mouthFunnel"] = min(1.0, base_shapes.get("mouthFunnel", 0.0)
                                     + envelope * max(0.0, 0.35 * np.sin(2 * np.pi * 3.1 * t + 0.7)))
        shapes["mouthPucker"] = min(1.0, base_shapes.get("mouthPucker", 0.0)
                                     + envelope * max(0.0, 0.3 * np.sin(2 * np.pi * 4.7 * t + 2.1)))
        shapes["mouthLeft"] = min(1.0, base_shapes.get("mouthLeft", 0.0)
                                   + envelope * max(0.0, 0.15 * np.sin(2 * np.pi * 2.3 * t + 1.4)))
        shapes["mouthRight"] = min(1.0, base_shapes.get("mouthRight", 0.0)
                                    + envelope * max(0.0, 0.15 * np.sin(2 * np.pi * 2.3 * t + 1.4 + np.pi)))

        if rms > avg_rms * 1.6 and rms > 0.03:
            shapes["browInnerUp"] = min(1.0, shapes.get("browInnerUp", 0.0) + 0.25)
            shapes["browOuterUpLeft"] = min(1.0, shapes.get("browOuterUpLeft", 0.0) + 0.15)
            shapes["browOuterUpRight"] = min(1.0, shapes.get("browOuterUpRight", 0.0) + 0.15)

        if blink_start is None and t >= next_blink_at:
            blink_start = t
        if blink_start is not None:
            phase = (t - blink_start) / BLINK_DURATION
            if phase >= 1.0:
                blink_start = None
                next_blink_at = t + random.uniform(*BLINK_INTERVAL_RANGE)
            else:
                blink_val = float(np.sin(phase * np.pi))  # 0 -> 1 -> 0
                shapes["eyeBlinkLeft"] = blink_val
                shapes["eyeBlinkRight"] = blink_val

        try:
            payload = {"blendshapes": shapes, "rotation_matrix": _head_sway_matrix(t)}
            sock.sendto(json.dumps(payload).encode("utf-8"), (UDP_IP, UDP_PORT))
        except OSError:
            pass

        if finished:
            raise sd.CallbackStop()

    stream = sd.OutputStream(samplerate=samplerate, channels=1, callback=callback)
    with stream:
        while stream.active:
            time.sleep(0.05)

    # 说完把嘴合上、眼睛睁开、头摆正，表情保留一下下再交还摄像头镜像
    rest_shapes = dict(base_shapes)
    rest_shapes.update({
        "jawOpen": 0.0, "eyeBlinkLeft": 0.0, "eyeBlinkRight": 0.0,
        "mouthFunnel": 0.0, "mouthPucker": 0.0, "mouthLeft": 0.0, "mouthRight": 0.0,
    })
    try:
        rest_payload = {"blendshapes": rest_shapes, "rotation_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}
        sock.sendto(json.dumps(rest_payload).encode("utf-8"), (UDP_IP, UDP_PORT))
    except OSError:
        pass


if __name__ == "__main__":
    main()
