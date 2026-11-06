"""
一键启动"桌宠"：一个命令把 Blender（pink.blend 里的 VRM 头像）+ face_capture.py（摄像头
表情捕捉）+ interactive_agent.py（语音对话，一个小对话框窗口）一起拉起来。你不用再分别
手动开 Blender、再开两个终端——跑这一个脚本就行：Blender 窗口显示 3D 角色，另外弹出的小
对话框里回车说话、看文字记录，两个窗口摆在一起用。

退出：把那个小对话框窗口关掉就行，Blender 和摄像头会跟着自动关掉（这个终端窗口在等
对话框关闭前会一直卡着不动，这是正常的）。也可以直接在这个终端按 Ctrl+C。

通过 BLENDER_EXE 环境变量指定 Blender 可执行文件，或把 blender 加入 PATH。
通过 BLEND_FILE 环境变量指定兼容的 .blend 场景，默认使用脚本目录下的 pink.blend。
"""
import os
import shutil
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
BLENDER_EXE = os.environ.get("BLENDER_EXE") or shutil.which("blender")
BLEND_FILE = os.path.abspath(os.path.expanduser(
    os.environ.get("BLEND_FILE", os.path.join(_HERE, "pink.blend"))
))
RECEIVER_SCRIPT = os.path.join(_HERE, "blender_receiver_vrm.py")
FACE_CAPTURE_SCRIPT = os.path.join(_HERE, "face_capture.py")
INTERACTIVE_AGENT_SCRIPT = os.path.join(_HERE, "interactive_agent.py")


def main():
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    if not BLENDER_EXE or not os.path.isfile(BLENDER_EXE):
        print(f"❌ 找不到 Blender：{BLENDER_EXE}")
        print("   请将 BLENDER_EXE 环境变量设为 Blender 可执行文件的完整路径，或把 blender 加入 PATH。")
        return 1
    if not os.path.exists(BLEND_FILE):
        print(f"❌ 找不到 {BLEND_FILE}")
        return 1

    if not os.environ.get("OPENAI_API_KEY"):
        print("❌ 请先设置 OPENAI_API_KEY 环境变量。语音对话会调用 OpenAI API。")
        return 1

    processes = []
    try:
        print("🚀 正在启动 Blender（VRM 角色）...")
        blender_proc = subprocess.Popen([BLENDER_EXE, BLEND_FILE, "--python", RECEIVER_SCRIPT], cwd=_HERE)
        processes.append(blender_proc)
        time.sleep(5)  # Give Blender time to open the scene and bind the UDP port.

        print("🎥 正在启动摄像头表情捕捉...")
        face_proc = subprocess.Popen([sys.executable, "-X", "utf8", FACE_CAPTURE_SCRIPT], cwd=_HERE)
        processes.append(face_proc)
        time.sleep(2)

        print("🎙️ 一切就绪，对话框窗口马上弹出来，在里面按回车说话（关掉那个窗口即可退出全部）...\n")
        subprocess.run([sys.executable, "-X", "utf8", INTERACTIVE_AGENT_SCRIPT], cwd=_HERE, check=True)
    except KeyboardInterrupt:
        pass
    finally:
        print("\n👋 正在关闭 Blender 和摄像头...")
        for proc in reversed(processes):
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
    return 0


if __name__ == "__main__":
    sys.exit(main())
