"""
和 blender_receiver.py 功能一样（同一个 UDP 端口/协议，监听 facial/face_capture.py 或
facial/interactive_agent.py 发来的 blendshape + 头部旋转数据），目标模型换成了
pink.blend（VRM 头像，正常导入、带完整骨骼），独立出一个文件，方便切换演示两个不同的模型
而不用互相改配置。

用法：在 Blender 里打开 pink.blend，Scripting 工作区里跑这个文件。

跟 blender_receiver.py 的关键差别（都是因为 pink.blend 是真正标准的 VRM 骨骼绑定，跟旧的
MetaHuman FBX 头结构不一样）：
- blendshape 目标依然是 "Face" 网格（118 个 shape key，完整包含 52 个标准 ARKit 名字，
  跟 face_capture.py/interactive_agent.py 发的名字直接对得上，不用改名映射），这部分逻辑
  跟 blender_receiver.py 完全一样，复用 _root_ancestor()/_collect_shape_key_objects()。
- 但头部旋转不能像旧模型那样直接转整个根物体了：pink.blend 里 Face/Body/Hair 都是
  Armature（240 根骨骼）的子物体，转 Armature 这个物体会让整个身体一起转，而不是只转头。
  正确做法是转骨骼本身——Face 用顶点组绑定在骨骼上（Armature 修改器），只要转
  "J_Bip_C_Head" 这根骨骼，蒙皮会自动带动 Face/Hair 一起变形，身体其它部分不受影响。
- 骨骼的静止姿态（rest pose）本身在 armature 空间里就带有旋转（不是单位矩阵，VRM 骨骼
  的朝向惯例跟世界坐标轴不对齐），所以不能像旧脚本那样直接把 rotation_euler 设成摄像头
  算出来的角度——那样会把骨骼的静止朝向直接覆盖掉，姿势会整个崩掉。做法是先在第一帧
  记下骨骼原本的静止旋转矩阵(head_rest_rot)，之后每一帧都是"静止朝向 + 摄像头这一帧的
  增量旋转"叠加着设，而不是替换。

⚠️ AXIS_REMAP 这个矩阵沿用了旧模型那份，但没有实际在 Blender 里转头验证过方向对不对
（VRM 模型朝向、骨骼 roll 都可能跟旧的 FBX 头不一样）。跑起来之后如果发现转头方向不对
或者头部姿势很怪，大概率是这里或者旋转叠加顺序（smoothed_quat.to_matrix() @ rest_rot
这一行）需要调整，照着报错现象试着交换乘法顺序或者调整 AXIS_REMAP 就行，不是逻辑性的
大问题。

演示的时候不想让人看到这是 Blender 在跑，脚本一启动会自动把窗口切成"只剩 3D 视图，没有
菜单栏/工具栏/侧边栏"的干净样子（等同于手动在视图里按 Ctrl+Space），细节见 _go_clean_
fullscreen()。这一步需要真正的 GUI 窗口环境（不能在 --background 无头模式下验证），
只在 Blender 图形界面里跑这个脚本时才会生效。想切回正常界面的话，鼠标移到视图里再按一次
Ctrl+Space 就行（Blender 自己的快捷键，不是这个脚本管的）。
"""
import json
import math
import random
import socket
import time

import bpy
from mathutils import Euler, Matrix, Quaternion

UDP_IP = "127.0.0.1"
UDP_PORT = 5006

TARGET_OBJECT_NAME = "Face"
ARMATURE_OBJECT_NAME = "Armature"
HEAD_BONE_NAME = "J_Bip_C_Head"

AXIS_REMAP = Matrix((
    (1, 0, 0),
    (0, 0, -1),
    (0, 1, 0),
))

SMOOTHING_FACTOR = 0.35

# 自动对焦头部（见 _focus_on_head）之后，把视角瞄准点沿世界 Z 轴往上挪一点，效果是让
# 角色在画面里往下"沉"一些（视角瞄准点变高 = 画面里同一个东西相对显得更低）。正面视角下
# 世界 Z 轴正好对应屏幕的"上"方向。单位是米，相对 view_selected() 算出来的取景距离的比例，
# 不是绝对值，这样不管镜头离多近都是同一个观感。嫌角色还是太高/太低就调这个数字。
HEAD_FRAME_VERTICAL_OFFSET_RATIO = 0.18

# ---- 空闲微动作（眨眼 / 眼球小幅扫视 / 挑眉 / 头部轻微摆动）----
# 背景：interactive_agent.py 讲话模式下只发嘴部 blendshape，没有摄像头驱动眼睛/头部，脸
# 长时间定住不动是恐怖谷效应最明显的来源之一。这里完全在 Blender 本地用随机/正弦波生成，
# 不需要额外录数据，跟 UDP 收到的嘴型/头部追踪数据叠加在一起用。
#
# 眨眼/眼神/挑眉这三样只在"当前没有人在实时控制表情"时才生效（见 _poll_udp 里的
# actively_driven 判断）：如果每 tick 都无条件覆盖，说话时 GPT 选的表情（比如 surprised
# 会让眉毛抬得很高）会被这里的小幅度挑眉动作直接盖掉，白做了 EMOTION_BLENDSHAPES 那套。
# 头部摆动不受这个限制，摄像头/说话自带的头部旋转之上永远叠加一点点空闲摆动，因为那是
# 相对增量叠加而不是覆盖，不会互相打架。
ENABLE_IDLE_MOTION = True

# 没有收到新的、带 blendshape 数据的 UDP 包超过这么久，就认为"现在没人在实时控制表情"，
# 空闲动作接管；一旦有新包进来立刻退出空闲、把控制权交还（下一个 tick 就生效）。
ACTIVE_DRIVE_TIMEOUT = 0.5  # 秒

# 空闲状态下，上一次说话/摄像头留下的表情值（比如说完一句"惊讶"的话，眉毛还抬着）每 tick
# 按这个系数往 0 衰减，不然摄像头镜像关掉之后（ENABLE_LIVE_MIRROR=False）没有人会再把
# 这个表情盖掉，会一直卡在那个夸张表情上。0.94 大约 1-2 秒衰减到看不太出来，3-4 秒基本归零。
IDLE_DECAY_RATE = 0.94

BLINK_MIN_INTERVAL = 2.0   # 秒，两次眨眼之间最短间隔
BLINK_MAX_INTERVAL = 6.0   # 秒，最长间隔
BLINK_DURATION = 0.22      # 秒，单次眨眼从睁到闭再到睁的总时长

GAZE_MIN_INTERVAL = 1.5    # 秒，眼球扫视换目标点的最短间隔
GAZE_MAX_INTERVAL = 4.0    # 秒，最长间隔
GAZE_MAX_AMOUNT = 0.22     # 0~1，左右扫视的最大幅度（1 是 blendshape 拉满，太大会很夸张）
GAZE_VERTICAL_RATIO = 0.35  # 上下扫视幅度相对左右的比例，人平时空闲时左右扫视比上下常见，
                             # 上下这里明显调小，效果是"正常左右晃一下"而不是眼珠到处乱转
GAZE_EASE_FACTOR = 0.08    # 每 tick（30fps）朝目标缓动的比例，越大转得越快越生硬

BROW_MIN_INTERVAL = 4.0    # 秒，两次挑眉之间最短间隔（比眨眼少见，不然会很聒噪）
BROW_MAX_INTERVAL = 9.0    # 秒，最长间隔
BROW_DURATION = 0.55       # 秒，一次挑眉从平静到最高再回落的总时长，比眨眼慢一点更自然
BROW_MAX_AMOUNT = 0.22     # 0~1，挑眉的最大幅度，故意很小——只是个自然的小动作，不是
                            # "惊讶表情"那种拉满的挑眉

HEAD_SWAY_AMPLITUDE_DEG = 1.8   # 度，头部摆动最大幅度，故意很小，像呼吸感而不是晃头

_state = bpy.app.driver_namespace.setdefault("_facial_udp_state_vrm", {})
_state["target"] = TARGET_OBJECT_NAME
_state.setdefault("smoothed_bs", {})
_state.setdefault("smoothed_quat", None)
_state.setdefault("target_bs", {})
_state.setdefault("target_quat", None)
_state.setdefault("head_rest_rot", None)
_state.setdefault("head_rest_loc", None)
_state.setdefault("blink_start", None)
_state.setdefault("next_blink_at", None)
_state.setdefault("gaze_cur", [0.0, 0.0])
_state.setdefault("gaze_target", [0.0, 0.0])
_state.setdefault("gaze_next_change_at", None)
_state.setdefault("brow_start", None)
_state.setdefault("next_brow_at", None)
_state.setdefault("idle_start_time", None)
_state.setdefault("idle_phase", None)
_state.setdefault("last_packet_at", None)

_armature_obj = bpy.data.objects.get(ARMATURE_OBJECT_NAME)
if _armature_obj is not None:
    _armature_obj.hide_set(True)  # 骨骼线本身已经在 pink.blend 里存成隐藏了，这里是双保险


def _go_clean_fullscreen():
    """把当前窗口切成只剩 3D 视图、没有菜单栏/工具栏/侧边栏的样子（演示的时候不想让人看到
    这是 Blender 在跑）。用的是 Blender 自带的"最大化区域并隐藏面板"（默认快捷键
    Ctrl+Space），效果等同于手动按一下那个快捷键。用 _state 里的标记位防止脚本被重复运行
    时把已经最大化的视图又切回去（这个操作本身是个开关，调两次等于没调）。
    """
    if _state.get("went_fullscreen"):
        return
    window = bpy.context.window
    if window is None:
        return
    screen = window.screen
    viewport_area = next((a for a in screen.areas if a.type == 'VIEW_3D'), None)
    if viewport_area is None:
        return

    with bpy.context.temp_override(window=window, area=viewport_area):
        bpy.ops.screen.screen_full_area(use_hide_panels=True)

    # screen_full_area 执行完以后，原来那个 viewport_area 对象已经失效了（全屏会把区域
    # 整个换掉，重新生成一个新的 area），继续用旧引用取 .spaces[0] 会因为集合是空的直接
    # 报 IndexError，而且是在模块顶层跑的，一崩溃后面绑 UDP 端口、注册定时器的代码全都
    # 不会执行——表现为头像彻底没反应，不是"偶尔崩"而是"必现"。这里改成崩溃后重新从
    # window.screen 里找一遍新的 VIEW_3D 区域。
    new_area = next((a for a in window.screen.areas if a.type == 'VIEW_3D'), None)
    if new_area is not None and new_area.spaces:
        space = new_area.spaces[0]
        space.overlay.show_overlays = False  # 网格线这些调试用的东西隐藏掉
        # 注意：不关 show_gizmo——那是导航小工具（右上角那个球 + 下面的手型平移按钮）的
        # 总开关，关掉整个视图右上角就什么都没有了，没法自己手动拖动调整视角。只是不想
        # 看到菜单栏/工具栏，跟这个导航小工具没关系，所以留着。

    _state["went_fullscreen"] = True


try:
    _go_clean_fullscreen()
except Exception as e:
    # 这一步纯粹是演示时候好看，不是核心功能——万一在某些窗口状态下又出别的问题，
    # 宁可保留菜单栏也不能让它把下面绑端口/注册定时器的代码一起拖垮。
    print(f"⚠️ 隐藏 Blender 界面失败（不影响表情/头部同步功能）：{e}")


def _focus_on_head():
    """打开就把视角对准脸部附近，方便盯着表情 debug（不然默认是打开文件时保存的那个
    视角，可能是全身或者别的角度，脸小看不清）。做法：选中 Face 这个网格（头部所在的
    那个网格对象），切到正面视角，再用 Blender 自带的"选中项归位"（数字键盘 . 那个功能）
    让视图自动缩放对准选中的东西——具体该拉多近让 Blender 自己根据 Face 网格的大小算，
    不用我们手动猜一个距离数字。"""
    face_obj = bpy.data.objects.get(TARGET_OBJECT_NAME)
    if face_obj is None:
        return
    window = bpy.context.window
    if window is None:
        return
    area = next((a for a in window.screen.areas if a.type == 'VIEW_3D'), None)
    if area is None:
        return
    region = next((r for r in area.regions if r.type == 'WINDOW'), None)
    if region is None:
        return

    # 记住原来选中的东西，弄完再还原，不影响其它可能依赖选中状态的操作
    prev_selected = list(bpy.context.selected_objects)
    prev_active = bpy.context.view_layer.objects.active

    for o in bpy.data.objects:
        o.select_set(False)
    face_obj.select_set(True)
    bpy.context.view_layer.objects.active = face_obj

    with bpy.context.temp_override(window=window, area=area, region=region):
        bpy.ops.view3d.viewnumpad(type='FRONT')
        bpy.ops.view3d.view_selected()

    space = area.spaces[0] if area.spaces else None
    if space is not None and space.region_3d is not None:
        rv3d = space.region_3d
        rv3d.view_location.z += rv3d.view_distance * HEAD_FRAME_VERTICAL_OFFSET_RATIO

    for o in bpy.data.objects:
        o.select_set(o in prev_selected)
    bpy.context.view_layer.objects.active = prev_active


try:
    _focus_on_head()
except Exception as e:
    print(f"⚠️ 自动对焦头部失败（不影响表情/头部同步功能）：{e}")

if "sock" not in _state:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setblocking(False)
    try:
        sock.bind((UDP_IP, UDP_PORT))
    except OSError as e:
        print(f"⚠️ 无法监听端口 {UDP_PORT}：{e}")
        sock = None
    _state["sock"] = sock

_old_timer = _state.get("timer_func")
if _old_timer is not None and bpy.app.timers.is_registered(_old_timer):
    bpy.app.timers.unregister(_old_timer)


def _root_ancestor(obj):
    while obj.parent is not None:
        obj = obj.parent
    return obj


def _collect_shape_key_objects(root):
    found = []
    stack = [root]
    while stack:
        o = stack.pop()
        if o.type == 'MESH' and o.data.shape_keys is not None:
            found.append(o)
        stack.extend(o.children)
    return found


def _idle_blink_value(now):
    """返回这一帧的眨眼强度（0=睁眼，1=完全闭眼），内部维护"下次眨眼时间/眨眼开始时间"
    的简单状态机。用 sin 曲线让睁->闭->睁一次成型，不用分别处理闭眼和睁眼两段。"""
    if _state["next_blink_at"] is None:
        _state["next_blink_at"] = now + random.uniform(BLINK_MIN_INTERVAL, BLINK_MAX_INTERVAL)

    if _state["blink_start"] is None and now >= _state["next_blink_at"]:
        _state["blink_start"] = now

    if _state["blink_start"] is None:
        return 0.0

    t = now - _state["blink_start"]
    if t >= BLINK_DURATION:
        _state["blink_start"] = None
        _state["next_blink_at"] = now + random.uniform(BLINK_MIN_INTERVAL, BLINK_MAX_INTERVAL)
        return 0.0

    return math.sin(math.pi * t / BLINK_DURATION)


def _idle_gaze_offsets(now):
    """返回这一帧的眼球扫视偏移 (gx, gy)，每隔随机几秒换一个新的随机目标点，每帧都朝目标
    缓动一点（而不是瞬间跳过去），看起来才像自然扫视而不是抽搐。"""
    if _state["gaze_next_change_at"] is None:
        _state["gaze_next_change_at"] = now + random.uniform(GAZE_MIN_INTERVAL, GAZE_MAX_INTERVAL)

    if now >= _state["gaze_next_change_at"]:
        vertical_amount = GAZE_MAX_AMOUNT * GAZE_VERTICAL_RATIO
        _state["gaze_target"] = [
            random.uniform(-GAZE_MAX_AMOUNT, GAZE_MAX_AMOUNT),
            random.uniform(-vertical_amount, vertical_amount),
        ]
        _state["gaze_next_change_at"] = now + random.uniform(GAZE_MIN_INTERVAL, GAZE_MAX_INTERVAL)

    cur = _state["gaze_cur"]
    target = _state["gaze_target"]
    cur[0] += (target[0] - cur[0]) * GAZE_EASE_FACTOR
    cur[1] += (target[1] - cur[1]) * GAZE_EASE_FACTOR
    return cur[0], cur[1]


def _idle_brow_value(now):
    """返回这一帧的挑眉强度（0=平静，1=挑眉拉满），状态机结构跟 _idle_blink_value 一模
    一样，只是间隔更长、曲线更慢，效果是偶尔很自然地"眉毛动一下"而不是眨眼那种规律的
    开合。"""
    if _state["next_brow_at"] is None:
        _state["next_brow_at"] = now + random.uniform(BROW_MIN_INTERVAL, BROW_MAX_INTERVAL)

    if _state["brow_start"] is None and now >= _state["next_brow_at"]:
        _state["brow_start"] = now

    if _state["brow_start"] is None:
        return 0.0

    t = now - _state["brow_start"]
    if t >= BROW_DURATION:
        _state["brow_start"] = None
        _state["next_brow_at"] = now + random.uniform(BROW_MIN_INTERVAL, BROW_MAX_INTERVAL)
        return 0.0

    return math.sin(math.pi * t / BROW_DURATION)


def _idle_blendshapes(now):
    """把眨眼 + 眼球扫视 + 挑眉换算成对应的 ARKit blendshape 名字。gx/gy 的正负方向具体
    对应角色的左看/右看、上看/下看没有在 Blender 里实际验证过（跟 AXIS_REMAP 那个头部
    旋转矩阵一样），但因为是随机来回扫视，两边概率均等，方向就算反了看起来也仍然自然，
    不影响效果，不用特意去核对。"""
    blink = _idle_blink_value(now)
    gx, gy = _idle_gaze_offsets(now)
    brow = _idle_brow_value(now) * BROW_MAX_AMOUNT

    return {
        "eyeBlinkLeft": blink,
        "eyeBlinkRight": blink,
        "eyeLookInLeft": max(0.0, gx),
        "eyeLookOutRight": max(0.0, gx),
        "eyeLookOutLeft": max(0.0, -gx),
        "eyeLookInRight": max(0.0, -gx),
        "eyeLookUpLeft": max(0.0, gy),
        "eyeLookUpRight": max(0.0, gy),
        "eyeLookDownLeft": max(0.0, -gy),
        "eyeLookDownRight": max(0.0, -gy),
        "browInnerUp": brow,
        "browOuterUpLeft": brow * 0.7,
        "browOuterUpRight": brow * 0.7,
    }


def _idle_head_quat(now):
    """头部轻微摆动：三个轴上频率、幅度都不同的正弦波叠加，幅度很小（HEAD_SWAY_AMPLITUDE_
    DEG），效果是"呼吸感"的轻微晃动而不是明显转头。相位在第一次调用时随机一次，不然每次
    重启 Blender 摆动节奏都完全一样，演示多次会让人看出是循环的。"""
    if _state["idle_start_time"] is None:
        _state["idle_start_time"] = now
        _state["idle_phase"] = (
            random.uniform(0, math.tau),
            random.uniform(0, math.tau),
            random.uniform(0, math.tau),
        )

    t = now - _state["idle_start_time"]
    px, py, pz = _state["idle_phase"]
    amp = math.radians(HEAD_SWAY_AMPLITUDE_DEG)
    ex = amp * math.sin(t * 0.31 + px)
    ey = amp * 0.6 * math.sin(t * 0.23 + py)
    ez = amp * 0.8 * math.sin(t * 0.19 + pz)
    return Euler((ex, ey, ez), 'XYZ').to_quaternion()


def _poll_udp():
    now = time.time()

    s = _state.get("sock")
    if s is None:
        return None

    obj = bpy.data.objects.get(_state["target"])
    if obj is None:
        return 1 / 30

    shape_key_objects = _collect_shape_key_objects(_root_ancestor(obj))
    armature_obj = bpy.data.objects.get(ARMATURE_OBJECT_NAME)
    head_bone = armature_obj.pose.bones.get(HEAD_BONE_NAME) if armature_obj else None

    if not shape_key_objects and head_bone is None:
        return 1 / 30

    # 静止姿态只在骨骼第一次可用时记一次，不依赖是否收到过 UDP 的 rotation_matrix——这样
    # 就算 interactive_agent.py 只发嘴型、从来不发头部数据，下面的空闲摆动照样能用这个基准。
    if head_bone is not None and _state["head_rest_rot"] is None:
        _state["head_rest_rot"] = head_bone.matrix.to_3x3()
        _state["head_rest_loc"] = head_bone.matrix.to_translation()

    latest = None
    try:
        while True:
            data, _addr = s.recvfrom(65536)
            latest = data
    except BlockingIOError:
        pass

    # 只在真的收到新数据时更新"目标值"(target_bs/target_quat)，不在这里直接做平滑/赋值。
    # 平滑和实际赋值放到下面，每次 poll(30fps) 都会做一遍 —— 这样即使摄像头/MediaPipe
    # 那边实际帧率比 30fps 低（很常见），Blender 这边也会持续朝着"最后一次收到的目标"
    # 缓动，而不是摄像头没发新数据的那些 tick 就直接定住不动、等下一帧数据来了再猛地
    # 跳一下，看起来更连贯。
    if latest is not None:
        try:
            payload = json.loads(latest.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            payload = {}

        bs = payload.get("blendshapes")
        if bs:
            _state["target_bs"] = bs
            _state["last_packet_at"] = now

        rot_rows = payload.get("rotation_matrix")
        if rot_rows is not None:
            r_mp = Matrix(rot_rows)
            r_blender = AXIS_REMAP @ r_mp @ AXIS_REMAP.transposed()
            _state["target_quat"] = r_blender.to_quaternion()

    idle_quat = None
    if ENABLE_IDLE_MOTION:
        idle_quat = _idle_head_quat(now)

        actively_driven = (
            _state["last_packet_at"] is not None
            and (now - _state["last_packet_at"]) < ACTIVE_DRIVE_TIMEOUT
        )
        if not actively_driven:
            # 没人在实时控制表情：先把上一轮留下的表情值往 0 衰减（见 IDLE_DECAY_RATE 的
            # 注释），再叠加眨眼/眼神/挑眉这些空闲小动作。实时控制中（说话/摄像头镜像）
            # 就完全不碰这些 key，把控制权彻底交给发送方，不然 GPT 选的表情会被这里盖掉。
            for name in _state["target_bs"]:
                _state["target_bs"][name] *= IDLE_DECAY_RATE
            combined_bs = dict(_state["target_bs"])
            combined_bs.update(_idle_blendshapes(now))
        else:
            combined_bs = dict(_state["target_bs"])
    else:
        combined_bs = dict(_state["target_bs"])

    smoothed_bs = _state["smoothed_bs"]
    for name, raw_score in combined_bs.items():
        prev = smoothed_bs.get(name, raw_score)
        smoothed_bs[name] = prev + (raw_score - prev) * SMOOTHING_FACTOR

    matched = 0
    for shape_obj in shape_key_objects:
        kb = shape_obj.data.shape_keys.key_blocks
        for name, score in smoothed_bs.items():
            if name in kb:
                kb[name].value = score
                matched += 1

    camera_quat = _state["target_quat"]
    if head_bone is not None and _state["head_rest_rot"] is not None:
        # 摄像头给的目标朝向（没有摄像头数据时就是单位四元数=不额外转）叠加上空闲摆动，
        # 两者都是"相对静止姿态的增量"，直接四元数相乘即可，顺序上 idle 在外层，效果是
        # 摆动永远叠在摄像头追踪结果之上，而不是被摄像头结果盖掉。
        combined_target_quat = camera_quat if camera_quat is not None else Quaternion()
        if idle_quat is not None:
            combined_target_quat = idle_quat @ combined_target_quat

        prev_quat = _state["smoothed_quat"]
        smoothed_quat = (
            combined_target_quat if prev_quat is None
            else prev_quat.slerp(combined_target_quat, SMOOTHING_FACTOR)
        )
        _state["smoothed_quat"] = smoothed_quat

        final_rot = (smoothed_quat.to_matrix() @ _state["head_rest_rot"]).to_quaternion()
        head_bone.matrix = Matrix.LocRotScale(_state["head_rest_loc"], final_rot, None)

    if matched > 0 or head_bone is not None:
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()

    return 1 / 30


_state["timer_func"] = _poll_udp

if _state["sock"] is not None:
    bpy.app.timers.register(_poll_udp)
    print(f"✅ 正在监听 {UDP_IP}:{UDP_PORT}，目标物体是 '{TARGET_OBJECT_NAME}' + 骨骼 '{HEAD_BONE_NAME}'（VRM 头像）。")
