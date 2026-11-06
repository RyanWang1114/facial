"""
Run inside Blender's Scripting workspace. TARGET_OBJECT_NAME just needs to be any one
mesh belonging to the face rig (e.g. the head). Real ARKit face rigs are often several
separate mesh objects sharing one parent - head, eyeLeft, eyeRight, teeth, ... - each
with its own Shape Keys block, so this script walks up to that shared root and drives
Shape Keys on every mesh underneath it whose key names match MediaPipe's ARKit blendshape
names (jawOpen, mouthSmileLeft, eyeLookInLeft, ...), ignoring names that don't match on
each object. Works with the single-object placeholder TestHead too.

Listens for the JSON payload that facial/face_capture.py streams over UDP - blendshape
scores plus a head-rotation matrix - and drives those Shape Key values plus the rig
root's rotation_euler in real time via a repeating timer. Both signals are exponentially
smoothed (see SMOOTHING_FACTOR) to reduce MediaPipe's per-frame jitter.

Click Play once; it keeps running in the background while you use the rest of Blender.
You can freely edit this file and rerun it (or change TARGET_OBJECT_NAME) any time - it
reuses the already-bound socket but always swaps in the latest polling logic, so edits
take effect immediately with no Blender restart. To stop it entirely, close/reload the
.blend file.
"""
import json
import socket

import bpy
from mathutils import Matrix

UDP_IP = "127.0.0.1"
UDP_PORT = 5006

# Change this any time and rerun the script - takes effect immediately, no restart needed.
TARGET_OBJECT_NAME = "head_lod0_ORIGINAL"

# MediaPipe's head-rotation matrix uses X-right/Y-up/Z-toward-viewer axes. Blender's
# front view looks down -Y (see blender_build_test_head.py), so "toward viewer" is -Y
# there instead of +Z. This matrix re-expresses MediaPipe's rotation in Blender's axes
# (a change-of-basis, not a guess). If the head still turns/nods the wrong way once you
# test it, your model's rest pose isn't facing -Y like TestHead's does - try swapping
# which row holds the -1 below.
AXIS_REMAP = Matrix((
    (1, 0, 0),
    (0, 0, -1),
    (0, 1, 0),
))

# Exponential smoothing applied every frame to both blendshape scores and head rotation,
# to take the edge off MediaPipe's per-frame jitter. Range (0, 1]: 1.0 = no smoothing
# (raw, snappiest, most jittery), smaller = smoother but laggier. Tweak and rerun any time.
SMOOTHING_FACTOR = 0.35

_state = bpy.app.driver_namespace.setdefault("_facial_udp_state", {})
_state["target"] = TARGET_OBJECT_NAME
_state.setdefault("smoothed_bs", {})
_state.setdefault("smoothed_quat", None)

if "sock" not in _state:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setblocking(False)
    try:
        sock.bind((UDP_IP, UDP_PORT))
    except OSError as e:
        print(f"⚠️ 无法监听端口 {UDP_PORT}：{e}")
        sock = None
    _state["sock"] = sock

# Unregister whichever _poll_udp closure a previous run of this script registered, so
# rerunning after an edit doesn't leave the OLD polling logic running forever alongside
# (or instead of) the new one - bpy.app.timers keeps calling whatever function object it
# was given, oblivious to changes made to this file on disk since then.
_old_timer = _state.get("timer_func")
if _old_timer is not None and bpy.app.timers.is_registered(_old_timer):
    bpy.app.timers.unregister(_old_timer)


def _root_ancestor(obj):
    """Walk up to the topmost parent. Face rigs like this one are usually a set of
    separate mesh objects (head, eyeLeft, eyeRight, teeth, ...) all parented under one
    root/armature - rotating just the head mesh leaves the others behind, which is why
    the head appeared to "vanish" while the eyes/teeth stayed put. Rotating the shared
    root instead turns the whole face together. If the target has no parent (e.g. the
    single-object TestHead placeholder), this just returns the object itself."""
    while obj.parent is not None:
        obj = obj.parent
    return obj


def _collect_shape_key_objects(root):
    """Every mesh in root's hierarchy (root included) that has a Shape Keys block - e.g.
    head + eyeLeft + eyeRight + teeth as separate objects under one shared parent. Found
    by walking the tree rather than hardcoding names, since which parts a given face rig
    splits into varies (this specific rig separates eyeball gaze and teeth/jaw from the
    head mesh, each needing its own shape keys driven)."""
    found = []
    stack = [root]
    while stack:
        o = stack.pop()
        if o.type == 'MESH' and o.data.shape_keys is not None:
            found.append(o)
        stack.extend(o.children)
    return found


def _poll_udp():
    s = _state.get("sock")
    if s is None:
        return None

    obj = bpy.data.objects.get(_state["target"])
    if obj is None:
        return 1 / 30

    rotation_target = _root_ancestor(obj)
    shape_key_objects = _collect_shape_key_objects(rotation_target)
    if not shape_key_objects:
        return 1 / 30

    latest = None
    try:
        while True:
            data, _addr = s.recvfrom(65536)
            latest = data
    except BlockingIOError:
        pass

    if latest is not None:
        try:
            payload = json.loads(latest.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            payload = {}

        bs = payload.get("blendshapes", {})
        rot_rows = payload.get("rotation_matrix")
        smoothed_bs = _state["smoothed_bs"]

        for name, raw_score in bs.items():
            prev = smoothed_bs.get(name, raw_score)
            smoothed_bs[name] = prev + (raw_score - prev) * SMOOTHING_FACTOR

        matched = 0
        for shape_obj in shape_key_objects:
            kb = shape_obj.data.shape_keys.key_blocks
            for name, score in smoothed_bs.items():
                if name in kb:
                    kb[name].value = score
                    matched += 1
            # mouthSmile fallback for the 3-key TestHead placeholder, whose key doesn't
            # match a single ARKit name (it's an average of the left/right pair).
            if "mouthSmile" in kb:
                left = smoothed_bs.get("mouthSmileLeft", 0.0)
                right = smoothed_bs.get("mouthSmileRight", 0.0)
                kb["mouthSmile"].value = (left + right) / 2

        if rot_rows is not None:
            rotation_target.rotation_mode = 'XYZ'
            r_mp = Matrix(rot_rows)
            r_blender = AXIS_REMAP @ r_mp @ AXIS_REMAP.transposed()
            target_quat = r_blender.to_quaternion()

            prev_quat = _state["smoothed_quat"]
            smoothed_quat = target_quat if prev_quat is None else prev_quat.slerp(target_quat, SMOOTHING_FACTOR)
            _state["smoothed_quat"] = smoothed_quat

            rotation_target.rotation_euler = smoothed_quat.to_euler('XYZ')

        if matched > 0 or rot_rows is not None:
            for window in bpy.context.window_manager.windows:
                for area in window.screen.areas:
                    if area.type == 'VIEW_3D':
                        area.tag_redraw()

    return 1 / 30  # poll again in ~33ms


_state["timer_func"] = _poll_udp

if _state["sock"] is not None:
    bpy.app.timers.register(_poll_udp)
    print(f"✅ 正在监听 {UDP_IP}:{UDP_PORT}，目标物体是 '{TARGET_OBJECT_NAME}'。")
