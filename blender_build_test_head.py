"""
Run inside Blender's Scripting workspace (Scripting tab -> open this file -> Run Script,
or paste its contents into a new text block and click the Play button).

Builds a placeholder "TestHead" (a squashed UV sphere) with three Shape Keys:
mouthOpen, mouthSmile, browInnerUp — just enough to prove the face_capture.py ->
Blender data pipeline end to end before investing in a real ARKit-blendshape face rig.
"""
import bpy
from mathutils import Vector

if "TestHead" in bpy.data.objects:
    bpy.data.objects.remove(bpy.data.objects["TestHead"], do_unlink=True)

bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, radius=1.0, location=(0, 0, 0))
obj = bpy.context.active_object
obj.name = "TestHead"
obj.scale = (0.9, 1.0, 1.1)
bpy.ops.object.transform_apply(scale=True)

mesh = obj.data
obj.shape_key_add(name="Basis")


def add_shapekey(name, selector, offset):
    sk = obj.shape_key_add(name=name)
    sk.interpolation = 'KEY_LINEAR'
    count = 0
    for v in mesh.vertices:
        if selector(v.co):
            sk.data[v.index].co = v.co + offset
            count += 1
    print(f"{name}: affected {count} vertices")
    return sk


# NOTE: this sphere faces -Y (Blender's default "front"). If mouthOpen/mouthSmile end up
# moving the back of the head instead of the front when you test the sliders, flip the
# sign of the Y comparisons below (change "< -0.6" to "> 0.6", etc.) and rerun.
add_shapekey("mouthOpen",
             lambda co: co.y < -0.6 and -0.5 < co.z < -0.1,
             Vector((0, 0.05, -0.15)))
add_shapekey("mouthSmile",
             lambda co: co.y < -0.6 and -0.35 < co.z < -0.15 and abs(co.x) > 0.15,
             Vector((0, -0.05, 0.08)))
add_shapekey("browInnerUp",
             lambda co: co.y < -0.5 and 0.35 < co.z < 0.6 and abs(co.x) < 0.4,
             Vector((0, 0, 0.1)))

for kb in mesh.shape_keys.key_blocks:
    kb.value = 0.0

print("TestHead built. Open the Object Data Properties tab (green triangle icon) -> "
      "Shape Keys panel and drag the value sliders to sanity-check the three keys "
      "before running blender_receiver.py.")
