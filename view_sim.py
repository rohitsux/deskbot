"""
DeskBot — Live room viewer with real SO-101 arms (MuJoCo Menagerie).

Two SO-101 arms sit on a charcoal desk inside a realistic room.
Objects from the current episode are shown and updated live from the server.

Run: python3 view_sim.py
"""

import time, json, pathlib, threading
import urllib.request
import mujoco
import mujoco.viewer
import numpy as np

ROOT     = pathlib.Path(__file__).parent
SO101    = ROOT / "assets/robotstudio_so101/so101.xml"
DESK_H   = 0.74
DESK_T   = 0.02
ARM_Z    = DESK_H + 0.005

# ── Build base room XML (no arms — attached via mjSpec) ──────────────────────
BASE_XML = f"""
<mujoco model="deskbot_room">
  <option gravity="0 0 -9.81" timestep="0.004"/>
  <compiler meshdir="{ROOT}/assets/robotstudio_so101/assets" autolimits="true"/>

  <asset>
    <texture name="floor_tex" type="2d" builtin="checker"
             rgb1="0.55 0.47 0.35" rgb2="0.48 0.41 0.30" width="512" height="512"/>
    <material name="floor_mat" texture="floor_tex" texrepeat="6 6" reflectance="0.05"/>
    <texture name="wall_tex" type="2d" builtin="flat"
             rgb1="0.88 0.86 0.82" width="512" height="512"/>
    <material name="wall_mat" texture="wall_tex"/>
    <texture name="desk_tex" type="2d" builtin="flat"
             rgb1="0.16 0.14 0.13" width="512" height="512"/>
    <material name="desk_mat" texture="desk_tex" reflectance="0.15"/>
  </asset>

  <default>
    <geom condim="4" solimp="0.9 0.95 0.001" solref="0.02 1"/>
  </default>

  <worldbody>
    <!-- Lighting: warm key + cool fill + ambient -->
    <light name="key"  pos="-0.5 -1.0 2.5" dir="0.3 0.6 -1"
           diffuse="1.0 0.95 0.85" specular="0.3 0.3 0.3" castshadow="true"/>
    <light name="fill" pos=" 1.0  1.0 2.0" dir="-0.4 -0.4 -1"
           diffuse="0.4 0.45 0.6" specular="0.0 0.0 0.0" castshadow="false"/>
    <light name="desk_led" pos="0 {DESK_H - 0.3:.3f} {DESK_H + 0.05:.3f}"
           dir="0 0 -1" diffuse="0.9 0.88 0.75" specular="0.1 0.1 0.1"
           castshadow="false"/>

    <!-- Floor -->
    <geom name="floor" type="plane" size="3 3 0.1"
          material="floor_mat" friction="0.8 0.005 0.0001"
          contype="1" conaffinity="1"/>

    <!-- Walls -->
    <geom name="wall_back"  type="box" size="2.5 0.05 1.5"
          pos="0 1.8 1.2" rgba="0.88 0.86 0.82 1" contype="0" conaffinity="0"/>
    <geom name="wall_left"  type="box" size="0.05 2.5 1.5"
          pos="-2.5 0 1.2" rgba="0.85 0.83 0.79 1" contype="0" conaffinity="0"/>
    <geom name="wall_right" type="box" size="0.05 2.5 1.5"
          pos=" 2.5 0 1.2" rgba="0.85 0.83 0.79 1" contype="0" conaffinity="0"/>
    <geom name="ceiling"    type="box" size="2.5 2.5 0.05"
          pos="0 0 2.65"   rgba="0.95 0.94 0.92 1" contype="0" conaffinity="0"/>

    <!-- Desk surface -->
    <body name="desk" pos="0 0.35 {DESK_H - DESK_T/2:.4f}">
      <geom type="box" size="0.30 0.28 {DESK_T/2:.4f}"
            material="desk_mat" friction="0.7 0.005 0.0001"
            contype="1" conaffinity="1"/>
    </body>
    <!-- Desk legs -->
    <geom type="box" size="0.02 0.02 {DESK_H/2:.3f}"
          pos="-0.27 0.08 {DESK_H/2:.3f}" rgba="0.12 0.11 0.10 1"
          contype="0" conaffinity="0"/>
    <geom type="box" size="0.02 0.02 {DESK_H/2:.3f}"
          pos=" 0.27 0.08 {DESK_H/2:.3f}" rgba="0.12 0.11 0.10 1"
          contype="0" conaffinity="0"/>
    <geom type="box" size="0.02 0.02 {DESK_H/2:.3f}"
          pos="-0.27 0.62 {DESK_H/2:.3f}" rgba="0.12 0.11 0.10 1"
          contype="0" conaffinity="0"/>
    <geom type="box" size="0.02 0.02 {DESK_H/2:.3f}"
          pos=" 0.27 0.62 {DESK_H/2:.3f}" rgba="0.12 0.11 0.10 1"
          contype="0" conaffinity="0"/>
    <!-- LED strip glow (thin bright strip at back of desk) -->
    <geom type="box" size="0.28 0.004 0.003"
          pos="0 0.63 {DESK_H + 0.012:.4f}" rgba="1.0 0.92 0.70 1"
          contype="0" conaffinity="0"/>

    <!-- Monitor -->
    <geom type="box" size="0.22 0.02 0.14"
          pos="0 0.60 {DESK_H + 0.18:.3f}" rgba="0.06 0.06 0.06 1"
          contype="0" conaffinity="0"/>
    <geom type="box" size="0.20 0.008 0.125"
          pos="0 0.595 {DESK_H + 0.18:.3f}" rgba="0.05 0.15 0.30 1"
          contype="0" conaffinity="0"/>
    <!-- Monitor stand -->
    <geom type="cylinder" size="0.015 0.07"
          pos="0 0.58 {DESK_H + 0.085:.3f}" rgba="0.1 0.1 0.1 1"
          contype="0" conaffinity="0"/>
    <geom type="box" size="0.08 0.07 0.008"
          pos="0 0.58 {DESK_H + 0.01:.3f}" rgba="0.1 0.1 0.1 1"
          contype="0" conaffinity="0"/>

    <!-- Keyboard -->
    <geom type="box" size="0.16 0.055 0.008"
          pos="0 0.38 {DESK_H + 0.012:.3f}" rgba="0.07 0.07 0.07 1"
          contype="0" conaffinity="0"/>
    <!-- RGB strip on keyboard -->
    <geom type="box" size="0.16 0.003 0.003"
          pos="0 0.325 {DESK_H + 0.012:.3f}" rgba="0.9 0.1 0.5 1"
          contype="0" conaffinity="0"/>

    <!-- Mousepad -->
    <geom type="box" size="0.12 0.09 0.003"
          pos="-0.19 0.38 {DESK_H + 0.007:.3f}" rgba="0.05 0.05 0.05 1"
          contype="0" conaffinity="0"/>

    <!-- Speaker (left) -->
    <geom type="cylinder" size="0.03 0.055"
          pos="-0.29 0.50 {DESK_H + 0.06:.3f}"
          euler="0 1.5708 0" rgba="0.08 0.08 0.08 1"
          contype="0" conaffinity="0"/>

    <!-- Wall shelves -->
    <geom type="box" size="0.40 0.09 0.015"
          pos="0 0.72 1.55" rgba="0.16 0.14 0.13 1"
          contype="0" conaffinity="0"/>
    <!-- Books on shelf -->
    <geom type="box" size="0.025 0.07 0.06" pos="-0.32 0.66 1.60" rgba="0.7 0.1 0.1 1" contype="0" conaffinity="0"/>
    <geom type="box" size="0.025 0.07 0.07" pos="-0.27 0.66 1.61" rgba="0.1 0.3 0.7 1" contype="0" conaffinity="0"/>
    <geom type="box" size="0.025 0.07 0.05" pos="-0.22 0.66 1.59" rgba="0.2 0.6 0.2 1" contype="0" conaffinity="0"/>
    <geom type="box" size="0.025 0.07 0.08" pos="-0.17 0.66 1.62" rgba="0.8 0.6 0.1 1" contype="0" conaffinity="0"/>
    <!-- Shelf LED -->
    <geom type="box" size="0.38 0.003 0.002" pos="0 0.64 1.537"
          rgba="1.0 0.90 0.65 1" contype="0" conaffinity="0"/>

    <!-- Chair (behind camera, just base visible) -->
    <geom type="box" size="0.22 0.22 0.03"
          pos="0 -0.5 0.50" rgba="0.25 0.24 0.23 1" contype="0" conaffinity="0"/>
    <geom type="cylinder" size="0.03 0.22"
          pos="0 -0.5 0.28" rgba="0.15 0.15 0.15 1" contype="0" conaffinity="0"/>

    <!-- Arm attachment frames (arms attached via mjSpec below) -->
    <body name="left_arm_mount"
          pos="-0.12 0.35 {ARM_Z:.4f}"
          euler="0 0 1.5708">
      <frame name="left_attach"/>
    </body>
    <body name="right_arm_mount"
          pos=" 0.12 0.35 {ARM_Z:.4f}"
          euler="0 0 -1.5708">
      <frame name="right_attach"/>
    </body>

  </worldbody>
</mujoco>
"""

def build_scene_with_objects(obj_list):
    """Rebuild the scene spec with object boxes on the desk."""
    spec = mujoco.MjSpec.from_string(BASE_XML)

    # Attach two SO-101 arms via frames (load separately — same spec can't attach twice)
    left_arm_spec  = mujoco.MjSpec.from_file(str(SO101))
    right_arm_spec = mujoco.MjSpec.from_file(str(SO101))
    left_frame  = spec.body("left_arm_mount").first_frame()
    right_frame = spec.body("right_arm_mount").first_frame()
    spec.attach(left_arm_spec,  prefix="left_",  frame=left_frame)
    spec.attach(right_arm_spec, prefix="right_", frame=right_frame)

    # Add object geoms (freejoint bodies)
    COLORS = {
        "mug_ceramic":    [0.75, 0.32, 0.18],
        "apple":          [0.85, 0.12, 0.10],
        "banana":         [0.95, 0.85, 0.08],
        "wooden_block":   [0.65, 0.42, 0.22],
        "notebook":       [0.20, 0.40, 0.70],
        "pen_holder":     [0.30, 0.30, 0.30],
        "scissors":       [0.50, 0.50, 0.55],
        "bowl":           [0.90, 0.88, 0.84],
        "foam_brick":     [0.15, 0.55, 0.85],
        "rubiks_cube":    [0.95, 0.55, 0.05],
        "wood_block":     [0.62, 0.40, 0.20],
        "orange":         [0.95, 0.50, 0.05],
        "lemon":          [0.95, 0.90, 0.10],
        "fork":           [0.75, 0.75, 0.78],
        "spoon":          [0.75, 0.75, 0.78],
        "large_marker":   [0.10, 0.10, 0.70],
        "cracker_box":    [0.85, 0.72, 0.40],
        "sugar_box":      [0.95, 0.95, 0.95],
        "mustard_bottle": [0.95, 0.82, 0.05],
        "soup_can":       [0.60, 0.18, 0.18],
    }
    SHAPES = {
        "mug_ceramic": "cylinder", "apple": "sphere", "orange": "sphere",
        "lemon": "sphere", "bowl": "cylinder", "mustard_bottle": "cylinder",
        "soup_can": "cylinder", "large_marker": "cylinder", "pen_holder": "cylinder",
    }
    SIZES = {
        "mug_ceramic":    [0.035, 0.035, 0.045],
        "apple":          [0.038, 0.038, 0.038],
        "banana":         [0.12,  0.025, 0.025],
        "notebook":       [0.09,  0.065, 0.008],
        "pen_holder":     [0.02,  0.02,  0.06],
        "scissors":       [0.09,  0.016, 0.008],
        "bowl":           [0.06,  0.06,  0.03],
        "foam_brick":     [0.055, 0.04,  0.03],
        "rubiks_cube":    [0.028, 0.028, 0.028],
        "orange":         [0.038, 0.038, 0.038],
        "lemon":          [0.032, 0.028, 0.028],
        "fork":           [0.10,  0.012, 0.005],
        "spoon":          [0.09,  0.015, 0.005],
        "large_marker":   [0.015, 0.015, 0.065],
        "cracker_box":    [0.075, 0.045, 0.055],
        "sugar_box":      [0.045, 0.035, 0.085],
        "mustard_bottle": [0.025, 0.025, 0.085],
        "soup_can":       [0.032, 0.032, 0.050],
        "wooden_block":   [0.04,  0.04,  0.04],
        "wood_block":     [0.04,  0.04,  0.04],
    }

    for obj in obj_list:
        oid  = obj["id"]
        pos  = obj["position"]
        col  = COLORS.get(oid, [0.6, 0.4, 0.2])
        sz   = SIZES.get(oid, [0.03, 0.03, 0.04])
        shp  = SHAPES.get(oid, "box")

        body = spec.worldbody.add_body()
        body.name = f"obj_{oid}"
        body.pos  = [pos[0], pos[1] + 0.35, pos[2]]  # offset for desk Y

        fj = body.add_freejoint()
        fj.name = f"fj_{oid}"

        g = body.add_geom()
        if shp == "sphere":
            g.type = mujoco.mjtGeom.mjGEOM_SPHERE
            g.size = [sz[0], 0, 0]
        elif shp == "cylinder":
            g.type = mujoco.mjtGeom.mjGEOM_CYLINDER
            g.size = [sz[0], sz[2]/2, 0]
        else:
            g.type = mujoco.mjtGeom.mjGEOM_BOX
            g.size = [sz[0]/2, sz[1]/2, sz[2]/2]
        g.rgba = [col[0], col[1], col[2], 1.0]
        g.mass = 0.15
        g.friction = [0.7, 0.005, 0.0001]
        g.contype  = 1
        g.conaffinity = 1

    model = spec.compile()
    data  = mujoco.MjData(model)
    return model, data


def fetch_state():
    try:
        with urllib.request.urlopen("http://localhost:8000/state", timeout=0.5) as r:
            return json.loads(r.read())
    except Exception:
        return None

def fetch_objects():
    try:
        # reset to get object list
        req = urllib.request.Request(
            "http://localhost:8000/reset",
            data=json.dumps({"task": "easy", "seed": 42}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=2) as r:
            return json.loads(r.read())["objects"]
    except Exception:
        return []

# ── Main ─────────────────────────────────────────────────────────────────────
print("Fetching scene from server...")
objects = fetch_objects()
if not objects:
    # Default 5 objects if server not running
    objects = [
        {"id": "mug_ceramic",  "position": [ 0.05,  0.05, 0.79]},
        {"id": "apple",        "position": [-0.08,  0.12, 0.79]},
        {"id": "banana",       "position": [ 0.10, -0.10, 0.78]},
        {"id": "wooden_block", "position": [-0.05, -0.08, 0.78]},
        {"id": "notebook",     "position": [ 0.00,  0.18, 0.755]},
    ]

model, data = build_scene_with_objects(objects)
mujoco.mj_resetData(model, data)
for _ in range(400):
    mujoco.mj_step(model, data)

print(f"Scene: {model.nbody} bodies | {model.njnt} joints | {model.ngeom} geoms")
print("Controls: left-drag=rotate  scroll=zoom  right-drag=pan")

with mujoco.viewer.launch_passive(model, data) as v:
    v.cam.lookat  = [0.0, 0.35, 0.85]
    v.cam.distance  = 1.8
    v.cam.elevation = -22
    v.cam.azimuth   = 200

    step = 0
    while v.is_running():
        mujoco.mj_step(model, data)
        if step % 120 == 0:
            v.sync()
        step += 1
