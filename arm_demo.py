"""
DeskBot — Full arm capability demo.

Both real SO-101 arms at front-left and front-right corners of desk.
Real Gazebo Fuel meshes: desk, keyboard, cup (physics freejoint), chair.
Sweeps all 6 DOF: shoulder pan, shoulder lift, elbow flex,
wrist flex, wrist roll, gripper. Cosine-interpolated keyframes.

Run: python3 arm_demo.py
"""
import math, pathlib
import mujoco
import mujoco.viewer
import numpy as np

ROOT  = pathlib.Path(__file__).parent
SO101 = ROOT / "assets/robotstudio_so101/so101.xml"

DESK_H = 0.74
DESK_T = 0.02
ARM_Z  = DESK_H + 0.005

# Absolute paths for mesh assets
DESK_OBJ   = ROOT / "assets/models/desk/meshes/desk.obj"
DESK_TEX   = ROOT / "assets/models/desk/meshes/Desk_Diffuse.png"
KB_OBJ     = ROOT / "assets/models/keyboard/meshes/model.obj"
KB_TEX     = ROOT / "assets/models/keyboard/materials/textures/texture.png"
CUP_OBJ    = ROOT / "assets/models/cup/mesh.obj"
CUP_TEX    = ROOT / "assets/models/cup/touxiang.png"
CHAIR_OBJ  = ROOT / "assets/models/chair/meshes/OfficeChairGrey.obj"
CHAIR_TEX  = ROOT / "assets/models/chair/meshes/OfficeChairGrey.png"

BASE_XML = f"""
<mujoco model="arm_demo">
  <option gravity="0 0 -9.81" timestep="0.004"/>
  <compiler meshdir="{ROOT}/assets/robotstudio_so101/assets" autolimits="true"/>
  <asset>
    <texture name="ft" type="2d" builtin="checker"
             rgb1="0.52 0.44 0.33" rgb2="0.46 0.39 0.29" width="512" height="512"/>
    <material name="fm" texture="ft" texrepeat="8 8" reflectance="0.04"/>

    <!-- Real desk -->
    <mesh name="desk_mesh" file="{DESK_OBJ}" scale="1.3 1.1 1.35"/>
    <texture name="desk_tex" type="2d" file="{DESK_TEX}"/>
    <material name="desk_mat" texture="desk_tex" reflectance="0.12" shininess="0.3"/>

    <!-- Real keyboard -->
    <mesh name="kb_mesh" file="{KB_OBJ}"/>
    <texture name="kb_tex" type="2d" file="{KB_TEX}"/>
    <material name="kb_mat" texture="kb_tex" reflectance="0.05"/>

    <!-- Real cup (scaled down from ~2.24 m bounding box) -->
    <mesh name="cup_mesh" file="{CUP_OBJ}" scale="0.05 0.05 0.05"/>
    <texture name="cup_tex" type="2d" file="{CUP_TEX}"/>
    <material name="cup_mat" texture="cup_tex" reflectance="0.15"/>

    <!-- Real office chair -->
    <mesh name="chair_mesh" file="{CHAIR_OBJ}"/>
    <texture name="chair_tex" type="2d" file="{CHAIR_TEX}"/>
    <material name="chair_mat" texture="chair_tex" reflectance="0.08"/>
  </asset>
  <default>
    <geom condim="4" solimp="0.9 0.95 0.001" solref="0.02 1"/>
  </default>
  <worldbody>
    <light name="key"  pos="-1.0 -2.0 3.5" dir="0.35 0.7 -1.0"
           diffuse="1.0 0.95 0.85" specular="0.4 0.4 0.35" castshadow="true"/>
    <light name="fill" pos=" 2.0  1.5 2.5" dir="-0.5 -0.4 -1.0"
           diffuse="0.30 0.36 0.52" specular="0.0 0.0 0.0" castshadow="false"/>
    <light name="rim"  pos=" 0.0  2.5 2.0" dir="0.0 -0.7 -0.4"
           diffuse="0.55 0.50 0.45" specular="0.1 0.1 0.1" castshadow="false"/>

    <!-- Floor -->
    <geom type="plane" size="4 4 0.1" material="fm"
          friction="0.8 0.005 0.0001" contype="1" conaffinity="1"/>

    <!-- Walls -->
    <geom type="box" size="4.0 0.06 2.0" pos="0  2.1 1.5" rgba="0.87 0.85 0.81 1" contype="0" conaffinity="0"/>
    <geom type="box" size="0.06 4.0 2.0" pos="-4.1 0 1.5" rgba="0.84 0.82 0.78 1" contype="0" conaffinity="0"/>
    <geom type="box" size="0.06 4.0 2.0" pos=" 4.1 0 1.5" rgba="0.84 0.82 0.78 1" contype="0" conaffinity="0"/>
    <geom type="box" size="4.1 4.1 0.05" pos="0 0 3.1"    rgba="0.94 0.93 0.91 1" contype="0" conaffinity="0"/>

    <!-- Real desk mesh — legs touch floor, surface at 0.74m -->
    <body name="desk" pos="0 0.35 0.0" euler="0 0 1.5708">
      <geom type="mesh" mesh="desk_mesh" material="desk_mat"
            friction="0.7 0.005 0.0001" contype="1" conaffinity="1"/>
    </body>

    <!-- Real keyboard — centred on desk surface -->
    <body name="keyboard" pos="0 0.38 {DESK_H + 0.012:.4f}">
      <geom type="mesh" mesh="kb_mesh" material="kb_mat"
            contype="1" conaffinity="1"/>
    </body>

    <!-- Real cup — freejoint so it has physics and can be knocked over -->
    <body name="cup" pos="-0.18 0.42 {DESK_H + 0.06:.4f}">
      <freejoint name="cup_joint"/>
      <geom type="mesh" mesh="cup_mesh" material="cup_mat"
            mass="0.25" friction="0.6 0.005 0.0001"
            contype="1" conaffinity="1"/>
    </body>

    <!-- Real office chair — behind desk -->
    <body name="chair" pos="0 -0.55 0.0">
      <geom type="mesh" mesh="chair_mesh" material="chair_mat"
            contype="0" conaffinity="0"/>
    </body>

    <!-- Shelf above monitor area -->
    <geom type="box" size="0.42 0.085 0.014" pos="0 0.73 1.52" rgba="0.14 0.12 0.11 1" contype="0" conaffinity="0"/>
    <geom type="box" size="0.38 0.003 0.002" pos="0 0.645 1.506" rgba="1.0 0.88 0.62 1" contype="0" conaffinity="0"/>
    <!-- Books on shelf -->
    <geom type="box" size="0.022 0.065 0.065" pos="-0.35 0.668 1.571" rgba="0.72 0.08 0.08 1" contype="0" conaffinity="0"/>
    <geom type="box" size="0.022 0.065 0.075" pos="-0.30 0.668 1.581" rgba="0.08 0.28 0.72 1" contype="0" conaffinity="0"/>
    <geom type="box" size="0.022 0.065 0.055" pos="-0.25 0.668 1.561" rgba="0.15 0.62 0.20 1" contype="0" conaffinity="0"/>
    <geom type="box" size="0.022 0.065 0.082" pos="-0.20 0.668 1.588" rgba="0.82 0.62 0.08 1" contype="0" conaffinity="0"/>
    <!-- Small plant on shelf -->
    <geom type="cylinder" size="0.022 0.030" pos="0.30 0.668 1.546" rgba="0.52 0.32 0.18 1" contype="0" conaffinity="0"/>
    <geom type="sphere"   size="0.030"       pos="0.30 0.668 1.600" rgba="0.15 0.55 0.12 1" contype="0" conaffinity="0"/>

    <!-- Arm mounts: front-left and front-right corners, angled inward 120 deg -->
    <body name="left_mount"  pos="-0.26 0.09 {ARM_Z:.4f}" euler="0 0  2.094"><frame name="lf"/></body>
    <body name="right_mount" pos=" 0.26 0.09 {ARM_Z:.4f}" euler="0 0 -2.094"><frame name="rf"/></body>
  </worldbody>
</mujoco>"""

spec       = mujoco.MjSpec.from_string(BASE_XML)
left_spec  = mujoco.MjSpec.from_file(str(SO101))
right_spec = mujoco.MjSpec.from_file(str(SO101))
spec.attach(left_spec,  prefix="L_", frame=spec.body("left_mount").first_frame())
spec.attach(right_spec, prefix="R_", frame=spec.body("right_mount").first_frame())
model = spec.compile()
data  = mujoco.MjData(model)
mujoco.mj_resetData(model, data)

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex",
          "wrist_flex", "wrist_roll", "gripper"]

def set_ctrl(prefix, joint, val):
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{prefix}{joint}")
    if aid >= 0:
        lo, hi = model.actuator_ctrlrange[aid]
        data.ctrl[aid] = float(np.clip(val, lo, hi))

def lerp(a, b, t):
    return a + (b - a) * t

# Keyframes — each is {joint: angle}
# Left arm values; right arm mirrors pan and roll (bilateral symmetry)
KEYFRAMES = [
    # 0: home rest
    dict(shoulder_pan=0.0, shoulder_lift=-0.8, elbow_flex=1.2,
         wrist_flex=-0.4, wrist_roll=0.0, gripper=0.0),
    # 1: full pan left — max shoulder_pan
    dict(shoulder_pan=1.80, shoulder_lift=-0.5, elbow_flex=0.8,
         wrist_flex=-0.3, wrist_roll=0.0, gripper=0.0),
    # 2: reach high — shoulder lift max
    dict(shoulder_pan=0.0, shoulder_lift=1.60, elbow_flex=-1.4,
         wrist_flex=1.2, wrist_roll=0.0, gripper=1.6),
    # 3: full elbow flex + wrist flex max
    dict(shoulder_pan=0.3, shoulder_lift=-0.4, elbow_flex=1.60,
         wrist_flex=-1.55, wrist_roll=0.0, gripper=0.0),
    # 4: wrist roll clockwise max
    dict(shoulder_pan=0.0, shoulder_lift=0.2, elbow_flex=0.5,
         wrist_flex=0.0, wrist_roll=2.60, gripper=0.8),
    # 5: wrist roll counter-clockwise max
    dict(shoulder_pan=0.0, shoulder_lift=0.2, elbow_flex=0.5,
         wrist_flex=0.0, wrist_roll=-2.60, gripper=0.8),
    # 6: gripper full open, arms extended forward-up
    dict(shoulder_pan=0.0, shoulder_lift=1.40, elbow_flex=-1.2,
         wrist_flex=0.8, wrist_roll=0.0, gripper=1.70),
    # 7: pan right max
    dict(shoulder_pan=-1.80, shoulder_lift=-0.5, elbow_flex=0.8,
         wrist_flex=-0.3, wrist_roll=0.0, gripper=0.0),
    # 8: back to home
    dict(shoulder_pan=0.0, shoulder_lift=-0.8, elbow_flex=1.2,
         wrist_flex=-0.4, wrist_roll=0.0, gripper=0.0),
]

MIRROR = {"shoulder_pan", "wrist_roll"}   # flip sign for right arm

# Settle to home pose
for j, v in KEYFRAMES[0].items():
    set_ctrl("L_", j, v)
    set_ctrl("R_", j, -v if j in MIRROR else v)
for _ in range(600):
    mujoco.mj_step(model, data)

print(f"SO-101 x2  |  {model.nbody} bodies  {model.njnt} joints  {model.ngeom} geoms")
print("Drag=rotate  Scroll=zoom  F=fullscreen  Esc=quit")

STEPS_PER_KEY = 480   # ~1.9s per transition at 240Hz

with mujoco.viewer.launch_passive(model, data) as v:
    v.cam.lookat   = [0.0, 0.30, 0.90]
    v.cam.distance = 1.50
    v.cam.elevation = -20
    v.cam.azimuth   = 185

    step = 0
    while v.is_running():
        ki = (step // STEPS_PER_KEY) % (len(KEYFRAMES) - 1)
        t  = (step % STEPS_PER_KEY) / STEPS_PER_KEY
        t  = (1 - math.cos(t * math.pi)) / 2   # cosine ease

        kf0, kf1 = KEYFRAMES[ki], KEYFRAMES[ki + 1]
        for j in JOINTS:
            val = lerp(kf0[j], kf1[j], t)
            set_ctrl("L_", j,  val)
            set_ctrl("R_", j, -val if j in MIRROR else val)

        mujoco.mj_step(model, data)
        if step % 4 == 0:
            v.sync()
        step += 1
