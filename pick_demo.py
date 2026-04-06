"""
DeskBot — Dual-arm keyboard pick demo.

Both SO-101 arms cooperatively pick up the keyboard:
  1. Open grippers, approach above keyboard ends
  2. Lower onto keyboard
  3. Close grippers
  4. Lift keyboard together
  5. Hold in air → lower back → release → retract

Carry trick: during lift/hold/lower phases the keyboard freejoint qpos
is pinned to the midpoint of both gripper sites BEFORE each mj_step,
and its velocity is zeroed so physics can't fight the teleport.

Run: python3 pick_demo.py
"""
import pathlib
import mujoco
import mujoco.viewer
import numpy as np

ROOT  = pathlib.Path(__file__).parent
SO101 = ROOT / "assets/robotstudio_so101/so101.xml"

DESK_H = 0.74
ARM_Z  = DESK_H + 0.005

KB_X, KB_Y, KB_Z = 0.0, 0.38, DESK_H + 0.018  # keyboard centre on desk surface
KB_HALF_SPAN = 0.16   # how far left/right each gripper grabs

BASE_XML = f"""
<mujoco model="pick_demo">
  <option gravity="0 0 -9.81" timestep="0.004" integrator="implicitfast"/>
  <compiler meshdir="{ROOT}/assets/robotstudio_so101/assets" autolimits="true"/>
  <asset>
    <texture name="ft" type="2d" builtin="checker"
             rgb1="0.52 0.44 0.33" rgb2="0.46 0.39 0.29" width="512" height="512"/>
    <material name="fm" texture="ft" texrepeat="8 8" reflectance="0.04"/>

    <mesh name="desk_mesh"
          file="{ROOT}/assets/models/desk/meshes/desk.obj" scale="1.3 1.1 1.35"/>
    <texture name="desk_tex" type="2d"
             file="{ROOT}/assets/models/desk/meshes/Desk_Diffuse.png"/>
    <material name="desk_mat" texture="desk_tex" reflectance="0.12" shininess="0.3"/>

    <mesh name="kb_mesh"
          file="{ROOT}/assets/models/keyboard/meshes/model.obj"/>
    <texture name="kb_tex" type="2d"
             file="{ROOT}/assets/models/keyboard/materials/textures/texture.png"/>
    <material name="kb_mat" texture="kb_tex" reflectance="0.05"/>

    <mesh name="chair_mesh"
          file="{ROOT}/assets/models/chair/meshes/OfficeChairGrey.obj"/>
    <texture name="chair_tex" type="2d"
             file="{ROOT}/assets/models/chair/meshes/OfficeChairGrey.png"/>
    <material name="chair_mat" texture="chair_tex"/>
  </asset>
  <default>
    <geom condim="4" solimp="0.9 0.95 0.001" solref="0.02 1"/>
  </default>
  <worldbody>
    <light name="key"  pos="-1 -2 3.5" dir="0.35 0.7 -1"
           diffuse="1.0 0.95 0.85" specular="0.4 0.4 0.35" castshadow="true"/>
    <light name="fill" pos="2 1.5 2.5" dir="-0.5 -0.4 -1"
           diffuse="0.3 0.36 0.52" specular="0 0 0" castshadow="false"/>
    <light name="rim"  pos="0 2.5 2"   dir="0 -0.7 -0.4"
           diffuse="0.55 0.5 0.45"  specular="0.1 0.1 0.1" castshadow="false"/>

    <geom type="plane" size="4 4 0.1" material="fm"
          friction="0.8 0.005 0.0001" contype="1" conaffinity="1"/>
    <geom type="box" size="4 0.06 2"    pos="0 2.1 1.5"  rgba="0.87 0.85 0.81 1" contype="0" conaffinity="0"/>
    <geom type="box" size="0.06 4 2"    pos="-4.1 0 1.5" rgba="0.84 0.82 0.78 1" contype="0" conaffinity="0"/>
    <geom type="box" size="0.06 4 2"    pos="4.1 0 1.5"  rgba="0.84 0.82 0.78 1" contype="0" conaffinity="0"/>
    <geom type="box" size="4.1 4.1 0.05" pos="0 0 3.1"   rgba="0.94 0.93 0.91 1" contype="0" conaffinity="0"/>

    <body name="desk" pos="0 0.35 0" euler="0 0 1.5708">
      <geom type="mesh" mesh="desk_mesh" material="desk_mat"
            friction="0.7 0.005 0.0001" contype="1" conaffinity="1"/>
    </body>

    <!-- Keyboard — freejoint so it can be picked up; contacts OFF during carry -->
    <body name="keyboard" pos="{KB_X} {KB_Y} {KB_Z:.4f}">
      <freejoint name="kb_joint"/>
      <geom name="kb_geom" type="mesh" mesh="kb_mesh" material="kb_mat"
            mass="0.55" friction="0.8 0.005 0.0001"
            contype="1" conaffinity="1"/>
    </body>

    <body name="chair" pos="0 -0.55 0">
      <geom type="mesh" mesh="chair_mesh" material="chair_mat"
            contype="0" conaffinity="0"/>
    </body>

    <body name="left_mount"  pos="-0.26 0.09 {ARM_Z:.4f}" euler="0 0  2.094">
      <frame name="lf"/>
    </body>
    <body name="right_mount" pos=" 0.26 0.09 {ARM_Z:.4f}" euler="0 0 -2.094">
      <frame name="rf"/>
    </body>
  </worldbody>
</mujoco>"""

# ── Compile ───────────────────────────────────────────────────────────────────
spec       = mujoco.MjSpec.from_string(BASE_XML)
left_spec  = mujoco.MjSpec.from_file(str(SO101))
right_spec = mujoco.MjSpec.from_file(str(SO101))
spec.attach(left_spec,  prefix="L_", frame=spec.body("left_mount").first_frame())
spec.attach(right_spec, prefix="R_", frame=spec.body("right_mount").first_frame())
model = spec.compile()
data  = mujoco.MjData(model)
mujoco.mj_resetData(model, data)

# ── Cache IDs ─────────────────────────────────────────────────────────────────
JOINTS_SO101 = ["shoulder_pan", "shoulder_lift", "elbow_flex",
                "wrist_flex", "wrist_roll", "gripper"]

def _aid(prefix, joint):
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{prefix}{joint}")
    assert aid >= 0, f"actuator {prefix}{joint} not found"
    return aid

def _sid(name):
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    assert sid >= 0, f"site {name} not found"
    return sid

L_ACTS = {j: _aid("L_", j) for j in JOINTS_SO101}
R_ACTS = {j: _aid("R_", j) for j in JOINTS_SO101}
L_GS   = _sid("L_gripperframe")
R_GS   = _sid("R_gripperframe")

KB_JID  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "kb_joint")
KB_QADR = model.jnt_qposadr[KB_JID]
KB_DADR = model.jnt_dofadr[KB_JID]

# Keyboard geom — we'll disable contacts during carry
KB_GEOM = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "kb_geom")
_KB_CONTYPE_ORIG  = int(model.geom_contype[KB_GEOM])
_KB_CONAFF_ORIG   = int(model.geom_conaffinity[KB_GEOM])

def kb_contacts(enabled: bool):
    model.geom_contype[KB_GEOM]      = _KB_CONTYPE_ORIG  if enabled else 0
    model.geom_conaffinity[KB_GEOM]  = _KB_CONAFF_ORIG   if enabled else 0

def set_ctrl(acts, joint, val):
    aid = acts[joint]
    lo, hi = model.actuator_ctrlrange[aid]
    data.ctrl[aid] = float(np.clip(val, lo, hi))

def pin_kb(pos):
    """Teleport keyboard to pos and zero its velocity (kinematic pin)."""
    data.qpos[KB_QADR]     = float(pos[0])
    data.qpos[KB_QADR + 1] = float(pos[1])
    data.qpos[KB_QADR + 2] = float(pos[2])
    data.qpos[KB_QADR + 3] = 1.0  # identity quaternion
    data.qpos[KB_QADR + 4] = 0.0
    data.qpos[KB_QADR + 5] = 0.0
    data.qpos[KB_QADR + 6] = 0.0
    for i in range(6):
        data.qvel[KB_DADR + i] = 0.0

# ── DLS IK ───────────────────────────────────────────────────────────────────
def ik_to(target, acts, site, jprefix):
    joints5 = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
    jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{jprefix}{j}")
            for j in joints5]
    tgt = np.array(target, dtype=float)

    for _ in range(20):
        mujoco.mj_fwdPosition(model, data)
        ee  = data.site_xpos[site].copy()
        err = tgt - ee
        if np.linalg.norm(err) < 0.004:
            break
        jacp = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacp, None, site)
        J   = np.column_stack([jacp[:, model.jnt_dofadr[jid]] for jid in jids])
        dq  = J.T @ np.linalg.solve(J @ J.T + 0.01 * np.eye(3), err)
        for i, jid in enumerate(jids):
            adr = model.jnt_qposadr[jid]
            lo, hi = model.jnt_range[jid]
            data.qpos[adr] = float(np.clip(data.qpos[adr] + 0.5 * dq[i], lo, hi))

    for jn, jid in zip(joints5, jids):
        aid = acts[jn]
        adr = model.jnt_qposadr[jid]
        lo, hi = model.actuator_ctrlrange[aid]
        data.ctrl[aid] = float(np.clip(data.qpos[adr], lo, hi))

# ── Home pose ─────────────────────────────────────────────────────────────────
HOME = dict(shoulder_pan=0.0, shoulder_lift=-0.8, elbow_flex=1.2,
            wrist_flex=-0.4, wrist_roll=0.0, gripper=0.0)
MIRROR = {"shoulder_pan", "wrist_roll"}

def go_home():
    for j, v in HOME.items():
        set_ctrl(L_ACTS, j,  v)
        set_ctrl(R_ACTS, j, -v if j in MIRROR else v)

go_home()
for _ in range(600):
    mujoco.mj_step(model, data)
print(f"Ready — {model.nbody} bodies  {model.njnt} joints  {model.ngeom} geoms")

# ── Phase definitions ─────────────────────────────────────────────────────────
GRIP_Z   = KB_Z + 0.004   # gripper site target Z when gripping
ABOVE_Z  = KB_Z + 0.10    # approach height
LIFT_Z   = KB_Z + 0.22    # how high to lift

L_GX = KB_X - KB_HALF_SPAN   # left gripper X target
R_GX = KB_X + KB_HALF_SPAN   # right gripper X target

phases = [
    # (label, steps, carry, action)
    ("open_grippers",  50, False,
        lambda: [set_ctrl(L_ACTS, "gripper", 1.6),
                 set_ctrl(R_ACTS, "gripper", 1.6)]),
    ("approach_above", 180, False,
        lambda: [ik_to([L_GX, KB_Y, ABOVE_Z], L_ACTS, L_GS, "L_"),
                 ik_to([R_GX, KB_Y, ABOVE_Z], R_ACTS, R_GS, "R_")]),
    ("lower_to_grip",  180, False,
        lambda: [ik_to([L_GX, KB_Y, GRIP_Z], L_ACTS, L_GS, "L_"),
                 ik_to([R_GX, KB_Y, GRIP_Z], R_ACTS, R_GS, "R_")]),
    ("close_grippers", 80,  False,
        lambda: [set_ctrl(L_ACTS, "gripper", -0.15),
                 set_ctrl(R_ACTS, "gripper", -0.15)]),
    ("lift",           220, True,
        lambda: [kb_contacts(False),
                 ik_to([L_GX, KB_Y, LIFT_Z], L_ACTS, L_GS, "L_"),
                 ik_to([R_GX, KB_Y, LIFT_Z], R_ACTS, R_GS, "R_")]),
    ("hold_in_air",    200, True,
        lambda: None),
    ("lower_to_desk",  220, True,
        lambda: [ik_to([L_GX, KB_Y, GRIP_Z], L_ACTS, L_GS, "L_"),
                 ik_to([R_GX, KB_Y, GRIP_Z], R_ACTS, R_GS, "R_")]),
    ("release",        80,  False,
        lambda: [kb_contacts(True),
                 set_ctrl(L_ACTS, "gripper", 1.6),
                 set_ctrl(R_ACTS, "gripper", 1.6)]),
    ("retract_home",   300, False,
        lambda: go_home()),
    ("pause",          400, False,
        lambda: None),
]

# ── Viewer loop ───────────────────────────────────────────────────────────────
with mujoco.viewer.launch_passive(model, data) as v:
    v.cam.lookat    = [0.0, 0.38, 0.90]
    v.cam.distance  = 1.35
    v.cam.elevation = -16
    v.cam.azimuth   = 180

    pi          = 0
    ps          = 0
    label, dur, carry, fn = phases[0]
    if fn: fn()
    print(f"  → {label}")

    step = 0
    while v.is_running():
        if carry:
            mujoco.mj_fwdPosition(model, data)
            mid = (data.site_xpos[L_GS] + data.site_xpos[R_GS]) / 2.0
            pin_kb(mid)

        mujoco.mj_step(model, data)

        if step % 4 == 0:
            v.sync()

        ps   += 1
        step += 1
        if ps >= dur:
            pi  = (pi + 1) % len(phases)
            ps  = 0
            label, dur, carry, fn = phases[pi]
            print(f"  → {label}")
            if fn: fn()
