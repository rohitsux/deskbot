"""
DeskBot visual test — real SO-101 arms, real desk, YCB objects, live pick & place.

Run:
    python3 visual_test.py
    python3 visual_test.py --task medium
    python3 visual_test.py --task hard
"""
import argparse, math, pathlib, time
import mujoco, mujoco.viewer
import numpy as np

ROOT   = pathlib.Path(__file__).parent
SO101  = ROOT / "assets/robotstudio_so101/so101.xml"
YCB    = ROOT / "assets/ycb/ycb"
DESK_H = 0.7448          # measured: probe settles at z=0.7498 (r=0.005) → surface=0.7448
ARM_Z  = DESK_H + 0.005

p = argparse.ArgumentParser()
p.add_argument("--task",     default="easy", choices=["easy","medium","hard"])
p.add_argument("--episodes", type=int, default=999)
p.add_argument("--seed",     type=int, default=42)
args = p.parse_args()

# ── Task objects ──────────────────────────────────────────────────────────────
TASK_OBJECTS = {
    "easy":   ["mug","notebook","pen_holder","block","scissors"],
    "medium": ["mug","notebook","pen_holder","block","scissors","apple","banana","bowl"],
    "hard":   ["mug","notebook","pen_holder","block","scissors",
               "apple","banana","bowl","cup","cube","marker","bottle"],
}

# YCB_MAP: oid → (ycb_dirname, z_offset, n_coacd)
# z_offset = -z_min of mesh → placing origin at DESK_H+z_offset puts bottom on desk
YCB_MAP = {
    "mug":    ("mug",    0.0274, 5),
    "apple":  ("apple",  0.0374, 5),
    "banana": ("banana", 0.0174, 4),
    "bowl":   ("bowl",   0.0276, 5),
    "cup":    ("a_cups", 0.0291, 5),
}

COLORS = {
    "notebook":  [0.18,0.36,0.72,1], "pen_holder":[0.22,0.62,0.28,1],
    "block":     [0.85,0.65,0.12,1], "scissors":  [0.60,0.60,0.65,1],
    "cube":      [0.28,0.72,0.72,1], "marker":    [0.82,0.28,0.72,1],
    "bottle":    [0.28,0.72,0.38,1],
}
SIZES = {   # [sx, sy, sz]  — for YCB: sz = z_offset (bottom of mesh to origin)
    "mug":       [0.047, 0.053, 0.0274],
    "notebook":  [0.095, 0.065, 0.007],
    "pen_holder":[0.022, 0.022, 0.060],
    "block":     [0.030, 0.030, 0.030],
    "scissors":  [0.012, 0.065, 0.008],
    "apple":     [0.038, 0.037, 0.0374],
    "banana":    [0.040, 0.101, 0.0174],
    "bowl":      [0.082, 0.081, 0.0276],
    "cup":       [0.029, 0.032, 0.0291],
    "cube":      [0.030, 0.030, 0.030],
    "marker":    [0.010, 0.010, 0.075],
    "bottle":    [0.025, 0.025, 0.080],
}

_TARGET_GRID = [
    [ 0.10, 0.12],[-0.10, 0.12],[ 0.10, 0.04],
    [-0.10, 0.04],[ 0.00, 0.08],[ 0.15, 0.12],
    [-0.15, 0.12],[ 0.15, 0.04],[-0.15, 0.04],
    [ 0.00, 0.14],[ 0.05, 0.08],[-0.05, 0.08],
]

obj_ids = TASK_OBJECTS[args.task]
targets = {oid: [_TARGET_GRID[i][0], _TARGET_GRID[i][1],
                 DESK_H + SIZES.get(oid, [0,0,0.030])[2] + 0.002]
           for i, oid in enumerate(obj_ids)}

# ── XML asset helpers ─────────────────────────────────────────────────────────
def _ycb_assets():
    xml = ""
    for oid in obj_ids:
        if oid not in YCB_MAP:
            continue
        dirname, _, n_col = YCB_MAP[oid]
        d = YCB / dirname
        xml += f'\n    <mesh name="{oid}_vis"  file="{d}/textured.obj"/>'
        for k in range(n_col):
            xml += f'\n    <mesh name="{oid}_col{k}" file="{d}/textured_coacd_{k}.stl"/>'
        xml += f'\n    <texture name="{oid}_tex" type="2d" file="{d}/texture_map.png"/>'
        xml += f'\n    <material name="{oid}_mat" texture="{oid}_tex" reflectance="0.08"/>'
    return xml

def _obj_bodies():
    xml = ""
    for i, oid in enumerate(obj_ids):
        sx, sy, sz = SIZES.get(oid, [0.025, 0.025, 0.030])
        x = -0.15 + (i % 4) * 0.10
        y = -0.20 + (i // 4) * 0.16
        z = DESK_H + sz + 0.004
        if oid in YCB_MAP:
            _, _, n_col = YCB_MAP[oid]
            xml += f"""
    <body name="obj_{oid}" pos="{x:.3f} {y:.3f} {z:.4f}">
      <freejoint name="fj_{oid}"/>
      <geom name="g_{oid}_vis" type="mesh" mesh="{oid}_vis" material="{oid}_mat"
            contype="0" conaffinity="0"/>"""
            for k in range(n_col):
                mass = "0.12" if k == 0 else "0"
                xml += f"""
      <geom name="g_{oid}_col{k}" type="mesh" mesh="{oid}_col{k}"
            contype="1" conaffinity="1" rgba="0 0 0 0" mass="{mass}"
            friction="0.8 0.005 0.0001"/>"""
            xml += "\n    </body>"
        else:
            c = COLORS.get(oid, [0.6, 0.4, 0.2, 1])
            xml += f"""
    <body name="obj_{oid}" pos="{x:.3f} {y:.3f} {z:.4f}">
      <freejoint name="fj_{oid}"/>
      <geom name="g_{oid}" type="box" size="{sx} {sy} {sz}"
            mass="0.15" rgba="{c[0]} {c[1]} {c[2]} 1"
            friction="0.8 0.005 0.0001" contype="1" conaffinity="1"/>
    </body>"""
    return xml

def _target_markers():
    xml = ""
    for i, oid in enumerate(obj_ids):
        if i >= len(_TARGET_GRID): break
        x, y = _TARGET_GRID[i]
        sx, sy, _ = SIZES.get(oid, [0.025, 0.025, 0.030])
        xml += f"""<geom type="cylinder" size="0.018 0.001"
          pos="{x} {y} {DESK_H+0.001:.4f}"
          rgba="0.05 0.95 0.15 0.90" contype="0" conaffinity="0"/>\n    """
    return xml

XML = f"""
<mujoco model="visual_test">
  <option gravity="0 0 -9.81" timestep="0.004" integrator="implicitfast"/>
  <compiler meshdir="{ROOT}/assets/robotstudio_so101/assets" autolimits="true"/>
  <asset>
    <texture name="ft" type="2d" builtin="checker"
             rgb1="0.52 0.44 0.33" rgb2="0.46 0.39 0.29" width="512" height="512"/>
    <material name="fm" texture="ft" texrepeat="8 8" reflectance="0.04"/>
    <mesh name="desk_mesh" file="{ROOT}/assets/models/desk/meshes/desk.obj" scale="1.3 1.1 1.35"/>
    <texture name="desk_tex" type="2d" file="{ROOT}/assets/models/desk/meshes/Desk_Diffuse.png"/>
    <material name="desk_mat" texture="desk_tex" reflectance="0.12" shininess="0.3"/>
    <mesh name="chair_mesh" file="{ROOT}/assets/models/chair/meshes/OfficeChairGrey.obj"/>
    <texture name="chair_tex" type="2d" file="{ROOT}/assets/models/chair/meshes/OfficeChairGrey.png"/>
    <material name="chair_mat" texture="chair_tex"/>{_ycb_assets()}
  </asset>
  <default>
    <geom condim="4" solimp="0.9 0.95 0.001" solref="0.02 1"/>
  </default>
  <worldbody>
    <light name="key"  pos="-1 -2 3.5" dir="0.35 0.7 -1"
           diffuse="1.0 0.95 0.85" specular="0.4 0.35 0.35" castshadow="true"/>
    <light name="fill" pos="2 1.5 2.5" dir="-0.5 -0.4 -1"
           diffuse="0.3 0.36 0.52"  specular="0 0 0" castshadow="false"/>
    <light name="rim"  pos="0 2.5 2"   dir="0 -0.7 -0.4"
           diffuse="0.55 0.5 0.45"  specular="0.1 0.1 0.1" castshadow="false"/>
    <geom type="plane" size="4 4 0.1" material="fm" contype="1" conaffinity="1"/>
    <geom type="box" size="4 0.06 2"     pos="0 2.1 1.5"  rgba="0.87 0.85 0.81 1" contype="0" conaffinity="0"/>
    <geom type="box" size="0.06 4 2"     pos="-4.1 0 1.5" rgba="0.84 0.82 0.78 1" contype="0" conaffinity="0"/>
    <geom type="box" size="0.06 4 2"     pos="4.1 0 1.5"  rgba="0.84 0.82 0.78 1" contype="0" conaffinity="0"/>
    <geom type="box" size="4.1 4.1 0.05" pos="0 0 3.1"    rgba="0.94 0.93 0.91 1" contype="0" conaffinity="0"/>
    <body name="desk" pos="0 0.35 0" euler="0 0 1.5708">
      <geom type="mesh" mesh="desk_mesh" material="desk_mat"
            contype="0" conaffinity="0"/>
    </body>
    <!-- Invisible precise collision surface — top face sits exactly at DESK_H -->
    <geom name="desk_surf" type="box" size="0.50 0.42 0.003"
          pos="0 0.05 {DESK_H-0.003:.4f}" rgba="0 0 0 0"
          friction="0.7 0.005 0.0001" contype="1" conaffinity="1"/>
    <body name="chair" pos="0 -0.55 0">
      <geom type="mesh" mesh="chair_mesh" material="chair_mat" contype="0" conaffinity="0"/>
    </body>
    {_target_markers()}
    {_obj_bodies()}
    <body name="left_mount"  pos="-0.30 0.22 {ARM_Z:.4f}" euler="0 0  2.356"><frame name="lf"/></body>
    <body name="right_mount" pos=" 0.30 0.22 {ARM_Z:.4f}" euler="0 0 -2.356"><frame name="rf"/></body>
  </worldbody>
</mujoco>"""

# ── Compile ONCE ──────────────────────────────────────────────────────────────
spec = mujoco.MjSpec.from_string(XML)
ls   = mujoco.MjSpec.from_file(str(SO101))
rs   = mujoco.MjSpec.from_file(str(SO101))
spec.attach(ls, prefix="L_", frame=spec.body("left_mount").first_frame())
spec.attach(rs, prefix="R_", frame=spec.body("right_mount").first_frame())
m = spec.compile()
d = mujoco.MjData(m)
mujoco.mj_resetData(m, d)

# ── Collision groups ──────────────────────────────────────────────────────────
# Bit layout (MuJoCo: collide iff contype_A & conaffinity_B != 0):
#   bit 0 (1) — objects
#   bit 1 (2) — left arm
#   bit 2 (4) — right arm
#   bits 0+1+2 (7) — desk/floor (collides with everything)
#
#   obj  vs obj   : (1&1)=1  ✓ objects stack
#   obj  vs desk  : (1&7)=1  ✓ objects rest on desk
#   L_   vs desk  : (2&7)=2  ✓ arm can't clip through desk
#   R_   vs desk  : (4&7)=4  ✓ arm can't clip through desk
#   L_   vs obj   : (2&1)=0  ✓ left arm passes through objects
#   R_   vs obj   : (4&1)=0  ✓ right arm passes through objects
#   L_   vs R_    : (2&4)=0  ✓ arms don't collide with each other
n_left = n_right = n_desk = 0
for i in range(m.ngeom):
    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
    if name.startswith("L_"):
        m.geom_contype[i]    = 2
        m.geom_conaffinity[i]= 2
        n_left += 1
    elif name.startswith("R_"):
        m.geom_contype[i]    = 4
        m.geom_conaffinity[i]= 4
        n_right += 1

# Desk surface and floor → contype=7, conaffinity=7 (collide with all arm bits)
gid_surf = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "desk_surf")
if gid_surf >= 0:
    m.geom_contype[gid_surf]    = 7
    m.geom_conaffinity[gid_surf]= 7
    n_desk += 1
for i in range(m.ngeom):
    if m.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE:
        m.geom_contype[i]    = 7
        m.geom_conaffinity[i]= 7
        n_desk += 1

print(f"Task={args.task}  {len(obj_ids)} objects  {m.nbody}b {m.njnt}j {m.ngeom}g")
print(f"Collision groups: L_arm={n_left}g(bit1)  R_arm={n_right}g(bit2)  desk/floor={n_desk}g(bits0-2)")

# ── ID lookups ────────────────────────────────────────────────────────────────
def _aid(prefix, j):
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{prefix}{j}")
def _sid(name):
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, name)
def _jid(name):
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
def _gid(name):
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)

L_SITE = _sid("L_gripperframe")
R_SITE = _sid("R_gripperframe")

# Store original appearance so we can restore after pick/place highlight
ORIG_RGBA  = {}
ORIG_MATID = {}
for oid in obj_ids:
    if oid in YCB_MAP:
        gid = _gid(f"g_{oid}_vis")
        if gid >= 0:
            ORIG_RGBA[oid]  = m.geom_rgba[gid].copy()
            ORIG_MATID[oid] = int(m.geom_matid[gid])
    else:
        gid = _gid(f"g_{oid}")
        if gid >= 0:
            ORIG_RGBA[oid] = m.geom_rgba[gid].copy()

# ── Helpers ───────────────────────────────────────────────────────────────────
JOINTS5 = ["shoulder_pan","shoulder_lift","elbow_flex","wrist_flex","wrist_roll"]

def set_ctrl(prefix, joint, val):
    aid = _aid(prefix, joint)
    if aid >= 0:
        lo, hi = m.actuator_ctrlrange[aid]
        d.ctrl[aid] = float(np.clip(val, lo, hi))

def get_pos(oid):
    jid = _jid(f"fj_{oid}")
    if jid < 0: return None
    a = m.jnt_qposadr[jid]
    return [float(d.qpos[a]), float(d.qpos[a+1]), float(d.qpos[a+2])]

def pin(oid, pos):
    jid = _jid(f"fj_{oid}")
    if jid < 0: return
    a = m.jnt_qposadr[jid]; va = m.jnt_dofadr[jid]
    d.qpos[a:a+3] = pos; d.qpos[a+3] = 1.0; d.qpos[a+4:a+7] = 0.0
    d.qvel[va:va+6] = 0.0

def contacts(oid, on):
    val = 1 if on else 0
    if oid in YCB_MAP:
        _, _, n_col = YCB_MAP[oid]
        for k in range(n_col):
            gid = _gid(f"g_{oid}_col{k}")
            if gid >= 0:
                m.geom_contype[gid]    = val
                m.geom_conaffinity[gid]= val
    else:
        gid = _gid(f"g_{oid}")
        if gid >= 0:
            m.geom_contype[gid]    = val
            m.geom_conaffinity[gid]= val

def dist(a, b):
    return math.sqrt(sum((x-y)**2 for x,y in zip(a,b)))

def color(oid, rgba):
    """Apply a solid color (disables texture for YCB objects)."""
    if oid in YCB_MAP:
        gid = _gid(f"g_{oid}_vis")
        if gid >= 0:
            m.geom_matid[gid] = -1   # disable material so rgba shows
            m.geom_rgba[gid]  = rgba
    else:
        gid = _gid(f"g_{oid}")
        if gid >= 0:
            m.geom_rgba[gid] = rgba

def restore_color(oid):
    """Restore original texture / color after a pick-place cycle."""
    if oid in YCB_MAP:
        gid = _gid(f"g_{oid}_vis")
        if gid >= 0:
            m.geom_matid[gid] = ORIG_MATID.get(oid, -1)
            m.geom_rgba[gid]  = ORIG_RGBA.get(oid, np.array([1,1,1,1]))
    else:
        gid = _gid(f"g_{oid}")
        if gid >= 0:
            m.geom_rgba[gid] = ORIG_RGBA.get(oid, np.array([0.6,0.4,0.2,1]))

HOME   = dict(shoulder_pan=0.0, shoulder_lift=-1.4, elbow_flex=0.5,
              wrist_flex=0.0, wrist_roll=0.0, gripper=0.0)
MIRROR = {"shoulder_pan","wrist_roll"}

def go_home():
    for j, val in HOME.items():
        set_ctrl("L_", j,  val)
        set_ctrl("R_", j, -val if j in MIRROR else val)

def step_n(n, v=None, carry_oid=None, site=None):
    """Step physics, optionally pin a carried object, sync viewer."""
    for i in range(n):
        if carry_oid is not None and site is not None:
            mujoco.mj_fwdPosition(m, d)
            pin(carry_oid, d.site_xpos[site].copy())
        mujoco.mj_step(m, d)
        if v is not None and i % 4 == 0:
            v.sync()

def ik_instant(target, prefix, site, n_iter=35):
    """Full DLS IK — tracks the object arc position each frame."""
    tgt  = np.array(target, dtype=float)
    jids = [_jid(f"{prefix}{j}") for j in JOINTS5]
    for _ in range(n_iter):
        mujoco.mj_fwdPosition(m, d)
        err = tgt - d.site_xpos[site]
        if np.linalg.norm(err) < 0.004: break
        jacp = np.zeros((3, m.nv))
        mujoco.mj_jacSite(m, d, jacp, None, site)
        valid = [j for j in jids if j >= 0]
        J  = np.column_stack([jacp[:, m.jnt_dofadr[j]] for j in valid])
        dq = J.T @ np.linalg.solve(J@J.T + 0.01*np.eye(3), err)
        for i, jid in enumerate(valid):
            adr = m.jnt_qposadr[jid]
            lo, hi = m.jnt_range[jid]
            d.qpos[adr] = float(np.clip(d.qpos[adr] + 0.2*dq[i], lo, hi))
    for j, jid in zip(JOINTS5, jids):
        if jid >= 0:
            aid = _aid(prefix, j)
            if aid >= 0:
                lo, hi = m.actuator_ctrlrange[aid]
                d.ctrl[aid] = float(np.clip(d.qpos[m.jnt_qposadr[jid]], lo, hi))

def move_to(target, prefix, site, v, n_phases=25, steps_per_phase=12, alpha=0.35):
    """Animated arm approach — interleaves IK with physics steps for smooth motion."""
    tgt  = np.array(target, dtype=float)
    jids = [_jid(f"{prefix}{j}") for j in JOINTS5]
    for _phase in range(n_phases):
        for _ in range(2):
            mujoco.mj_fwdPosition(m, d)
            err = tgt - d.site_xpos[site]
            if np.linalg.norm(err) < 0.005: break
            jacp = np.zeros((3, m.nv))
            mujoco.mj_jacSite(m, d, jacp, None, site)
            valid = [j for j in jids if j >= 0]
            J  = np.column_stack([jacp[:, m.jnt_dofadr[j]] for j in valid])
            dq = J.T @ np.linalg.solve(J@J.T + 0.01*np.eye(3), err)
            for i, jid in enumerate(valid):
                adr = m.jnt_qposadr[jid]
                lo, hi = m.jnt_range[jid]
                d.qpos[adr] = float(np.clip(d.qpos[adr] + alpha*dq[i], lo, hi))
        for j, jid in zip(JOINTS5, jids):
            if jid >= 0:
                aid = _aid(prefix, j)
                if aid >= 0:
                    lo, hi = m.actuator_ctrlrange[aid]
                    d.ctrl[aid] = float(np.clip(d.qpos[m.jnt_qposadr[jid]], lo, hi))
        for _ in range(steps_per_phase):
            mujoco.mj_step(m, d)
        if v is not None:
            v.sync()
            time.sleep(0.020)
        mujoco.mj_fwdPosition(m, d)
        if np.linalg.norm(tgt - d.site_xpos[site]) < 0.005:
            break

# Arc carry constants
ARC_STEPS  = 80     # frames along the arc
ARC_HEIGHT = 0.09   # peak rise (m)
LAND_GAP   = 0.025  # release height above target — object free-falls this distance

def carry_arc(oid, arm, tgt, v):
    """
    Object-driven arc carry: object follows a precise parabola src→tgt.
    Arm IK tracks each frame. Arc ends LAND_GAP above target so the object
    settles under gravity — no explosive penetration-resolution forces.
    """
    prefix = "L_" if arm == "left" else "R_"
    site   = L_SITE if arm == "left" else R_SITE

    # Always read REAL-TIME position — object may have shifted since do_pick
    src      = np.array(get_pos(oid), dtype=float)
    dst      = np.array(tgt, dtype=float)
    dst_land = dst.copy()
    dst_land[2] = tgt[2] + LAND_GAP

    contacts(oid, False)

    # Warm up IK at actual object position so arm is there before arc starts
    for _ in range(5):
        ik_instant(src.tolist(), prefix, site, n_iter=50)
        mujoco.mj_step(m, d)
    v.sync()

    for i in range(ARC_STEPS):
        t = i / (ARC_STEPS - 1)
        s = t*t*(3 - 2*t)                       # smooth-step easing
        pos     = src + s * (dst_land - src)
        pos[2] += ARC_HEIGHT * 4*t*(1 - t)      # parabolic lift at midpoint

        pin(oid, pos.tolist())
        ik_instant(pos.tolist(), prefix, site)
        mujoco.mj_step(m, d)
        if i % 2 == 0:
            v.sync()
            time.sleep(0.012)

    # Physics landing: re-enable contacts, let gravity settle the object
    contacts(oid, True)
    for i in range(80):
        mujoco.mj_step(m, d)
        if i % 3 == 0:
            v.sync()
            time.sleep(0.012)

    color(oid, [0.1, 0.85, 0.2, 1.0])   # green = placed
    v.sync()

# ── Reset: scatter objects on desk ───────────────────────────────────────────
def scatter(seed):
    # Step 1: park all objects below the desk (contacts off) so arm home move is clean
    for oid in obj_ids:
        contacts(oid, False)
        jid = _jid(f"fj_{oid}")
        if jid >= 0:
            a = m.jnt_qposadr[jid]
            d.qpos[a:a+3] = [0.0, 0.0, 0.3]   # below desk, out of the way
            d.qpos[a+3] = 1.0; d.qpos[a+4:a+7] = 0.0
    go_home()
    step_n(200)   # arms settle to home pose with no objects on desk

    # Step 2: place objects in FRONT zone (y ≤ 0.02) — arm mounts are at y=0.09
    rng = np.random.default_rng(seed)
    placed = []
    for oid in obj_ids:
        _, _, sz = SIZES.get(oid, [0.025, 0.025, 0.030])
        for _ in range(120):
            x = float(rng.uniform(-0.16, 0.16))
            y = float(rng.uniform( 0.02, 0.14))
            if all(dist([x,y],[px,py]) > 0.14 for px,py in placed):
                break
        placed.append((x, y))
        pin(oid, [x, y, DESK_H + sz + 0.012])   # start 12mm above desk, fall in
        restore_color(oid)

    # Step 3: enable contacts and let objects fully settle
    for oid in obj_ids:
        contacts(oid, True)
    step_n(400)

# ── Pick / place ──────────────────────────────────────────────────────────────
LIFT = 0.11

def do_pick(oid, arm, v):
    prefix = "L_" if arm == "left" else "R_"
    site   = L_SITE if arm == "left" else R_SITE
    pos    = get_pos(oid)
    if pos is None: return
    above  = [pos[0], pos[1], pos[2] + LIFT]

    color(oid, [1.0, 0.85, 0.0, 1.0])   # yellow = being picked
    v.sync()
    set_ctrl(prefix, "gripper", 1.6)     # open gripper
    step_n(25, v=v)
    move_to(above, prefix, site, v)      # swing above object
    move_to(pos,   prefix, site, v)      # descend to object
    set_ctrl(prefix, "gripper", -0.15)   # close gripper
    step_n(40, v=v)

# ── Initial settle ────────────────────────────────────────────────────────────
scatter(args.seed)
print("Drag=rotate  Scroll=zoom  Esc=quit\n")

with mujoco.viewer.launch_passive(m, d) as v:
    v.cam.lookat    = [0.0, 0.08, 0.78]
    v.cam.distance  = 1.55
    v.cam.elevation = -48
    v.cam.azimuth   = 205

    for ep in range(args.episodes):
        if not v.is_running(): break
        scatter(args.seed + ep)
        v.sync()
        print(f"── Episode {ep+1} | seed={args.seed+ep} ──")

        step = 0
        for oid in obj_ids:
            if not v.is_running(): break
            tgt = targets[oid]
            pos = get_pos(oid)
            if pos and dist(pos, tgt) < 0.05:
                color(oid, [0.1, 0.85, 0.2, 1.0]); continue

            # Assign arm by x-position so arms never cross each other
            arm = "left" if (pos[0] < 0) else "right"
            print(f"  pick  {oid:12s}  ({arm})")
            do_pick(oid, arm, v)
            print(f"  carry {oid:12s}  → {[round(x,2) for x in tgt]}")
            carry_arc(oid, arm, tgt, v)
            step += 1

        placed = sum(1 for oid in obj_ids
                     if (pos := get_pos(oid)) and dist(pos, targets[oid]) < 0.06)
        print(f"  ✓ {placed}/{len(obj_ids)}  score={placed/len(obj_ids):.2f}\n")

        go_home(); step_n(120)
        v.sync(); time.sleep(2.0)
