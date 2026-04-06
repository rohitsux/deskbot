"""
DualArmRobot — MuJoCo dual-arm controller.

Pick/place/push with physical carry:
  - Arm IK drives to target via damped-least-squares Jacobian iterations.
  - While an object is "held", its freejoint qpos is teleported to the EE
    site position every simulation step (update_carry).
  - Gripper open/close is a slide-joint position actuator.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional

import mujoco
import numpy as np

if TYPE_CHECKING:
    from deskbot.simulation.objects import ObjectSpawner

# ── Joint / site names ────────────────────────────────────────────────────────
_LEFT_JOINTS   = ["left_j0",  "left_j1",  "left_j2",  "left_j3",  "left_j4"]
_RIGHT_JOINTS  = ["right_j0", "right_j1", "right_j2", "right_j3", "right_j4"]
_LEFT_GRIPPER  = "left_j5"
_RIGHT_GRIPPER = "right_j5"
_LEFT_EE_SITE  = "left_ee"
_RIGHT_EE_SITE = "right_ee"

# Home pose (5 arm joints, radians)
_HOME_LEFT  = [0.0, -0.5, 0.8, -0.3, 0.0]
_HOME_RIGHT = [0.0, -0.5, 0.8, -0.3, 0.0]

# Approach lift above pick/place target (metres)
_APPROACH_LIFT = 0.08


class DualArmRobot:
    """Controls both arms inside a MuJoCo model/data pair."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self._model = model
        self._data  = data

        # arm → object_id currently held (None = empty)
        self._held: Dict[str, Optional[str]] = {"left": None, "right": None}

        # Cache actuator indices
        self._act_idx: Dict[str, int] = {}
        for name in _LEFT_JOINTS + _RIGHT_JOINTS + [_LEFT_GRIPPER, _RIGHT_GRIPPER]:
            aid = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_ACTUATOR, name.replace("j", "a", 1)
            )
            if aid >= 0:
                self._act_idx[name] = aid

        # Cache joint qpos addresses
        self._jnt_qpos: Dict[str, int] = {}
        for name in _LEFT_JOINTS + _RIGHT_JOINTS + [_LEFT_GRIPPER, _RIGHT_GRIPPER]:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid >= 0:
                self._jnt_qpos[name] = model.jnt_qposadr[jid]

        # Cache site indices for end-effector positions
        self._site_idx: Dict[str, int] = {}
        for name in [_LEFT_EE_SITE, _RIGHT_EE_SITE]:
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
            if sid >= 0:
                self._site_idx[name] = sid

        self._reset_to_home()

    # ── State accessors ───────────────────────────────────────────────────────

    def get_joint_states(self) -> List[float]:
        """12 joint angles: [left_0..5, right_0..5]."""
        result = []
        for name in _LEFT_JOINTS + [_LEFT_GRIPPER]:
            adr = self._jnt_qpos.get(name)
            result.append(float(self._data.qpos[adr]) if adr is not None else 0.0)
        for name in _RIGHT_JOINTS + [_RIGHT_GRIPPER]:
            adr = self._jnt_qpos.get(name)
            result.append(float(self._data.qpos[adr]) if adr is not None else 0.0)
        return result

    def get_gripper_states(self) -> List[bool]:
        """[left_closed, right_closed]."""
        result = []
        for name in [_LEFT_GRIPPER, _RIGHT_GRIPPER]:
            adr = self._jnt_qpos.get(name)
            val = float(self._data.qpos[adr]) if adr is not None else 1.0
            result.append(val < 0.005)
        return result

    def get_held(self, arm: str) -> Optional[str]:
        return self._held.get(arm)

    def set_held(self, arm: str, object_id: str) -> None:
        self._held[arm] = object_id

    def release(self, arm: str) -> None:
        self._held[arm] = None

    def get_ee_position(self, arm: str) -> np.ndarray:
        """Return [x, y, z] of the end-effector site."""
        site_name = _LEFT_EE_SITE if arm == "left" else _RIGHT_EE_SITE
        sid = self._site_idx.get(site_name)
        if sid is not None:
            mujoco.mj_fwdPosition(self._model, self._data)
            return self._data.site_xpos[sid].copy()
        return np.zeros(3)

    # ── Carry ─────────────────────────────────────────────────────────────────

    def update_carry(self, spawner: "ObjectSpawner") -> None:
        """
        Teleport held objects to the EE site every sim step so they
        physically follow the arm.  Call this inside your step loop.
        """
        for arm in ("left", "right"):
            oid = self._held.get(arm)
            if oid is None:
                continue
            ee_pos = self.get_ee_position(arm)
            spawner.teleport(oid, ee_pos.tolist())

    # ── High-level actions ────────────────────────────────────────────────────

    def pick(self, target_position: List[float], arm: str) -> bool:
        """
        Pick sequence:
          1. Open gripper
          2. IK to approach (target + lift)
          3. Settle (80 steps)
          4. IK to target
          5. Close gripper (60 steps)
        """
        target = list(target_position)
        approach = [target[0], target[1], target[2] + _APPROACH_LIFT]

        self._set_gripper(arm, closed=False)
        self._step_sim(20)

        self._ik_solve_and_set(approach, arm)
        self._step_sim(80)

        self._ik_solve_and_set(target, arm)
        self._step_sim(80)

        self._set_gripper(arm, closed=True)
        self._step_sim(60)
        return True

    def place(self, target_position: List[float], arm: str,
              spawner: Optional["ObjectSpawner"] = None) -> bool:
        """
        Place sequence:
          1. IK to approach (target + lift), carry object along
          2. Settle (80 steps)
          3. IK to target
          4. Open gripper (60 steps)
          5. Teleport object to exact target position
          6. Release held
        """
        target = list(target_position)
        approach = [target[0], target[1], target[2] + _APPROACH_LIFT]

        oid = self._held.get(arm)

        self._ik_solve_and_set(approach, arm)
        self._step_sim_with_carry(80, arm, oid, spawner)

        self._ik_solve_and_set(target, arm)
        self._step_sim_with_carry(80, arm, oid, spawner)

        self._set_gripper(arm, closed=False)
        self._step_sim(40)

        # Precise final placement
        if oid and spawner:
            spawner.teleport(oid, target)

        self._held[arm] = None
        return True

    def push(self, from_pos: List[float], to_pos: List[float], arm: str) -> bool:
        """
        Push sequence: IK to from → IK to to.
        """
        self._set_gripper(arm, closed=False)
        self._ik_solve_and_set(from_pos, arm)
        self._step_sim(80)
        self._ik_solve_and_set(to_pos, arm)
        self._step_sim(80)
        return True

    def home(self, arm: str) -> None:
        """Return arm to home pose."""
        joints = _LEFT_JOINTS if arm == "left" else _RIGHT_JOINTS
        home   = _HOME_LEFT   if arm == "left" else _HOME_RIGHT
        act_prefix = "left" if arm == "left" else "right"
        for i, val in enumerate(home):
            aname = f"{act_prefix}_a{i}"
            aid = self._act_idx.get(aname.replace("_a", "_j").replace("left_j", "left_j").replace("right_j","right_j"))
            # look up by standard name pattern
            aid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, aname)
            if aid >= 0:
                lo = self._model.actuator_ctrlrange[aid, 0]
                hi = self._model.actuator_ctrlrange[aid, 1]
                self._data.ctrl[aid] = float(np.clip(val, lo, hi))
        self._step_sim(60)

    # ── IK ────────────────────────────────────────────────────────────────────

    def _ik_solve_and_set(self, target_pos: List[float], arm: str) -> None:
        """Run IK and apply result to ctrl targets, then zero joint velocities
        to prevent QACC instability from stale qvel after a qpos teleport."""
        angles = self._ik_solve(target_pos, arm)
        self._set_joints(angles, arm)
        self._zero_joint_velocities(arm)

    def _ik_solve(self, target_pos: List[float], arm: str) -> List[float]:
        """
        Iterative Jacobian damped-least-squares IK.
        Returns 5 arm joint angles (not gripper).
        """
        site_name  = _LEFT_EE_SITE  if arm == "left" else _RIGHT_EE_SITE
        joint_names = _LEFT_JOINTS  if arm == "left" else _RIGHT_JOINTS
        sid = self._site_idx.get(site_name)

        target  = np.array(target_pos, dtype=float)
        alpha   = 0.5
        damping = 0.01

        for _ in range(15):
            mujoco.mj_fwdPosition(self._model, self._data)
            if sid is not None:
                ee_pos = self._data.site_xpos[sid].copy()
            else:
                ee_pos = np.zeros(3)

            error = target - ee_pos
            if np.linalg.norm(error) < 0.004:
                break

            n = len(joint_names)
            jacp = np.zeros((3, self._model.nv))
            mujoco.mj_jacSite(self._model, self._data, jacp, None, sid or 0)

            J = np.zeros((3, n))
            for i, jname in enumerate(joint_names):
                jid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, jname)
                if jid >= 0:
                    dv = self._model.jnt_dofadr[jid]
                    J[:, i] = jacp[:, dv]

            JJT = J @ J.T + damping * np.eye(3)
            dq  = J.T @ np.linalg.solve(JJT, error)

            for i, jname in enumerate(joint_names):
                jid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, jname)
                if jid >= 0:
                    adr = self._model.jnt_qposadr[jid]
                    lo  = self._model.jnt_range[jid, 0]
                    hi  = self._model.jnt_range[jid, 1]
                    self._data.qpos[adr] = float(
                        np.clip(self._data.qpos[adr] + alpha * dq[i], lo, hi)
                    )

        arm_angles = [float(self._data.qpos[self._jnt_qpos[n]])
                      for n in joint_names if n in self._jnt_qpos]
        gripper_name = _LEFT_GRIPPER if arm == "left" else _RIGHT_GRIPPER
        gripper_adr  = self._jnt_qpos.get(gripper_name)
        gripper_val  = float(self._data.qpos[gripper_adr]) if gripper_adr is not None else 0.0
        return arm_angles + [gripper_val]

    # ── Low-level helpers ─────────────────────────────────────────────────────

    def _zero_joint_velocities(self, arm: str) -> None:
        """Zero qvel for arm joints after an IK qpos teleport.

        Without this, stale velocities from the previous state cause MuJoCo to
        compute huge accelerations (QACC NaN) when the position jumps suddenly.
        """
        joint_names = _LEFT_JOINTS if arm == "left" else _RIGHT_JOINTS
        for jname in joint_names:
            jid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid >= 0:
                dv = self._model.jnt_dofadr[jid]
                self._data.qvel[dv] = 0.0

    def _set_joints(self, angles: List[float], arm: str) -> None:
        """Write arm joint ctrl targets."""
        act_prefix = "left" if arm == "left" else "right"
        for i, angle in enumerate(angles[:5]):
            aname = f"{act_prefix}_a{i}"
            aid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, aname)
            if aid >= 0:
                lo = self._model.actuator_ctrlrange[aid, 0]
                hi = self._model.actuator_ctrlrange[aid, 1]
                self._data.ctrl[aid] = float(np.clip(angle, lo, hi))

    def _set_gripper(self, arm: str, closed: bool) -> None:
        aname = "left_a5" if arm == "left" else "right_a5"
        aid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, aname)
        if aid >= 0:
            self._data.ctrl[aid] = 0.0 if closed else 0.04

    def _reset_to_home(self) -> None:
        for arm, joints, home in [
            ("left",  _LEFT_JOINTS,  _HOME_LEFT),
            ("right", _RIGHT_JOINTS, _HOME_RIGHT),
        ]:
            for i, (name, val) in enumerate(zip(joints, home)):
                # Sync qpos to home
                adr = self._jnt_qpos.get(name)
                if adr is not None:
                    self._data.qpos[adr] = val
                # Sync ctrl to home — without this, ctrl=0 after resetData
                # causes large control errors during the settle phase → QACC NaN
                aname = f"{arm}_a{i}"
                aid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, aname)
                if aid >= 0:
                    lo = self._model.actuator_ctrlrange[aid, 0]
                    hi = self._model.actuator_ctrlrange[aid, 1]
                    self._data.ctrl[aid] = float(np.clip(val, lo, hi))
            # Gripper open
            aname = f"{arm}_a5"
            aid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, aname)
            if aid >= 0:
                self._data.ctrl[aid] = 0.04
        mujoco.mj_forward(self._model, self._data)
        self._held = {"left": None, "right": None}

    def _step_sim(self, steps: int) -> None:
        for _ in range(steps):
            mujoco.mj_step(self._model, self._data)

    def _step_sim_with_carry(
        self,
        steps: int,
        arm: str,
        oid: Optional[str],
        spawner: Optional["ObjectSpawner"],
    ) -> None:
        """Step simulation while teleporting held object to EE each step."""
        site_name = _LEFT_EE_SITE if arm == "left" else _RIGHT_EE_SITE
        sid = self._site_idx.get(site_name)

        for _ in range(steps):
            mujoco.mj_step(self._model, self._data)
            if oid and spawner and sid is not None:
                ee = self._data.site_xpos[sid]
                spawner.teleport(oid, [float(ee[0]), float(ee[1]), float(ee[2])])


# Fallback import for DESK_HEIGHT used in error path
try:
    from deskbot.simulation.scene import DESK_HEIGHT  # noqa: E402
except ImportError:
    DESK_HEIGHT = 0.74
