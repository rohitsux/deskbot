"""
ObjectSpawner — MuJoCo object state manager.

Objects are baked into the MJCF XML by DeskScene at reset time.
This class tracks their body IDs and provides position read/write via
MuJoCo qpos (freejoint: 7 DOF per object: x y z qw qx qy qz).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import mujoco
import numpy as np


class ObjectSpawner:
    """
    Tracks and manipulates dynamic objects in a MuJoCo model.

    Parameters
    ----------
    model   : mujoco.MjModel — the compiled model (objects already in XML)
    data    : mujoco.MjData  — live simulation state
    catalogue : dict         — object catalogue from objects.json
    specs   : list of dicts  — placement specs used to build the XML
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        catalogue: Dict[str, dict],
        specs: List[dict],
    ) -> None:
        self._model = model
        self._data = data
        self._catalogue = catalogue

        # Map object_id → (body_id, freejoint_qpos_address)
        self._body_ids: Dict[str, int] = {}
        self._qpos_adr: Dict[str, int] = {}

        for spec in specs:
            oid = spec["id"]
            body_name = f"obj_{oid}"
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if bid >= 0:
                self._body_ids[oid] = bid
                # Freejoint for this body: find its joint
                joint_name = f"fj_{oid}"
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                if jid >= 0:
                    self._qpos_adr[oid] = model.jnt_qposadr[jid]

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def get_position(self, object_id: str) -> Optional[List[float]]:
        """Return current [x, y, z] of object."""
        adr = self._qpos_adr.get(object_id)
        if adr is None:
            return None
        return [float(self._data.qpos[adr]),
                float(self._data.qpos[adr + 1]),
                float(self._data.qpos[adr + 2])]

    def teleport(self, object_id: str, position: List[float]) -> None:
        """Instantly move object to position (identity orientation)."""
        adr = self._qpos_adr.get(object_id)
        if adr is None:
            return
        self._data.qpos[adr]     = float(position[0])
        self._data.qpos[adr + 1] = float(position[1])
        self._data.qpos[adr + 2] = float(position[2])
        # Reset orientation to identity quaternion (w x y z)
        self._data.qpos[adr + 3] = 1.0
        self._data.qpos[adr + 4] = 0.0
        self._data.qpos[adr + 5] = 0.0
        self._data.qpos[adr + 6] = 0.0
        # Zero velocities (freejoint vel is at qvel_adr)
        jid = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_JOINT, f"fj_{object_id}"
        )
        if jid >= 0:
            vadr = self._model.jnt_dofadr[jid]
            for i in range(6):
                self._data.qvel[vadr + i] = 0.0
        mujoco.mj_forward(self._model, self._data)

    def nudge(self, object_id: str, velocity: List[float]) -> None:
        """Apply a linear velocity impulse to an object."""
        jid = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_JOINT, f"fj_{object_id}"
        )
        if jid >= 0:
            vadr = self._model.jnt_dofadr[jid]
            for i, v in enumerate(velocity[:3]):
                self._data.qvel[vadr + i] += float(v)

    def get_catalogue(self) -> Dict[str, dict]:
        return self._catalogue
