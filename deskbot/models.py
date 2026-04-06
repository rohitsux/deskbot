"""
Pydantic v2 models — shared by server.py and simulation layer.
All OpenEnv API shapes live here.
"""

from __future__ import annotations

from typing import List, Optional, Literal

from pydantic import BaseModel, Field


# ------------------------------------------------------------------ #
# Action                                                               #
# ------------------------------------------------------------------ #

class DeskAction(BaseModel):
    action_type: Literal["pick", "place", "push", "home"]
    object_id: Optional[str] = None          # pick / push
    arm: Optional[Literal["left", "right"]] = "right"
    target: Optional[List[float]] = None     # place: [x, y, z]
    direction: Optional[List[float]] = None  # push: [dx, dy]


# ------------------------------------------------------------------ #
# Observation                                                          #
# ------------------------------------------------------------------ #

class ObjectState(BaseModel):
    id: str
    position: List[float]      # [x, y, z] metres
    held: bool = False
    fragile: bool = False
    material: str = "plastic"

class TargetState(BaseModel):
    object_id: str
    position: List[float]      # [x, y, z] metres

class DeskObservation(BaseModel):
    episode_id: str
    objects: List[ObjectState]
    joint_states: List[float] = Field(default_factory=lambda: [0.0] * 12)
    gripper_states: List[bool] = Field(default_factory=lambda: [False, False])
    targets: List[TargetState] = Field(default_factory=list)
    step_count: int = 0


# ------------------------------------------------------------------ #
# Step result                                                          #
# ------------------------------------------------------------------ #

class StepInfo(BaseModel):
    cleanliness: float = 0.0
    order: float = 1.0
    safety: float = 1.0
    violations: List[dict] = Field(default_factory=list)
    action_error: Optional[str] = None

class StepResult(BaseModel):
    observation: DeskObservation
    reward: float
    done: bool
    info: StepInfo


# ------------------------------------------------------------------ #
# State                                                                #
# ------------------------------------------------------------------ #

class DeskState(BaseModel):
    episode_id: str
    step_count: int


# ------------------------------------------------------------------ #
# /tasks endpoint response                                             #
# ------------------------------------------------------------------ #

class TaskSchema(BaseModel):
    name: str
    display_name: str
    difficulty: str
    max_steps: int
    num_objects: int
    action_fields: List[str]

class TasksResponse(BaseModel):
    tasks: List[TaskSchema]


# ------------------------------------------------------------------ #
# /grader endpoint                                                     #
# ------------------------------------------------------------------ #

class GraderRequest(BaseModel):
    task: str
    episode_id: str
    trajectory: List[dict] = Field(default_factory=list)

class GraderResponse(BaseModel):
    task: str
    score: float
    components: dict = Field(default_factory=dict)


# ------------------------------------------------------------------ #
# /baseline endpoint response                                          #
# ------------------------------------------------------------------ #

class BaselineScores(BaseModel):
    easy: float
    medium: float
    hard: float
    agent: str = "gpt-4o-mini"
