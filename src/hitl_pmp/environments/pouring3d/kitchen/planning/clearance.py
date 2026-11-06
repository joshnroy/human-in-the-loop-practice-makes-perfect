"""The planning margin as an invariant over every configuration a kitchen skill commands.

The planned motions (`parameterized_skills`) keep `PLANNING_MARGIN_M` from the scene in
the PyBullet planning scene, and BiRRT refuses a start inside that margin. The scripted
segments between them (approaches, descents, lifts, the dumpling carry, releases) were
checked only against MuJoCo's own robot model, at a threshold 1 mm looser, so a skill
could end, or fail, with the arm inside the margin: on 2026-11-04 a lost dumpling left
the upper arm 14.4 mm from the stovetop, and no later motion could be planned from there.

`PlanningClearance` answers the planner's own question, in the planner's own scene, for
any commanded configuration: how close does the robot come to a scene body? It is the
start check, applied before the motion instead of after it.

What it does not count, because the skill must touch it:

- whatever the hand holds, and the dumplings riding in a held jug;
- in a dumpling's contact phases, the gripper (palm and fingers) against the dumplings,
  the pan and the worktop or hob under them;
- in the jug's grasp and placement phases, the gripper against the jug and the worktop.

The arm itself, shoulder to bracelet, is held to the full margin in every phase.
"""

from typing import Any

import numpy as np
import pybullet as p
from pydantic import BaseModel, ConfigDict
from spatialmath import SE2

from .utils import MIRRORED_MOVABLES, PLANNING_MARGIN_M, KitchenPyBulletSim

# A dumpling's contact phases: the gripper may touch the dumplings and their support.
DUMPLING_CONTACT_PHASES = frozenset({
    "descend to shrimp body",
    "settle shrimp approach",
    "pinch shrimp",
    "retain shrimp",
    "secure shrimp grasp",
    "lift shrimp",
    "release failed shrimp pick",
    "retreat after failed shrimp pick",
    "release shrimp into pitcher",
    "release held shrimp in place",
    "withdraw after releasing shrimp in place",
    "settle released shrimp",
})
DUMPLING_SUPPORTS = ("tray", "prep_table", "stovetop")
# The jug's grasp and placement: the gripper may touch the jug and the worktop under it.
PITCHER_CONTACT_PHASES = frozenset({
    "approach pitcher",
    "align fingers across handle",
    "descend to pitcher handle",
    "close fingers on pitcher handle",
    "lift pitcher by handle",
    "lower pitcher to floor",
    "align pitcher above table",
    "seat pitcher on table",
    "release pitcher",
    "withdraw gripper",
})
PITCHER_SUPPORTS = ("pitcher", "prep_table")


class Closest(BaseModel):
    """The robot's closest counted approach to the scene: distance (m), link, body."""

    model_config = ConfigDict(frozen=True)

    distance: float
    link: str
    body: str

    def describe(self) -> str:
        return f"planning scene {self.link}/{self.body} {self.distance * 1000:.1f} mm"


class PlanningClearance(BaseModel):
    """One PyBullet planning scene, kept for a controller's life and re-posed per query."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    sim: Any
    links: dict[int, str]
    gripper: frozenset[int]
    bodies: dict[str, int]

    @classmethod
    def for_controller(cls, *, controller: Any) -> "PlanningClearance":
        sim = KitchenPyBulletSim.from_controller(controller=controller)
        robot, client = sim.robot.robot_id, sim.physics_client_id
        links = {-1: "base"} | {
            i: p.getJointInfo(robot, i, physicsClientId=client)[12].decode()
            for i in range(p.getNumJoints(robot, physicsClientId=client))
        }
        # Everything distal of the wrist's tool flange: the Robotiq palm and fingers.
        flange = next(i for i, n in links.items() if n == "robotiq_arg2f_base_link")
        return cls(
            sim=sim,
            links=links,
            gripper=frozenset(i for i in links if i >= flange),
            bodies=sim.named_collision_bodies(),
        )

    def close(self) -> None:
        self.sim.close()

    def exempt(self, *, controller: Any) -> tuple[frozenset[str], frozenset[str]]:
        """(bodies never counted, bodies the gripper alone may touch) in this phase."""
        c = controller
        ignored: set[str] = set()
        if c.held is not None:
            ignored.add(c.held)
        if c.held == "pitcher":
            ignored.update(
                name
                for name, at in zip(c.scene.cube_names, c.scene.cube_locations(), strict=True)
                if at == "pitcher"
            )
        touchable: set[str] = set()
        if c.phase in DUMPLING_CONTACT_PHASES:
            touchable.update((*c.scene.cube_names, *DUMPLING_SUPPORTS))
        if c.phase in PITCHER_CONTACT_PHASES:
            # The dumplings in the jug ride with it from the moment the hand closes.
            touchable.update(PITCHER_SUPPORTS)
            touchable.update(
                name
                for name, at in zip(c.scene.cube_names, c.scene.cube_locations(), strict=True)
                if at == "pitcher"
            )
        return frozenset(ignored), frozenset(touchable)

    def closest(
        self,
        *,
        controller: Any,
        base: np.ndarray,
        arm: np.ndarray,
        cap: float = 0.1,
        contact: bool = True,
    ) -> Closest:
        """The robot at (`base`, `arm`), fingers as the planner models them, against the
        scene where MuJoCo has it now; capped at `cap`. With `contact` false nothing but
        the held object is exempt: the planner's own start check."""
        c = controller
        sim = self.sim
        for name in (*MIRRORED_MOVABLES, *c.scene.cube_names):
            body = c.model.body(name).id
            sim.place(name=name, position=c.data.xpos[body], wxyz=c.data.xquat[body])
        sim.set_robot_base(SE2(float(base[0]), float(base[1]), float(base[2])))
        joints = list(sim.robot.get_joint_positions())
        joints[:7] = [float(v) for v in arm]
        sim.robot.set_joints(joints)
        ignored, touchable = self.exempt(controller=c)
        if not contact:
            touchable = frozenset()
        best = Closest(distance=cap, link="", body="")
        for name, body in self.bodies.items():
            owner = name.split(":")[1] if name.startswith("collider:") else name
            if owner in ignored:
                continue
            for point in p.getClosestPoints(
                sim.robot.robot_id, body, distance=cap, physicsClientId=sim.physics_client_id
            ):
                link = int(point[3])
                if owner in touchable and link in self.gripper:
                    continue
                if float(point[8]) < best.distance:
                    best = Closest(distance=float(point[8]), link=self.links[link], body=name)
        return best

    def violation(self, *, controller: Any, base: np.ndarray, arm: np.ndarray) -> Closest | None:
        """The closest approach if it is inside the planning margin, else None."""
        closest = self.closest(controller=controller, base=base, arm=arm)
        return closest if closest.distance < PLANNING_MARGIN_M else None
