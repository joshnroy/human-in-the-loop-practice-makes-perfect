"""Bounded contact-separating recovery for a physically released floor tool."""

import mujoco
import numpy as np
from pybullet_helpers.geometry import Pose

from hitl_pmp.environments.sweep_simple3d.controllers import FloorPrimitives
from hitl_pmp.environments.sweep_simple3d.native_palm import NativePalmClearance
from hitl_pmp.environments.sweep_simple3d.physical.motion import ExecutionError

ContactMap = dict[tuple[int, int], float]


class ReleaseContacts:
    """Native collision-only queries on scratch data; never advance live physics."""

    def __init__(self, *, primitive: "FloorPrimitives") -> None:
        self.primitive = primitive
        self.query = NativePalmClearance(
            model=primitive.session.mj_model, live_data=primitive.session.mj_data
        )
        model = self.query.model
        self.robot_geoms = {
            g
            for g in range(model.ngeom)
            if (
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])) or ""
            ).startswith("robot_")
        }
        self.tool_geoms = set(self.query.tool_geoms)

    def contacts(self, *, arm: np.ndarray | None = None) -> ContactMap:
        query = self.query
        query.data.qpos[:] = query.live_data.qpos
        if arm is not None:
            query.data.qpos[np.asarray(query.joint_addresses[:7])] = arm
        mujoco.mj_fwdPosition(query.model, query.data)
        contacts: ContactMap = {}
        for contact in query.data.contact[: query.data.ncon]:
            first, second = int(contact.geom1), int(contact.geom2)
            pair = (min(first, second), max(first, second))
            if contact.dist <= 0 and any(g in self.robot_geoms for g in pair):
                contacts[pair] = min(float(contact.dist), contacts.get(pair, 0.0))
        return contacts

    def tool_contact(self, *, contacts: ContactMap) -> bool:
        return any(any(g in self.tool_geoms for g in pair) for pair in contacts)

    @staticmethod
    def separating(*, previous: ContactMap, current: ContactMap) -> bool:
        """No new pair, recontact, or deeper penetration; no clearance tolerance."""
        return all(pair in previous and depth >= previous[pair] for pair, depth in current.items())

    @staticmethod
    def record(*, contacts: ContactMap) -> list[dict]:
        return [{"geoms": list(pair), "distance": depth} for pair, depth in contacts.items()]


def separate_released_tool(*, primitive: "FloorPrimitives") -> bool:
    """Try one checked 3 cm downward retreat, then require native tool separation.

    Called only after normal ReturnRobotToStart planning fails. The tool remains
    a native dynamic object. Its existing contacts are permitted only while
    monotonically separating; all other arm collision checks remain active.
    """
    session, scene = primitive.session, primitive.scene
    query = ReleaseContacts(primitive=primitive)
    initial = query.contacts()
    if session.gripper() >= 0.2 or not query.tool_contact(contacts=initial):
        return False
    scene.sync()
    start = session.arm().copy()
    ee = scene.ee_now()
    bodies = scene.bodies() - {scene.wiper_body}
    target = Pose(tuple(np.asarray(ee.position) + [0.0, 0.0, -0.03]), ee.orientation)
    path = scene.linear_path(
        start=start, target=target, bodies=bodies | {scene.wiper_body}, max_jump=0.6
    )
    if path is None:
        return False
    if not scene.within_arm_limits(arm=start) or scene.in_collision(
        joints=scene.planning_fingers(arm=start), bodies=bodies
    ):
        return False
    previous = initial
    arm = start
    for waypoint in path:
        steps = max(1, int(np.ceil(np.max(np.abs(waypoint - arm)) / 0.001)))
        for fraction in np.linspace(0.0, 1.0, steps + 1)[1:]:
            candidate = arm + fraction * (waypoint - arm)
            current = query.contacts(arm=candidate)
            if (
                not query.separating(previous=previous, current=current)
                or not scene.within_arm_limits(arm=candidate)
                or scene.in_collision(joints=scene.planning_fingers(arm=candidate), bodies=bodies)
            ):
                return False
            previous = current
        arm = waypoint
    if query.tool_contact(contacts=previous):
        return False
    session._write(
        record={
            "kind": "release_separation_checked",
            "t": session.ticks,
            "displacement": [0.0, 0.0, -0.03],
            "waypoints": len(path),
            "initial_contacts": query.record(contacts=initial),
            "max_ticks": 180,
        }
    )
    previous = initial

    def guard() -> None:
        nonlocal previous
        current = query.contacts()
        actual = session.arm()
        scene.sync()
        valid = (
            query.separating(previous=previous, current=current)
            and scene.within_arm_limits(arm=actual)
            and not scene.in_collision(
                joints=scene.planning_fingers(arm=actual),
                bodies=scene.bodies() - {scene.wiper_body},
            )
        )
        session._write(
            record={
                "kind": "release_separation_tick",
                "t": session.ticks,
                "contacts": query.record(contacts=current),
                "valid": bool(valid),
            }
        )
        if not valid:
            raise ExecutionError("Released-tool separation gained or deepened a native contact")
        previous = current

    converged = primitive.motion.follow(path=path, grip=0.0, max_ticks=180, tick_guard=guard)
    return bool(converged and not query.tool_contact(contacts=query.contacts()))
