"""The declared start distribution and exact upstream goal-region checks.

No placement, reset, rejection sampling, or simulator step occurs here. The
caller records rejected seeds and never substitutes a new sample for that seed.
"""

from typing import Any, ClassVar

import numpy as np
from pydantic import BaseModel
from scipy.spatial.transform import Rotation

from .session import SweepDrawerSession
from .types import SweepDrawerScene as S


class StartValidation(BaseModel):
    seed: int
    valid: bool
    checks: dict[str, bool]
    poses: dict[str, list[float]]
    reasons: tuple[str, ...]


class SweepRegions:
    # Shared with the physical reset's existing outcome tolerances. Cube yaw is
    # unrestricted by the task JSON; wiper yaw is fixed at zero.
    WIPER_YAW_TOLERANCE_DEG: ClassVar[float] = 5.0
    DRAWER_CLOSED_TOLERANCE: ClassVar[float] = 0.01

    @staticmethod
    def core(*, session: SweepDrawerSession) -> Any:
        return session.env.unwrapped._object_centric_env

    @staticmethod
    def contains(
        *, session: SweepDrawerSession, name: str, region: str, support_tolerance: float = 0.0
    ) -> bool:
        core = SweepRegions.core(session=session)
        config = core.task_config["regions"][region]
        if name == S.ROBOT:
            position = np.array((*session.base()[:2], 0.0), dtype=np.float32)
        else:
            position = np.asarray(session.position(name=name), dtype=np.float32)
        target = config["target"]
        entity = (
            core._ground_fixture
            if target == "ground"
            else core._fixtures_dict.get(target, core._objects_dict.get(target))
        )
        if entity is None:
            raise ValueError(f"Unknown declared region target {target!r}")
        if entity.check_in_region(position, region, core._robot_env):
            return True
        # Resting MuJoCo bodies settle slightly below their nominal supporting
        # plane. Apply the existing robot reset support tolerance only when
        # validating starts/resets, never when scoring the upstream task goal.
        if support_tolerance:
            for delta in (-support_tolerance, support_tolerance):
                shifted = position.copy()
                shifted[2] += delta
                if entity.check_in_region(shifted, region, core._robot_env):
                    return True
        return False

    @staticmethod
    def in_goal(*, session: SweepDrawerSession, cube: str) -> bool:
        core = SweepRegions.core(session=session)
        predicates = [
            p for p in core.task_config["goal_state"] if isinstance(p, list) and p[1] == cube
        ]
        assert len(predicates) == 1 and predicates[0][0] in ("on", "in")
        return SweepRegions.contains(session=session, name=cube, region=predicates[0][2])

    @staticmethod
    def yaw_matches(*, yaw: float, ranges: list[list[float]], tolerance_deg: float = 0.0) -> bool:
        degrees = float(np.degrees(yaw) % 360)
        for low, high in ranges:
            if high - low >= 360:
                return True
            for candidate in (degrees - 360, degrees, degrees + 360):
                if low - tolerance_deg <= candidate <= high + tolerance_deg:
                    return True
        return False

    @staticmethod
    def validate(*, session: SweepDrawerSession) -> StartValidation:
        core = SweepRegions.core(session=session)
        checks: dict[str, bool] = {}
        poses: dict[str, list[float]] = {}
        for relation, name, region in core.task_config["initial_state"]:
            assert relation in ("in", "on")
            checks[f"{name}:region"] = SweepRegions.contains(
                session=session,
                name=name,
                region=region,
                support_tolerance=0.01 if name == S.WIPER else 0.0,
            )
            config = core.task_config["regions"][region]
            if name == S.ROBOT:
                x, y, yaw = session.base()
                poses[name] = [x, y, yaw]
            else:
                quat = session.quaternion(name=name)
                yaw = float(Rotation.from_quat(quat).as_euler("zyx")[0])
                poses[name] = [*map(float, session.position(name=name)), *map(float, quat)]
            tolerance = SweepRegions.WIPER_YAW_TOLERANCE_DEG if name == S.WIPER else 0.0
            checks[f"{name}:yaw"] = SweepRegions.yaw_matches(
                yaw=yaw, ranges=config.get("yaw_ranges", [[0, 360]]), tolerance_deg=tolerance
            )
        checks["drawer:closed"] = session.drawer_pos() < SweepRegions.DRAWER_CLOSED_TOLERANCE
        reasons = tuple(name for name, passed in checks.items() if not passed)
        return StartValidation(
            seed=session.seed, valid=not reasons, checks=checks, poses=poses, reasons=reasons
        )
