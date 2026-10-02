"""Direct native goal-region checks and separately labeled start support tolerance."""

import numpy as np

from hitl_pmp.environments.sweep_drawer3d.start_regions import StartValidation, SweepRegions

from .session import SweepSimpleSession


class SimpleRegions:
    @staticmethod
    def contains(
        *, session: SweepSimpleSession, name: str, region: str, support_tolerance: float = 0.0
    ) -> bool:
        core = session.env.unwrapped._object_centric_env
        assert core.task_config["regions"][region]["target"] == "ground"
        position = np.asarray(
            (*session.base()[:2], 0.0) if name == "robot" else session.position(name=name),
            dtype=np.float32,
        )
        for offset in (0.0, -support_tolerance, support_tolerance):
            sample = position.copy()
            sample[2] += offset
            if core._ground_fixture.check_in_region(sample, region, core._robot_env):
                return True
        return False

    @staticmethod
    def validate(*, session: SweepSimpleSession) -> StartValidation:
        core = session.env.unwrapped._object_centric_env
        checks = {}
        poses = {}
        for _, name, region in core.task_config["initial_state"]:
            config = core.task_config["regions"][region]
            checks[name + ":region"] = SimpleRegions.contains(
                session=session,
                name=name,
                region=region,
                support_tolerance=0.01 if name == "wiper_0" else 0.0,
            )
            yaw = session.base()[2] if name == "robot" else session.yaw(name=name)
            if name == "wiper_0":
                from scipy.spatial.transform import Rotation

                vertical_axis = Rotation.from_quat(session.quaternion(name=name)).as_matrix()[:, 2]
                checks[name + ":upright"] = bool(
                    np.allclose(vertical_axis, (0.0, 0.0, 1.0), atol=1e-3, rtol=0.0)
                )
            checks[name + ":yaw"] = SweepRegions.yaw_matches(
                yaw=yaw, ranges=config.get("yaw_ranges", [[0, 360]])
            )
            poses[name] = (
                [*session.base()]
                if name == "robot"
                else [*session.position(name=name).tolist(), yaw]
            )
        return StartValidation(
            seed=session.seed,
            valid=all(checks.values()),
            checks=checks,
            poses=poses,
            reasons=tuple(k for k, v in checks.items() if not v),
        )
