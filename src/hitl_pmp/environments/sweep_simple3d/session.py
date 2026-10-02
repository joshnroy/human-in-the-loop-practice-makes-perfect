"""Reuse the physical recorder and simulator access for the native floor scene."""

from typing import Any, ClassVar

from hitl_pmp.environments.sweep_drawer3d.session import SweepDrawerSession


class SweepSimpleSession(SweepDrawerSession):
    ENV_ID: ClassVar[str] = (
        "kinder/SweepSimple3D-o5-sweep_the_blocks_to_the_left_side_of_the_kitchen_island-v0"
    )

    def wiper_initial_region(self) -> tuple[Any, str]:
        core = self.env.unwrapped._object_centric_env
        region = next(
            clause[2] for clause in core.task_config["initial_state"] if clause[1] == "wiper_0"
        )
        return core._ground_fixture, region
