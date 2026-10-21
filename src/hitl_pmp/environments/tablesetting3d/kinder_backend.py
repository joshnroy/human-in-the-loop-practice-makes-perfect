"""Lazy access to the native KinDER skill-level table-setting simulator."""

from typing import Any


class KinderBackend:
    @staticmethod
    def create(*, human_competence: float) -> Any:
        from kinder.envs.dynamic3d.tablesetting3d import TableSetting3DEnv

        return TableSetting3DEnv(human_competence=human_competence, render_mode="rgb_array")
