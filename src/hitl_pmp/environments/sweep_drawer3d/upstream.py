"""Direct reuse of the pinned stock state abstractor without resetting practice.

The upstream constructor calls ``sim.reset()`` solely to obtain an object-state
snapshot for its private planning simulator. Supply an immutable snapshot facade,
never the live practice simulator. All predicate evaluation and goal derivation
remain the actual upstream methods, including their intentionally coarse tests.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict


class SweepSnapshot(BaseModel):
    """Read-only constructor input; reset returns a copy, changes no simulator."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    state: Any
    robot_name: str = "robot"

    def reset(self) -> tuple[Any, dict[str, Any]]:
        return self.state.copy(), {}


class SweepUpstreamFacts:
    @staticmethod
    def create(*, state: Any) -> Any:
        from kinder_models.dynamic3d.sweep3D.state_abstractions import Sweep3DStateAbstractor

        return Sweep3DStateAbstractor(SweepSnapshot(state=state))

    @staticmethod
    def feature(*, atom: Any) -> str:
        name = atom.predicate.name
        objects = tuple(obj.name for obj in atom.objects)
        if name in ("DrawerOpen", "DrawerClosed", "HandEmpty"):
            return name
        if name == "Holding" and objects == ("robot", "wiper_0"):
            return "HoldingWiper"
        if name in ("OnTable", "InDrawer"):
            movable = objects[0]
            if movable == "wiper_0":
                return name + "Wiper"
            if movable.startswith("cube_"):
                return name + movable.removeprefix("cube_")
        raise ValueError(f"Unmapped upstream Sweep atom: {atom}")

    @staticmethod
    def observed(*, abstractor: Any, state: Any) -> frozenset[str]:
        return frozenset(
            SweepUpstreamFacts.feature(atom=atom)
            for atom in abstractor.state_abstractor(state).atoms
        )

    @staticmethod
    def goal(*, abstractor: Any, state: Any) -> frozenset[str]:
        return frozenset(
            SweepUpstreamFacts.feature(atom=atom)
            for atom in abstractor.goal_deriver(state).atoms
        )

    @staticmethod
    def close(*, abstractor: Any) -> None:
        abstractor._pybullet_sim.close()
