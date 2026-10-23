"""RoboCode's shelf practice, step by step, sorted into what it was spent on."""

import json
from pathlib import Path
from typing import Any

import pytest

from analysis.robocode_practice_categories import (
    CATEGORIES,
    PracticeCategories,
)

BOX = {"x": 1.235, "y": 0.0, "z": 0.0, "qw": 1.0, "qx": 0.0, "qy": 0.0, "qz": 0.0}
BOX |= {"bb_x": 0.25, "bb_y": 0.5, "bb_z": 0.2}
SHELF = {"cube_0": (1.48, -0.03, 0.29), "cube_1": (1.55, 0.0, 0.29), "cube_2": (1.6, 0.05, 0.29)}
IN_BOX = (1.235, 0.0, 0.03)
FLOOR = (0.8, 0.4, 0.02)


class Log:
    """A practice.jsonl written the way `World.dispatch` writes it."""

    def __init__(self) -> None:
        self.cubes = {name: list(p) for name, p in SHELF.items()}
        self.velocity = {name: [0.0, 0.0, 0.0] for name in SHELF}
        self.wiper = [1.5, 0.0, 0.81]
        self.tool = [0.5, 0.0, 0.6]
        self.grip = 0.0
        self.events: list[dict[str, Any]] = []
        self.write(operation="observe")

    def observation(self) -> dict[str, Any]:
        def movable(*, name: str, p: list[float], v: list[float]) -> dict[str, Any]:
            features = dict(x=p[0], y=p[1], z=p[2], qw=1.0, qx=0.0, qy=0.0, qz=0.0)
            features |= dict(vx=v[0], vy=v[1], vz=v[2], wx=0.0, wy=0.0, wz=0.0)
            return {"name": name, "type": "mujoco_movable_object", "features": features}

        robot = {"gripper_command": self.grip, "pos_tool_x": self.tool[0]}
        robot |= {"pos_tool_y": self.tool[1], "pos_tool_z": self.tool[2]}
        objects = [
            {"name": "box_0", "type": "mujoco_movable_object", "features": dict(BOX)},
            *(movable(name=n, p=p, v=self.velocity[n]) for n, p in self.cubes.items()),
            movable(name="wiper_0", p=self.wiper, v=[0.0, 0.0, 0.0]),
            {"name": "robot", "type": "mujoco_tidybot_robot", "features": robot},
        ]
        return {"objects": objects}

    def write(self, *, operation: str, error: bool = False, **request: Any) -> None:
        response = (
            {"error": "ValueError: bounds"}
            if error
            else {"result": {"observation": self.observation(), "operation": operation}}
        )
        self.events.append({"request": {"operation": operation, **request}, "response": response})

    def step(self, *, n: int = 1) -> None:
        for _ in range(n):
            self.write(operation="step", action=[0.0] * 11)

    def carry(self, *, cube: str, path: list[tuple[float, float, float]]) -> None:
        """Close on `cube`, carry it along `path`, open, and let it come to rest."""
        self.tool = list(self.cubes[cube])
        self.grip = 1.0
        self.step()
        for point in path:
            self.tool = list(point)
            self.cubes[cube] = list(point)
            self.velocity[cube] = [0.3, 0.0, 0.0]
            self.step()
        self.grip = 0.0
        self.tool = [self.tool[0], self.tool[1], self.tool[2] + 0.1]
        self.step()  # released, still falling
        self.velocity[cube] = [0.0, 0.0, 0.0]
        self.step()  # at rest

    def save(self, *, path: Path) -> Path:
        path.write_text("".join(json.dumps(e) + "\n" for e in self.events))
        return path


@pytest.fixture(name="log")
def _log(*, tmp_path: Path) -> Path:
    log = Log()
    log.step(n=2)  # 2 robot_only
    log.carry(cube="cube_0", path=[(1.4, 0.0, 0.4), IN_BOX])  # 5 cube_into_box
    log.carry(cube="cube_1", path=[(1.3, 0.3, 0.4), FLOOR])  # 5 dropped
    log.carry(cube="cube_1", path=[(1.0, 0.2, 0.3), (1.235, 0.15, 0.03)])  # 5 into box
    log.carry(cube="cube_0", path=[(1.2, 0.0, 0.3), FLOOR])  # 5 out of box
    log.carry(cube="cube_0", path=[(0.9, 0.3, 0.1), FLOOR])  # 5 floor repick
    log.write(operation="step", error=True)  # refused: no physics, not counted
    log.cubes["cube_2"][0] -= 0.05  # pushed off its spot by the arm
    log.step()  # 1 cube_pushed
    log.tool, log.grip = list(log.wiper), 1.0
    log.step()
    log.wiper = [1.2, 0.0, 0.7]
    log.tool = list(log.wiper)
    log.step()  # 2 wiper_handling
    log.write(operation="request_help", intervention_id="reset_cubes")
    log.write(operation="request_help", intervention_id="reset_wiper")
    log.write(operation="request_help", intervention_id="reset_cubes")
    log.grip = 0.0
    log.tool = [0.5, 0.0, 0.6]
    log.step(n=3)  # 3 robot_only
    return log.save(path=tmp_path / "practice.jsonl")


def test_every_counted_step_gets_exactly_one_category(*, log: Path) -> None:
    summary = PracticeCategories.summarize(practice=log)
    assert summary.robot_steps == 2 + 5 * 5 + 1 + 2 + 3
    assert sum(summary.steps.values()) == summary.robot_steps
    assert set(summary.steps) == set(CATEGORIES)


def test_the_categories_are_what_the_cubes_did(*, log: Path) -> None:
    steps = PracticeCategories.summarize(practice=log).steps
    assert steps["cube_into_box"] == 10
    assert steps["cube_dropped_to_floor"] == 5
    assert steps["cube_out_of_box"] == 5
    assert steps["cube_floor_repick"] == 5
    assert steps["cube_within_box"] == steps["cube_shelf_to_shelf"] == 0
    assert steps["cube_swept_with_wiper"] == steps["cube_carry_unfinished"] == 0
    assert steps["cube_pushed"] == 1
    assert steps["wiper_handling"] == 2
    assert steps["robot_only"] == 5


def test_carries_are_counted_as_transitions(*, log: Path) -> None:
    carries = PracticeCategories.summarize(practice=log).carries
    assert carries == {
        "shelf->box": 1,
        "shelf->floor": 1,
        "floor->box": 1,
        "box->floor": 1,
        "floor->floor": 1,
    }


def test_human_requests_are_counted_by_skill_not_as_steps(*, log: Path) -> None:
    summary = PracticeCategories.summarize(practice=log)
    assert summary.human_requests == {"reset_cubes": 2, "reset_wiper": 1}


def test_regions_read_the_live_box_pose() -> None:
    box = dict(BOX, x=0.5)
    assert PracticeCategories.region(position=(0.5, 0.0, 0.03), box=box) == "box"
    assert PracticeCategories.region(position=IN_BOX, box=box) == "floor"
    assert PracticeCategories.region(position=(1.5, 0.0, 0.29), box=box) == "shelf"
    # A box turned 90 degrees swaps its footprint's axes.
    turned = dict(BOX, qw=0.7071068, qz=0.7071068)
    assert PracticeCategories.region(position=(1.235 + 0.2, 0.0, 0.03), box=turned) == "box"
    assert PracticeCategories.region(position=(1.235, 0.2, 0.03), box=turned) == "floor"


def test_a_figure_and_json_are_written(*, log: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    PracticeCategories.main(argv=["--practice", str(log), "--output-dir", str(out)])
    written = json.loads((out / "practice_categories.json").read_text())
    assert written["robot_steps"] == 33
    assert (out / "practice_categories.png").stat().st_size > 0
