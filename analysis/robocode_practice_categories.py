"""What RoboCode spent its SweepOffShelf3D practice on, step by step.

    python -m analysis.robocode_practice_categories --practice RUN/practice.jsonl \\
        --output-dir OUT

Post-run only: it reads the `practice.jsonl` that `full_agentic.world.World.dispatch`
appends for every request, and never drives a run. Each line holds the request and
the host's response, whose `observation` is the object state after it. RoboCode may
drop, re-pick and take cubes back out of the box itself (Josh, 2026-10-22, no
restriction); this sorts its counted robot steps by what the cubes and the wiper did,
so that is visible next to the planner arms, whose skills cannot do all of it.

Only what the log records is used: object poses and velocities, the box's live pose
and size, the pinch-site tool pose and the gripper command. Nothing reads a predicate,
because the agent never saw one. So "held" is inferred, not measured: an object is
*carried* while the gripper command is closed (> `GRIP_CLOSED`) and the cube's centre is
within `CARRY_RADIUS` of the tool; the wiper, gripped at its handle, within
`WIPER_RADIUS`.

A cube is in one of three *regions*: `box` (centre inside the box's footprint, in the
box's own yaw frame, and below its rim), `shelf` (any other cube resting higher than
`SHELF_MIN_Z`: a cupboard board or the cupboard top) or `floor` (everything else).

A *carry* runs from the first step that ends with a cube carried until that cube is at
rest again (speed under `REST_SPEED`) after release, and is named by its regions at
those two ends, e.g. `shelf->box`. Every robot step gets one category, first match:

- a carry: `cube_into_box` (ends in the box, from anywhere else), `cube_out_of_box`
  (starts in the box, ends outside), `cube_within_box`, `cube_dropped_to_floor`
  (shelf to floor), `cube_floor_repick` (starts on the floor, does not reach the box),
  `cube_shelf_to_shelf`, or `cube_carry_unfinished` (still carried when the log ends);
- `cube_swept_with_wiper`: the wiper carried or moving, and some cube displaced;
- `wiper_handling`: the wiper carried or moving, no cube displaced;
- `cube_pushed`: a cube displaced (more than `MOVE_EPS`) with nothing carried;
- `robot_only`: nothing but the robot moved.

Human requests are not robot steps; they are counted by skill. A request the host
refused before execution (an error response) is not counted at all.
"""

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import BaseModel

GRIP_CLOSED = 0.5
CARRY_RADIUS = 0.05
WIPER_RADIUS = 0.15
SHELF_MIN_Z = 0.15
REST_SPEED = 0.02
MOVE_EPS = 0.002
CARRY_CATEGORIES = (
    "cube_into_box",
    "cube_out_of_box",
    "cube_within_box",
    "cube_dropped_to_floor",
    "cube_floor_repick",
    "cube_shelf_to_shelf",
    "cube_carry_unfinished",
)
CATEGORIES = (
    *CARRY_CATEGORIES,
    "cube_swept_with_wiper",
    "wiper_handling",
    "cube_pushed",
    "robot_only",
)
WIPER = "wiper_0"
BOX = "box_0"
ROBOCODE_COLOUR = "#CC79A7"


class CategorySummary(BaseModel):
    robot_steps: int
    steps: dict[str, int]
    carries: dict[str, int]
    human_requests: dict[str, int]


class Carry(BaseModel):
    cube: str
    start: str
    steps: int = 0


class PracticeCategories:
    """A static-method container, never instantiated."""

    @staticmethod
    def region(*, position: tuple[float, float, float], box: dict[str, float]) -> str:
        yaw = 2 * math.atan2(box["qz"], box["qw"])
        dx, dy = position[0] - box["x"], position[1] - box["y"]
        local_x = math.cos(yaw) * dx + math.sin(yaw) * dy
        local_y = -math.sin(yaw) * dx + math.cos(yaw) * dy
        inside = abs(local_x) <= box["bb_x"] / 2 and abs(local_y) <= box["bb_y"] / 2
        if inside and position[2] < box["z"] + box["bb_z"]:
            return "box"
        return "shelf" if position[2] > SHELF_MIN_Z else "floor"

    @staticmethod
    def carry_category(*, start: str, end: str) -> str:
        if end == "box":
            return "cube_within_box" if start == "box" else "cube_into_box"
        if start == "box":
            return "cube_out_of_box"
        if start == "floor":
            return "cube_floor_repick"
        return "cube_dropped_to_floor" if end == "floor" else "cube_shelf_to_shelf"

    @staticmethod
    def objects(*, observation: dict[str, Any]) -> dict[str, dict[str, float]]:
        return {o["name"]: o["features"] for o in observation["objects"]}

    @staticmethod
    def position(features: dict[str, float]) -> tuple[float, float, float]:  # noqa: PLR0917
        return (features["x"], features["y"], features["z"])

    @staticmethod
    def carried(*, objects: dict[str, dict[str, float]]) -> str | None:
        robot = objects["robot"]
        if robot["gripper_command"] <= GRIP_CLOSED:
            return None
        tool = (robot["pos_tool_x"], robot["pos_tool_y"], robot["pos_tool_z"])
        cubes = [
            (math.dist(tool, PracticeCategories.position(f)), name)
            for name, f in objects.items()
            if name.startswith("cube_")
        ]
        distance, nearest = min(cubes)
        if distance <= CARRY_RADIUS:
            return nearest
        if WIPER in objects and (
            math.dist(tool, PracticeCategories.position(objects[WIPER])) <= WIPER_RADIUS
        ):
            return WIPER
        return None

    @staticmethod
    def displaced(
        *, before: dict[str, dict[str, float]], after: dict[str, dict[str, float]], name: str
    ) -> bool:
        return (
            math.dist(
                PracticeCategories.position(before[name]), PracticeCategories.position(after[name])
            )
            > MOVE_EPS
        )

    @staticmethod
    def summarize(*, practice: Path) -> CategorySummary:
        steps: Counter[str] = Counter(dict.fromkeys(CATEGORIES, 0))
        carries: Counter[str] = Counter()
        human: Counter[str] = Counter()
        state: dict[str, dict[str, float]] | None = None
        carry: Carry | None = None
        robot_steps = 0

        def finish(*, end: str) -> None:
            assert carry is not None
            steps[PracticeCategories.carry_category(start=carry.start, end=end)] += carry.steps
            carries[f"{carry.start}->{end}"] += 1

        with practice.open() as stream:
            for line in stream:
                event = json.loads(line)
                result = event["response"].get("result")
                if result is None:
                    continue
                after = PracticeCategories.objects(observation=result["observation"])
                operation = event["request"]["operation"]
                if operation == "request_help":
                    human[event["request"]["intervention_id"]] += 1
                if operation != "step" or state is None:
                    state = after
                    continue
                robot_steps += 1
                before, state = state, after
                box = after[BOX]
                held = PracticeCategories.carried(objects=after)
                cube = held if held is not None and held.startswith("cube_") else None
                if carry is not None and cube is not None and cube != carry.cube:
                    finish(
                        end=PracticeCategories.region(
                            position=PracticeCategories.position(after[carry.cube]), box=box
                        )
                    )
                    carry = None
                if carry is None and cube is not None:
                    start = PracticeCategories.region(
                        position=PracticeCategories.position(before[cube]), box=box
                    )
                    carry = Carry(cube=cube, start=start)
                if carry is not None:
                    carry.steps += 1
                    if cube is None:
                        velocity = [after[carry.cube][k] for k in ("vx", "vy", "vz")]
                        speed = math.dist((0.0, 0.0, 0.0), velocity)
                        if speed < REST_SPEED:
                            finish(
                                end=PracticeCategories.region(
                                    position=PracticeCategories.position(after[carry.cube]),
                                    box=box,
                                )
                            )
                            carry = None
                    continue
                cubes_moved = any(
                    PracticeCategories.displaced(before=before, after=after, name=name)
                    for name in after
                    if name.startswith("cube_")
                )
                wiper_active = held == WIPER or (
                    WIPER in after
                    and PracticeCategories.displaced(before=before, after=after, name=WIPER)
                )
                if wiper_active:
                    steps["cube_swept_with_wiper" if cubes_moved else "wiper_handling"] += 1
                elif cubes_moved:
                    steps["cube_pushed"] += 1
                else:
                    steps["robot_only"] += 1
        if carry is not None and state is not None:
            if PracticeCategories.carried(objects=state) != carry.cube:
                finish(
                    end=PracticeCategories.region(
                        position=PracticeCategories.position(state[carry.cube]), box=state[BOX]
                    )
                )
            else:
                steps["cube_carry_unfinished"] += carry.steps
                carries[f"{carry.start}->carried"] += 1
        return CategorySummary(
            robot_steps=robot_steps,
            steps=dict(steps),
            carries=dict(carries),
            human_requests=dict(human),
        )

    @staticmethod
    def figure(*, summary: CategorySummary, path: Path, title: str) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        names = list(CATEGORIES)[::-1]
        counts = [summary.steps[name] for name in names]
        total = summary.robot_steps
        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.barh(names, counts, color=ROBOCODE_COLOUR)
        for y, count in enumerate(counts):
            ax.text(count, y, f" {count}/{total}", va="center", fontsize=8)
        ax.set_xlabel("robot control steps")
        human = ", ".join(f"{k} {v}" for k, v in sorted(summary.human_requests.items())) or "none"
        ax.set_title(f"{title}\nrobot steps by category (of {total}); human requests: {human}")
        ax.set_xlim(0, max(max(counts), 1) * 1.25)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

    @staticmethod
    def main(*, argv: list[str] | None = None) -> int:
        parser = argparse.ArgumentParser(
            description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
        )
        parser.add_argument("--practice", type=Path, required=True)
        parser.add_argument("--output-dir", type=Path, required=True)
        parser.add_argument("--title", default="SweepOffShelf3D, RoboCode practice")
        args = parser.parse_args(argv)
        summary = PracticeCategories.summarize(practice=args.practice)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "practice_categories.json").write_text(
            summary.model_dump_json(indent=2) + "\n"
        )
        PracticeCategories.figure(
            summary=summary, path=args.output_dir / "practice_categories.png", title=args.title
        )
        for name in CATEGORIES:
            print(f"{name:24s} {summary.steps[name]}/{summary.robot_steps}")
        print("carries:", summary.carries)
        print("human requests:", summary.human_requests)
        return 0


if __name__ == "__main__":
    raise SystemExit(PracticeCategories.main())
