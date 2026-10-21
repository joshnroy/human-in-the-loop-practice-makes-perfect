"""Failed robot skills harvested from a run's recording as benchmark states."""

import json
from pathlib import Path

from hitl_pmp.environments.sweepoffshelf3d.skill_benchmark.harvest import HarvestedStates

FEATURES = {
    "robot": [0.0, 0.0, 0.0, 0.0],
    "cube_0": [1.48, -0.03, 0.289],
    "cube_1": [1.55, 0.005, 0.289],
    "wiper_0": [1.51, 0.04, 0.546, 1.5708],
    "box_0": [1.12, -0.24, 1.35, 0.24, 0.2],
}


def _action(*, skill: str, objects: list[str], success: bool, seq: int) -> dict:
    return {
        "kind": "action",
        "seq": seq,
        "actor": "robot",
        "skill": skill,
        "objects": objects,
        "success": success,
        "outcome": None if success else "no grasp planned",
    }


def _write(*, path: Path, records: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return path


def _tick(*, features: dict) -> dict:
    return {"kind": "tick", "state": features}


def test_a_failed_robot_skill_is_harvested_with_the_state_before_it(*, tmp_path: Path) -> None:
    held = {**FEATURES, "robot": [0.0, 0.0, 0.0, 1.0]}
    moved = {**FEATURES, "cube_0": [0.9, 0.4, 0.011]}
    _write(
        path=tmp_path / "run" / "recording.jsonl",
        records=[
            {"kind": "header", "environment": "sweepoffshelf3d"},
            {"kind": "segment", "scene_seed": 3},
            _tick(features=FEATURES),
            _action(
                skill="random: PickCubeFromShelf", objects=["robot", "cube_0"], success=False, seq=0
            ),
            _tick(features=FEATURES),
            _action(skill="PickWiper", objects=["robot", "wiper_0"], success=True, seq=1),
            _tick(features=held),
            _action(
                skill="PlaceCubeInBox", objects=["robot", "cube_1", "box_0"], success=False, seq=2
            ),
            _tick(features=moved),
            _action(skill="PickCubeFromFloor", objects=["robot", "cube_0"], success=False, seq=3),
            # No tick since the last action: nothing to harvest from.
            _action(skill="PickCubeFromFloor", objects=["robot", "cube_0"], success=False, seq=4),
            _tick(features=FEATURES),
            # The same moment as seq 0: deduplicated.
            _action(skill="PickCubeFromShelf", objects=["robot", "cube_0"], success=False, seq=5),
        ],
    )
    harvests = HarvestedStates.candidates(paths=[tmp_path])
    assert [(h.skill, h.cube, h.scene_seed) for h in harvests] == [
        ("PickCubeFromFloor", "cube_0", 3),
        ("PickCubeFromShelf", "cube_0", 3),
    ]
    specs = HarvestedStates.specs(harvests=harvests)
    assert [s.family for s in specs] == ["harvested", "harvested"]
    assert specs[1].cases[0].skill == "PickCubeFromShelf"
    assert "no grasp planned" in specs[1].provenance


def test_other_domains_recordings_are_ignored_and_the_cap_holds(*, tmp_path: Path) -> None:
    _write(
        path=tmp_path / "a" / "recording.jsonl",
        records=[
            {"kind": "header", "environment": "tossing3d"},
            _tick(features=FEATURES),
            _action(skill="PickWiper", objects=["robot", "wiper_0"], success=False, seq=0),
        ],
    )
    records: list[dict] = [{"kind": "header", "environment": "sweepoffshelf3d"}]
    for i in range(5):
        records += [
            _tick(features={**FEATURES, "cube_0": [1.40 + 0.05 * i, 0.0, 0.289]}),
            _action(skill="PickCubeFromShelf", objects=["robot", "cube_0"], success=False, seq=i),
        ]
    _write(path=tmp_path / "b" / "recording.jsonl", records=records)
    harvests = HarvestedStates.candidates(paths=[tmp_path], per_skill=3)
    assert [h.skill for h in harvests] == ["PickCubeFromShelf"] * 3


def test_a_skill_the_goal_latch_cut_short_is_not_harvested_as_a_failure(*, tmp_path: Path) -> None:
    latched = {
        **_action(skill="PickCubeFromShelf", objects=["robot", "cube_0"], success=False, seq=0),
        "goal_reached": True,
    }
    _write(
        path=tmp_path / "run" / "recording.jsonl",
        records=[
            {"kind": "header", "environment": "sweepoffshelf3d"},
            {"kind": "segment", "scene_seed": 3},
            _tick(features=FEATURES),
            latched,
        ],
    )
    assert HarvestedStates.candidates(paths=[tmp_path]) == []
