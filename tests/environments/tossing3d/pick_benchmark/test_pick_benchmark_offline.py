"""The Tossing3D pick benchmark's offline half: labels, verdicts, placements, harvest."""

import json
import math
from pathlib import Path

import pytest

from hitl_pmp.environments.tossing3d.pick_benchmark.harvest import PickHarvest
from hitl_pmp.environments.tossing3d.pick_benchmark.states import PickPlacements, PickStateSet
from hitl_pmp.environments.tossing3d.pick_benchmark.types import PickResult, PickState
from hitl_pmp.environments.tossing3d.pick_benchmark.verdicts import PickVerdicts


@pytest.mark.parametrize(
    ("steps", "terminated", "error", "holding", "label"),
    [
        (70, True, None, True, "success"),
        (0, False, "ValueError: No collision-free cube grasp", False, "no_plan"),
        (55, True, None, False, "execution_failed"),
        (12, False, "AssertionError: lift failed", False, "execution_failed"),
        (400, False, None, False, "budget"),
    ],
)
def test_a_pick_is_labelled_by_what_its_dispatch_did(
    *, steps: int, terminated: bool, error: str | None, holding: bool, label: str
) -> None:
    assert (
        PickVerdicts.label(steps=steps, terminated=terminated, error=error, holding=holding)
        == label
    )


def _result(
    *, skill: str, label: str, in_target: bool = True, family: str = "f", **kw
) -> PickResult:
    return PickResult(
        case_id=f"{family}-{label}-{len(kw)}-{in_target}",
        state_id="s",
        family=family,
        skill=skill,
        label=label,
        in_target=in_target,
        **kw,
    )


def test_every_reachable_state_must_succeed_and_states_outside_the_operator_do_not_count() -> None:
    results = [
        _result(skill="PickCube", label="success", dry_run_plannable=True, pybullet="planned"),
        _result(skill="PickCube", label="success", family="g", dry_run_plannable=True),
        _result(skill="PickCube", label="no_plan", in_target=False, dry_run_plannable=False),
        _result(skill="PickCubeFromBin", label="no_plan", dry_run_plannable=False),
        _result(
            skill="PickCubeFromBin",
            label="success",
            family="h",
            dry_run_plannable=False,
            pybullet="planned",
        ),
    ]
    verdicts = {v.skill: v for v in PickVerdicts.judge(results=results)}
    cube, from_bin = verdicts["PickCube"], verdicts["PickCubeFromBin"]
    assert (cube.successes, cube.total, cube.passed, cube.outside_operator) == (2, 2, True, 1)
    assert (from_bin.successes, from_bin.total, from_bin.passed) == (1, 2, False)
    # The dry run predicted no plan for a pick that then succeeded: a disagreement.
    assert (from_bin.dry_run_plannable, from_bin.dry_run_agrees) == (0, 1)
    assert from_bin.per_family == {"f": [0, 1], "h": [1, 1]}
    assert not PickVerdicts.passed(verdicts=list(verdicts.values()))
    assert not PickVerdicts.passed(verdicts=[v.model_copy(update={"total": 0}) for v in [cube]])


def test_in_bin_placements_sit_the_requested_gap_from_each_wall_and_turn_with_the_bin() -> None:
    inner, half = 0.13, 0.025
    for yaw in (0.0, 0.5):
        places = PickPlacements.in_bin(
            bin_xy=(1.0, 2.0), bin_yaw=yaw, inner_half=inner, cube_half=half, gaps=(0.0, 0.03)
        )
        labels = [label for label, _, _ in places]
        assert len(labels) == len(set(labels)) == 4 * 2 + 4 * 2
        for label, x, y in places:
            dx, dy = x - 1.0, y - 2.0
            lx = math.cos(-yaw) * dx - math.sin(-yaw) * dy
            ly = math.sin(-yaw) * dx + math.cos(-yaw) * dy
            gap = float(label.split("-g")[1])
            assert max(abs(lx), abs(ly)) == pytest.approx(inner - half - gap, abs=1e-9)
            if "corner" in label:
                assert abs(lx) == pytest.approx(abs(ly), abs=1e-9)


def test_floor_placements_stay_outside_the_bin_and_on_the_robots_side_of_the_barrier() -> None:
    for _, x, y in PickPlacements.around_bin(
        bin_xy=(0.5, 1.0), bin_yaw=0.0, outer_half=0.15, cube_half=0.025, gaps=(0.005, 0.02)
    ):
        assert max(abs(x - 0.5), abs(y - 1.0)) >= 0.15 + 0.025 + 0.005 - 1e-9
    for _, x, _ in PickPlacements.beside_barrier(
        barrier_x=1.3, barrier_half=0.03, robot_x=-0.2, cube_half=0.025, gaps=(0.005,), ys=(0.0,)
    ):
        assert x == pytest.approx(1.3 - 0.03 - 0.005 - 0.025)
    walls = {"south": ("y", -3.0), "west": ("x", -2.0)}
    places = {
        label: (x, y)
        for label, x, y in PickPlacements.near_room_walls(
            walls=walls, cube_half=0.025, gaps=(0.01,), along=0.5
        )
    }
    assert places["wall-south-g0.01"] == pytest.approx((0.5, -3.0 + 0.035))
    assert places["wall-west-g0.01"] == pytest.approx((-2.0 + 0.035, 0.5))


def test_the_state_set_is_append_only(*, tmp_path: Path) -> None:
    def state(*, state_id: str, skill: str = "PickCube") -> PickState:
        return PickState(
            state_id=state_id,
            family="start",
            seed=0,
            layout="barrier",
            scene_fingerprint="scene",
            plain={"cube_0": [0.1]},
            atoms=["HandEmpty(robot)"],
            skill=skill,
            in_target=True,
        )

    path = tmp_path / "states.jsonl"
    assert PickStateSet.append(path=path, states=[state(state_id="a"), state(state_id="b")]) == [
        "a",
        "b",
    ]
    assert PickStateSet.append(
        path=path, states=[state(state_id="a", skill="PickCubeFromBin"), state(state_id="c")]
    ) == ["c"]
    assert [(s.state_id, s.skill) for s in PickStateSet.load(path=path)] == [
        ("a", "PickCube"),
        ("b", "PickCube"),
        ("c", "PickCube"),
    ]


def test_harvest_takes_the_plannable_false_resets_and_their_last_tick(*, tmp_path: Path) -> None:
    log = tmp_path / "run" / "tossing3d_state_log.jsonl"
    log.parent.mkdir()
    ticks = [
        {"kind": "header", "seed": 0},
        {"kind": "tick", "state": {"cube_0": [1.0]}, "timestamp": "2026-09-28T14:00:00+00:00"},
        {"kind": "tick", "state": {"cube_0": [2.0]}, "timestamp": "2026-09-28T14:00:05+00:00"},
        {"kind": "skill", "name": "PickCube", "timestamp": "2026-09-28T14:00:06+00:00"},
        {"kind": "tick", "state": {"cube_0": [3.0]}, "timestamp": "2026-09-28T14:00:09+00:00"},
    ]
    log.write_text("".join(json.dumps(t) + "\n" for t in ticks))
    attribution = {
        "exp22c/ModelB-lam3e-6/0": {
            "run_dir": str(log.parent),
            "resets": [
                {
                    "pp": False,
                    "anchor_t": "2026-09-28T14:00:05+00:00",
                    "d4": "in bin near wall",
                    "cycle": 1,
                },
                {
                    "pp": True,
                    "anchor_t": "2026-09-28T14:00:05+00:00",
                    "d4": "open floor",
                    "cycle": 2,
                },
                {
                    "pp": False,
                    "anchor_t": "2026-09-28T14:00:09+00:00",
                    "d4": "across barrier (gate)",
                    "cycle": 3,
                },
                {
                    "pp": False,
                    "anchor_t": None,
                    "d4": "consecutive reset (no geometry)",
                    "cycle": 4,
                },
                {
                    "pp": False,
                    "anchor_t": "2026-09-28T14:00:09.500000+00:00",
                    "d4": "tipped bin",
                    "cycle": 5,
                },
            ],
        },
        "exp22c/ModelA-lam3e-6/0": {"run_dir": "/nowhere", "resets": [{"pp": False}]},
    }
    harvests = PickHarvest.refusals(attribution=attribution, run_prefix="exp22c/ModelB-lam3e-6/")
    assert [(h.d4, h.state) for h in harvests] == [
        ("in bin near wall", {"cube_0": [2.0]}),
        ("tipped bin", {"cube_0": [3.0]}),
    ]
    assert harvests[0].source.endswith("exp22c/ModelB-lam3e-6/0#cycle1")
