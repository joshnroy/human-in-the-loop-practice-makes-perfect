from analysis.render_tossing3d_presentation import (
    _cycle_end_reason,
    _decision_values,
    _is_applicable,
    reset_destination,
)


def test_reset_destination_reads_the_bound_side_object() -> None:
    decision = {
        "action": {
            "skill": {"name": "ask_for_reset_cube_bin_only"},
            "objects": [{"name": "robot"}, {"name": "opposite_side"}],
        }
    }
    assert reset_destination(decision) == "OPPOSITE SIDE"


def test_reset_destination_ignores_non_reset_actions() -> None:
    decision = {
        "action": {
            "skill": {"name": "PickCube"},
            "objects": [{"name": "robot_side"}],
        }
    }
    assert reset_destination(decision) is None


def test_decision_values_preserve_all_logged_root_actions() -> None:
    decision = {
        "action": {"skill": {"name": "PickCube"}},
        "value": 0.8,
        "action_values": {"PickCube": 0.8, "OpenGripper": 0.6, "STOP": 0.5},
    }
    assert _decision_values(decision) == {
        "PickCube": 0.8,
        "OpenGripper": 0.6,
        "STOP": 0.5,
    }


def test_cycle_end_reason_reports_stop() -> None:
    assert _cycle_end_reason({"action": "STOP"}) == "PRACTICE CYCLE ENDED — STOP SELECTED"


def test_same_side_open_gripper_requires_closed_empty() -> None:
    assert not _is_applicable({"atoms": ["name='HandEmpty'"]}, "OpenGripper")
    assert _is_applicable({"atoms": ["name='ClosedEmpty'"]}, "OpenGripper")


def test_expectimax_values_render_without_sampled_path_fields() -> None:
    import numpy as np

    from analysis.render_tossing3d_presentation import _compose

    frame = _compose(
        frame=np.zeros((4, 4, 3), dtype=np.uint8),
        state={"cube_0": [0, 0, 1], "bin_0": [3, 0, 0]},
        seed=0,
        cycle=0,
        total_cycles=10,
        step=1,
        transitions=1,
        current_skill="PickCube",
        current_objects=(),
        history=[],
        samples=100,
        horizon=6,
        decision={
            "action": {"skill": {"name": "PickCube"}},
            "search": [
                {"event": "stop_value", "node": 0, "value": 0.5},
                {
                    "event": "action_value",
                    "node": 0,
                    "action": {"skill": {"name": "PickCube"}},
                    "value": 0.8,
                },
            ],
        },
    )
    assert frame.shape == (960, 1920, 3)


def test_renderer_retains_stop_only_cycles_and_excludes_between_cycle_evaluation(
    *, tmp_path, monkeypatch
) -> None:
    import json
    from types import SimpleNamespace

    import numpy as np

    from analysis import render_tossing3d_presentation as renderer

    state = {"cube_0": [0, 0, 1], "bin_0": [3, 0, 0]}

    def stamp(*, seconds):
        return f"2026-09-22T00:00:{seconds:02d}+00:00"

    header = {
        "kind": "header",
        "variant": "o1",
        "scene_bg": False,
        "canonical_seed": 125,
        "seed": 0,
        "test_env_seed_offset": 10000,
    }
    rows = [
        header,
        {"kind": "tick", "timestamp": stamp(seconds=2), "state": state},
        {"kind": "tick", "timestamp": stamp(seconds=4), "state": state},
    ]
    events = [
        {"event": "session_start", "cycle": 0, "timestamp": stamp(seconds=1)},
        {
            "event": "decision",
            "cycle": 0,
            "decision": 0,
            "action": "STOP",
            "timestamp": stamp(seconds=2),
        },
        {"event": "refit", "cycle": 0, "timestamp": stamp(seconds=3)},
        {"event": "session_start", "cycle": 1, "timestamp": stamp(seconds=5)},
        {
            "event": "decision",
            "cycle": 1,
            "decision": 1,
            "action": "STOP",
            "timestamp": stamp(seconds=6),
        },
        {"event": "refit", "cycle": 1, "timestamp": stamp(seconds=7)},
    ]
    (tmp_path / "tossing3d_state_log.jsonl").write_text("\n".join(map(json.dumps, rows)))
    (tmp_path / "pomdp_decisions.jsonl").write_text("\n".join(map(json.dumps, events)))
    restored = []
    backend = SimpleNamespace(
        render=lambda **_: np.zeros((4, 4, 3), dtype=np.uint8),
        snapshot=lambda: state,
        snapshot_to_plain=lambda **_: state,
        abstraction_diagnostics=lambda: {},
        render_fps=lambda: 1,
    )
    env = SimpleNamespace(
        backend=lambda: backend,
        restore_plain_snapshot=lambda **kw: restored.append(kw),
        close=lambda: None,
    )
    problem = SimpleNamespace(
        env=env,
        hard_reset=lambda: None,
        reset_to_task=lambda **_: None,
        sample_train_task=lambda: None,
    )
    monkeypatch.setattr(renderer.Tossing3DCli, "build_problem", lambda **_: problem)
    composed = []
    monkeypatch.setattr(
        renderer, "_compose", lambda **kw: composed.append(kw) or np.zeros((1, 1, 3))
    )
    frames = []
    video = SimpleNamespace(
        append=lambda **kw: frames.append(kw), close=lambda: None, frames_written=3
    )
    monkeypatch.setattr(renderer, "VideoStream", lambda **_: video)
    assert (
        renderer.render(run=tmp_path, output=tmp_path / "video.mp4", realistic_background=False)
        == 3
    )
    assert len(restored) == 1
    assert [item["cycle"] for item in composed if item["banner"]] == [0, 1]
    assert composed[-1]["step"] == 0
