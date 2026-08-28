"""The recoverable layout is explicit and never changes the stock benchmark."""

import json

from hitl_pmp.cli import Cli
from hitl_pmp.environments.tossing3d.cli import Tossing3DCli
from hitl_pmp.environments.tossing3d.layout import Tossing3DLayout
from hitl_pmp.environments.tossing3d.state_log import StateLogHeader


def test_stock_layout_has_no_task_override() -> None:
    args = Cli.parse_args(argv=["--env", "tossing3d", "--method", "ees"])
    env = Tossing3DCli.build_problem(args=args).env
    assert env.layout is Tossing3DLayout.BARRIER
    assert env.backend().task_config_path is None


def test_same_side_layout_reaches_backend_and_has_space_to_throw() -> None:
    args = Cli.parse_args(argv=["--env", "tossing3d", "--method", "ees", "--layout", "same-side"])
    env = Tossing3DCli.build_problem(args=args).env
    path = env.backend().task_config_path
    assert path is not None
    config = json.loads(path.read_text())
    regions = config["regions"]
    bin_x_min, _, bin_x_max, _ = regions["bin_init_region"]["ranges"][0]
    barrier_x = regions["barrier_init_region"]["ranges"][0][0]
    assert bin_x_max + 0.15 < barrier_x
    assert bin_x_min - 1.45 > -2.5
    assert regions["blocks_goal_region"]["target"] == "bin_0"


def test_state_log_preserves_layout_and_reads_legacy_headers() -> None:
    fields = dict(
        variant="o1", scene_bg=True, canonical_seed=125, seed=0, test_env_seed_offset=10000
    )
    legacy = StateLogHeader(**fields)
    assert legacy.layout is Tossing3DLayout.BARRIER
    header = StateLogHeader(**fields, layout=Tossing3DLayout.SAME_SIDE)
    assert StateLogHeader.model_validate_json(header.model_dump_json()).layout is (
        Tossing3DLayout.SAME_SIDE
    )


def test_live_scene_matches_selected_layout(*, tmp_path) -> None:
    import importlib.util

    import pytest

    if importlib.util.find_spec("kinder") is None:
        pytest.skip("KINDER is not installed")
    from scripts.tossing3d_layout_demo import LayoutDemo

    records = LayoutDemo.capture(output_dir=tmp_path, seed=125)
    stock, same_side = records
    for record in records:
        objects = record["objects"]
        assert objects["robot"]["x"] < objects["cuboid_barrier"]["x"]
        assert objects["cube_0"]["x"] < objects["cuboid_barrier"]["x"]
    assert stock["objects"]["bin_0"]["x"] > stock["objects"]["cuboid_barrier"]["x"]
    assert same_side["objects"]["bin_0"]["x"] + 0.15 < same_side["objects"]["cuboid_barrier"]["x"]
    assert (tmp_path / "same-side.png").stat().st_size > 1000
    assert json.loads((tmp_path / "poses.json").read_text()) == records


def test_far_side_evaluation_does_not_change_practice_layout() -> None:
    args = Cli.parse_args(
        argv=[
            "--env",
            "tossing3d",
            "--method",
            "ees",
            "--layout",
            "same-side",
            "--evaluation-layout",
            "barrier",
            "--num-test-tasks",
            "10",
        ]
    )
    practice = Tossing3DCli.build_problem(args=args)
    evaluation = Tossing3DCli.build_evaluation_problem(args=args)
    assert practice.env.layout is Tossing3DLayout.SAME_SIDE
    assert evaluation.env.layout is Tossing3DLayout.BARRIER
    assert evaluation.env.backend().task_config_path is None
    assert args.layout == "same-side"
    assert practice.env is not evaluation.env
    assert [practice.tasks.draw_scene_seed(rng=practice.tasks.test_rng) for _ in range(10)] == [
        evaluation.tasks.draw_scene_seed(rng=evaluation.tasks.test_rng) for _ in range(10)
    ]


def test_evaluation_layout_defaults_to_practice_layout() -> None:
    args = Cli.parse_args(argv=["--env", "tossing3d", "--method", "ees", "--layout", "same-side"])
    assert Tossing3DCli.build_evaluation_problem(args=args).env.layout is Tossing3DLayout.SAME_SIDE


def test_mixed_layout_runner_keeps_replay_logs_separate(*, tmp_path, monkeypatch) -> None:
    from hitl_pmp.method_runner import MethodRunner

    args = Cli.parse_args(
        argv=[
            "--env",
            "tossing3d",
            "--method",
            "ees",
            "--layout",
            "same-side",
            "--evaluation-layout",
            "barrier",
            "--output-dir",
            str(tmp_path),
        ]
    )
    seen = []

    def inspect_run(**kwargs):
        seen.append((kwargs["problem"].env.layout, kwargs["evaluation_problem"].env.layout))

    monkeypatch.setattr(MethodRunner, "run", inspect_run)
    monkeypatch.setattr(Tossing3DCli, "resolve_render_fps", lambda **kwargs: 20)
    Tossing3DCli.run_method(
        args=args, method_factory=lambda context: None, num_cycles=1, max_steps_per_interaction=20
    )
    assert seen == [(Tossing3DLayout.SAME_SIDE, Tossing3DLayout.BARRIER)]
    practice = json.loads((tmp_path / "tossing3d_state_log.jsonl").read_text().splitlines()[0])
    evaluation = json.loads(
        (tmp_path / "tossing3d_evaluation_state_log.jsonl").read_text().splitlines()[0]
    )
    assert practice["layout"] == "same-side"
    assert evaluation["layout"] == "barrier"
