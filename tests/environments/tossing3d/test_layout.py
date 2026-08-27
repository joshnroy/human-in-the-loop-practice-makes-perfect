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
