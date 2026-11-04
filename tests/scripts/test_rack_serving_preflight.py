import sys

import pytest

from hitl_pmp.environments.racktoserving3d.config import RackServingConfig
from scripts import rack_serving_preflight


class _Built(Exception):
    pass


def _config(*, monkeypatch, tmp_path, argv: list[str]) -> RackServingConfig:
    built = {}

    def capture(*, config, **kwargs):
        built["config"] = config
        raise _Built

    monkeypatch.setattr(rack_serving_preflight, "RackServingEnvironment", capture)
    monkeypatch.setattr(
        sys, "argv", ["rack_serving_preflight", "--output", str(tmp_path / "out"), *argv]
    )
    with pytest.raises(_Built):
        rack_serving_preflight.main()
    return built["config"]


def test_preflight_closes_on_food_as_the_environment_does_by_default(*, monkeypatch, tmp_path):
    # The force-limited food grip closes fully; a script default of its own (0.8)
    # silently tested a different controller from the environment's.
    config = _config(monkeypatch=monkeypatch, tmp_path=tmp_path, argv=[])
    assert config.food_grip_command == RackServingConfig().food_grip_command
    assert config.food_grip_force == RackServingConfig().food_grip_force


def test_preflight_still_honours_an_explicit_food_grip_command(*, monkeypatch, tmp_path):
    config = _config(
        monkeypatch=monkeypatch, tmp_path=tmp_path, argv=["--food-grip-command", "0.8"]
    )
    assert config.food_grip_command == 0.8
