"""A current run's CLI configuration must replay through the same parser."""

import pytest

from hitl_pmp.evaluate_snapshot import ConfigReplay
from tests.practice_cost_fixtures import CostFixtures


@pytest.mark.parametrize("method", ["ees", "pomdp"])
def test_default_run_configuration_round_trips(*, tmp_path, method):
    args = CostFixtures.parse(
        argv=[
            "--env",
            "tossing3d_real",
            "--method",
            method,
            "--robot",
            "fake",
            "--output-dir",
            str(tmp_path / "run"),
            "--practice-step-budget",
            "3000",
        ]
    )
    recorded = ConfigReplay.strings(args=args)
    restored = ConfigReplay.strings(args=ConfigReplay.args(recorded=recorded, flags=[]))
    assert {k: v for k, v in restored.items() if k not in ConfigReplay.NOT_REPLAYED} == {
        k: v for k, v in recorded.items() if k not in ConfigReplay.NOT_REPLAYED
    }
