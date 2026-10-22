"""The system prompt states the per-instance evaluation deadline the runner enforces.

It used to say a fixed "60 seconds"; since #456 each held-out task's sandbox deadline is
`FullAgenticRunner.evaluation_settings`' startup allowance plus a per-step allowance, so
the prompt renders that same number from the evaluation budget.
"""

from pathlib import Path

import pytest

from hitl_pmp.agentic_runtime.sandbox import SandboxSettings
from hitl_pmp.full_agentic.runner import FullAgenticRunner


def deadline(*, budget: int) -> float:
    settings = SandboxSettings(image="unused", robocode_checkout=Path("/rc"))
    return FullAgenticRunner.evaluation_settings(settings=settings, budget=budget).timeout_seconds


@pytest.mark.parametrize(
    ("budget", "seconds"),
    [
        (500, "245"),  # Tossing3D: the runner's default --evaluation-control-steps
        (3000, "870"),  # SweepOffShelf3D's launch budget
    ],
)
def test_the_prompt_states_the_runners_own_deadline(*, budget: int, seconds: str) -> None:
    assert f"{deadline(budget=budget):g}" == seconds
    prompt = FullAgenticRunner.system_prompt(evaluation_control_steps=budget)
    assert f"given {seconds} seconds of wall-clock per instance;" in prompt
    assert "60 seconds" not in prompt


def test_a_fractional_deadline_is_stated_exactly() -> None:
    prompt = FullAgenticRunner.system_prompt(evaluation_control_steps=1)
    assert f"given {deadline(budget=1):g} seconds" in prompt
    assert "given 120.25 seconds of wall-clock per instance;" in prompt


def test_the_default_budget_is_the_parsers() -> None:
    args = FullAgenticRunner.parse_args(
        argv=["--env", "tossing3d", "--output", "o", "--sandbox-settings", "s"]
    )
    assert args.evaluation_control_steps == 500
