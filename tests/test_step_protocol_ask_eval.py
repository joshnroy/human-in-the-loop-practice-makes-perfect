import builtins

import pytest

from hitl_pmp.step_protocol import StepPracticeRunner


def test_evaluation_runs_without_the_environment_variable(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("HITL_ASK_BEFORE_EVAL", raising=False)
    monkeypatch.setattr(builtins, "input", lambda _prompt: pytest.fail("must not ask"))
    assert StepPracticeRunner.evaluation_wanted(completed_cycle=0)


@pytest.mark.parametrize(
    ("answers", "wanted"), [(["y"], True), (["s"], False), (["x", "S"], False)]
)
def test_the_person_decides_when_asked(
    *, monkeypatch: pytest.MonkeyPatch, answers: list[str], wanted: bool
) -> None:
    monkeypatch.setenv("HITL_ASK_BEFORE_EVAL", "1")
    replies = iter(answers)
    monkeypatch.setattr(builtins, "input", lambda _prompt: next(replies))
    assert StepPracticeRunner.evaluation_wanted(completed_cycle=3) is wanted
