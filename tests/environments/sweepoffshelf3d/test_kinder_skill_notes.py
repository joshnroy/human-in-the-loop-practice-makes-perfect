"""`KinderSkills.drive` records every candidate's fate as a structured `SkillNote`.

No simulator: the controller and the scene are stubs, so what is pinned is only the
bookkeeping -- a rejected candidate is noted with its parameters and reason, and one
that executes is noted as executed."""

import importlib.util
from types import SimpleNamespace

import numpy as np
import pytest

from hitl_pmp.environments.sweepoffshelf3d.kinder_skills import KinderSkills

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("bilevel_planning") is None, reason="needs bilevel_planning"
)


class Refuses:
    def reset(self, state, params):  # noqa: PLR0917 -- upstream's positional signature
        raise AssertionError("Motion planning failed")


class Plans:
    def __init__(self) -> None:
        self.steps = 2

    def reset(self, state, params):  # noqa: PLR0917
        pass

    def terminated(self) -> bool:
        return self.steps == 0

    def step(self) -> np.ndarray:
        self.steps -= 1
        return np.zeros(11)

    def observe(self, state) -> None:  # noqa: PLR0917
        pass


def _skills() -> KinderSkills:
    shelf = SimpleNamespace(env=None, core=SimpleNamespace(), ticks=0)
    skills = KinderSkills(shelf=shelf, backend=None)  # type: ignore[arg-type]
    skills.state = lambda: None  # type: ignore[method-assign]
    skills.plan_blocked = lambda *, controller: None  # type: ignore[method-assign]
    skills.sync = lambda: None  # type: ignore[method-assign]

    def env_step(*, action: np.ndarray, label: str) -> None:
        shelf.ticks += 1

    skills.env_step = env_step  # type: ignore[method-assign]
    return skills


def test_a_candidate_that_does_not_plan_is_noted_with_its_parameters() -> None:
    skills = _skills()
    assert skills.drive(controller=Refuses(), params=(0.7, -np.pi), label="pick_wiper") == (
        "planned-out"
    )
    (note,) = skills.notes
    assert (note.stage, note.kind) == ("pick_wiper", "no_plan")
    assert note.params == pytest.approx([0.7, -np.pi])
    assert "Motion planning failed" in note.reason


def test_a_plan_refused_by_the_mujoco_recheck_is_noted_as_blocked() -> None:
    skills = _skills()
    skills.plan_blocked = lambda *, controller: "approach plan hits board_3"  # type: ignore[method-assign]
    assert skills.drive(controller=Plans(), params=(0.6, 0.0), label="pick_wiper") == (
        "planned-out"
    )
    assert [(n.kind, n.reason) for n in skills.notes] == [
        ("plan_blocked", "approach plan hits board_3")
    ]


def test_the_executed_candidate_is_noted_with_its_control_steps() -> None:
    skills = _skills()
    skills.drive(controller=Refuses(), params=(0.7, 0.0), label="pick_wiper")
    assert skills.drive(controller=Plans(), params=(0.6, 0.0), label="pick_wiper") == "done"
    assert [n.kind for n in skills.notes] == ["no_plan", "executed"]
    assert skills.notes[-1].reason == "2 control steps"
