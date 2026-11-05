"""The belief-space decision log is byte-stable across `PYTHONHASHSEED`.

Skills, atoms and ground skills hold frozensets, whose iteration order follows string
hashing and so changes from process to process, and predicates hold lambdas, whose
default repr carries a memory address. Both reached the log, so two identically seeded
runs wrote different bytes. Each run here is its own process with its own hash seed, and
wall-clock fields are pinned so that only those two sources could differ.
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from hitl_pmp.core.method.types import GroundSkill
from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.skill_provider import Tossing3DSkillProvider
from hitl_pmp.methods.belief_space.decision_log_encoding import DecisionLogEncoding
from hitl_pmp.methods.belief_space.tossing3d_method import Tossing3DPomdpMethod
from hitl_pmp.planning.grounding import SkillGrounder

_ADDRESS = re.compile(r" at 0x[0-9a-fA-F]+")

# One decision and one execution-cost record from a Tossing3D method under step
# accounting, so the log carries a searched action, symbolic atoms, failure records and
# the cost forecasts keyed by ground skill.
_SCRIPT = """
import sys
import time
from pathlib import Path

from hitl_pmp.core.log_timing import LogTiming
from hitl_pmp.core.practice_costs import PracticeAccounting
from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.skill_provider import Tossing3DSkillProvider
from hitl_pmp.methods.belief_space.tossing3d_method import Tossing3DPomdpMethod
from hitl_pmp.planning.grounding import SkillGrounder

LogTiming.fields = staticmethod(lambda: {"timestamp": "fixed", "elapsed_seconds": 0.0})
time.perf_counter = lambda: 0.0

log = Path(sys.argv[1])
env = Tossing3DEnvironment(scene_bg=False)
method = Tossing3DPomdpMethod(
    env=env,
    skill_provider=Tossing3DSkillProvider(env=env),
    seed=0,
    pomdp_search_depth=2,
    pomdp_solver=sys.argv[2],
    decision_log=log,
)
method.configure_practice_accounting(accounting=PracticeAccounting())
universe = SkillGrounder.all_possible_ground_atoms(
    objects=method.objects(), predicates=method.predicates()
)
pick = next(
    skill
    for skill in SkillGrounder.applicable_ground_skills(
        skills=method.skills(), objects=method.objects(), true_atoms=universe
    )
    if skill.skill.name == "PickCube"
)
method.select_skill_to_practice(true_atoms=pick.preconditions)
method.record_action_cost(ground_skill=pick)
method.observe_execution_cost(cost=12, steps=12, complete=True)
method.observe_symbolic_transition(
    ground_skill=pick,
    before_atoms=pick.preconditions,
    after_atoms=pick.preconditions | pick.add_effects,
    success=False,
    was_random_exploration=False,
)
method.select_skill_to_practice(true_atoms=pick.preconditions)
"""


def _log_bytes(*, tmp_path: Path, hash_seed: str, solver: str) -> bytes:
    log = tmp_path / f"decisions-{solver}-{hash_seed}.jsonl"
    subprocess.run(
        [sys.executable, "-c", _SCRIPT, str(log), solver],
        env={**os.environ, "PYTHONHASHSEED": hash_seed},
        check=True,
    )
    return log.read_bytes()


@pytest.mark.parametrize("solver", ["expectimax", "determinized_astar"])
def test_the_decision_log_is_byte_identical_under_two_hash_seeds(
    *, tmp_path: Path, solver: str
) -> None:
    first = _log_bytes(tmp_path=tmp_path, hash_seed="1", solver=solver)
    second = _log_bytes(tmp_path=tmp_path, hash_seed="2", solver=solver)
    assert b" at 0x" not in first
    assert first == second


def _ground_skills() -> list[GroundSkill]:
    env = Tossing3DEnvironment(scene_bg=False)
    method = Tossing3DPomdpMethod(
        env=env,
        skill_provider=Tossing3DSkillProvider(env=env),
        seed=0,
    )
    universe = SkillGrounder.all_possible_ground_atoms(
        objects=method.objects(), predicates=method.predicates()
    )
    return SkillGrounder.applicable_ground_skills(
        skills=method.skills(), objects=method.objects(), true_atoms=universe
    )


def _unordered(*, value: object) -> object:
    if isinstance(value, dict):
        return {key: _unordered(value=item) for key, item in value.items()}
    if isinstance(value, list):
        return sorted((_unordered(value=item) for item in value), key=json.dumps)
    if isinstance(value, str):
        return _ADDRESS.sub("", value)
    return value


def test_only_set_order_and_addresses_change_in_a_logged_action() -> None:
    ground_skills = _ground_skills()
    assert any(len(skill.skill.preconditions) > 1 for skill in ground_skills)
    for ground_skill in ground_skills:
        logged = DecisionLogEncoding.value(value=ground_skill)
        dumped = ground_skill.model_dump(mode="json", fallback=str)
        assert " at 0x" not in json.dumps(logged)
        assert _unordered(value=logged) == _unordered(value=dumped)


def test_only_set_order_and_addresses_change_in_a_logged_text() -> None:
    ground_skills = _ground_skills()
    for ground_skill in ground_skills:
        logged = DecisionLogEncoding.text(value=ground_skill)
        assert " at 0x" not in logged
        assert sorted(logged) == sorted(_ADDRESS.sub("", str(ground_skill)))
    atom = next(iter(ground_skills[0].preconditions))
    assert DecisionLogEncoding.text(value=atom) == _ADDRESS.sub("", str(atom))


def test_forecast_keys_are_re_rendered_with_their_suffix_and_order_kept() -> None:
    first, second = _ground_skills()[:2]
    costs = {str(second) + " (execution forecast)": 2.0, str(first): 1.0, "other": 3.0}
    assert DecisionLogEncoding.ground_skill_keys(costs=costs, ground_skills=(first, second)) == {
        DecisionLogEncoding.text(value=second) + " (execution forecast)": 2.0,
        DecisionLogEncoding.text(value=first): 1.0,
        "other": 3.0,
    }
