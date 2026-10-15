"""A POMDP-chosen human reset reaches the harness with the destination its decision
event names, so `human_resets.jsonl` and `pomdp_decisions.jsonl` agree.

No simulator: the planner is stubbed to choose the reset, and the episode is stepped
on a hand-built `Tossing3DState` whose atoms satisfy the reset's precondition."""

import json
from pathlib import Path

import numpy as np
import pytest

from hitl_pmp.core.method.method import HumanCubeBinResetRequested
from hitl_pmp.core.method.types import GroundAtom
from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.predicates import ROBOT_AT_SIDE
from hitl_pmp.environments.tossing3d.sides import Tossing3DSides
from hitl_pmp.environments.tossing3d.skill_provider import Tossing3DSkillProvider
from hitl_pmp.environments.tossing3d.types import Tossing3DState
from hitl_pmp.methods.belief_space.tossing3d_method import Tossing3DPomdpMethod
from hitl_pmp.methods.practice_makes_perfect.ees_method import _EesEpisode


@pytest.mark.parametrize("destination", ["robot_side", "opposite_side"])
def test_the_dispatched_destination_is_the_decision_events_last_argument(
    *, destination: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = Tossing3DEnvironment(scene_bg=False)
    provider = Tossing3DSkillProvider(env=env, human_reset_practice_cost=0.25)
    decision_log = tmp_path / "pomdp_decisions.jsonl"
    method = Tossing3DPomdpMethod(
        env=env,
        skill_provider=provider,
        seed=17,
        pomdp_search_depth=1,
        pomdp_num_samples=2,
        sampler_max_train_iters=1,
        decision_log=decision_log,
    )
    reset = next(
        ground
        for ground in provider.human_cube_bin_reset_skills()
        if ground.objects[-1].name == destination
    )
    planner = method.pomdp_planner
    assert planner is not None
    monkeypatch.setattr(planner, "solve", lambda **kwargs: (0.0, reset))
    robot_at_side = GroundAtom(
        predicate=ROBOT_AT_SIDE, objects=(env.robot, env.barrier, Tossing3DSides.robot)
    )
    state = Tossing3DState(
        data={obj: np.zeros(obj.type.dim) for obj in method.objects()},
        abstract_atoms=frozenset({
            (robot_at_side.predicate.name, tuple(obj.name for obj in robot_at_side.objects))
        }),
    )
    env.current_state = state
    episode = _EesEpisode(method=method, goal=frozenset(), practicing=True, pursue_goal=False)

    with pytest.raises(HumanCubeBinResetRequested) as excinfo:
        episode.step(state=state)

    decisions = [
        event
        for event in map(json.loads, decision_log.read_text().splitlines())
        if event["event"] == "decision"
    ]
    assert decisions[-1]["action"]["objects"][-1]["name"] == destination
    assert excinfo.value.destination == destination
    assert excinfo.value.skill == decisions[-1]["action"]["skill"]["name"]
