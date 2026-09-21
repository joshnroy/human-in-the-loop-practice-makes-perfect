"""Proposal rejection must precede every selection mode and execution accounting."""

import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from hitl_pmp.core.method.method import InteractionComplete, NoFeasibleParametersError
from hitl_pmp.core.method.types import GroundSkill, SamplerConsultation
from hitl_pmp.environments.lightswitch.environment import LightSwitchEnvironment
from hitl_pmp.environments.lightswitch.skill_provider import LightSwitchSkillProvider
from hitl_pmp.environments.lightswitch.skills import LightSwitchSkills
from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.kinder_backend import KinderBackend
from hitl_pmp.environments.tossing3d.skill_provider import Tossing3DSkillProvider
from hitl_pmp.environments.tossing3d.skills import Tossing3DSkills
from hitl_pmp.environments.tossing3d.types import (
    PlanarCollisionBox,
    TossFeasibilityGeometry,
    Tossing3DState,
)
from hitl_pmp.methods.belief_space.tossing3d_method import Tossing3DPomdpMethod
from hitl_pmp.methods.practice_makes_perfect.ees_method import EesMethod, _EesEpisode
from hitl_pmp.methods.practice_makes_perfect.wrapped_sampler import LearnedSkillSampler
from hitl_pmp.sampler_draws import SAMPLER_PROPOSALS_FILENAME, SamplerDrawRecorder


def _lightswitch(*, tmp_path: Path, **kwargs):
    env = LightSwitchEnvironment(grid_size=4)
    method = EesMethod(
        env=env,
        skill_provider=LightSwitchSkillProvider(env=env),
        num_candidates=4,
        max_proposals_per_candidate=3,
        draw_recorder=SamplerDrawRecorder(output_path=tmp_path / "sampler_draws.jsonl"),
        **kwargs,
    )
    ground = GroundSkill(
        skill=LightSwitchSkills.TURN_ON_LIGHT,
        objects=(env.robot, env.get_cells()[-1], env.light),
    )
    return method, ground, env.build_initial_state(light_level=0.2, light_target=0.8)


@pytest.mark.parametrize(
    "mode,explore,epsilon,consultation",
    [
        ("uniform", False, 0.5, SamplerConsultation.UNINFORMATIVE),
        ("epsilon", True, 1.0, SamplerConsultation.EPSILON_RANDOM),
        ("informed", False, 0.5, SamplerConsultation.INFORMED),
    ],
)
def test_all_selection_modes_receive_only_accepted_proposals(
    *, monkeypatch, tmp_path, mode, explore, epsilon, consultation
) -> None:
    method, ground, state = _lightswitch(tmp_path=tmp_path, exploration_epsilon=epsilon)
    proposals = [0.1, -0.9, 0.3, -0.8, 0.5, 0.8]
    draw = Mock(side_effect=[np.array([v]) for v in proposals])
    monkeypatch.setattr(LightSwitchSkillProvider, "sample_params", draw)
    monkeypatch.setattr(
        LightSwitchSkillProvider,
        "parameter_rejection_reason",
        lambda self, **kwargs: "blocked" if kwargs["params"][0] < 0 else None,
    )
    seen = []

    def score(self, *, sampler_inputs):  # noqa: PLR0917 (bound method replacement)
        del self
        seen.extend(row[-1] for row in sampler_inputs)
        return [0.5 if mode == "uniform" else row[-1] for row in sampler_inputs]

    monkeypatch.setattr(LearnedSkillSampler, "score_inputs", score)
    labeled, record = method.execute_ground_skill(ground_skill=ground, state=state, explore=explore)
    assert seen == [0.1, 0.3, 0.5, 0.8]
    assert draw.call_count == 6
    assert record is not None and record.consultation == consultation
    assert record.params[0] in seen
    assert labeled.action[1] == record.params[0]
    assert record.sampler_input[-1] == record.params[0]
    if mode == "informed":
        assert record.params == [0.8]
    assert method.sampler(skill_name=ground.skill.name, param_dim=1).num_observations == 0
    log = json.loads((tmp_path / SAMPLER_PROPOSALS_FILENAME).read_text())
    assert log["rejection_reasons"] == {"blocked": 2}
    assert log["accepted_candidates"] == 4
    assert log["sampled_proposals"] == 6
    assert not (tmp_path / "sampler_draws.jsonl").exists()


def test_default_hook_preserves_original_rng_and_sampler_choice(*, tmp_path) -> None:
    method, ground, state = _lightswitch(tmp_path=tmp_path, seed=21)
    proposal_rng = np.random.default_rng(21)
    candidates = [
        method.skill_provider.sample_params(ground_skill=ground, rng=proposal_rng) for _ in range(4)
    ]
    sampler = LearnedSkillSampler(skill_name=ground.skill.name, param_dim=1, seed=21)
    expected = sampler.sample(
        candidates=candidates,
        sampler_inputs=[
            method.sampler_input_row(ground_skill=ground, params=p, state=state) for p in candidates
        ],
        explore=True,
    )
    _, record = method.execute_ground_skill(ground_skill=ground, state=state, explore=True)
    assert record is not None and record.params == expected.params.tolist()
    assert record.consultation == expected.consultation
    assert method._rng.bit_generator.state == proposal_rng.bit_generator.state  # noqa: SLF001


def test_all_rejected_batch_is_typed_and_does_not_consult_sampler(*, monkeypatch, tmp_path) -> None:
    method, ground, state = _lightswitch(tmp_path=tmp_path)
    gate = Mock(return_value="blocked")
    compute = Mock(side_effect=AssertionError("No action should be constructed"))
    monkeypatch.setattr(LightSwitchSkillProvider, "parameter_rejection_reason", gate)
    monkeypatch.setattr(LightSwitchSkillProvider, "compute_action", compute)
    with pytest.raises(NoFeasibleParametersError) as caught:
        method.execute_ground_skill(ground_skill=ground, state=state, explore=True)
    assert gate.call_count == 12
    assert caught.value.skill_name == ground.skill.name
    assert caught.value.diagnostics.accepted_candidates == 0
    assert caught.value.diagnostics.rejection_reasons == {"blocked": 12}
    assert method._samplers == {}  # noqa: SLF001
    compute.assert_not_called()
    assert not (tmp_path / "sampler_draws.jsonl").exists()


def test_limit_with_some_accepted_candidates_still_selects(*, monkeypatch, tmp_path) -> None:
    method, ground, state = _lightswitch(tmp_path=tmp_path)
    monkeypatch.setattr(
        LightSwitchSkillProvider,
        "parameter_rejection_reason",
        Mock(side_effect=[None] + ["blocked"] * 11),
    )
    _, record = method.execute_ground_skill(ground_skill=ground, state=state, explore=True)
    assert record is not None
    log = json.loads((tmp_path / SAMPLER_PROPOSALS_FILENAME).read_text())
    assert log["accepted_candidates"] == 1
    assert log["sampled_proposals"] == log["max_proposals"] == 12


def test_rejection_settles_previous_real_attempt_only_once(*, monkeypatch, tmp_path) -> None:
    method, ground, state = _lightswitch(tmp_path=tmp_path)
    monkeypatch.setattr(EesMethod, "abstract_state", lambda self, **kwargs: ground.preconditions)
    costs = Mock()
    monkeypatch.setattr(EesMethod, "record_action_cost", costs)
    episode = _EesEpisode(method=method, goal=frozenset(), practicing=True)
    episode._plan = [ground]  # noqa: SLF001 (isolate dispatch)
    episode.step(state=state)
    assert costs.call_count == 1
    monkeypatch.setattr(
        LightSwitchSkillProvider, "parameter_rejection_reason", Mock(return_value="blocked")
    )
    episode._plan = [ground]  # noqa: SLF001
    with pytest.raises(InteractionComplete):
        episode.step(state=state)
    episode.observe_pending(state=state, true_atoms=ground.add_effects)
    assert costs.call_count == 1
    assert method.total_observations() == 1
    assert method.sampler(skill_name=ground.skill.name, param_dim=1).num_observations == 1
    assert len((tmp_path / "sampler_draws.jsonl").read_text().splitlines()) == 1
    assert episode._pending is None  # noqa: SLF001


@pytest.mark.parametrize("practicing", [False, True])
@pytest.mark.parametrize("reject", [False, True])
def test_pomdp_dispatch_cost_and_pending_require_constructed_action(
    *, monkeypatch, tmp_path, practicing, reject
) -> None:
    env = Tossing3DEnvironment(scene_bg=False)
    method = Tossing3DPomdpMethod(
        env=env,
        skill_provider=Tossing3DSkillProvider(env=env),
        num_candidates=2,
        max_proposals_per_candidate=2,
        pomdp_num_particles=16,
        decision_log=tmp_path / "decisions.jsonl",
    )
    ground = GroundSkill(
        skill=Tossing3DSkills.MOVE_TO_TOSS_LOCATION_AND_TOSS,
        objects=(env.robot, env.bin, env.cube, env.barrier),
    )
    state = Tossing3DState(
        data={obj: np.zeros(obj.type.dim) for obj in method.objects()},
        abstract_atoms=frozenset(),
    )
    monkeypatch.setattr(EesMethod, "abstract_state", lambda self, **kwargs: ground.preconditions)
    monkeypatch.setattr(
        Tossing3DSkillProvider,
        "parameter_rejection_reason",
        lambda self, **kwargs: "blocked" if reject else None,
    )
    # Tossing3D replans every practice action, so a preloaded plan alone does
    # not isolate dispatch accounting from the planner's STOP criterion.
    monkeypatch.setattr(_EesEpisode, "_next_plan", lambda self, **kwargs: [ground])
    episode = _EesEpisode(method=method, goal=frozenset(), practicing=practicing)
    episode._plan = [ground]  # noqa: SLF001 (isolate dispatch from search)
    method.observe_practice_action_budget(remaining_actions=7)
    before = method.pomdp_state.model_dump(mode="json")
    if reject and practicing:
        with pytest.raises(InteractionComplete) as caught:
            episode.step(state=state)
        assert not caught.value.planner_stop
    else:
        action = episode.step(state=state)
        if reject:
            assert np.array_equal(action.action, env.noop_action())
            assert action.label == "no-op (no feasible parameters)"
    assert method._remaining_practice_actions == 7  # noqa: SLF001 (harness owns budget)
    assert method._pending_reset is None  # noqa: SLF001
    assert method.pomdp_state.accumulated_cost == (1.0 if practicing and not reject else 0.0)
    if reject:
        assert episode._pending is None  # noqa: SLF001
        assert episode._pending_sampler_record is None  # noqa: SLF001
        episode.observe_pending(state=state, true_atoms=ground.add_effects)
        assert method.pomdp_state.model_dump(mode="json") == before
        assert method._samplers == {}  # noqa: SLF001
        assert method.practice_outcomes() == {}
    else:
        assert episode._pending == ground  # noqa: SLF001
        assert episode._pending_sampler_record is not None  # noqa: SLF001
    events = [json.loads(line) for line in (tmp_path / "decisions.jsonl").read_text().splitlines()]
    sampling = [event for event in events if event["event"] == "parameter_sampling"]
    assert len(sampling) == 1
    assert sampling[0]["accepted_candidates"] == (0 if reject else 2)
    assert len([event for event in events if event["event"] == "dispatch"]) == int(
        practicing and not reject
    )
    assert env._backend is None  # noqa: SLF001


def test_tossing_integration_uses_supplied_evaluation_snapshot(*, monkeypatch) -> None:
    env = Tossing3DEnvironment(scene_bg=False)
    provider = Tossing3DSkillProvider(env=env)
    method = EesMethod(env=env, skill_provider=provider, num_candidates=2)
    ground = GroundSkill(
        skill=Tossing3DSkills.MOVE_TO_TOSS_LOCATION_AND_TOSS,
        objects=(env.robot, env.bin, env.cube, env.barrier),
    )
    snapshot = object()
    state = Tossing3DState(
        data={obj: np.zeros(obj.type.dim) for obj in method.objects()}, object_centric=snapshot
    )
    geometry = TossFeasibilityGeometry(
        robot_pose=(0.0, 0.0, 0.0),
        robot_size=(0.55, 0.55),
        bin_pose=(3.0, 0.0, 0.0),
        obstacles=(
            PlanarCollisionBox(name="barrier", center=(1.3, 0), width=0.06, height=10, yaw=0),
        ),
        sampling_x_bounds=(-2.5, 2.5),
        sampling_y_bounds=(-2.5, 2.5),
    )

    def geometry_for_state(*, snapshot):
        assert snapshot is state.object_centric
        return geometry

    monkeypatch.setattr(KinderBackend, "toss_feasibility_geometry", geometry_for_state)
    values = [np.array([d, 0.0, 360.0, 500.0]) for d in [1.35, 2.5, 1.7, 2.6]]
    proposals = Mock(side_effect=values)
    monkeypatch.setattr(Tossing3DSkillProvider, "sample_params", proposals)
    _, record = method.execute_ground_skill(ground_skill=ground, state=state, explore=False)
    assert record is not None and record.params[0] in (2.5, 2.6)
    assert record.params[1:] == [0.0, 360.0, 500.0]
    assert not record.records_training_row
    assert proposals.call_count == 4
    assert env._backend is None  # noqa: SLF001 (no live training environment read)
