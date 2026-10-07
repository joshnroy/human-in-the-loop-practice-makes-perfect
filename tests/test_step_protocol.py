import pickle

import numpy as np
import pytest

from hitl_pmp.core.control_steps import ControlStepLimitReached
from hitl_pmp.environments.lightswitch.environment import LightSwitchEnvironment
from hitl_pmp.environments.lightswitch.skill_provider import LightSwitchSkillProvider
from hitl_pmp.methods.practice_makes_perfect.ees_method import EesMethod
from hitl_pmp.step_protocol import DeploymentSnapshot, PracticeClock


def test_clock_uses_prepared_per_skill_duration_and_actual_robot_cost():
    from hitl_pmp.core.practice_costs import (
        ChargeFunction,
        HumanCharge,
        PracticeAccounting,
        PracticeCosts,
    )

    ledger = PracticeAccounting(
        costs=PracticeCosts(
            robot_step=ChargeFunction(value=2),
            human_skills={
                "near": HumanCharge(cost=ChargeFunction(value=7), duration=ChargeFunction(value=3))
            },
        )
    )
    seen = []
    clock = PracticeClock(
        budget=5,
        interval=2,
        accounting=ledger,
        on_measure=lambda: seen.append((clock.steps, ledger.total_cost)),
    )
    clock.robot_active = True
    clock.before_robot_step()
    clock.after_robot_step()
    clock.robot_active = False
    charge = clock.before_human(skill="near")
    clock.human_invoked(charge=charge)
    assert seen == [(4, 9)]
    assert clock.steps == ledger.practice_steps == 4
    with pytest.raises(ControlStepLimitReached):
        clock.before_human(skill="near")
    assert ledger.total_cost == 9


def test_tossing_snapshot_with_observed_competence_round_trips():
    from hitl_pmp.core.method.types import GroundSkill
    from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
    from hitl_pmp.environments.tossing3d.skill_provider import Tossing3DSkillProvider

    env = Tossing3DEnvironment()
    provider = Tossing3DSkillProvider(env=env)
    method = EesMethod(env=env, skill_provider=provider)
    skill = provider.skills()[0]
    objects = tuple(
        next(o for o in provider.objects() if o.type == p.type) for p in skill.parameters
    )
    ground = GroundSkill(skill=skill, objects=objects)
    method.competence_model(ground_skill=ground).observe(success=True)
    raw = DeploymentSnapshot.encode(method=method)
    restored = DeploymentSnapshot.restore(raw=raw, env=env, provider=provider)
    assert restored.current_competences() == method.current_competences()


def test_evaluation_clock_stops_at_goal_or_exact_budget():
    from hitl_pmp.step_protocol import EvaluationClock

    clock = EvaluationClock(budget=2, check_goal=lambda: False)
    for _ in range(2):
        clock.before()
        clock.after()
    with pytest.raises(ControlStepLimitReached):
        clock.before()
    assert clock.steps == 2
    successful = EvaluationClock(budget=20, check_goal=lambda: True)
    successful.before()
    successful.after()
    with pytest.raises(ControlStepLimitReached):
        successful.before()
    assert successful.solved and successful.steps == 1


def test_measurement_inside_skill_does_not_end_it():
    measurements = []
    clock = PracticeClock(budget=9, interval=3, on_measure=lambda: measurements.append(clock.steps))
    clock.robot_active = True
    for _ in range(8):
        clock.before_robot_step()
        clock.after_robot_step()
    assert measurements == [3, 6]
    assert clock.robot_active
    assert clock.steps == 8


def test_human_counts_once_and_reset_physics_does_not_count():
    measurements = []
    clock = PracticeClock(budget=9, interval=3, on_measure=lambda: measurements.append(clock.steps))
    clock.robot_active = True
    for _ in range(2):
        clock.after_robot_step()
    clock.robot_active = False
    for _ in range(10):
        clock.after_robot_step()
    clock.before_human()
    clock.human_invoked()
    assert (clock.robot_steps, clock.human_invocations, clock.steps) == (2, 1, 3)
    assert measurements == [3]


def test_budget_stops_before_extra_physics():
    clock = PracticeClock(budget=3, interval=2, on_measure=lambda: None)
    clock.robot_active = True
    for _ in range(3):
        clock.before_robot_step()
        clock.after_robot_step()
    with pytest.raises(ControlStepLimitReached):
        clock.before_robot_step()
    with pytest.raises(ControlStepLimitReached):
        clock.before_human()
    assert clock.steps == 3


def test_custom_human_duration_checks_remaining_budget():
    clock = PracticeClock(budget=3, interval=2, human_steps=2, on_measure=lambda: None)
    clock.before_human()
    clock.human_invoked()
    with pytest.raises(ControlStepLimitReached):
        clock.before_human()
    assert clock.steps == 2


def test_snapshot_freezes_sampler_without_refitting_or_sharing_rng():
    env = LightSwitchEnvironment(grid_size=4)
    provider = LightSwitchSkillProvider(env=env)
    method = EesMethod(env=env, skill_provider=provider, seed=7)
    sampler = method.sampler(skill_name="test", param_dim=1)
    sampler.observe(sampler_input=[1.0, 0.3], success=True)
    sampler.fit()
    before_rng = pickle.dumps(method._rng.bit_generator.state)
    raw = DeploymentSnapshot.encode(method=method)
    sampler.observe(sampler_input=[1.0, 0.8], success=False)
    fresh = LightSwitchEnvironment(grid_size=4)
    restored = DeploymentSnapshot.restore(
        raw=raw, env=fresh, provider=LightSwitchSkillProvider(env=fresh)
    )
    assert restored.env is fresh
    assert restored._samplers["test"].num_observations == 1
    assert sampler.num_observations == 2
    assert restored._samplers["test"].score_inputs(sampler_inputs=[[1.0, 0.2]]) == [1.0]
    restored._rng.random(10)
    assert pickle.dumps(method._rng.bit_generator.state) == before_rng
    assert np.array_equal(
        restored._samplers["test"]._classifier._input_shift,
        sampler._classifier._input_shift,
    )
