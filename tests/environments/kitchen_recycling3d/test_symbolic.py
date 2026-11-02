"""Kitchen symbolic integration without launching physics or an experiment."""

from pathlib import Path

import numpy as np
import pytest
from pydantic import Field, ValidationError

from hitl_pmp.core.method.method import HumanCubeBinResetRequested
from hitl_pmp.core.method.types import GroundSkill, SamplerConsultation
from hitl_pmp.core.problem.tasks.types import Task
from hitl_pmp.environments.kitchen_recycling3d.domain import Calibration, KitchenConfig
from hitl_pmp.environments.kitchen_recycling3d.environment import KitchenRecyclingEnvironment
from hitl_pmp.environments.kitchen_recycling3d.predicates import (
    BIN_AT_TARGET,
    HOLDING,
    IN_BIN,
    PICKABLE,
    KitchenRecyclingPredicates,
    bin_retrievable_predicate,
)
from hitl_pmp.environments.kitchen_recycling3d.problem import KitchenRecyclingProblem
from hitl_pmp.environments.kitchen_recycling3d.skill_provider import (
    KitchenRecyclingOracle,
    KitchenRecyclingSkillProvider,
)
from hitl_pmp.environments.kitchen_recycling3d.tasks import KitchenRecyclingTasks
from hitl_pmp.environments.kitchen_recycling3d.types import (
    NEAR_BIN_POSITION,
    KitchenObjects,
    KitchenState,
)
from hitl_pmp.humans.oracle import UnconditionalHumanOracle
from hitl_pmp.methods.belief_space.kitchen_recycling_domain import KitchenRecyclingPracticeDomain
from hitl_pmp.methods.belief_space.tossing3d_method import Tossing3DPomdpMethod
from hitl_pmp.methods.practice_makes_perfect.ees_method import EesMethod, _EesEpisode
from hitl_pmp.planning.fast_downward import FastDownwardPlanner, PlanningFailure
from hitl_pmp.planning.grounding import SkillGrounder


def _state(*, config=None, seed=0, near=False, training=True):
    config = config or KitchenConfig()
    objects = KitchenObjects(config=config)
    data = {obj: np.zeros(obj.type.dim) for obj in objects.all_objects()}
    state = KitchenState(data=data)
    for feature, value in (
        ("hand_empty", 1),
        ("gripper_open", 1),
        ("training", training),
        ("seed", seed),
    ):
        state.set(obj=objects.robot, feature_name=feature, feature_val=value)
    for item in objects.items:
        for feature in ("qw", "reachable", "on_kitchen_floor"):
            state.set(obj=item, feature_name=feature, feature_val=1)
    state.set(obj=objects.bin, feature_name="qw", feature_val=1)
    state.set(obj=objects.bin, feature_name="near", feature_val=near)
    state.set(obj=objects.bin, feature_name="at_target", feature_val=not near)
    for axis, value in zip(
        ("x", "y", "z"), NEAR_BIN_POSITION if near else config.bin_position, strict=True
    ):
        state.set(obj=objects.bin, feature_name=axis, feature_val=value)
    return state


class FakeKitchenEnvironment(KitchenRecyclingEnvironment):
    resets: list[int] = Field(default_factory=list)
    human_requests: list[str] = Field(default_factory=list)
    robot_actions: list = Field(default_factory=list)

    def model_post_init(self, __context):
        self.current_state = _state(
            config=self.config, seed=self.seed, training=self.phase == "training"
        )

    def get_current_state(self):
        return self.current_state.model_copy(deep=True)

    def reset_to_seed(self, *, seed):
        self.resets.append(seed)
        self.seed = seed
        self.current_state = _state(
            config=self.config, seed=seed, training=self.phase == "training"
        )
        return self.get_current_state()

    def reset_to_state(self, *, state):
        self.current_state = state.model_copy(deep=True)
        return self.get_current_state()

    def reset_movables(self, *, destination=None):
        assert self.phase == "training"
        self.human_requests.append(destination)
        item = self.objects.by_name[destination.split(":", 1)[1]]
        for feature, value in (
            ("settled", 0),
            ("blocked", 0),
            ("on_kitchen_floor", 1),
            ("reachable", 1),
        ):
            self.current_state.set(obj=item, feature_name=feature, feature_val=value)
        return True

    def take_action(self, *, action):
        self.robot_actions.append(action)
        opcode, item_index = map(int, action[:2])
        if opcode not in (0, 1):
            return self.get_current_state()
        item = self.items[item_index]
        for feature, value in (
            ("held", opcode == 0),
            ("settled", opcode == 1),
            ("on_kitchen_floor", 0),
        ):
            self.current_state.set(obj=item, feature_name=feature, feature_val=value)
        self.current_state.set(obj=self.robot, feature_name="hand_empty", feature_val=opcode == 1)
        self.current_state.set(obj=self.robot, feature_name="gripper_open", feature_val=opcode == 1)
        return self.get_current_state()


def _provider(*, phase="training", config=None):
    return KitchenRecyclingSkillProvider(
        env=FakeKitchenEnvironment(config=config or KitchenConfig(), phase=phase)
    )


def _abstract(*, provider, state):
    return SkillGrounder.abstract_state(
        state=state, objects=provider.objects(), predicates=provider.predicates()
    )


def _applicable(*, provider, state):
    return SkillGrounder.applicable_ground_skills(
        skills=provider.skills(),
        objects=provider.objects(),
        true_atoms=_abstract(provider=provider, state=state),
    )


def _ground(*, provider, name, item_name="gatorade_20oz"):
    skill = next(skill for skill in provider.skills() if skill.name == name)
    objects = provider.env.objects
    bound = (
        (objects.robot, objects.bin)
        if len(skill.parameters) == 2
        else (objects.robot, objects.by_name[item_name], objects.bin)
    )
    return GroundSkill(skill=skill, objects=bound)


def _apply(*, ground, atoms):
    assert ground.preconditions <= atoms
    ignored = ground.expected_ignore_effects(before_atoms=atoms)
    retained = frozenset(atom for atom in atoms if atom.predicate not in ignored)
    return (retained - ground.delete_effects) | ground.expected_add_effects(before_atoms=atoms)


def test_schema_copies_feature_arrays_and_checks_dimensions():
    objects = KitchenObjects(config=KitchenConfig())
    features = np.zeros(objects.robot.type.dim)
    state = KitchenState(data={objects.robot: features}, snapshot={"opaque": 1})
    features[:] = 42
    assert np.all(state[objects.robot] == 0)
    assert state.snapshot == {"opaque": 1}
    assert {"seed", "skills_taken", "gripper_open"} <= set(objects.robot.type.feature_names)
    assert "blocked" in objects.item_type.feature_names
    with pytest.raises(ValidationError):
        KitchenState(data={objects.robot: np.zeros(1)})


@pytest.mark.parametrize("near", [False, True])
def test_identity_and_context_pin_exactly_one_pick_per_object(*, near):
    provider = _provider()
    state = _state(near=near)
    applicable = _applicable(provider=provider, state=state)
    context = "near" if near else "far"
    assert {ground.skill.name for ground in applicable} == {
        f"Pick_{item.name}_{context}" for item in provider.env.items
    }
    assert len(applicable) == 3


def test_predicates_use_supplied_observation_not_live_environment():
    provider = _provider()
    observed = provider.env.get_current_state()
    before = _abstract(provider=provider, state=observed)
    provider.env.current_state.set(obj=provider.env.bin, feature_name="near", feature_val=1)
    assert _abstract(provider=provider, state=observed) == before
    assert _abstract(provider=provider, state=provider.env.get_current_state()) != before


def test_goal_requires_bottle_and_can_settled_and_bin_at_target():
    provider = _provider()
    tasks = KitchenRecyclingTasks(env=provider.env)
    state = _state()
    for name in provider.env.config.required_objects:
        state.set(obj=provider.env.objects.by_name[name], feature_name="settled", feature_val=1)
    assert tasks.goal().is_satisfied(state=state)
    assert len(tasks.goal().atoms) == 3
    state.set(obj=provider.env.bin, feature_name="at_target", feature_val=0)
    assert not tasks.goal().is_satisfied(state=state)
    state.set(obj=provider.env.bin, feature_name="at_target", feature_val=1)
    state.set(obj=provider.env.items[1], feature_name="settled", feature_val=0)
    assert not tasks.goal().is_satisfied(state=state)


def test_reachable_in_bin_retrieval_and_blocked_pick_refusal():
    provider = _provider()
    state = _state(near=True)
    item = provider.env.items[1]
    state.set(obj=item, feature_name="settled", feature_val=1)
    assert PICKABLE.holds(state, (item,))
    assert any(ground.objects[1] == item for ground in _applicable(provider=provider, state=state))
    state.set(obj=item, feature_name="blocked", feature_val=1)
    assert not PICKABLE.holds(state, (item,))
    assert all(ground.objects[1] != item for ground in _applicable(provider=provider, state=state))


@pytest.mark.parametrize("context", ["near", "far"])
def test_pick_is_fixed_and_toss_keeps_three_shared_learner_parameters(*, context):
    config = KitchenConfig(
        calibrations={
            f"gatorade_20oz/{context}": Calibration(grasp_offset=0.047, toss_azimuth=0.23)
        }
    )
    provider = _provider(config=config)
    state = _state(config=config, near=context == "near")
    rng = np.random.default_rng(7)
    pick = _ground(provider=provider, name=f"Pick_gatorade_20oz_{context}")
    assert provider.sample_params(ground_skill=pick, rng=rng).shape == (0,)
    np.testing.assert_allclose(
        provider.compute_action(ground_skill=pick, params=np.zeros(0), state=state),
        [0, 1, 0.047, 0, 0, 0],
    )
    toss = _ground(provider=provider, name=f"Toss_gatorade_20oz_{context}")
    assert provider.sample_params(ground_skill=toss, rng=rng).shape == (3,)
    np.testing.assert_allclose(
        provider.compute_action(ground_skill=toss, params=np.array([1.7, 140, 700]), state=state),
        [1, 1, 1.7, 0.23, 140, 700],
    )
    with pytest.raises(ValueError, match="dimension"):
        provider.compute_action(ground_skill=toss, params=np.zeros(4), state=state)
    wrong = GroundSkill(
        skill=toss.skill, objects=(provider.env.robot, provider.env.items[0], provider.env.bin)
    )
    with pytest.raises(ValueError, match="identity"):
        provider.compute_action(ground_skill=wrong, params=np.zeros(3), state=state)


def test_ees_has_distinct_parent_samplers_and_keeps_unparameterized_open_gripper():
    provider = _provider()
    method = EesMethod(env=provider.env, skill_provider=provider)
    tosses = [skill for skill in method.skills() if skill.name.startswith("Toss_")]
    assert len(tosses) == 6
    samplers = [
        method.sampler(skill_name=skill.name, param_dim=skill.param_dim) for skill in tosses
    ]
    assert len({id(sampler) for sampler in samplers}) == 6
    opens = [skill for skill in method.skills() if skill.name.startswith("OpenGripper")]
    assert len(opens) == 8
    assert all(skill.param_dim == 0 for skill in opens)
    assert not any("dump" in skill.name.lower() for skill in method.skills())


@pytest.mark.parametrize("sharing", ["grounded", "parent", "rate"])
def test_ees_never_pools_object_context_competence_examples_or_counts(*, sharing):
    provider = _provider()
    method = EesMethod(env=provider.env, skill_provider=provider, competence_sharing=sharing)
    grounds = [
        _ground(provider=provider, name=f"Toss_{item.name}_{context}", item_name=item.name)
        for item in provider.env.items
        for context in ("near", "far")
    ]
    practiced = grounds[0]
    method.observe_outcome(ground_skill=practiced, success=False)
    method.record_practice_attempt(
        skill_name=practiced.skill.name, success=False, consultation=SamplerConsultation.INFORMED
    )
    method.observe_sampler_outcome(
        skill_name=practiced.skill.name, param_dim=3, sampler_input=[1, 0, 0, 0], success=False
    )
    for index, ground in enumerate(grounds):
        expected = int(index == 0)
        assert method.forecast_competence_model(ground_skill=ground).num_observations == expected
        assert (
            method.sampler(skill_name=ground.skill.name, param_dim=3).num_observations == expected
        )
    assert set(method.practice_outcomes()) == {practiced.skill.name}


def test_near_toss_success_in_shallow_bin_is_not_mislabeled_as_failure():
    provider = _provider()
    ground = _ground(provider=provider, name="Toss_gatorade_20oz_near")
    after = _state(near=True)
    after.set(obj=provider.env.items[1], feature_name="settled", feature_val=1)
    assert ground.add_effects <= _abstract(provider=provider, state=after)


def test_sampler_features_exclude_seed_clock_and_training_phase():
    provider = _provider()
    ground = _ground(provider=provider, name="Toss_gatorade_20oz_far")
    first = _state(seed=1, training=True)
    second = _state(seed=99, training=False)
    second.set(obj=provider.env.robot, feature_name="skills_taken", feature_val=123)
    params = np.array([1.5, 130, 720])
    assert provider.hand_selected_feature_transform(
        ground_skill=ground, state=first, params=params
    ) == provider.hand_selected_feature_transform(ground_skill=ground, state=second, params=params)


@pytest.mark.parametrize("held", [False, True])
def test_open_gripper_is_applicable_for_held_and_closed_empty_states(*, held):
    provider = _provider()
    state = _state()
    state.set(obj=provider.env.robot, feature_name="hand_empty", feature_val=0)
    state.set(obj=provider.env.robot, feature_name="gripper_open", feature_val=0)
    state.set(obj=provider.env.items[1], feature_name="held", feature_val=held)
    name = "OpenGripper_gatorade_20oz_far" if held else "OpenGripperEmpty_far"
    opened = next(
        ground
        for ground in _applicable(provider=provider, state=state)
        if ground.skill.name == name
    )
    assert provider.compute_action(ground_skill=opened, params=np.zeros(0), state=state)[0] == 2


def test_human_requests_are_training_only_separate_and_cost_twelve():
    provider = _provider()
    humans = provider.human_cube_bin_reset_skills()
    assert len(humans) == 8
    assert all(ground.evaluate_practice_cost() == 12 for ground in humans)
    assert {provider.movables_reset_destination(ground_skill=ground) for ground in humans} == {
        "recover:cube_control",
        "recover:gatorade_20oz",
        "recover:food_can",
        "relocate_bin_near",
        "restore_bin_target",
    }
    assert _provider(phase="deployment").human_cube_bin_reset_skills() == ()
    provider.offer_human_reset = False
    assert provider.human_cube_bin_reset_skills() == ()
    provider.offer_human_reset = True
    deployment = _abstract(provider=provider, state=_state(training=False))
    assert all(not ground.preconditions <= deployment for ground in humans)
    for ground in humans:
        with pytest.raises(ValueError, match="robot"):
            provider.compute_action(ground_skill=ground, params=np.zeros(0), state=_state())


def test_human_recovery_dispatches_existing_protocol_without_robot_action():
    provider = _provider()
    item = provider.env.items[1]
    provider.env.current_state.set(obj=item, feature_name="on_kitchen_floor", feature_val=0)
    provider.env.current_state.set(obj=item, feature_name="reachable", feature_val=0)
    method = EesMethod(env=provider.env, skill_provider=provider)
    goal = KitchenRecyclingPredicates.goal_atoms(config=provider.env.config)
    episode = _EesEpisode(method=method, goal=goal, practicing=True)
    recovery = next(
        ground
        for ground in provider.human_cube_bin_reset_skills()
        if ground.skill.name == "Recover_gatorade_20oz_far"
    )
    episode._plan = [recovery]
    with pytest.raises(HumanCubeBinResetRequested) as raised:
        episode.step(state=provider.env.get_current_state())
    assert raised.value.destination == "recover:gatorade_20oz"
    assert raised.value.cost == 12
    assert provider.env.robot_actions == []
    tasks = KitchenRecyclingTasks(env=provider.env)
    problem = KitchenRecyclingProblem(env=provider.env, tasks=tasks, human=UnconditionalHumanOracle)
    problem.execute_movables_reset(destination=raised.value.destination)
    method.observe_help_granted(state=provider.env.get_current_state())
    assert provider.env.human_requests == ["recover:gatorade_20oz"]
    assert method.practice_outcomes()[recovery.skill.name].num_successes == 1


def test_relocation_and_restoration_do_not_claim_transporting_bin_contents():
    provider = _provider()
    bin_skills = [
        ground for ground in provider.human_cube_bin_reset_skills() if len(ground.objects) == 2
    ]
    assert len(bin_skills) == 2
    for ground in bin_skills:
        assert IN_BIN in ground.ignore_effects
        assert all(atom.predicate != IN_BIN for atom in ground.add_effects)
    restored = next(ground for ground in bin_skills if ground.skill.name == "RestoreBinTarget")
    assert any(atom.predicate == BIN_AT_TARGET for atom in restored.add_effects)


@pytest.mark.parametrize("variant", ["ees", "algorithm3", "curtis"])
@pytest.mark.parametrize("skill_name", ["RelocateBinNear", "RestoreBinTarget"])
@pytest.mark.parametrize("outcome", ["floor_release", "unreachable_release", "bin_not_moved"])
def test_human_bin_move_competence_ignores_incidental_item_outcomes(
    *, variant, skill_name, outcome, monkeypatch
):
    cube = KitchenConfig().object(name="cube_control")
    config = KitchenConfig(objects=(cube,), required_objects=(cube.name,))
    provider = _provider(config=config)
    before_near = skill_name == "RestoreBinTarget"
    before = _state(config=config, near=before_near)
    item = provider.env.items[0]
    before.set(obj=item, feature_name="settled", feature_val=1)
    before.set(obj=item, feature_name="on_kitchen_floor", feature_val=0)
    before.set(obj=item, feature_name="reachable", feature_val=before_near)
    for axis in ("x", "y", "z"):
        before.set(
            obj=item,
            feature_name=axis,
            feature_val=before.get(obj=provider.env.bin, feature_name=axis),
        )
    provider.env.current_state = before
    diagnostics = []

    def record(*args, **kwargs):
        del args
        diagnostics.append(kwargs)

    if variant == "ees":
        method = EesMethod(env=provider.env, skill_provider=provider)
    else:
        monkeypatch.setattr(Tossing3DPomdpMethod, "record_diagnostic", record)
        method = Tossing3DPomdpMethod(
            env=provider.env,
            skill_provider=provider,
            pomdp_domain=KitchenRecyclingPracticeDomain.for_provider(provider=provider),
            pomdp_solver="determinized_astar",
            pomdp_determinized_variant=variant,
            pomdp_num_particles=32,
            pomdp_num_samples=2,
            pomdp_search_depth=0,
        )
    ground = next(
        ground
        for ground in provider.human_cube_bin_reset_skills()
        if ground.skill.name == skill_name
    )
    assert ground.ignore_effects == frozenset({IN_BIN})
    assert all(atom.objects == (provider.env.bin,) for atom in ground.add_effects)
    episode = _EesEpisode(
        method=method, goal=KitchenRecyclingPredicates.goal_atoms(config=config), practicing=True
    )
    with pytest.raises(HumanCubeBinResetRequested):
        episode._dispatch_human_reset(
            ground_skill=ground, true_atoms=_abstract(provider=provider, state=before)
        )
    moved = outcome != "bin_not_moved"
    after = _state(config=config, near=not before_near if moved else before_near)
    for axis in ("x", "y", "z"):
        after.set(obj=item, feature_name=axis, feature_val=before.get(obj=item, feature_name=axis))
    after.set(
        obj=item, feature_name="on_kitchen_floor", feature_val=outcome != "unreachable_release"
    )
    after.set(obj=item, feature_name="reachable", feature_val=outcome != "unreachable_release")
    method.observe_help_granted(state=after)
    tally = method.practice_outcomes()[skill_name]
    assert tally.num_attempts == 1
    assert tally.num_successes == int(moved)
    if variant != "ees":
        result = next(event for event in diagnostics if event["event"] == "outcome")
        assert result["success"] == moved
    if skill_name == "RestoreBinTarget" and outcome == "floor_release":
        observed = _abstract(provider=provider, state=after)
        forecast = _apply(ground=ground, atoms=_abstract(provider=provider, state=before))
        assert observed != forecast
        assert PICKABLE.holds(after, (item,))
        assert any(atom.predicate == PICKABLE and atom.objects == (item,) for atom in forecast)


@pytest.mark.parametrize("near", [False, True])
def test_toss_forecast_enables_retrieval_only_with_a_geometric_certificate(*, near):
    provider = _provider()
    context = "near" if near else "far"
    atoms = _abstract(provider=provider, state=_state(near=near))
    picked = _ground(provider=provider, name=f"Pick_gatorade_20oz_{context}")
    thrown = _ground(provider=provider, name=f"Toss_gatorade_20oz_{context}")
    after = _apply(ground=thrown, atoms=_apply(ground=picked, atoms=atoms))
    assert (picked.preconditions <= after) == near


def test_near_label_alone_does_not_certify_reachability():
    config = KitchenConfig(near_distance=2.0)
    provider = _provider(config=config)
    state = _state(config=config, near=True)
    state.set(obj=provider.env.bin, feature_name="x", feature_val=config.bin_position[0])
    predicate = bin_retrievable_predicate(config=config)
    assert not predicate.holds(state, (provider.env.bin,))
    picked = _ground(provider=provider, name="Pick_gatorade_20oz_near")
    thrown = _ground(provider=provider, name="Toss_gatorade_20oz_near")
    after = _apply(
        ground=thrown, atoms=_apply(ground=picked, atoms=_abstract(provider=provider, state=state))
    )
    assert not picked.preconditions <= after


def test_bin_motion_updates_retrieval_certificate_without_moving_item_facts():
    provider = _provider()
    humans = {ground.skill.name: ground for ground in provider.human_cube_bin_reset_skills()}
    atoms = _abstract(provider=provider, state=_state())
    near = _apply(ground=humans["RelocateBinNear"], atoms=atoms)
    predicate = bin_retrievable_predicate(config=provider.env.config)
    assert any(atom.predicate == predicate for atom in near)
    assert {atom for atom in atoms if atom.predicate == PICKABLE} <= near
    far = _apply(ground=humans["RestoreBinTarget"], atoms=near)
    assert not any(atom.predicate == predicate for atom in far)
    assert {atom for atom in atoms if atom.predicate == PICKABLE} <= far


def test_bin_relocation_does_not_certify_reach_when_config_cannot_support_it():
    provider = _provider(config=KitchenConfig(direct_placement_reach=0.01))
    relocation = next(
        ground
        for ground in provider.human_cube_bin_reset_skills()
        if ground.skill.name == "RelocateBinNear"
    )
    predicate = bin_retrievable_predicate(config=provider.env.config)
    assert not any(atom.predicate == predicate for atom in relocation.add_effects)


@pytest.mark.parametrize("context", ["near", "far"])
@pytest.mark.parametrize("name", ["cube_control", "gatorade_20oz", "food_can"])
def test_successful_held_open_gripper_forecasts_safe_floor_release_and_repick(*, context, name):
    provider = _provider()
    state = _state(near=context == "near")
    picked = _ground(provider=provider, name=f"Pick_{name}_{context}", item_name=name)
    opened = _ground(provider=provider, name=f"OpenGripper_{name}_{context}", item_name=name)
    atoms = _apply(ground=picked, atoms=_abstract(provider=provider, state=state))
    after = _apply(ground=opened, atoms=atoms)
    assert picked.preconditions <= after
    assert {atom.predicate.name for atom in opened.add_effects} == {
        "HandEmpty",
        "OnKitchenFloor",
        "Pickable",
    }
    assert {atom.predicate.name for atom in opened.delete_effects} == {
        "Holding",
        "HandOccupied",
        "InBin",
        "NeedsRecovery",
    }
    assert opened.add_effects <= _abstract(provider=provider, state=state)
    assert not opened.delete_effects & _abstract(provider=provider, state=state)
    assert picked in _applicable(provider=provider, state=state)


@pytest.mark.parametrize("context", ["near", "far"])
@pytest.mark.parametrize(
    "failure", [None, "off_floor", "unreachable", "blocked", "held", "in_bin", "closed"]
)
def test_ees_scores_held_open_gripper_from_actual_safe_release_postconditions(
    *, context, failure, monkeypatch
):
    cube = KitchenConfig().object(name="cube_control")
    config = KitchenConfig(objects=(cube,), required_objects=(cube.name,))
    provider = _provider(config=config)
    method = EesMethod(env=provider.env, skill_provider=provider)
    opened = _ground(
        provider=provider, name=f"OpenGripper_cube_control_{context}", item_name=cube.name
    )
    before = _state(config=config, near=context == "near")
    before.set(obj=provider.env.items[0], feature_name="held", feature_val=1)
    before.set(obj=provider.env.items[0], feature_name="on_kitchen_floor", feature_val=0)
    before.set(obj=provider.env.robot, feature_name="hand_empty", feature_val=0)
    before.set(obj=provider.env.robot, feature_name="gripper_open", feature_val=0)
    episode = _EesEpisode(
        method=method, goal=KitchenRecyclingPredicates.goal_atoms(config=config), practicing=True
    )
    episode._plan = [opened]
    action = episode.step(state=before)
    assert action.action[0] == 2
    after = _state(config=config, near=context == "near")
    failures = {
        "off_floor": ("on_kitchen_floor", 0),
        "unreachable": ("reachable", 0),
        "blocked": ("blocked", 1),
        "held": ("held", 1),
        "in_bin": ("settled", 1),
    }
    if failure in failures:
        feature, value = failures[failure]
        after.set(obj=provider.env.items[0], feature_name=feature, feature_val=value)
    if failure in ("held", "closed"):
        after.set(obj=provider.env.robot, feature_name="hand_empty", feature_val=0)
        after.set(obj=provider.env.robot, feature_name="gripper_open", feature_val=0)
    transitions = []

    def capture(*args, **kwargs):
        del args
        transitions.append(kwargs)

    monkeypatch.setattr(EesMethod, "observe_symbolic_transition", capture)
    observed = _abstract(provider=provider, state=after)
    episode.observe_pending(true_atoms=observed, state=after)
    tally = method.practice_outcomes()[opened.skill.name]
    assert tally.num_attempts == 1
    assert tally.num_successes == int(failure is None)
    assert tally.num_unparameterized_attempts == 1
    assert transitions[0]["success"] == (failure is None)
    assert transitions[0]["after_atoms"] == observed
    assert _abstract(provider=provider, state=after) == observed


@pytest.mark.parametrize("context", ["near", "far"])
def test_closed_empty_open_gripper_contract_is_unchanged(*, context):
    provider = _provider()
    opened = _ground(provider=provider, name=f"OpenGripperEmpty_{context}")
    assert {atom.predicate.name for atom in opened.add_effects} == {"HandEmpty"}
    assert {atom.predicate.name for atom in opened.delete_effects} == {
        "HandOccupied",
        "ClosedEmpty",
    }
    assert opened.skill.param_dim == 0


@pytest.mark.parametrize("context", ["near", "far"])
def test_planner_uses_held_open_gripper_then_autonomous_repick_without_human(*, context):
    if not Path(FastDownwardPlanner.fd_dir(), "fast-downward.py").exists():
        pytest.skip("Fast Downward is not installed")
    cube = KitchenConfig().object(name="cube_control")
    config = KitchenConfig(objects=(cube,), required_objects=(cube.name,))
    provider = _provider(config=config)
    picked = _ground(provider=provider, name=f"Pick_cube_control_{context}", item_name=cube.name)
    opened = _ground(
        provider=provider, name=f"OpenGripper_cube_control_{context}", item_name=cube.name
    )
    atoms = _abstract(provider=provider, state=_state(config=config, near=context == "near"))
    held = _apply(ground=picked, atoms=atoms)
    arguments = dict(
        skills=provider.skills(),
        predicates=provider.predicates(),
        types=provider.types(),
        objects=provider.objects(),
    )
    release = FastDownwardPlanner.plan(**arguments, init_atoms=held, goal=opened.add_effects)
    assert release == [opened]
    after = _apply(ground=release[0], atoms=held)
    repick = FastDownwardPlanner.plan(**arguments, init_atoms=after, goal=picked.add_effects)
    assert repick == [picked]


def test_planner_can_retrieve_after_near_toss_but_must_retoss_after_restoring_bin():
    if not Path(FastDownwardPlanner.fd_dir(), "fast-downward.py").exists():
        pytest.skip("Fast Downward is not installed")
    cube = KitchenConfig().object(name="cube_control")
    config = KitchenConfig(objects=(cube,), required_objects=(cube.name,))
    provider = _provider(config=config)
    picked = _ground(provider=provider, name="Pick_cube_control_near", item_name=cube.name)
    thrown = _ground(provider=provider, name="Toss_cube_control_near", item_name=cube.name)
    near = _abstract(provider=provider, state=_state(config=config, near=True))
    settled = _apply(ground=thrown, atoms=_apply(ground=picked, atoms=near))
    arguments = dict(
        predicates=provider.predicates(), types=provider.types(), objects=provider.objects()
    )
    held = frozenset(atom for atom in picked.add_effects if atom.predicate == HOLDING)
    retrieval = FastDownwardPlanner.plan(
        **arguments, skills=provider.skills(), init_atoms=settled, goal=held
    )
    assert [ground.skill.name for ground in retrieval] == ["Pick_cube_control_near"]
    goal = KitchenRecyclingPredicates.goal_atoms(config=config)
    with pytest.raises(PlanningFailure):
        FastDownwardPlanner.plan(
            **arguments, skills=provider.skills(), init_atoms=settled, goal=goal
        )
    restored = FastDownwardPlanner.plan(
        **arguments,
        skills=(
            *provider.skills(),
            *(ground.skill for ground in provider.human_cube_bin_reset_skills()),
        ),
        init_atoms=settled,
        goal=goal,
    )
    assert [ground.skill.name for ground in restored] == [
        "RestoreBinTarget",
        "Pick_cube_control_far",
        "Toss_cube_control_far",
    ]


def test_task_streams_are_independent_and_never_sampling_does_not_reset_or_draw():
    first = KitchenRecyclingTasks(env=FakeKitchenEnvironment(), seed=13)
    second = KitchenRecyclingTasks(env=FakeKitchenEnvironment(), seed=13)
    first.env.current_state.set(obj=first.env.items[1], feature_name="blocked", feature_val=1)
    in_place = first.sample_train_task_in_place()
    assert in_place.initial_state.get(obj=first.env.items[1], feature_name="blocked") == 1
    assert first.env.resets == []
    first.sample_test_task()
    first.sample_test_task()
    first.sample_train_task()
    second.sample_train_task()
    assert first.env.resets[-1] == second.env.resets[-1]
    second.sample_test_task()
    assert first.env.resets[0] == second.env.resets[-1]


def test_calibration_oracle_and_problem_run_both_required_objects_only():
    provider = _provider(phase="deployment")
    tasks = KitchenRecyclingTasks(env=provider.env)
    problem = KitchenRecyclingProblem(env=provider.env, tasks=tasks)
    task = tasks.sample_test_task()
    oracle = KitchenRecyclingOracle(env=provider.env)
    success, frames, trace = problem.run_task_episode(
        task=task, policy=lambda state: oracle.get_labeled_action(state=state, goal=task.goal)
    )
    assert success
    assert frames == []
    assert len(trace.actions) == 4
    assert len(trace.states) == 5
    assert {int(action.action[1]) for action in trace.actions} == {1, 2}
    assert all(
        not HOLDING.holds(trace.states[-1], (provider.env.robot, item))
        for item in provider.env.items
    )


def test_fast_downward_solves_both_items_with_four_robot_skills():
    if not Path(FastDownwardPlanner.fd_dir(), "fast-downward.py").exists():
        pytest.skip("Fast Downward is not installed")
    provider = _provider(phase="deployment")
    state = provider.env.get_current_state()
    task = Task(initial_state=state, goal=KitchenRecyclingTasks(env=provider.env).goal())
    plan = FastDownwardPlanner.plan(
        skills=provider.skills(),
        predicates=provider.predicates(),
        types=provider.types(),
        objects=provider.objects(),
        init_atoms=_abstract(provider=provider, state=state),
        goal=task.goal.atoms,
    )
    assert len(plan) == 4
    assert {ground.objects[1].name for ground in plan} == {"gatorade_20oz", "food_can"}


@pytest.mark.parametrize("name", ["cube_control", "gatorade_20oz", "food_can"])
def test_single_object_configuration_limits_objects_skills_tasks_and_action_indices(*, name):
    original = KitchenConfig()
    config = KitchenConfig(objects=(original.object(name=name),), required_objects=(name,))
    provider = _provider(config=config)
    assert len(provider.env.items) == 1
    assert {obj.name for obj in provider.objects()} == {"robot", "bin", name}
    assert len(provider.skills()) == 8
    assert len(provider.human_cube_bin_reset_skills()) == 4
    excluded = {spec.name for spec in original.objects} - {name}
    all_skills = (
        *provider.skills(),
        *(ground.skill for ground in provider.human_cube_bin_reset_skills()),
    )
    assert all(other not in skill.name for skill in all_skills for other in excluded)
    assert {
        predicate.name for predicate in provider.predicates() if predicate.name.startswith("Is_")
    } == {f"Is_{name}"}
    state = provider.env.get_current_state()
    applicable = _applicable(provider=provider, state=state)
    assert len(applicable) == 1
    assert applicable[0].skill.name == f"Pick_{name}_far"
    action = provider.compute_action(ground_skill=applicable[0], params=np.zeros(0), state=state)
    assert action[1] == 0
    train_tasks = KitchenRecyclingTasks(env=provider.env)
    train = train_tasks.sample_train_task_in_place()
    assert len(train.goal.atoms) == 2
    assert set(train.initial_state.data) == set(provider.objects())
    evaluation = _provider(config=config, phase="deployment")
    test_tasks = KitchenRecyclingTasks(env=evaluation.env)
    task = test_tasks.sample_test_task()
    assert train.goal == task.goal
    assert len(task.initial_state.data) == 3
    problem = KitchenRecyclingProblem(env=evaluation.env, tasks=test_tasks)
    oracle = KitchenRecyclingOracle(env=evaluation.env)
    success, _, trace = problem.run_task_episode(
        task=task, policy=lambda state: oracle.get_labeled_action(state=state, goal=task.goal)
    )
    assert success
    assert [int(labeled.action[1]) for labeled in trace.actions] == [0, 0]
    assert len(trace.actions) == 2
