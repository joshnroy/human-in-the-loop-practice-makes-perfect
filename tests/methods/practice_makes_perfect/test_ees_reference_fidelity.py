"""EES reference audit: vendor 5bd3f5bd and release 5c5307e.

Both references cache hypothetical-cost plans per (train/replan, index), including
failed plans; deque positions deliberately retain cached plans after eviction.
Both sum costs and normalize by ALL distinct seen training indices only with UCB.
The release observes every outcome once, plus non-exploration outcomes again;
the vendor observes only non-exploration outcomes once. Neither implements human
resets. The existing double-observe flag selects these two different behaviors.

The port has no upstream training index: retained Task object identity is its
local substitute. Distinct tasks with equal symbolic states must not collapse.
The legacy recent-training-subset default remains configurable; reference order
requires reproduce_predicators_seen_task_order=True. All tests use robot costs
only, including paramless skills, with no execution/reset price in target scores.
"""

import ast
import json
import logging
import math
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from hitl_pmp.core.method.method import HumanCubeBinResetRequested
from hitl_pmp.core.method.types import GroundSkill, LabeledAction, Skill
from hitl_pmp.core.problem.tasks.types import Task
from hitl_pmp.environments.lightswitch.environment import LightSwitchEnvironment
from hitl_pmp.environments.lightswitch.predicates import ROBOT_IN_CELL
from hitl_pmp.environments.lightswitch.skill_provider import LightSwitchSkillProvider
from hitl_pmp.environments.lightswitch.tasks import LightSwitchTasks
from hitl_pmp.methods.practice_makes_perfect.ees_method import EesMethod, _EesEpisode
from hitl_pmp.planning.fast_downward import PlanningFailure
from hitl_pmp.results_writer.jsonl_log import JsonlLog


def build(**kwargs):
    environment = LightSwitchEnvironment(grid_size=4)
    return EesMethod(
        env=environment,
        skill_provider=LightSwitchSkillProvider(env=environment),
        **kwargs,
    )


def action(*, name):
    return GroundSkill(
        skill=Skill(
            name=name,
            parameters=(),
            preconditions=frozenset(),
            add_effects=frozenset(),
            delete_effects=frozenset(),
            param_dim=0,
            practice_cost=1.0,
        ),
        objects=(),
    )


def two_routes(*, monkeypatch, **kwargs):
    method = build(use_ucb_bonus=False, **kwargs)
    safe = action(name="existing route")
    promising = action(name="alternative route")
    method.competence_model(ground_skill=safe)
    method.competence_model(ground_skill=promising)
    current = {safe: 0.6, promising: 0.4}
    predicted = {safe: 0.61, promising: 0.9}
    for ground in current:
        model = method.competence_model(ground_skill=ground)
        model.alpha = 10 * current[ground]
        model.beta = 10 * (1 - current[ground])
    monkeypatch.setitem(
        method.__dict__,
        "forecast_competence_gain",
        lambda **kwargs: predicted[kwargs["ground_skill"]] - current[kwargs["ground_skill"]],
    )
    calls = []

    def plan(**kwargs):
        calls.append(kwargs)
        return [min(kwargs["costs"], key=kwargs["costs"].get)]

    monkeypatch.setitem(method.__dict__, "plan_to", plan)
    method.record_seen_task(init_atoms=frozenset(), goal=frozenset())
    return method, safe, promising, calls


def test_cold_cache_plans_with_candidate_hypothesis(*, monkeypatch):
    method, _safe, promising, calls = two_routes(monkeypatch=monkeypatch)
    assert method.score_ground_skill(ground_skill=promising) == pytest.approx(math.log(0.9))
    assert len(calls) == 1
    assert calls[0]["costs"][promising] == pytest.approx(-math.log(0.9))


def test_shared_cache_is_not_fresh_replanning_for_every_candidate(*, monkeypatch):
    method, safe, promising, calls = two_routes(monkeypatch=monkeypatch, replan_frequency=2)
    assert method.score_ground_skill(ground_skill=safe) == pytest.approx(math.log(0.61))
    assert method.score_ground_skill(ground_skill=promising) == pytest.approx(math.log(0.6))
    assert len(calls) == 1
    assert method.score_ground_skill(ground_skill=promising) == pytest.approx(math.log(0.9))
    assert len(calls) == 2


def test_new_task_gets_its_own_cold_cache(*, monkeypatch):
    method, safe, promising, calls = two_routes(monkeypatch=monkeypatch)
    method.score_ground_skill(ground_skill=safe)
    method.record_seen_task(init_atoms=frozenset(), goal=frozenset())
    assert method.score_ground_skill(ground_skill=promising) == pytest.approx(
        math.log(0.6) + math.log(0.9)
    )
    assert len(calls) == 2


@pytest.mark.parametrize("use_ucb", [False, True])
def test_normalization_counts_all_seen_tasks_not_successful_or_capped_plans(
    *, monkeypatch, use_ucb
):
    method, safe, _promising, calls = two_routes(
        monkeypatch=monkeypatch, planning_progress_max_tasks=2
    )
    method.use_ucb_bonus = use_ucb
    method.explore_bonus = 0
    method.record_seen_task(init_atoms=frozenset(), goal=frozenset())
    method.record_seen_task(init_atoms=frozenset(), goal=frozenset())
    method._replanning_tasks.append((frozenset(), frozenset()))

    def plan(**kwargs):
        calls.append(kwargs)
        if len(calls) == 2:
            raise PlanningFailure("unreachable")
        return [safe]

    monkeypatch.setitem(method.__dict__, "plan_to", plan)
    expected = 2 * math.log(0.61) / (3 if use_ucb else 1)
    assert method.score_ground_skill(ground_skill=safe) == pytest.approx(expected)
    assert method.score_ground_skill(ground_skill=safe) == pytest.approx(expected)
    assert len(calls) == 3


def test_seen_training_identity_is_distinct_and_recorded_on_policy_execution(*, monkeypatch):
    method = build()
    task = LightSwitchTasks(env=method.env, seed=0).sample_train_task()
    other = Task(initial_state=task.initial_state, goal=task.goal)
    monkeypatch.setattr(_EesEpisode, "step", lambda self, **kwargs: None)
    first_policy = method.get_practice_policy(task=task)
    assert not method._seen_tasks
    first_policy(task.initial_state)
    first_policy(task.initial_state)
    method.get_practice_policy(task=task)(task.initial_state)
    assert len(method._seen_tasks) == 1
    method.get_practice_policy(task=other)(other.initial_state)
    assert len(method._seen_tasks) == 2


def test_replanning_queue_records_nonempty_setup_goals_before_success_or_failure(*, monkeypatch):
    method = build()
    state = method.env.build_initial_state(light_level=0, light_target=0.5)
    target = next(
        ground for ground in method.all_practice_groundings() if ground.skill.name == "TurnOnLight"
    )
    monkeypatch.setitem(method.__dict__, "choose_practice_target", lambda: [target])
    monkeypatch.setitem(method.__dict__, "random_choice", lambda **kwargs: target)

    def failed(**kwargs):
        assert method._replanning_tasks[-1] == (kwargs["init_atoms"], kwargs["goal"])
        raise PlanningFailure("unreachable")

    monkeypatch.setitem(method.__dict__, "plan_to", failed)
    episode = _EesEpisode(method=method, goal=frozenset(), practicing=True)
    episode._practice_plan(true_atoms=frozenset())
    assert len(method._replanning_tasks) == 1
    monkeypatch.setitem(method.__dict__, "plan_to", lambda **kwargs: [])
    episode._practice_plan(true_atoms=target.preconditions)
    assert len(method._replanning_tasks) == 2
    assert method._replanning_tasks[-1] == (target.preconditions, target.preconditions)
    empty = action(name="OpenGripper")
    monkeypatch.setitem(method.__dict__, "choose_practice_target", lambda: [empty])
    episode._practice_plan(true_atoms=frozenset())
    assert len(method._replanning_tasks) == 2
    goals = [
        frozenset({ROBOT_IN_CELL(state=state, objects=(method.env.robot, cell))})
        for cell in method.env.get_cells()
    ]
    for goal in goals:
        _EesEpisode(method=method, goal=goal, practicing=True)._next_plan(true_atoms=frozenset())
    assert method._replanning_tasks.maxlen == 5
    assert len(method._replanning_tasks) == 5
    assert [goal for _init, goal in method._replanning_tasks][-4:] == goals


def test_evaluation_and_hypothetical_plans_do_not_record_replanning_tasks(*, monkeypatch):
    method, safe, _promising, _calls = two_routes(monkeypatch=monkeypatch)
    method.score_ground_skill(ground_skill=safe)
    state = method.env.build_initial_state(light_level=0, light_target=0.5)
    goal = frozenset({
        ROBOT_IN_CELL(state=state, objects=(method.env.robot, method.env.get_cells()[0]))
    })
    _EesEpisode(method=method, goal=goal, practicing=False)._next_plan(true_atoms=frozenset())
    assert not method._replanning_tasks


def test_replanning_cache_uses_reference_deque_positions_after_eviction(*, monkeypatch):
    method, safe, promising, calls = two_routes(monkeypatch=monkeypatch)
    method._replanning_tasks.extend([(frozenset(), frozenset())] * 5)
    method.score_ground_skill(ground_skill=safe)
    assert len(calls) == 6
    state = method.env.build_initial_state(light_level=0, light_target=0.5)
    new_goal = frozenset({
        ROBOT_IN_CELL(state=state, objects=(method.env.robot, method.env.get_cells()[0]))
    })
    method._replanning_tasks.append((frozenset(), new_goal))
    assert method.score_ground_skill(ground_skill=promising) == pytest.approx(6 * math.log(0.6))
    assert len(calls) == 6


def test_end_cycle_clears_explorer_caches_without_baseline_planning(*, monkeypatch):
    method, safe, promising, calls = two_routes(monkeypatch=monkeypatch)
    method._replanning_tasks.append((frozenset(), frozenset()))
    method.score_ground_skill(ground_skill=safe)
    assert len(calls) == 2
    method.end_cycle()
    assert len(calls) == 2
    assert not method._replanning_tasks
    assert method.score_ground_skill(ground_skill=promising) == pytest.approx(math.log(0.9))
    assert len(calls) == 3


@pytest.mark.parametrize("suffix", [".jsonl", ".jsonl.zst"])
def test_decision_logging_does_not_change_scores_planner_calls_or_rng(
    *, monkeypatch, tmp_path, suffix
):
    path = tmp_path / f"ees{suffix}"
    plain, _safe, _promising, plain_calls = two_routes(monkeypatch=monkeypatch)
    logged, _safe, _promising, logged_calls = two_routes(
        monkeypatch=monkeypatch, ees_decision_log=path
    )
    assert plain.choose_practice_target() == logged.choose_practice_target()
    assert plain_calls == logged_calls
    assert plain._rng.bit_generator.state == logged._rng.bit_generator.state
    with JsonlLog.open_text(path=path) as stream:
        records = [json.loads(line) for line in stream]
    scores = [record for record in records if record["event"] == "score"]
    assert len(scores) == 2
    assert scores[0]["base_score"] == pytest.approx(math.log(0.61))
    assert scores[0]["cache_refreshes"] == 1
    assert scores[1]["cache_refreshes"] == 0
    assert scores[0]["normalization_denominator"] == 1
    assert records[-1]["event"] == "ranking"


def test_goal_horizon_rejects_negative_values():
    with pytest.raises(ValidationError, match="goal_pursuit_horizon"):
        build(goal_pursuit_horizon=-1)


@pytest.mark.parametrize("human", [False, True])
def test_dispatch_diagnostics_show_actual_setup_and_never_evaluation(
    *, monkeypatch, tmp_path, human
):
    path = tmp_path / "nested" / "ees.jsonl"
    method = build(ees_decision_log=path, goal_pursuit_horizon=5)
    setup = action(name="reset" if human else "PickCube")
    target = action(name="Toss")
    state = method.env.build_initial_state(light_level=0, light_target=0.5)
    monkeypatch.setitem(
        method.__dict__,
        "execute_ground_skill",
        lambda **kwargs: (LabeledAction(action=np.zeros(2), label="setup"), None),
    )
    monkeypatch.setattr(
        LightSwitchSkillProvider, "is_movables_reset_skill", lambda self, **kwargs: human
    )
    monkeypatch.setattr(
        LightSwitchSkillProvider, "movables_reset_destination", lambda self, **kwargs: "robot_side"
    )
    for practicing in (True, False):
        episode = _EesEpisode(method=method, goal=frozenset(), practicing=practicing)
        episode._goal_phase_done = True
        episode._plan = [setup, target]
        episode._practice_target = target
        if human:
            with pytest.raises(HumanCubeBinResetRequested):
                episode.step(state=state)
        else:
            episode.step(state=state)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["event"] == "dispatch"
    assert records[0]["phase"] == "practice_setup"
    assert records[0]["ground_skill"]["skill"] == setup.skill.name
    assert records[0]["target"]["skill"] == "Toss"
    assert records[0]["human"] is human
    assert records[0]["goal_pursuit_remaining"] == 5
    assert records[0]["true_atoms"]


@pytest.mark.parametrize("phase", ["goal", "practice_target", "random"])
def test_dispatch_diagnostics_preserve_phase_target_and_horizon(*, monkeypatch, tmp_path, phase):
    path = tmp_path / "ees.jsonl"
    method = build(ees_decision_log=path, goal_pursuit_horizon=5)
    target = action(name="OpenGripper")
    state = method.env.build_initial_state(light_level=0, light_target=0.5)
    action_result = (LabeledAction(action=np.zeros(2), label="open"), None)
    monkeypatch.setitem(method.__dict__, "execute_ground_skill", lambda **kwargs: action_result)
    monkeypatch.setitem(
        method.__dict__, "execute_random_ground_skill", lambda **kwargs: action_result
    )
    episode = _EesEpisode(method=method, goal=frozenset(), practicing=True)
    episode._goal_phase_done = phase != "goal"
    episode._random_mode = phase == "random"
    episode._plan = [target]
    episode._practice_target = target if phase == "practice_target" else None
    episode.step(state=state)
    record = json.loads(path.read_text())
    assert record["event"] == "dispatch"
    assert record["phase"] == phase
    assert record["goal_pursuit_remaining"] == (4 if phase == "goal" else 5)
    expected_target = {"skill": "OpenGripper", "objects": []}
    assert record["ground_skill"] == expected_target
    assert record["target"] == (expected_target if phase == "practice_target" else None)


def test_perfect_candidates_log_null_scores_without_dropping_targets(*, monkeypatch, tmp_path):
    path = tmp_path / "ees.jsonl"
    method, safe, promising, calls = two_routes(monkeypatch=monkeypatch, ees_decision_log=path)
    method.reproduce_predicators_skip_perfect = True
    for candidate in (safe, promising):
        method.observe_outcome(ground_skill=candidate, success=True)
    assert set(method.choose_practice_target()) == {safe, promising}
    assert calls == []
    record = json.loads(path.read_text())
    assert record["event"] == "ranking"
    assert len(record["candidates"]) == 2
    assert all(candidate["score"] is None for candidate in record["candidates"])
    assert all(candidate["declined_perfect"] for candidate in record["candidates"])


def reference_class(*, source_name, frequency):
    """Execute the actual audited methods, not a handwritten reference replica.

    Reference checkouts are optional locally; the self-contained regressions
    above remain mandatory when neither third-party source is installed.
    """
    if source_name == "release":
        source = Path("/tmp/ees-release-audit/active_sampler_explorer.py")
    else:
        relative = Path("reference/predicators/predicators/explorers/active_sampler_explorer.py")
        source = next(
            (
                parent / relative
                for parent in Path(__file__).resolve().parents
                if (parent / relative).is_file()
            ),
            None,
        )
    if source is None or not source.is_file():
        pytest.skip(f"optional audited {source_name} source unavailable")
    tree = ast.parse(source.read_text())
    explorer = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    names = {
        "_score_ground_op_planning_progress",
        "_get_task_plan_for_task",
        "_update_ground_op_hist",
    }
    methods = [
        node for node in explorer.body if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert len(methods) == len(names)
    for method in methods:
        method.returns = None
        for argument in method.args.args:
            argument.annotation = None
    extracted = ast.Module(
        body=[
            ast.ClassDef(name="Reference", bases=[], keywords=[], body=methods, decorator_list=[])
        ],
        type_ignores=[],
    )
    configuration = SimpleNamespace(
        skill_competence_model_lookahead=1,
        active_sampler_explorer_skip_perfect=False,
        active_sampler_explorer_planning_progress_max_tasks=10,
        active_sampler_explorer_replan_frequency=frequency,
        active_sampler_learning_save_every_datum=False,
        timeout=1,
        sesame_task_planning_heuristic="unused",
    )

    def planner(*args, **kwargs):
        costs = kwargs["ground_op_costs"]
        return [SimpleNamespace(op=min(costs, key=costs.get))], None, None

    namespace = {
        "np": np,
        "logging": logging,
        "CFG": configuration,
        "run_task_plan_once": planner,
        "DefaultState": object(),
        "PlanningFailure": PlanningFailure,
        "PlanningTimeout": TimeoutError,
    }
    exec(compile(ast.fix_missing_locations(extracted), str(source), "exec"), namespace)
    return namespace["Reference"]


class ReferenceAction:
    def __init__(self, *, name):
        self.short_str = name


@pytest.mark.parametrize("source_name", ["vendor", "release"])
@pytest.mark.parametrize("promising_first", [False, True])
def test_actual_reference_scoring_matches_shared_caches(
    *, monkeypatch, source_name, promising_first
):
    reference = reference_class(source_name=source_name, frequency=2)()
    method, safe, promising, _calls = two_routes(monkeypatch=monkeypatch, replan_frequency=2)
    reference_safe = ReferenceAction(name="existing route")
    reference_promising = ReferenceAction(name="alternative route")
    reference._competence_models = {
        reference_safe: SimpleNamespace(
            get_current_competence=lambda: 0.6, predict_competence=lambda count: 0.61
        ),
        reference_promising: SimpleNamespace(
            get_current_competence=lambda: 0.4, predict_competence=lambda count: 0.9
        ),
    }
    reference._ground_op_hist = {ground: [True, False] for ground in reference._competence_models}
    reference._seen_train_task_idxs = {0}
    reference._train_tasks = [SimpleNamespace(init=object())]
    reference._replanning_tasks = deque([SimpleNamespace(init=object())], maxlen=5)
    reference._task_plan_calls_since_replan = {}
    reference._task_plan_cache = {}
    reference._nsrts = reference._predicates = reference._types = set()
    reference._seed = 0
    reference._default_cost = method.default_cost()
    method._replanning_tasks.append((frozenset(), frozenset()))
    sequence = [(safe, reference_safe), (promising, reference_promising)]
    if promising_first:
        sequence.reverse()
    for ground, reference_ground in sequence * 3:
        assert method.score_ground_skill(ground_skill=ground) == pytest.approx(
            reference._score_ground_op_planning_progress(reference_ground)
        )
        assert method._task_plan_calls_since_replan == reference._task_plan_calls_since_replan


@pytest.mark.parametrize("source_name", ["vendor", "release"])
@pytest.mark.parametrize("random", [False, True])
def test_observation_flag_matches_actual_reference_update(*, source_name, random):
    reference = reference_class(source_name=source_name, frequency=100)()
    ground = ReferenceAction(name="OpenGripper")
    observations = []
    reference._last_executed_nsrt = SimpleNamespace(
        op=ground, short_str=ground.short_str, add_effects=[]
    )
    reference._last_executed_nsrt_was_exploration = random
    reference._ground_op_hist = {}
    reference._competence_models = {ground: SimpleNamespace(observe=observations.append)}
    reference._update_ground_op_hist(object())
    method = build(reproduce_predicators_double_observe=source_name == "release")
    candidate = action(name="OpenGripper")
    method.observe_outcome(ground_skill=candidate, success=True, was_random_exploration=random)
    assert method.competence_model(ground_skill=candidate).num_observations == len(observations)
    assert method._all_attempt_outcomes[candidate] == reference._ground_op_hist[ground]
