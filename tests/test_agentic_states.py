# ruff: noqa: PLR0917 -- pytest fixtures
"""Language state routing must replace, rather than decorate, predicates."""

import json

import pytest

from hitl_pmp.agentic_states.budget import SharedBudget
from hitl_pmp.agentic_states.graph import LanguageManifest, StateGraph


def manifest():
    return LanguageManifest.model_validate({
        "revision": "r1",
        "clusters": [
            {"cluster_id": "loose", "description": "cube is loose"},
            {"cluster_id": "held", "description": "cube is held"},
            {"cluster_id": "goal", "description": "cube is delivered"},
        ],
        "goal_clusters": ["goal"],
        "edges": [
            {"source": "loose", "action": "PickCube", "success": "held", "failure": "loose"},
            {
                "source": "held",
                "action": "MoveToTossLocationAndToss",
                "success": "goal",
                "failure": "loose",
            },
            {"source": "goal", "action": "PickCube", "success": "held", "failure": "goal"},
            {
                "source": "loose",
                "action": "reset_cube_and_bin_near",
                "success": "goal",
                "failure": "loose",
            },
            {"source": "loose", "action": "reset_cube_far", "success": "held", "failure": "loose"},
        ],
    })


def test_state_graph_gates_actions_and_labels_outcomes():
    graph = StateGraph(manifest=manifest())
    assert graph.actions(cluster="held") == ["MoveToTossLocationAndToss"]
    assert graph.outcome(source="loose", action="PickCube", destination="held") is True
    assert graph.outcome(source="loose", action="PickCube", destination="loose") is False
    assert graph.outcome(source="loose", action="PickCube", destination=None) is None
    assert graph.actions(cluster=None) == []
    assert graph.deployment_action(cluster="loose") == "PickCube"
    assert graph.deployment_action(cluster="goal") is None


def test_graph_rejects_ambiguous_outcome_and_unbound_controller():
    raw = manifest().model_dump()
    raw["edges"][0]["failure"] = "held"
    with pytest.raises(ValueError):
        LanguageManifest.model_validate(raw)
    raw = manifest().model_dump()
    raw["edges"][0]["action"] = "arbitrary_simulator_reset"
    with pytest.raises(ValueError):
        LanguageManifest.model_validate(raw)


def test_shared_budget_counts_resumed_coding_and_fresh_classification(tmp_path):
    coding = SharedBudget(path=tmp_path / "budget.json", limit=20, session="coding")
    judge = SharedBudget(path=coding.path, limit=20, session="classification-1")
    with coding.lock():
        coding.record(cumulative=2)
    with judge.lock():
        judge.record(cumulative=1)
    with coding.lock():
        coding.record(cumulative=4)
    assert coding.spent == 5
    assert judge.remaining == 15
    with coding.lock(), pytest.raises(ValueError):
        coding.record(cumulative=3)
    assert json.loads(coding.path.read_text())["sessions"] == {"coding": 4, "classification-1": 1}


def test_chain_human_probability_is_bayesian_and_cost_is_measured():
    from hitl_pmp.agentic_states.model import ChainModel, ClusterState, OptionAction
    from hitl_pmp.environments.tossing3d.skill_provider import Tossing3DSkillProvider
    from hitl_pmp.hybrid_skills.execution import HybridEnvironment
    from hitl_pmp.hybrid_skills.method import HybridMethod

    env = HybridEnvironment(scene_bg=False)
    method = HybridMethod(
        env=env,
        skill_provider=Tossing3DSkillProvider(env=env),
        seed=0,
        pomdp_inference_engine="grid",
    )
    model = ChainModel(manifest=manifest(), method=method)
    ground = model.bindings["reset_cube_and_bin_near"]
    state = method.pomdp_state.model_copy(update={"expected_execution_costs": {str(ground): 123}})
    outcomes = model.transition_outcomes(
        environment_state=ClusterState(cluster_id="loose"),
        practice_action=OptionAction(option_id="reset_cube_and_bin_near"),
        belief_state=state,
    )
    k = state.skill_beliefs[ground.skill.name].mean_competence()
    assert 0 < k < 1
    assert [x[2] for x in outcomes] == pytest.approx([k, 1 - k])
    assert all(x[1] == 123 for x in outcomes)
    after = model.compute_next_belief_state(
        belief_state=state,
        environment_state=ClusterState(cluster_id="loose"),
        potential_next_environment_state=outcomes[0][0],
        practice_action=OptionAction(option_id="reset_cube_and_bin_near"),
    )
    assert after.skill_beliefs[ground.skill.name].mean_competence() > k
    assert model.bindings["reset_cube_far"] != ground


def test_language_dispatch_does_not_consult_fixed_predicates(monkeypatch):
    from hitl_pmp.agentic_states.method import SkillsStatesMethod
    from hitl_pmp.agentic_states.model import OptionAction
    from hitl_pmp.core.practice_costs import PracticeAccounting, PracticeCosts
    from hitl_pmp.environments.tossing3d.skill_provider import Tossing3DSkillProvider
    from hitl_pmp.hybrid_skills.execution import HybridEnvironment

    env = HybridEnvironment(scene_bg=False)
    method = SkillsStatesMethod(
        env=env,
        skill_provider=Tossing3DSkillProvider(env=env),
        seed=0,
        pomdp_inference_engine="grid",
    )
    method.configure_practice_accounting(accounting=PracticeAccounting(costs=PracticeCosts()))
    method.accept_manifest(manifest=manifest())
    monkeypatch.setattr(SkillsStatesMethod, "classify", lambda self: "loose")
    monkeypatch.setattr(
        SkillsStatesMethod,
        "predicates",
        lambda self: (_ for _ in ()).throw(AssertionError("fixed predicates used")),
    )
    method.pomdp_planner.solve = lambda **kwargs: (1, OptionAction(option_id="PickCube"))
    action = method.get_practice_policy(task=None)(None)
    assert int(action.action[0]) == env.pick_cube_id


def test_snapshot_freezes_language_states_and_skill_code(tmp_path):
    from hitl_pmp.agentic_states.execution import StatesSnapshot
    from hitl_pmp.agentic_states.method import SkillsStatesMethod
    from hitl_pmp.environments.tossing3d.skill_provider import Tossing3DSkillProvider
    from hitl_pmp.hybrid_skills.artifacts import SkillBundle
    from hitl_pmp.hybrid_skills.execution import HybridEnvironment

    env = HybridEnvironment(scene_bg=False)
    env.bundle = SkillBundle(files={"skills.py": "raise RuntimeError('do not load on host')"})
    provider = Tossing3DSkillProvider(env=env)
    method = SkillsStatesMethod(
        env=env, skill_provider=provider, seed=0, pomdp_inference_engine="grid"
    )
    method.accept_manifest(manifest=manifest())
    raw = StatesSnapshot.encode(method=method)
    env.bundle = SkillBundle(files={"skills.py": "pass"})
    restored = StatesSnapshot.restore(raw=raw, env=env, provider=provider)
    assert restored.graph.manifest == manifest()
    assert "do not load on host" in env.bundle.files["skills.py"]


def test_real_expectimax_searches_language_edges_with_finite_costs():
    from hitl_pmp.agentic_states.method import SkillsStatesMethod
    from hitl_pmp.agentic_states.model import ClusterState
    from hitl_pmp.core.practice_costs import PracticeAccounting, PracticeCosts
    from hitl_pmp.environments.tossing3d.skill_provider import Tossing3DSkillProvider
    from hitl_pmp.hybrid_skills.execution import HybridEnvironment
    from hitl_pmp.methods.belief_space.types.stop_action import StopAction

    env = HybridEnvironment(scene_bg=False)
    method = SkillsStatesMethod(
        env=env,
        skill_provider=Tossing3DSkillProvider(env=env),
        seed=0,
        pomdp_inference_engine="grid",
        pomdp_solver="expectimax",
    )
    method.configure_practice_accounting(accounting=PracticeAccounting(costs=PracticeCosts()))
    method.accept_manifest(manifest=manifest())
    value, action = method.pomdp_planner.solve(
        environment_state=ClusterState(cluster_id="loose"),
        belief_state=method.pomdp_state,
        summed_cost=0,
        horizon=2,
        model=method._chain,
    )
    assert isinstance(value, float)
    assert isinstance(action, StopAction) or action.option_id in method._chain.graph.actions(
        "loose"
    )


@pytest.mark.parametrize(
    "action,side", [("reset_cube_far", "opposite_side"), ("reset_cube_and_bin_near", "robot_side")]
)
def test_language_planner_can_dispatch_both_human_sides(monkeypatch, action, side):
    from hitl_pmp.agentic_states.method import SkillsStatesMethod
    from hitl_pmp.agentic_states.model import OptionAction
    from hitl_pmp.core.method.method import HumanCubeBinResetRequested
    from hitl_pmp.core.practice_costs import PracticeAccounting, PracticeCosts
    from hitl_pmp.environments.tossing3d.skill_provider import Tossing3DSkillProvider
    from hitl_pmp.hybrid_skills.execution import HybridEnvironment

    env = HybridEnvironment(scene_bg=False)
    method = SkillsStatesMethod(
        env=env,
        skill_provider=Tossing3DSkillProvider(env=env),
        seed=0,
        pomdp_inference_engine="grid",
    )
    costs = PracticeCosts.model_validate({
        "human_skills": {
            name: {"cost": {"value": 100}, "duration": {"value": 1}}
            for name in ["reset_cube_far", "reset_cube_and_bin_near"]
        }
    })
    method.configure_practice_accounting(accounting=PracticeAccounting(costs=costs))
    method.accept_manifest(manifest=manifest())
    monkeypatch.setattr(SkillsStatesMethod, "classify", lambda self: "loose")
    method.pomdp_planner.solve = lambda **kwargs: (1, OptionAction(option_id=action))
    with pytest.raises(HumanCubeBinResetRequested) as exc:
        method.get_practice_policy(task=None)(None)
    assert exc.value.destination == side and exc.value.cost == 100
    before = method.pomdp_state.model_dump_json()
    method.observe_environment_reset(state=None)
    assert method.pomdp_state.model_dump_json() == before
    assert method._pending_language == ("loose", action)


def test_unknown_endpoint_does_not_poison_competence():
    from hitl_pmp.agentic_states.method import SkillsStatesMethod
    from hitl_pmp.environments.tossing3d.skill_provider import Tossing3DSkillProvider
    from hitl_pmp.hybrid_skills.execution import HybridEnvironment

    env = HybridEnvironment(scene_bg=False)
    method = SkillsStatesMethod(
        env=env,
        skill_provider=Tossing3DSkillProvider(env=env),
        seed=0,
        pomdp_inference_engine="grid",
    )
    method.accept_manifest(manifest=manifest())
    method._pending_language = ("loose", "PickCube")
    before = method.pomdp_state.model_dump_json()
    method.settle(cluster=None)
    assert method.pomdp_state.model_dump_json() == before
    assert not method._chain.counts


def test_evaluation_ledger_does_not_reduce_adaptation_budget(tmp_path):
    training = SharedBudget(path=tmp_path / "model_budget.json", limit=20, session="coding")
    evaluation = SharedBudget(
        path=tmp_path / "evaluation_model_cost.json", limit=None, session="eval-1"
    )
    with training.lock():
        training.record(cumulative=19)
    with evaluation.lock():
        evaluation.record(cumulative=25)
    assert training.remaining == 1
    assert evaluation.spent == 25
    assert evaluation.remaining is None
    assert not evaluation.exhausted
    with training.lock():
        training.record(cumulative=20)
    assert training.exhausted
    assert not evaluation.exhausted


def test_classifier_routes_evaluation_to_separate_uncapped_ledger(tmp_path):
    from hitl_pmp.agentic_states.classifier import LanguageClassifier

    common = dict(
        settings=None,
        output=tmp_path / "classifications",
        budget_path=tmp_path / "model_budget.json",
        budget_limit=20,
    )
    practice = LanguageClassifier(**common, phase="practice")
    evaluation = LanguageClassifier(**common, phase="evaluation")
    assert practice.budget_path == tmp_path / "model_budget.json"
    assert practice.budget_limit == 20
    assert evaluation.budget_path == tmp_path / "evaluation_model_cost.json"
    assert evaluation.budget_limit is None
    with pytest.raises(ValueError):
        LanguageClassifier(**common, phase="typo")


def test_paid_eval_still_runs_when_adaptation_exhausted(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import hitl_pmp.agentic_states.classifier as module
    from hitl_pmp.hybrid_skills.artifacts import ModelBudgetExhausted

    budget_path = tmp_path / "model_budget.json"
    training = SharedBudget(path=budget_path, limit=20, session="coding")
    training.record(cumulative=20)
    before = budget_path.read_bytes()
    calls = []

    class Transport:
        def __init__(self, **kwargs):
            pass

        def modules(self):
            return (
                SimpleNamespace(SandboxConfig=SimpleNamespace),
                SimpleNamespace(create_backend=lambda settings: None),
                SimpleNamespace(DictConfig=dict),
            )

        def run_with_recovery(self, *, config, **kwargs):
            calls.append(config)
            (config.sandbox_dir / "judgment.json").write_text(
                '{"cluster_id": "loose", "reason": "cube is loose"}'
            )
            return SimpleNamespace(success=True, error=None, total_cost_usd=0.25)

    monkeypatch.setattr(module, "DeadlineTransport", Transport)
    common = dict(
        settings=SimpleNamespace(robocode_checkout=tmp_path),
        output=tmp_path / "classifications",
        budget_path=budget_path,
        budget_limit=20,
    )
    evaluation = module.LanguageClassifier(**common, phase="evaluation")
    assert evaluation.classify(manifest=manifest(), observation={}).cluster_id == "loose"
    assert calls[0].max_budget_usd == 0.0  # Native backend: no model-dollar cap.
    assert budget_path.read_bytes() == before
    assert json.loads(evaluation.budget_path.read_text())["spent"] == 0.25
    practice = module.LanguageClassifier(**common, phase="practice")
    with pytest.raises(ModelBudgetExhausted):
        practice.classify(manifest=manifest(), observation={})
    assert len(calls) == 1


def robot_only_manifest_raw():
    raw = manifest().model_dump()
    raw["edges"] = [e for e in raw["edges"] if not e["action"].startswith("reset_cube")]
    return raw


def test_manifest_requires_human_skills_only_when_the_reset_is_offered():
    from hitl_pmp.agentic_states.graph import HUMANS

    assert LanguageManifest.for_run(human_reset_offered=True) is LanguageManifest
    robot_only = LanguageManifest.for_run(human_reset_offered=False)
    accepted = robot_only.model_validate(robot_only_manifest_raw())
    assert not {e.action for e in accepted.edges} & HUMANS
    with pytest.raises(ValueError):
        LanguageManifest.model_validate(robot_only_manifest_raw())
    with pytest.raises(ValueError):
        robot_only.model_validate(manifest().model_dump())
    # The schema the coding model reads names only the actions the run offers.
    enum = robot_only.model_json_schema()["$defs"]["RobotEdge"]["properties"]["action"]["enum"]
    assert enum == ["PickCube", "MoveToTossLocationAndToss", "OpenGripper"]
    unchanged = LanguageManifest.model_json_schema()["$defs"]["Edge"]["properties"]["action"]
    assert unchanged["enum"] == [*enum, "reset_cube_far", "reset_cube_and_bin_near"]
    assert "reset_cube" not in json.dumps(robot_only.model_json_schema())


def test_chain_and_snapshot_accept_a_robot_only_graph_without_the_reset():
    from hitl_pmp.agentic_states.execution import StatesSnapshot
    from hitl_pmp.agentic_states.method import SkillsStatesMethod
    from hitl_pmp.agentic_states.model import ClusterState
    from hitl_pmp.environments.tossing3d.skill_provider import Tossing3DSkillProvider
    from hitl_pmp.hybrid_skills.artifacts import SkillBundle
    from hitl_pmp.hybrid_skills.execution import HybridEnvironment

    env = HybridEnvironment(scene_bg=False)
    env.bundle = SkillBundle(files={"skills.py": "pass"})
    provider = Tossing3DSkillProvider(env=env, offer_human_reset=False)
    method = SkillsStatesMethod(
        env=env, skill_provider=provider, seed=0, pomdp_inference_engine="grid"
    )
    robot_only = LanguageManifest.for_run(human_reset_offered=False)
    accepted = robot_only.model_validate(robot_only_manifest_raw())
    method.accept_manifest(manifest=accepted)
    assert set(method._chain.bindings) == {"PickCube", "MoveToTossLocationAndToss", "OpenGripper"}
    valid = method._chain.get_valid_actions(environment_state=ClusterState(cluster_id="loose"))
    assert [a.option_id for a in valid] == ["PickCube"]
    with pytest.raises(ValueError, match="unbound"):
        method.accept_manifest(manifest=manifest())
    restored = StatesSnapshot.restore(
        raw=StatesSnapshot.encode(method=method), env=env, provider=provider
    )
    assert isinstance(restored.graph.manifest, robot_only)
    assert restored.graph.manifest.model_dump() == accepted.model_dump()
    assert restored.graph.deployment_action(cluster="loose") == "PickCube"


def test_language_learner_and_classifier_never_mention_a_human_without_the_reset(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    import hitl_pmp.agentic_states.classifier as classifier_module
    from hitl_pmp.agentic_runtime.sandbox import SandboxSettings
    from hitl_pmp.agentic_states.learner import STATE_CONTRACT, SkillsStatesLearner
    from hitl_pmp.core.practice_costs import PracticeCosts
    from hitl_pmp.environments.tossing3d.agentic_bridge import Tossing3DAgenticBridge
    from hitl_pmp.hybrid_skills import learner as learner_module
    from hitl_pmp.hybrid_skills.execution import HybridEnvironment

    assert SkillsStatesLearner.state_contract(human_reset_offered=True) == STATE_CONTRACT
    assert "reset_cube_far" in STATE_CONTRACT and "human" in STATE_CONTRACT
    contract = SkillsStatesLearner.state_contract(human_reset_offered=False)
    assert "human" not in contract.lower() and "reset_cube" not in contract
    assert "PickCube, MoveToTossLocationAndToss and OpenGripper" in contract
    monkeypatch.setattr(Tossing3DAgenticBridge, "observe", lambda self: {"objects": []})
    calls = []

    class Transport:
        def __init__(self, **kwargs):
            pass

        def modules(self):
            return (
                SimpleNamespace(SandboxConfig=lambda **kw: SimpleNamespace(**kw)),
                SimpleNamespace(create_backend=lambda config: None),
                SimpleNamespace(DictConfig=lambda x: x),
            )

        def run_with_recovery(self, *, config, **kwargs):
            calls.append(config)
            if config.output_filename == "judgment.json":
                (config.sandbox_dir / "judgment.json").write_text(
                    '{"cluster_id": "loose", "reason": "cube is loose"}'
                )
            else:
                (config.sandbox_dir / "submission" / "states.json").write_text(
                    json.dumps(robot_only_manifest_raw())
                )
            return SimpleNamespace(success=True, error=None, total_cost_usd=1)

    monkeypatch.setattr(learner_module, "DeadlineTransport", Transport)
    monkeypatch.setattr(classifier_module, "DeadlineTransport", Transport)
    env = HybridEnvironment(scene_bg=False)
    settings = SandboxSettings(
        image="fake", robocode_checkout=tmp_path, artifact_dir=tmp_path / "trajectories"
    )
    accepted = []
    learner = SkillsStatesLearner(
        on_accept=lambda *, manifest: accepted.append(manifest),
        output=tmp_path,
        env=env,
        settings=settings,
        costs=PracticeCosts(),
        human_reset_offered=False,
    )
    learner.revise()
    assert len(accepted) == 1
    assert accepted[0].__class__ is LanguageManifest.for_run(human_reset_offered=False)
    for name in (
        "task_prompt.md",
        "system_prompt.md",
        "coding/sandbox/STATE_CONTRACT.md",
        "coding/sandbox/states.schema.json",
        "coding/sandbox/evidence/accounting.json",
    ):
        text = (tmp_path / name).read_text().lower()
        assert "human" not in text and "reset_cube" not in text, name
    assert "human" not in (calls[0].prompt + calls[0].system_prompt).lower()
    classifier = classifier_module.LanguageClassifier(
        settings=SimpleNamespace(robocode_checkout=tmp_path),
        output=tmp_path / "classifications",
        budget_path=tmp_path / "model_budget.json",
        budget_limit=20,
        phase="practice",
    )
    assert classifier.classify(manifest=accepted[0], observation={}).cluster_id == "loose"
    assert "human" not in (calls[1].prompt + calls[1].system_prompt).lower()
