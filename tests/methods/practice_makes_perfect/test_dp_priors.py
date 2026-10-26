import hashlib
import json
from pathlib import Path

import pytest

from hitl_pmp.core.practice_costs import ChargeFunction, PracticeAccounting, PracticeCosts
from hitl_pmp.environments.sweepoffshelf3d.environment import SweepOffShelf3DEnvironment
from hitl_pmp.environments.sweepoffshelf3d.skill_provider import SweepOffShelf3DSkillProvider
from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.skill_provider import Tossing3DSkillProvider
from hitl_pmp.methods.practice_makes_perfect.always_ask_reset import AlwaysAskReset
from hitl_pmp.methods.practice_makes_perfect.dp_mdp import DpMdp, DpTaskModel, DpTransition
from hitl_pmp.methods.practice_makes_perfect.dp_method import DpMethod
from hitl_pmp.methods.practice_makes_perfect.dp_priors import DpPriorCalibration, DpSkillPrior
from hitl_pmp.methods.practice_makes_perfect.ees_method import EesMethod

from .test_dp_method import _build, _execute, _ground, _human


@pytest.fixture
def calibration(*, tmp_path):
    root = Path(__file__).resolve().parents[3]
    source = "src/hitl_pmp/methods/practice_makes_perfect/dp_priors.py"
    report = {
        "schema_version": 1,
        "environment": "sweepoffshelf3d",
        "layout": "test",
        "variant": "test",
        "skills": {
            skill.name: {"alpha": 0.2, "beta": 1.8, "initial_learning_rate": 0.005}
            for skill in SweepOffShelf3DSkillProvider(env=SweepOffShelf3DEnvironment()).skills()
        },
        "fit_seeds": [2000, 2001, 2002],
        "validation_seeds": [2010, 2011, 2012],
        "source_sha256": {source: hashlib.sha256((root / source).read_bytes()).hexdigest()},
        "validation_passed": True,
        "uses_experiment_evaluations": False,
    }
    path = tmp_path / "priors.json"
    path.write_text(json.dumps(report))
    return path, report


def test_calibrated_prior_is_source_bound_and_accepts_fractional_beta_parameters(*, calibration):
    path, _report = calibration
    prior = DpPriorCalibration.load(path=path).skills["PickWiper"]
    assert prior.alpha == 0.2 and prior.beta == 1.8
    assert prior.initial_learning_rate == 0.005


def test_packaged_tossing3d_priors_cover_every_robot_skill_without_changing_toss_fit():
    path = Path(__file__).resolve().parents[3] / "configs/tossing3d_dp_priors.json"
    calibration = DpPriorCalibration.model_validate_json(path.read_text())
    provider = Tossing3DSkillProvider(env=Tossing3DEnvironment())
    calibration.verify_robot_skills(skills=provider.skills())
    assert calibration.skills["OpenGripper"].initial_learning_rate == 0
    toss = calibration.skills["MoveToTossLocationAndToss"]
    assert toss.alpha / (toss.alpha + toss.beta) == pytest.approx(0.1326530612244898)
    assert toss.initial_learning_rate == pytest.approx(0.0053677721088435364)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("validation_passed", False, "validation_passed"),
        ("uses_experiment_evaluations", True, "uses_experiment_evaluations"),
        ("fit_seeds", [0], "experiment seeds"),
        ("validation_seeds", [2000], "disjoint"),
        ("source_sha256", {"../outside.py": "0" * 64}, "invalid calibration"),
        (
            "source_sha256",
            {"src/hitl_pmp/methods/practice_makes_perfect/dp_priors.py": "0" * 64},
            "stale prior",
        ),
    ],
)
def test_bad_calibration_cannot_be_loaded(*, calibration, field, value, message):
    path, report = calibration
    report[field] = value
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match=message):
        DpPriorCalibration.load(path=path)


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf")])
def test_nonphysical_learning_rate_is_rejected(*, value):
    with pytest.raises(ValueError):
        DpSkillPrior(alpha=1, beta=1, initial_learning_rate=value)


@pytest.mark.parametrize("missing", ["PickCube", "MoveToTossLocationAndToss", "OpenGripper"])
@pytest.mark.parametrize("always_ask", [False, True])
def test_all_shared_robot_skills_are_required_before_any_planning(
    *, calibration, missing, always_ask
):
    path, report = calibration
    env = Tossing3DEnvironment()
    provider = Tossing3DSkillProvider(env=env)
    report["skills"] = {
        skill.name: {"alpha": 1.9, "beta": 0.1, "initial_learning_rate": 0.0}
        for skill in provider.skills()
        if skill.name != missing
    }
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match=f"robot skills lack calibrated DP priors.*{missing}"):
        DpMethod(
            env=env,
            skill_provider=provider,
            dp_budget=2000,
            dp_prior_calibration=path,
            always_ask_reset=AlwaysAskReset(cost=5) if always_ask else None,
        )


@pytest.mark.parametrize("always_ask", [False, True])
def test_new_provider_robot_skill_automatically_requires_a_prior(
    *, calibration, monkeypatch, always_ask
):
    path, report = calibration
    env = Tossing3DEnvironment()
    provider = Tossing3DSkillProvider(env=env)
    skills = provider.skills()
    report["skills"] = {
        skill.name: {"alpha": 1.9, "beta": 0.1, "initial_learning_rate": 0.0} for skill in skills
    }
    path.write_text(json.dumps(report))
    future = skills[-1].model_copy(update={"name": "FutureRecovery"})
    monkeypatch.setattr(
        Tossing3DSkillProvider, "skills", lambda *_args, **_kwargs: (*skills, future)
    )
    with pytest.raises(ValueError, match="robot skills lack calibrated DP priors.*FutureRecovery"):
        DpMethod(
            env=env,
            skill_provider=provider,
            dp_budget=2000,
            dp_prior_calibration=path,
            always_ask_reset=AlwaysAskReset(cost=5) if always_ask else None,
        )


@pytest.mark.parametrize("always_ask", [False, True])
def test_shared_robot_controllers_and_human_practice_resets_are_preserved(
    *, calibration, always_ask
):
    path, report = calibration
    env = Tossing3DEnvironment()
    provider = Tossing3DSkillProvider(env=env)
    report["skills"] = {
        skill.name: {"alpha": 1.9, "beta": 0.1, "initial_learning_rate": 0.0}
        for skill in provider.skills()
    }
    path.write_text(json.dumps(report))
    method = DpMethod(
        env=env,
        skill_provider=provider,
        dp_budget=2000,
        dp_prior_calibration=path,
        always_ask_reset=AlwaysAskReset(cost=5) if always_ask else None,
    )
    ees = EesMethod(
        env=env,
        skill_provider=provider,
        always_ask_reset=AlwaysAskReset(cost=5) if always_ask else None,
    )
    assert method.skill_provider is ees.skill_provider is provider
    assert method.skills() == ees.skills() == provider.skills()
    assert DpMethod.execute_ground_skill is EesMethod.execute_ground_skill
    assert method.offered_partial_resets() == ees.offered_partial_resets()
    resets = provider.human_cube_bin_reset_skills()
    assert len(resets) == 2
    assert method.offered_partial_resets() == (() if always_ask else resets)
    for reset in resets:
        assert method.human_skill_competence(ground_skill=reset) == pytest.approx(10 / 11)
    assert method.score_human_skills
    assert (method.always_ask_reset is not None) is always_ask


@pytest.mark.parametrize("offer_human_reset", [False, True])
def test_complete_robot_coverage_does_not_require_human_priors(*, calibration, offer_human_reset):
    path, report = calibration
    env = Tossing3DEnvironment()
    provider = Tossing3DSkillProvider(env=env, offer_human_reset=offer_human_reset)
    report["skills"] = {
        skill.name: {"alpha": 1.9, "beta": 0.1, "initial_learning_rate": 0.0}
        for skill in provider.skills()
    }
    path.write_text(json.dumps(report))
    method = DpMethod(env=env, skill_provider=provider, dp_budget=2000, dp_prior_calibration=path)
    assert set(method._dp_skill_priors) == {skill.name for skill in provider.skills()}


@pytest.mark.parametrize("complete", [False, True])
def test_tossing3d_cli_checks_recovery_coverage_before_practice(
    *, calibration, monkeypatch, complete
):
    from types import SimpleNamespace

    from hitl_pmp.cli import Cli
    from hitl_pmp.environments.tossing3d.cli import Tossing3DCli

    path, report = calibration
    env = Tossing3DEnvironment()
    provider = Tossing3DSkillProvider(env=env)
    report.update(environment="tossing3d", layout="barrier", variant="o1")
    report["skills"] = {
        skill.name: {"alpha": 1.9, "beta": 0.1, "initial_learning_rate": 0.0}
        for skill in provider.skills()
        if complete or skill.name != "OpenGripper"
    }
    path.write_text(json.dumps(report))
    built = []

    def run_method(*, method_factory, **kwargs):
        built.append(method_factory(SimpleNamespace(env=env, skill_provider=provider)))

    monkeypatch.setattr(Tossing3DCli, "run_method", staticmethod(run_method))
    argv = [
        "--env",
        "tossing3d",
        "--method",
        "dp",
        "--dp-budget",
        "2000",
        "--dp-prior-calibration",
        str(path),
    ]
    if complete:
        Cli.main(argv=argv)
        assert built[0].skills() == provider.skills()
    else:
        with pytest.raises(ValueError, match="robot skills lack calibrated DP priors.*OpenGripper"):
            Cli.main(argv=argv)
        assert built == []


def test_recovery_transition_uses_its_calibrated_prior_without_crashing(
    *, calibration, monkeypatch
):
    path, report = calibration
    env = Tossing3DEnvironment()
    provider = Tossing3DSkillProvider(env=env)
    report["skills"] = {
        skill.name: {"alpha": 1.9, "beta": 0.1, "initial_learning_rate": 0.0}
        for skill in provider.skills()
    }
    path.write_text(json.dumps(report))
    method = DpMethod(
        env=env,
        skill_provider=provider,
        dp_budget=2000,
        dp_prior_calibration=path,
        dp_solver="enumeration",
    )
    method.configure_practice_accounting(
        accounting=PracticeAccounting(costs=PracticeCosts(robot_step=ChargeFunction(value=0.01)))
    )
    recovery = _ground(method=method, name="OpenGripper")
    monkeypatch.setattr(
        DpTaskModel,
        "build",
        staticmethod(
            lambda **kwargs: DpMdp(
                initial_probabilities={0: 1},
                terminal_rewards={1: 1},
                transitions=[DpTransition(state=0, skill=recovery, successor=1)],
            )
        ),
    )
    method.choose_practice_target()
    assert method.dp_allocation()[recovery] == 0
    linear = method._linear_model(history=None, ground_skill=recovery)
    assert linear.get_current_competence() == pytest.approx(0.95)
    assert linear.learning_rate() == 0


def test_calibration_changes_dp_robot_beliefs_but_not_ees_or_human_priors(*, calibration):
    path, _report = calibration
    env = SweepOffShelf3DEnvironment()
    provider = SweepOffShelf3DSkillProvider(env=env)
    method = DpMethod(env=env, skill_provider=provider, dp_budget=2000, dp_prior_calibration=path)
    ground = _ground(method=method, name="PickWiper")
    history = method.competence_model(ground_skill=ground)
    assert history.get_current_competence() == pytest.approx(0.1)
    linear = method._linear_model(history=history, ground_skill=ground)
    assert linear.get_current_competence() == pytest.approx(0.1)
    assert linear.learning_rate() == pytest.approx(0.005)
    assert linear.predict_competence(num_attempts=1) == pytest.approx(0.105)
    assert method.human_skill_competence(ground_skill=_human(method=method)) == pytest.approx(
        10 / 11
    )
    ees = EesMethod(env=env, skill_provider=provider)
    assert ees.competence_model(ground_skill=ground).get_current_competence() == pytest.approx(
        10 / 11
    )


def test_calibration_for_unknown_skills_is_rejected(*, calibration):
    path, report = calibration
    report["skills"]["NotASkill"] = report["skills"].pop("PickWiper")
    path.write_text(json.dumps(report))
    env = SweepOffShelf3DEnvironment()
    with pytest.raises(ValueError, match="absent from this domain"):
        DpMethod(
            env=env,
            skill_provider=SweepOffShelf3DSkillProvider(env=env),
            dp_budget=2000,
            dp_prior_calibration=path,
        )


def test_calibrated_once_allocation_is_not_exhausted_after_two_trials(*, calibration, monkeypatch):
    path, _report = calibration
    method = _build()
    method = method.model_copy(
        update={"dp_budget_unit": "episodes", "dp_allocation_schedule": "once"}
    )
    method.dp_prior_calibration = path
    method._human_skills_are_first_class()
    pick = _ground(method=method, name="PickWiper")
    monkeypatch.setattr(
        DpTaskModel,
        "build",
        staticmethod(
            lambda **kwargs: DpMdp(
                initial_probabilities={0: 1},
                terminal_rewards={1: 1},
                transitions=[DpTransition(state=0, skill=pick, successor=1)],
            )
        ),
    )
    assert method.choose_practice_target()[0] == pick
    assert method.dp_allocation()[pick] == 100
    _execute(method=method, ground=pick)
    _execute(method=method, ground=pick)
    assert method.practice_complete() is False
    assert method.dp_remaining_allocation()[pick] == 98


def test_calibrated_mode_refuses_unmeasured_deployment_skills(*, calibration, monkeypatch):
    path, _report = calibration
    method = _build()
    method.dp_prior_calibration = path
    method._human_skills_are_first_class()
    method._dp_skill_priors.pop("ReturnWiper")
    back = _ground(method=method, name="ReturnWiper")
    monkeypatch.setattr(
        DpTaskModel,
        "build",
        staticmethod(
            lambda **kwargs: DpMdp(
                initial_probabilities={0: 1},
                terminal_rewards={1: 1},
                transitions=[DpTransition(state=0, skill=back, successor=1)],
            )
        ),
    )
    with pytest.raises(ValueError, match="lack calibrated DP priors"):
        method.choose_practice_target()
