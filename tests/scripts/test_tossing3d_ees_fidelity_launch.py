import copy
import json
from itertools import product
from pathlib import Path

import pytest

from hitl_pmp.cli import Cli
from scripts.tossing3d_ees_fidelity_launch import (
    SEEDS,
    FreezeOverrides,
    arms,
    build_manifest,
    main,
    source_checks,
    validate_manifest,
)

TEST_HORIZON = 5


@pytest.fixture(scope="module")
def full_manifest():
    return build_manifest(size="full", goal_horizon=TEST_HORIZON, results_root=Path("/unused"))


@pytest.mark.parametrize("size", ["full", "pilot"])
def test_complete_matrix_and_seed_major_order(*, size):
    manifest = build_manifest(
        size=size,
        goal_horizon=TEST_HORIZON,
        results_root=Path("/unused"),
    )
    rows = manifest["runs"]
    names = [arm.name for arm in arms(goal_horizon=TEST_HORIZON)]
    assert len(names) == len(set(names)) == 20
    assert [(row["arm"], row["seed"]) for row in rows] == [
        (name, seed) for seed in SEEDS[size] for name in names
    ]
    assert len({row["output_dir"] for row in rows}) == 20 * len(SEEDS[size])
    assert len({row["id"] for row in rows}) == len(rows)
    cells = {
        (
            row["config"]["competence_sharing"],
            row["config"]["practice_groundings"],
            row["config"]["reproduce_predicators_skip_perfect"],
            row["config"]["goal_pursuit_horizon"] > 0,
        )
        for row in rows
        if row["config"]["method"] == "ees"
    }
    assert cells == set(
        product(("parent", "grounded"), ("observed", "all"), (True, False), (False, True))
    )
    assert manifest["schedule"] == {
        "ordering": "seed-major",
        "dependencies": [],
        "barriers": [],
        "max_concurrency": None,
        "concurrency_policy": "maximum scheduler permits",
    }
    for row in rows:
        args = Cli.parse_args(argv=row["argv"])
        assert args.practice_step_budget == (160000 if size == "full" else 6400)
        assert (
            args.measurement_interval_steps,
            args.num_test_tasks,
            args.evaluation_control_steps,
        ) == (3200, 10, 500)
        assert args.stop_after_perfect_evaluations == 0
        assert args.goal_pursuit_horizon == (TEST_HORIZON if row["arm"].endswith("goalon") else 0)
        assert args.goal_pursuit_interval == args.goal_pursuit_init_cycles == 1
        assert args.reproduce_predicators_explore_target_only
        assert args.exploration_epsilon == 0.5
        assert args.sampler_max_train_iters == 10000
        assert args.competence_window_size == args.competence_recency_size == 5
        assert not args.ees_reset_gate
        assert not args.always_ask_reset
        assert not args.enable_bin_pouring
        assert (args.bin_mass_kg, args.cube_mass_kg) == (1.0, 0.1)
        assert args.seed == row["seed"]
        assert str(args.output_dir) == row["output_dir"]
        if args.method == "ees":
            assert args.ees_score_human_skills is True
            assert args.record_ees_decisions is True
        else:
            assert args.ees_score_human_skills is None
            assert args.record_ees_decisions is False
            assert args.preset == "tossing3d-canonical"
            assert args.pomdp_search_depth == 6
            curtis = row["arm"].startswith("determinized-curtis")
            assert args.pomdp_solver == ("determinized_astar" if curtis else "expectimax")
            assert args.pomdp_observation_probability_weight == (0.01 if curtis else 0.0)
            if curtis:
                assert args.pomdp_determinized_variant == "curtis"
                assert args.pomdp_max_search_iterations == 100
                assert args.pomdp_draw_scope == "node"
                assert args.pomdp_num_rollouts is None


@pytest.mark.parametrize("horizon", [None, 0, -1, -10, True, 1.5])
def test_reject_unspecified_nonpositive_or_noninteger_horizon(*, horizon):
    with pytest.raises(ValueError, match="positive integer"):
        build_manifest(size="full", goal_horizon=horizon, results_root=Path("/unused"))


def test_determinism_and_source_checks(*, full_manifest):
    validate_manifest(manifest=full_manifest)
    assert source_checks()["inherited_method_defaults"]["score_human_skills"] is True
    assert full_manifest["provenance_pending"]
    assert len(full_manifest["source"]["head"]) == 40
    assert all(len(digest) == 64 for digest in full_manifest["source"]["sha256"].values())


@pytest.mark.parametrize("change", ["duplicate", "missing", "order", "argv", "config", "source"])
def test_validation_rejects_tampering(*, full_manifest, change):
    manifest = copy.deepcopy(full_manifest)
    rows = manifest["runs"]
    if change == "duplicate":
        rows[1] = rows[0]
    elif change == "missing":
        rows.pop()
    elif change == "order":
        rows[0], rows[1] = rows[1], rows[0]
    elif change == "argv":
        rows[0]["argv"] += ["--stop-after-perfect-evaluations", "3"]
    elif change == "config":
        rows[0]["config"]["reproduce_predicators_explore_target_only"] = False
    else:
        manifest["source"]["sha256"]["scripts/tossing3d_launch.py"] = "0" * 64
    with pytest.raises(ValueError, match="frozen matrix or current source"):
        validate_manifest(manifest=manifest)


def test_freeze_overrides_are_shared_and_recorded():
    overrides = {"sampler_max_train_iters": 123, "planning_timeout": 2.0}
    manifest = build_manifest(
        size="pilot",
        goal_horizon=TEST_HORIZON,
        results_root=Path("/unused"),
        freeze_overrides=overrides,
    )
    assert manifest["options"]["freeze_overrides"] == overrides
    for row in manifest["runs"]:
        assert row["config"]["sampler_max_train_iters"] == 123
        assert row["config"]["planning_timeout"] == 2.0
    validate_manifest(manifest=manifest)


@pytest.mark.parametrize(
    "overrides",
    [
        {"goal_pursuit_horizon": 0},
        {"stop_after_perfect_evaluations": 3},
        {"exploration_epsilon": 1.1},
        {"planning_timeout": float("inf")},
        {"sampler_max_train_iters": 0},
        {"competence_window_size": True},
    ],
)
def test_bad_or_protocol_changing_overrides_rejected(*, overrides):
    with pytest.raises(ValueError):
        FreezeOverrides.model_validate(overrides)


def test_source_default_drift_rejected(*, monkeypatch):
    from hitl_pmp.methods.belief_space.tossing3d_method import Tossing3DPomdpMethod

    monkeypatch.setattr(Tossing3DPomdpMethod.model_fields["exploration_epsilon"], "default", 0.1)
    with pytest.raises(ValueError, match="source default drift"):
        source_checks()


def test_all_arms_pass_real_method_cli_and_reach_constructor(*, full_manifest, monkeypatch):
    from hitl_pmp.cli import METHODS
    from hitl_pmp.core.method.skill_provider import DomainContext
    from hitl_pmp.environments.tossing3d.cli import Tossing3DCli
    from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
    from hitl_pmp.environments.tossing3d.skill_provider import (
        Tossing3DOracle,
        Tossing3DSkillProvider,
    )

    built = []

    def capture(*, args, method_factory, **unused):
        env = Tossing3DEnvironment(scene_bg=False, enable_bin_pouring=False)
        method = method_factory(
            DomainContext(
                env=env,
                skill_provider=Tossing3DSkillProvider(env=env),
                oracle=Tossing3DOracle(env=env),
            )
        )
        built.append(method)
        for name in (
            "goal_pursuit_horizon",
            "goal_pursuit_init_cycles",
            "goal_pursuit_interval",
            "exploration_epsilon",
            "sampler_max_train_iters",
            "competence_window_size",
            "competence_recency_size",
            "reproduce_predicators_explore_target_only",
            "reproduce_predicators_double_observe",
            "reproduce_predicators_skip_perfect",
            "reproduce_predicators_practice_target_history",
            "reproduce_predicators_seen_task_order",
            "reproduce_predicators_random_when_stranded",
            "competence_sharing",
            "practice_groundings",
        ):
            assert getattr(method, name) == getattr(args, name)
        assert method.score_human_skills is True
        assert method.reset_cost_gate is False
        assert method.always_ask_reset is None
        assert method.draw_recorder is not None
        if args.method == "ees":
            assert method.ees_decision_log == args.output_dir / "ees_decisions.jsonl"
        if args.method == "pomdp":
            assert args.ees_score_human_skills is None
            assert args.record_ees_decisions is False
            assert method.ees_decision_log is None
            assert method.pomdp_solver == args.pomdp_solver
            assert method.pomdp_search_depth == args.pomdp_search_depth
            assert method.pomdp_max_search_iterations == args.pomdp_max_search_iterations
            assert method.pomdp_observation_probability_weight == (
                args.pomdp_observation_probability_weight
            )
            assert method.decision_log is not None

    monkeypatch.setattr(Tossing3DCli, "run_method", staticmethod(capture))
    for row in full_manifest["runs"][:20]:
        args = Cli.parse_args(argv=row["argv"])
        Cli.require_explicit_observation_weight(argv=row["argv"], args=args)
        METHODS[args.method].run(args=args, env_cli=Tossing3DCli)
    assert len(built) == 20


def test_cli_prepares_validates_and_refuses_overwrite(*, tmp_path):
    output = tmp_path / "manifest.json"
    overrides = tmp_path / "freeze.json"
    overrides.write_text('{"sampler_max_train_iters": 123}')
    argv = [
        "--size",
        "pilot",
        "--goal-horizon",
        str(TEST_HORIZON),
        "--results-root",
        str(tmp_path / "results"),
        "--freeze-overrides",
        str(overrides),
    ]
    assert main(argv=["check", *argv]) == 0
    assert main(argv=["prepare", *argv, "--out", str(output)]) == 0
    assert len(json.loads(output.read_text())["runs"]) == 40
    assert main(argv=["validate", "--manifest", str(output)]) == 0
    with pytest.raises(SystemExit):
        main(argv=["prepare", *argv, "--out", str(output)])
    assert not (tmp_path / "results").exists()
    assert sorted(path.name for path in tmp_path.iterdir()) == ["freeze.json", "manifest.json"]


@pytest.mark.parametrize("horizon_args", [[], ["--goal-horizon", "0"], ["--goal-horizon", "-1"]])
def test_cli_requires_explicit_positive_horizon(*, tmp_path, horizon_args):
    with pytest.raises(SystemExit):
        main(
            argv=[
                "prepare",
                "--results-root",
                str(tmp_path),
                "--out",
                str(tmp_path / "manifest.json"),
                *horizon_args,
            ]
        )
    assert not (tmp_path / "manifest.json").exists()
