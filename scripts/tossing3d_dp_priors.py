"""Collect independent shared-learner curves and fit DP-specific competence priors."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from hitl_pmp.config_snapshot import ConfigSnapshot
from hitl_pmp.core.method.types import GroundSkill
from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.predicates import CLOSED_EMPTY
from hitl_pmp.environments.tossing3d.sides import Tossing3DSides
from hitl_pmp.environments.tossing3d.skill_provider import Tossing3DSkillProvider
from hitl_pmp.environments.tossing3d.skills import Tossing3DSkills
from hitl_pmp.methods.practice_makes_perfect.ees_method import EesMethod
from hitl_pmp.step_protocol import DeploymentSnapshot

REPO = Path(__file__).resolve().parents[1]
FIT_SEEDS = (2000, 2001, 2002)
VALIDATION_SEEDS = (2010, 2011, 2012)
CHECKPOINTS = (0, 20, 60)
PROBES = 16
REFIT_INTERVAL = 10
PRIOR_STRENGTH = 2.0
PICK = "PickCube"
TOSS = "MoveToTossLocationAndToss"
OPEN = "OpenGripper"
SIMULATOR_PINS = {
    "kinder": "490831fa54a40c4b2b4570d45bb08e3dd63c24ef",
    "kinder_models": "032b42634a015c7f710667b3a791836fa839f6ff",
}


class Tossing3DDpPriors:
    @staticmethod
    def open_protocol() -> dict:
        return {
            "environment": "tossing3d",
            "layout": "barrier",
            "variant": "o1",
            "probes_per_seed": PROBES,
            "scene_rng_stream": 900003,
            "setup": "shared PickCube controller then reset_movables; verify ClosedEmpty",
            "controller": "shared parameterless OpenGripper via EesMethod.execute_ground_skill",
            "fit_rule": "Jeffreys-smoothed fit success rate; strength two; zero learning rate",
            "uses_experiment_evaluations": False,
        }

    @staticmethod
    def collect_open(*, seed: int, output: Path) -> dict:
        if seed not in (*FIT_SEEDS, *VALIDATION_SEEDS):
            raise ValueError("prior calibration requires a reserved fit or validation seed")
        for package, expected in SIMULATOR_PINS.items():
            actual, dirty = ConfigSnapshot._repo_state(package=package)
            if dirty or actual != expected:
                raise ValueError(f"OpenGripper probe simulator mismatch: {package}")
        output.mkdir(parents=True, exist_ok=False)
        env = Tossing3DEnvironment()
        provider = Tossing3DSkillProvider(env=env)
        method = EesMethod(env=env, skill_provider=provider, seed=seed)
        pick = GroundSkill(
            skill=Tossing3DSkills.PICK_CUBE,
            objects=(env.robot, env.cube, env.barrier, Tossing3DSides.robot, env.bin),
        )
        recovery = GroundSkill(
            skill=Tossing3DSkills.OPEN_GRIPPER, objects=(env.robot, env.cube, env.bin)
        )
        metadata = {
            "seed": seed,
            "split": "fit" if seed in FIT_SEEDS else "validation",
            "protocol": Tossing3DDpPriors.open_protocol(),
            "source_sha256": Tossing3DDpPriors.source_hashes(),
            "config_snapshot": ConfigSnapshot.collect(
                args=argparse.Namespace(seed=seed), fd_exec_path=None
            ).model_dump(),
            "complete": False,
        }
        metadata_path = output / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        scenes = np.random.default_rng(np.random.SeedSequence([seed, 900003]))
        completed = 0
        try:
            for _attempt in range(PROBES * 10):
                scene = int(scenes.integers(2**31 - 1))
                state = env.reset_to_seed(seed=scene)
                before = method.abstract_state(state=state)
                if not pick.preconditions <= before:
                    raise RuntimeError("OpenGripper probe pick preconditions do not hold")
                action, _record = method.execute_ground_skill(
                    ground_skill=pick, state=state, explore=False
                )
                state = env.take_action(action=action.action)
                picked = pick.expected_add_effects(before_atoms=before) <= method.abstract_state(
                    state=state
                )
                setup_error = env.last_skill_error()
                with (output / "rollouts.jsonl").open("a") as stream:
                    stream.write(
                        json.dumps({
                            "phase": "setup",
                            "skill": PICK,
                            "scene_seed": scene,
                            "success": picked,
                            "skill_error": setup_error,
                        })
                        + "\n"
                    )
                if setup_error and "Robot starts in collision with" in setup_error:
                    raise RuntimeError(f"invalid OpenGripper setup start contact: {setup_error}")
                if not picked:
                    continue
                if not env.reset_movables():
                    raise RuntimeError("OpenGripper probe movable reset failed")
                state = env.get_current_state()
                before = method.abstract_state(state=state)
                closed = CLOSED_EMPTY.holds(state, (env.robot, env.cube))
                if not closed or not recovery.preconditions <= before:
                    raise RuntimeError(
                        "OpenGripper probe requires actual ClosedEmpty preconditions"
                    )
                action, record = method.execute_ground_skill(
                    ground_skill=recovery, state=state, explore=False
                )
                if record is not None:
                    raise RuntimeError("parameterless OpenGripper unexpectedly used a sampler")
                state = env.take_action(action=action.action)
                success = recovery.expected_add_effects(
                    before_atoms=before
                ) <= method.abstract_state(state=state)
                error = env.last_skill_error()
                with (output / "rollouts.jsonl").open("a") as stream:
                    stream.write(
                        json.dumps({
                            "phase": "probe",
                            "skill": OPEN,
                            "scene_seed": scene,
                            "success": success,
                            "preconditions_satisfied": True,
                            "closed_empty": closed,
                            "consultation": "no_sampler",
                            "skill_error": error,
                        })
                        + "\n"
                    )
                if error and "Robot starts in collision with" in error:
                    raise RuntimeError(f"invalid OpenGripper probe start contact: {error}")
                completed += 1
                print(
                    json.dumps({"seed": seed, "probes": completed, "success": success}), flush=True
                )
                if completed == PROBES:
                    break
            if completed != PROBES:
                raise RuntimeError("insufficient ClosedEmpty OpenGripper probes")
            metadata["complete"] = True
            metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        finally:
            env.close()
        return metadata

    @staticmethod
    def source_hashes() -> dict[str, str]:
        files = {
            Path(__file__).resolve(),
            REPO / "src/hitl_pmp/step_protocol.py",
            REPO / "src/hitl_pmp/core/method/types.py",
            *list((REPO / "src/hitl_pmp/environments/tossing3d").glob("*.py")),
            *[
                path
                for path in (REPO / "src/hitl_pmp/methods/practice_makes_perfect").glob("*.py")
                if not path.name.startswith("dp_") and path.name != "cli.py"
            ],
        }
        return {
            str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(files)
        }

    @staticmethod
    def protocol() -> dict:
        return {
            "environment": "tossing3d",
            "layout": "barrier",
            "variant": "o1",
            "checkpoints": list(CHECKPOINTS),
            "probes_per_checkpoint": PROBES,
            "refit_interval_target_rollouts": REFIT_INTERVAL,
            "sampler_max_train_iters": 10000,
            "exploration_epsilon": 0.5,
            "num_candidates": 100,
            "prior_strength": PRIOR_STRENGTH,
            "scene_reset": "fresh independent scene before each pick/toss pair",
            "probe_policy": "greedy deployment clone; no training or training-RNG mutation",
            "fit_rule": "Jeffreys-smoothed initial probe rate; nonnegative anchored linear slope",
            "uses_experiment_evaluations": False,
        }

    @staticmethod
    def collect(*, seed: int, output: Path) -> dict:
        if seed not in (*FIT_SEEDS, *VALIDATION_SEEDS):
            raise ValueError("prior calibration requires a reserved fit or validation seed")
        output.mkdir(parents=True, exist_ok=False)
        env = Tossing3DEnvironment()
        provider = Tossing3DSkillProvider(env=env)
        learner = EesMethod(
            env=env,
            skill_provider=provider,
            seed=seed,
            sampler_max_train_iters=10000,
            exploration_epsilon=0.5,
            num_candidates=100,
        )
        pick = GroundSkill(
            skill=Tossing3DSkills.PICK_CUBE,
            objects=(env.robot, env.cube, env.barrier, Tossing3DSides.robot, env.bin),
        )
        toss = GroundSkill(
            skill=Tossing3DSkills.MOVE_TO_TOSS_LOCATION_AND_TOSS,
            objects=(env.robot, env.bin, env.cube, env.barrier, Tossing3DSides.opposite),
        )
        train_scenes = np.random.default_rng(np.random.SeedSequence([seed, 900001]))
        probe_scenes = np.random.default_rng(np.random.SeedSequence([seed, 900002]))
        metadata = {
            "seed": seed,
            "split": "fit" if seed in FIT_SEEDS else "validation",
            "protocol": Tossing3DDpPriors.protocol(),
            "source_sha256": Tossing3DDpPriors.source_hashes(),
            "config_snapshot": ConfigSnapshot.collect(
                args=argparse.Namespace(seed=seed), fd_exec_path=None
            ).model_dump(),
            "complete": False,
        }
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        trained = 0
        try:
            for checkpoint in CHECKPOINTS:
                setup_attempts = 0
                while trained < checkpoint:
                    setup_attempts += 1
                    if setup_attempts > max(CHECKPOINTS) * 10:
                        raise RuntimeError("too many failed setups during calibration training")
                    scene = int(train_scenes.integers(2**31 - 1))
                    attempted = Tossing3DDpPriors.pair(
                        method=learner,
                        env=env,
                        pick=pick,
                        toss=toss,
                        scene=scene,
                        training=True,
                        trained=trained,
                        checkpoint=checkpoint,
                        output=output,
                    )
                    if attempted:
                        trained += 1
                        if trained % REFIT_INTERVAL == 0:
                            learner.fit_samplers()
                before = DeploymentSnapshot.encode(method=learner)
                probe = DeploymentSnapshot.restore(raw=before, env=env, provider=provider)
                completed = 0
                for _attempt in range(PROBES * 10):
                    scene = int(probe_scenes.integers(2**31 - 1))
                    if Tossing3DDpPriors.pair(
                        method=probe,
                        env=env,
                        pick=pick,
                        toss=toss,
                        scene=scene,
                        training=False,
                        trained=trained,
                        checkpoint=checkpoint,
                        output=output,
                    ):
                        completed += 1
                    if completed == PROBES:
                        break
                if completed != PROBES:
                    raise RuntimeError("insufficient reachable toss probes; inspect setup failures")
                if DeploymentSnapshot.encode(method=learner) != before:
                    raise RuntimeError("competence probes mutated the training learner")
                print(
                    json.dumps({"seed": seed, "checkpoint": checkpoint, "probes": completed}),
                    flush=True,
                )
            metadata["complete"] = True
            (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        finally:
            env.close()
        return metadata

    @staticmethod
    def pair(
        *,
        method: EesMethod,
        env: Tossing3DEnvironment,
        pick: GroundSkill,
        toss: GroundSkill,
        scene: int,
        training: bool,
        trained: int,
        checkpoint: int,
        output: Path,
    ) -> bool:
        state = env.reset_to_seed(seed=scene)
        for ground in (pick, toss):
            before_atoms = method.abstract_state(state=state)
            if not ground.preconditions <= before_atoms:
                raise RuntimeError(f"calibration preconditions do not hold: {ground}")
            action, record = method.execute_ground_skill(
                ground_skill=ground, state=state, explore=training and ground == toss
            )
            state = env.take_action(action=action.action)
            success = ground.expected_add_effects(
                before_atoms=before_atoms
            ) <= method.abstract_state(state=state)
            error = env.last_skill_error()
            entry = {
                "phase": "training" if training else "probe",
                "checkpoint": checkpoint,
                "training_episodes": trained,
                "scene_seed": scene,
                "skill": ground.skill.name,
                "success": success,
                "consultation": record.consultation.value if record is not None else "no_sampler",
                "skill_error": error,
            }
            with (output / "rollouts.jsonl").open("a") as stream:
                stream.write(json.dumps(entry) + "\n")
            if error and "Robot starts in collision with" in error:
                raise RuntimeError(f"calibration encountered invalid start contact: {error}")
            if training and record is not None:
                method.observe_sampler_outcome(
                    skill_name=record.skill_name,
                    param_dim=record.param_dim,
                    sampler_input=record.sampler_input,
                    success=success,
                )
            if ground == pick and not success:
                return False
        return True

    @staticmethod
    def fit(*, root: Path, open_root: Path | None = None) -> dict:
        if open_root is None:
            raise ValueError("OpenGripper requires independent ClosedEmpty competence probes")
        runs: list[dict] = []
        collector_hashes: set[str] = set()
        stacks: set[tuple[str, str, str]] = set()
        driver = str(Path(__file__).resolve().relative_to(REPO))
        for seed in (*FIT_SEEDS, *VALIDATION_SEEDS):
            directory = root / str(seed)
            metadata = json.loads((directory / "metadata.json").read_text())
            if not metadata["complete"] or metadata["seed"] != seed:
                raise ValueError("incomplete or mismatched prior calibration run")
            recorded = dict(metadata["source_sha256"])
            current = Tossing3DDpPriors.source_hashes()
            collector_hashes.add(recorded.pop(driver, "missing"))
            current.pop(driver)
            if recorded != current:
                raise ValueError("stale shared-learner or simulator source")
            if metadata["protocol"] != Tossing3DDpPriors.protocol():
                raise ValueError("mismatched prior calibration protocol")
            snapshot = metadata["config_snapshot"]
            if (
                snapshot["kindergarden_commit"] != SIMULATOR_PINS["kinder"]
                or snapshot["kinder_models_commit"] != SIMULATOR_PINS["kinder_models"]
                or snapshot["kindergarden_dirty"]
                or snapshot["kinder_models_dirty"]
            ):
                raise ValueError("prior calibration used unpinned or modified simulators")
            stacks.add(
                tuple(snapshot[key] for key in ("python_version", "torch_version", "numpy_version"))
            )
            rows = [
                json.loads(line) for line in (directory / "rollouts.jsonl").read_text().splitlines()
            ]
            for checkpoint in CHECKPOINTS:
                probes = [
                    row
                    for row in rows
                    if row["phase"] == "probe"
                    and row["checkpoint"] == checkpoint
                    and row["skill"] == TOSS
                ]
                if len(probes) != PROBES:
                    raise ValueError("incomplete checkpoint probes")
            runs.append({"seed": seed, "rows": rows})
        if len(collector_hashes) != 1 or "missing" in collector_hashes:
            raise ValueError("missing or inconsistent collection-driver provenance")
        if len(stacks) != 1:
            raise ValueError("calibration numerical environments differ")
        skills = {}
        diagnostics = {}
        for name in (PICK, TOSS):
            fit_rows = [
                row
                for run in runs
                if run["seed"] in FIT_SEEDS
                for row in run["rows"]
                if row["phase"] == "probe" and row["skill"] == name
            ]
            initial = [row for row in fit_rows if row["checkpoint"] == 0]
            probability = (sum(row["success"] for row in initial) + 0.5) / (len(initial) + 1)
            denominator = sum(row["checkpoint"] ** 2 for row in fit_rows)
            rate = (
                max(
                    0.0,
                    sum(row["checkpoint"] * (row["success"] - probability) for row in fit_rows)
                    / denominator,
                )
                if name == TOSS
                else 0.0
            )
            skills[name] = {
                "alpha": PRIOR_STRENGTH * probability,
                "beta": PRIOR_STRENGTH * (1 - probability),
                "initial_learning_rate": rate,
            }
            held_out = [
                row
                for run in runs
                if run["seed"] in VALIDATION_SEEDS
                for row in run["rows"]
                if row["phase"] == "probe" and row["skill"] == name
            ]
            calibrated_brier = sum(
                (min(1, probability + rate * row["checkpoint"]) - row["success"]) ** 2
                for row in held_out
            ) / len(held_out)
            old_brier = sum(
                (min(1, 10 / 11 + 0.1 * row["checkpoint"]) - row["success"]) ** 2
                for row in held_out
            ) / len(held_out)
            diagnostics[name] = {
                "initial_competence": probability,
                "initial_probe_successes": sum(row["success"] for row in initial),
                "initial_probe_trials": len(initial),
                "predicted_saturation_rollouts": math.ceil((1 - probability) / rate)
                if rate > 0
                else None,
                "validation_brier": calibrated_brier,
                "old_defaults_validation_brier": old_brier,
                "checkpoints": {
                    split: [
                        {
                            "training_episodes": checkpoint,
                            "successes": sum(
                                row["success"]
                                for run in runs
                                if run["seed"] in seeds
                                for row in run["rows"]
                                if row["phase"] == "probe"
                                and row["skill"] == name
                                and row["checkpoint"] == checkpoint
                            ),
                            "trials": sum(
                                1
                                for run in runs
                                if run["seed"] in seeds
                                for row in run["rows"]
                                if row["phase"] == "probe"
                                and row["skill"] == name
                                and row["checkpoint"] == checkpoint
                            ),
                        }
                        for checkpoint in CHECKPOINTS
                    ]
                    for split, seeds in (("fit", FIT_SEEDS), ("validation", VALIDATION_SEEDS))
                },
            }
        open_prior, open_diagnostics, open_hashes = Tossing3DDpPriors.fit_open(
            root=open_root, numerical_stack=next(iter(stacks))
        )
        skills[OPEN] = open_prior
        diagnostics[OPEN] = open_diagnostics
        passed = all(
            item["validation_brier"] <= item["old_defaults_validation_brier"]
            for item in diagnostics.values()
        )
        return {
            "schema_version": 1,
            "environment": "tossing3d",
            "layout": "barrier",
            "variant": "o1",
            "skills": skills,
            "diagnostics": diagnostics,
            "validation_passed": passed,
            "fit_seeds": list(FIT_SEEDS),
            "validation_seeds": list(VALIDATION_SEEDS),
            "protocol": Tossing3DDpPriors.protocol(),
            "source_sha256": Tossing3DDpPriors.source_hashes(),
            "collection_driver_sha256": sorted(collector_hashes),
            "open_gripper_collection_driver_sha256": open_hashes,
            "open_gripper_protocol": Tossing3DDpPriors.open_protocol(),
            "simulator_pins": SIMULATOR_PINS,
            "numerical_stack": dict(
                zip(
                    ("python_version", "torch_version", "numpy_version"),
                    next(iter(stacks)),
                    strict=True,
                )
            ),
            "uses_experiment_evaluations": False,
            "limitations": [
                "Fresh calibration scenes differ from reset-free experiment states.",
                "Refits occur every ten training tosses, not at experiment period boundaries.",
                "Only 0--60 rollouts are probed; predicted saturation is not validated mastery.",
                "PickCube has no learned parameters; its learning-rate prior is zero.",
                "OpenGripper is probed after pick and movable reset, not every failed-grasp state.",
                "OpenGripper has no learned parameters; its learning-rate prior is zero.",
            ],
        }

    @staticmethod
    def fit_open(
        *, root: Path, numerical_stack: tuple[str, str, str]
    ) -> tuple[dict, dict, list[str]]:
        rows_by_seed = {}
        hashes = set()
        driver = str(Path(__file__).resolve().relative_to(REPO))
        for seed in (*FIT_SEEDS, *VALIDATION_SEEDS):
            directory = root / str(seed)
            metadata = json.loads((directory / "metadata.json").read_text())
            if not metadata["complete"] or metadata["seed"] != seed:
                raise ValueError("incomplete or mismatched OpenGripper probe run")
            if metadata["protocol"] != Tossing3DDpPriors.open_protocol():
                raise ValueError("mismatched OpenGripper probe protocol")
            recorded = dict(metadata["source_sha256"])
            current = Tossing3DDpPriors.source_hashes()
            hashes.add(recorded.pop(driver, "missing"))
            current.pop(driver)
            if recorded != current:
                raise ValueError("stale OpenGripper shared-learner or simulator source")
            snapshot = metadata["config_snapshot"]
            if (
                snapshot["kindergarden_commit"] != SIMULATOR_PINS["kinder"]
                or snapshot["kinder_models_commit"] != SIMULATOR_PINS["kinder_models"]
                or snapshot["kindergarden_dirty"]
                or snapshot["kinder_models_dirty"]
                or tuple(
                    snapshot[key] for key in ("python_version", "torch_version", "numpy_version")
                )
                != numerical_stack
            ):
                raise ValueError("OpenGripper probes used a different simulator or numerical stack")
            rows = [
                json.loads(line) for line in (directory / "rollouts.jsonl").read_text().splitlines()
            ]
            probes = [row for row in rows if row["phase"] == "probe" and row["skill"] == OPEN]
            if len(probes) != PROBES or len({row["scene_seed"] for row in probes}) != PROBES:
                raise ValueError("incomplete or repeated OpenGripper probes")
            if any(
                row.get("closed_empty") is not True
                or row.get("preconditions_satisfied") is not True
                or row.get("consultation") != "no_sampler"
                or not isinstance(row["success"], bool)
                or "Robot starts in collision with" in (row.get("skill_error") or "")
                for row in probes
            ):
                raise ValueError("invalid ClosedEmpty OpenGripper probes")
            rows_by_seed[seed] = probes
        if len(hashes) != 1 or "missing" in hashes:
            raise ValueError("missing or inconsistent OpenGripper collection-driver provenance")
        fitted = [row for seed in FIT_SEEDS for row in rows_by_seed[seed]]
        held_out = [row for seed in VALIDATION_SEEDS for row in rows_by_seed[seed]]
        successes = sum(row["success"] for row in fitted)
        probability = (successes + 0.5) / (len(fitted) + 1)
        return (
            {
                "alpha": PRIOR_STRENGTH * probability,
                "beta": PRIOR_STRENGTH * (1 - probability),
                "initial_learning_rate": 0.0,
            },
            {
                "initial_competence": probability,
                "initial_probe_successes": successes,
                "initial_probe_trials": len(fitted),
                "predicted_saturation_rollouts": None,
                "validation_probe_successes": sum(row["success"] for row in held_out),
                "validation_probe_trials": len(held_out),
                "validation_brier": sum((probability - row["success"]) ** 2 for row in held_out)
                / len(held_out),
                "old_defaults_validation_brier": sum(
                    (10 / 11 - row["success"]) ** 2 for row in held_out
                )
                / len(held_out),
            },
            sorted(hashes),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    collect = subparsers.add_parser("collect")
    collect.add_argument("--seed", type=int, required=True)
    collect.add_argument("--output", type=Path, required=True)
    collect_open = subparsers.add_parser("collect-open")
    collect_open.add_argument("--seed", type=int, required=True)
    collect_open.add_argument("--output", type=Path, required=True)
    fit = subparsers.add_parser("fit")
    fit.add_argument("--root", type=Path, required=True)
    fit.add_argument("--output", type=Path, required=True)
    fit.add_argument("--open-root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "collect":
        Tossing3DDpPriors.collect(seed=args.seed, output=args.output)
    elif args.command == "collect-open":
        Tossing3DDpPriors.collect_open(seed=args.seed, output=args.output)
    else:
        report = Tossing3DDpPriors.fit(root=args.root, open_root=args.open_root)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        if not report["validation_passed"]:
            raise SystemExit("calibrated priors failed the held-out predictive check")


if __name__ == "__main__":
    main()
