"""Collect independent shared-learner curves and fit DP-specific competence priors.

Open probes cover closed-empty recovery and release immediately after a successful
Pick separately. DP currently requires one scalar prior per skill. Fitting must
explicitly select ``equal-context``: the arithmetic mean of the two independently
Jeffreys-smoothed fit rates, with total Beta strength two and zero learning rate.
This is a balanced calibration benchmark, not an estimated deployment mixture or
evidence of equal context difficulty. Validation never chooses the weights; the
scalar must pass the existing Brier comparison in each context separately.
"""

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

from hitl_pmp.config_snapshot import ConfigSnapshot
from hitl_pmp.core.method.types import GroundSkill
from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.predicates import CLOSED_EMPTY, HOLDING
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
OPEN_CONTEXTS = {"closed_empty": 900003, "holding": 900004}
ENVIRONMENT_SETTINGS = {
    "layout": "barrier",
    "variant": "o1",
    "bin_mass_kg": 1.0,
    "cube_mass_kg": 0.1,
}
COMPILED_MASSES = {"bin_0": 1.0, "cube_0": 0.1}
SIMULATOR_PINS = {
    "kinder": "490831fa54a40c4b2b4570d45bb08e3dd63c24ef",
    "kinder_models": "032b42634a015c7f710667b3a791836fa839f6ff",
}


class Tossing3DDpPriors:
    @staticmethod
    def verify_metadata(*, metadata: dict, seed: int) -> None:
        if metadata.get("split") != ("fit" if seed in FIT_SEEDS else "validation"):
            raise ValueError("mismatched calibration seed split")
        masses = metadata.get("compiled_masses_kg", {})
        if set(masses) != set(COMPILED_MASSES) or any(
            not isinstance(masses[name], (int, float))
            or not math.isclose(masses[name], expected, rel_tol=1e-9, abs_tol=1e-12)
            for name, expected in COMPILED_MASSES.items()
        ):
            raise ValueError("missing or mismatched compiled calibration masses")

    @staticmethod
    def verify_simulators() -> None:
        for package, expected in SIMULATOR_PINS.items():
            actual, dirty = ConfigSnapshot._repo_state(package=package)
            if dirty or actual != expected:
                raise ValueError(f"prior calibration simulator mismatch: {package}")

    @staticmethod
    def compiled_masses(*, env: Tossing3DEnvironment) -> dict[str, float]:
        scene = env.kinder_backend()._object_centric()
        model = scene._robot_env.sim.model.mj_model
        masses = {
            name: float(model.body_subtreemass[model.body(name).id]) for name in COMPILED_MASSES
        }
        if any(
            not math.isclose(masses[name], expected, rel_tol=1e-9, abs_tol=1e-12)
            for name, expected in COMPILED_MASSES.items()
        ):
            raise ValueError(f"prior calibration compiled mass mismatch: {masses}")
        return masses

    @staticmethod
    def open_protocol(*, enable_bin_pouring: bool = True) -> dict:
        return {
            "environment": "tossing3d",
            **ENVIRONMENT_SETTINGS,
            "enable_bin_pouring": enable_bin_pouring,
            "probes_per_seed_per_context": PROBES,
            "context_scene_rng_streams": dict(OPEN_CONTEXTS),
            "setup": {
                "closed_empty": "shared PickCube then reset_movables; verify ClosedEmpty",
                "holding": "shared PickCube; retain held cube without reset; verify Holding",
            },
            "controller": "shared parameterless OpenGripper via EesMethod.execute_ground_skill",
            "fit_rule": "separate Jeffreys-smoothed fit rates; explicit scalar aggregation",
            "uses_experiment_evaluations": False,
        }

    @staticmethod
    def collect_open(*, seed: int, output: Path, enable_bin_pouring: bool = True) -> dict:
        if seed not in (*FIT_SEEDS, *VALIDATION_SEEDS):
            raise ValueError("prior calibration requires a reserved fit or validation seed")
        Tossing3DDpPriors.verify_simulators()
        started = time.monotonic()
        output.mkdir(parents=True, exist_ok=False)
        env = Tossing3DEnvironment(**ENVIRONMENT_SETTINGS, enable_bin_pouring=enable_bin_pouring)
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
            "protocol": Tossing3DDpPriors.open_protocol(enable_bin_pouring=enable_bin_pouring),
            "source_sha256": Tossing3DDpPriors.source_hashes(),
            "config_snapshot": ConfigSnapshot.collect(
                args=argparse.Namespace(
                    seed=seed, **ENVIRONMENT_SETTINGS, enable_bin_pouring=enable_bin_pouring
                ),
                fd_exec_path=None,
            ).model_dump(),
            "complete": False,
        }
        metadata_path = output / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        metadata["context_elapsed_seconds"] = {}
        try:
            for context, rng_stream in OPEN_CONTEXTS.items():
                context_started = time.monotonic()
                scenes = np.random.default_rng(np.random.SeedSequence([seed, rng_stream]))
                completed = 0
                for _attempt in range(PROBES * 10):
                    scene = int(scenes.integers(2**31 - 1))
                    state = env.reset_to_seed(seed=scene)
                    metadata["compiled_masses_kg"] = Tossing3DDpPriors.compiled_masses(env=env)
                    before = method.abstract_state(state=state)
                    if not pick.preconditions <= before:
                        raise RuntimeError("OpenGripper probe pick preconditions do not hold")
                    action, _record = method.execute_ground_skill(
                        ground_skill=pick, state=state, explore=False
                    )
                    state = env.take_action(action=action.action)
                    picked = pick.expected_add_effects(
                        before_atoms=before
                    ) <= method.abstract_state(state=state)
                    setup_error = env.last_skill_error()
                    with (output / "rollouts.jsonl").open("a") as stream:
                        stream.write(
                            json.dumps({
                                "phase": "setup",
                                "skill": PICK,
                                "scene_seed": scene,
                                "context": context,
                                "success": picked,
                                "skill_error": setup_error,
                            })
                            + "\n"
                        )
                    if setup_error and "Robot starts in collision with" in setup_error:
                        raise RuntimeError(
                            f"invalid OpenGripper setup start contact: {setup_error}"
                        )
                    if not picked:
                        continue
                    if context == "closed_empty" and not env.reset_movables():
                        raise RuntimeError("OpenGripper probe movable reset failed")
                    state = env.get_current_state()
                    before = method.abstract_state(state=state)
                    closed = CLOSED_EMPTY.holds(state, (env.robot, env.cube))
                    holding = HOLDING.holds(state, (env.robot, env.cube))
                    context_valid = (
                        closed and not holding
                        if context == "closed_empty"
                        else holding and not closed
                    )
                    if not context_valid or not recovery.preconditions <= before:
                        raise RuntimeError(
                            f"OpenGripper probe requires actual {context} preconditions"
                        )
                    action, record = method.execute_ground_skill(
                        ground_skill=recovery, state=state, explore=False
                    )
                    if record is not None:
                        raise RuntimeError("parameterless OpenGripper unexpectedly used a sampler")
                    state = env.take_action(action=action.action)
                    after = method.abstract_state(state=state)
                    added = recovery.expected_add_effects(before_atoms=before) <= after
                    deleted = not bool(recovery.delete_effects & after)
                    error = env.last_skill_error()
                    success = added and deleted and error is None
                    with (output / "rollouts.jsonl").open("a") as stream:
                        stream.write(
                            json.dumps({
                                "phase": "probe",
                                "skill": OPEN,
                                "scene_seed": scene,
                                "context": context,
                                "success": success,
                                "preconditions_satisfied": True,
                                "closed_empty": closed,
                                "holding": holding,
                                "add_effects_satisfied": added,
                                "delete_effects_satisfied": deleted,
                                "consultation": "no_sampler",
                                "skill_error": error,
                            })
                            + "\n"
                        )
                    if error and "Robot starts in collision with" in error:
                        raise RuntimeError(f"invalid OpenGripper probe start contact: {error}")
                    completed += 1
                    print(
                        json.dumps({
                            "seed": seed,
                            "context": context,
                            "probes": completed,
                            "success": success,
                        }),
                        flush=True,
                    )
                    if completed == PROBES:
                        break
                if completed != PROBES:
                    raise RuntimeError(f"insufficient {context} OpenGripper probes")
                metadata["context_elapsed_seconds"][context] = time.monotonic() - context_started
            metadata["complete"] = True
        finally:
            metadata["elapsed_seconds"] = time.monotonic() - started
            metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
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
    def protocol(*, enable_bin_pouring: bool = True) -> dict:
        return {
            "environment": "tossing3d",
            **ENVIRONMENT_SETTINGS,
            "enable_bin_pouring": enable_bin_pouring,
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
    def collect(*, seed: int, output: Path, enable_bin_pouring: bool = True) -> dict:
        if seed not in (*FIT_SEEDS, *VALIDATION_SEEDS):
            raise ValueError("prior calibration requires a reserved fit or validation seed")
        Tossing3DDpPriors.verify_simulators()
        started = time.monotonic()
        output.mkdir(parents=True, exist_ok=False)
        env = Tossing3DEnvironment(**ENVIRONMENT_SETTINGS, enable_bin_pouring=enable_bin_pouring)
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
            objects=(env.robot, env.bin, env.cube, env.barrier),
        )
        train_scenes = np.random.default_rng(np.random.SeedSequence([seed, 900001]))
        probe_scenes = np.random.default_rng(np.random.SeedSequence([seed, 900002]))
        metadata = {
            "seed": seed,
            "split": "fit" if seed in FIT_SEEDS else "validation",
            "protocol": Tossing3DDpPriors.protocol(enable_bin_pouring=enable_bin_pouring),
            "source_sha256": Tossing3DDpPriors.source_hashes(),
            "config_snapshot": ConfigSnapshot.collect(
                args=argparse.Namespace(
                    seed=seed, **ENVIRONMENT_SETTINGS, enable_bin_pouring=enable_bin_pouring
                ),
                fd_exec_path=None,
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
            metadata["compiled_masses_kg"] = Tossing3DDpPriors.compiled_masses(env=env)
            metadata["complete"] = True
        finally:
            metadata["elapsed_seconds"] = time.monotonic() - started
            (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
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
        Tossing3DDpPriors.compiled_masses(env=env)
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
    def fit(
        *,
        root: Path,
        open_root: Path | None = None,
        enable_bin_pouring: bool = True,
        open_aggregation: str | None = None,
    ) -> dict:
        if open_root is None:
            raise ValueError("OpenGripper requires independent closed-empty and holding probes")
        if open_aggregation != "equal-context":
            raise ValueError("explicit OpenGripper aggregation required: equal-context")
        runs: list[dict] = []
        collector_hashes: set[str] = set()
        stacks: set[tuple[str, str, str]] = set()
        driver = str(Path(__file__).resolve().relative_to(REPO))
        for seed in (*FIT_SEEDS, *VALIDATION_SEEDS):
            directory = root / str(seed)
            metadata = json.loads((directory / "metadata.json").read_text())
            if not metadata["complete"] or metadata["seed"] != seed:
                raise ValueError("incomplete or mismatched prior calibration run")
            Tossing3DDpPriors.verify_metadata(metadata=metadata, seed=seed)
            recorded = dict(metadata["source_sha256"])
            current = Tossing3DDpPriors.source_hashes()
            collector_hashes.add(recorded.pop(driver, "missing"))
            current.pop(driver)
            if recorded != current:
                raise ValueError("stale shared-learner or simulator source")
            if metadata["protocol"] != Tossing3DDpPriors.protocol(
                enable_bin_pouring=enable_bin_pouring
            ):
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
            root=open_root,
            numerical_stack=next(iter(stacks)),
            enable_bin_pouring=enable_bin_pouring,
            aggregation=open_aggregation,
        )
        skills[OPEN] = open_prior
        diagnostics[OPEN] = open_diagnostics
        provider = Tossing3DSkillProvider(
            env=Tossing3DEnvironment(**ENVIRONMENT_SETTINGS, enable_bin_pouring=enable_bin_pouring)
        )
        names = {skill.name for skill in provider.skills()}
        if missing := names - set(skills):
            raise ValueError(f"robot skills lack calibrated DP priors: {sorted(missing)}")
        if unknown := set(skills) - names:
            raise ValueError(f"calibrated skills absent from this domain: {sorted(unknown)}")
        passed = open_diagnostics["validation_passed"] and all(
            item["validation_brier"] <= item["old_defaults_validation_brier"]
            for item in diagnostics.values()
        )
        return {
            "schema_version": 1,
            "environment": "tossing3d",
            **ENVIRONMENT_SETTINGS,
            "enable_bin_pouring": enable_bin_pouring,
            "skills": skills,
            "diagnostics": diagnostics,
            "validation_passed": passed,
            "fit_seeds": list(FIT_SEEDS),
            "validation_seeds": list(VALIDATION_SEEDS),
            "protocol": Tossing3DDpPriors.protocol(enable_bin_pouring=enable_bin_pouring),
            "source_sha256": Tossing3DDpPriors.source_hashes(),
            "collection_driver_sha256": sorted(collector_hashes),
            "open_gripper_collection_driver_sha256": open_hashes,
            "open_gripper_protocol": Tossing3DDpPriors.open_protocol(
                enable_bin_pouring=enable_bin_pouring
            ),
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
                "OpenGripper contexts are post-Pick holding and post-reset closed-empty only.",
                "Equal-context aggregation is a benchmark choice, not measured deployment weights.",
                "Releasing a held cube does not guarantee a recoverable landing.",
                "OpenGripper has no learned parameters; its learning-rate prior is zero.",
            ],
        }

    @staticmethod
    def fit_open(
        *,
        root: Path,
        numerical_stack: tuple[str, str, str],
        aggregation: str,
        enable_bin_pouring: bool = True,
    ) -> tuple[dict, dict, list[str]]:
        if aggregation != "equal-context":
            raise ValueError("explicit OpenGripper aggregation required: equal-context")
        rows_by_seed = {}
        hashes = set()
        driver = str(Path(__file__).resolve().relative_to(REPO))
        for seed in (*FIT_SEEDS, *VALIDATION_SEEDS):
            directory = root / str(seed)
            metadata = json.loads((directory / "metadata.json").read_text())
            if not metadata["complete"] or metadata["seed"] != seed:
                raise ValueError("incomplete or mismatched OpenGripper probe run")
            Tossing3DDpPriors.verify_metadata(metadata=metadata, seed=seed)
            if metadata["protocol"] != Tossing3DDpPriors.open_protocol(
                enable_bin_pouring=enable_bin_pouring
            ):
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
            if len(probes) != PROBES * len(OPEN_CONTEXTS):
                raise ValueError("incomplete or repeated OpenGripper probes")
            for context in OPEN_CONTEXTS:
                selected = [row for row in probes if row.get("context") == context]
                if (
                    len(selected) != PROBES
                    or len({row["scene_seed"] for row in selected}) != PROBES
                ):
                    raise ValueError(f"incomplete or repeated {context} OpenGripper probes")
            if any(
                row.get("closed_empty") is not (row.get("context") == "closed_empty")
                or row.get("holding") is not (row.get("context") == "holding")
                or row.get("preconditions_satisfied") is not True
                or row.get("consultation") != "no_sampler"
                or not isinstance(row["success"], bool)
                or not isinstance(row.get("add_effects_satisfied"), bool)
                or not isinstance(row.get("delete_effects_satisfied"), bool)
                or row["success"]
                != (
                    row["add_effects_satisfied"]
                    and row["delete_effects_satisfied"]
                    and row.get("skill_error") is None
                )
                or "Robot starts in collision with" in (row.get("skill_error") or "")
                for row in probes
            ):
                raise ValueError("invalid context-specific OpenGripper probes")
            rows_by_seed[seed] = probes
        if len(hashes) != 1 or "missing" in hashes:
            raise ValueError("missing or inconsistent OpenGripper collection-driver provenance")
        contexts = {}
        for context in OPEN_CONTEXTS:
            fitted = [
                row for seed in FIT_SEEDS for row in rows_by_seed[seed] if row["context"] == context
            ]
            held_out = [
                row
                for seed in VALIDATION_SEEDS
                for row in rows_by_seed[seed]
                if row["context"] == context
            ]
            successes = sum(row["success"] for row in fitted)
            estimate = (successes + 0.5) / (len(fitted) + 1)
            contexts[context] = {
                "initial_competence": estimate,
                "initial_probe_successes": successes,
                "initial_probe_trials": len(fitted),
                "validation_probe_successes": sum(row["success"] for row in held_out),
                "validation_probe_trials": len(held_out),
                "validation_brier": sum((estimate - row["success"]) ** 2 for row in held_out)
                / len(held_out),
                "old_defaults_validation_brier": sum(
                    (10 / 11 - row["success"]) ** 2 for row in held_out
                )
                / len(held_out),
            }
        probability = sum(item["initial_competence"] for item in contexts.values()) / len(contexts)
        for context, item in contexts.items():
            held_out = [
                row
                for seed in VALIDATION_SEEDS
                for row in rows_by_seed[seed]
                if row["context"] == context
            ]
            item["scalar_validation_brier"] = sum(
                (probability - row["success"]) ** 2 for row in held_out
            ) / len(held_out)
            item["validation_passed"] = (
                item["scalar_validation_brier"] <= item["old_defaults_validation_brier"]
            )
        return (
            {
                "alpha": PRIOR_STRENGTH * probability,
                "beta": PRIOR_STRENGTH * (1 - probability),
                "initial_learning_rate": 0.0,
            },
            {
                "initial_competence": probability,
                "contexts": contexts,
                "aggregation": {
                    "name": aggregation,
                    "weights": dict.fromkeys(OPEN_CONTEXTS, 1 / len(OPEN_CONTEXTS)),
                    "rule": "mean of context-specific Jeffreys-smoothed fit rates",
                    "interpretation": "balanced calibration benchmark, not deployment frequencies",
                    "prior_strength": PRIOR_STRENGTH,
                    "uses_validation_outcomes": False,
                },
                "initial_probe_successes": sum(
                    item["initial_probe_successes"] for item in contexts.values()
                ),
                "initial_probe_trials": sum(
                    item["initial_probe_trials"] for item in contexts.values()
                ),
                "predicted_saturation_rollouts": None,
                "validation_probe_successes": sum(
                    item["validation_probe_successes"] for item in contexts.values()
                ),
                "validation_probe_trials": sum(
                    item["validation_probe_trials"] for item in contexts.values()
                ),
                "validation_brier": sum(
                    item["scalar_validation_brier"] for item in contexts.values()
                )
                / len(contexts),
                "old_defaults_validation_brier": sum(
                    item["old_defaults_validation_brier"] for item in contexts.values()
                )
                / len(contexts),
                "validation_passed": all(item["validation_passed"] for item in contexts.values()),
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
    fit.add_argument(
        "--open-aggregation",
        choices=["equal-context"],
        required=True,
        help="Explicit balanced benchmark: equal weight to the two smoothed Open context rates; "
        "not an estimate of deployment context frequencies.",
    )
    for command in (collect, collect_open, fit):
        command.add_argument(
            "--no-bin-pouring",
            dest="enable_bin_pouring",
            action="store_false",
            default=True,
            help="Explicitly calibrate the original three robot skills without bin pouring.",
        )
    args = parser.parse_args()
    if args.command == "collect":
        Tossing3DDpPriors.collect(
            seed=args.seed, output=args.output, enable_bin_pouring=args.enable_bin_pouring
        )
    elif args.command == "collect-open":
        Tossing3DDpPriors.collect_open(
            seed=args.seed, output=args.output, enable_bin_pouring=args.enable_bin_pouring
        )
    else:
        report = Tossing3DDpPriors.fit(
            root=args.root,
            open_root=args.open_root,
            enable_bin_pouring=args.enable_bin_pouring,
            open_aggregation=args.open_aggregation,
        )
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        if not report["validation_passed"]:
            raise SystemExit("calibrated priors failed the held-out predictive check")


if __name__ == "__main__":
    main()
