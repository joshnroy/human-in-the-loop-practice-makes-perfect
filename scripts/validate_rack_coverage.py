"""Bounded, serial existential rack coverage with durable per-attempt evidence.

Example: scripts/with_env.sh python scripts/validate_rack_coverage.py --dry-run
    --output artifacts/rack-coverage-plan --seeds 0 1 --params 0 0.15 0 --params 0.01 0.10 10

A parameter set is GatherAndSweep(delta metres, d metres, phi degrees).

Default is dry-run. Native execution requires --run --physical-witness-confirmed;
that flag attests main has cleared physical validity, it does not establish it.
Run native experiments through external run-heavy, memory caps and timeout.
Different states may use different successful param_sets. Fixed layouts are seed-
independent repeats, not additional layout coverage. Finite coverage is not proof.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import os
import signal
import subprocess
import sys
import time
import traceback
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any


class Coverage:
    """Serial supervisor; each native attempt owns a fresh disposable process."""

    @staticmethod
    def params_entry(*, params: tuple[float, ...]) -> dict[str, float]:
        """One GatherAndSweep parameter set as its manifest entry."""
        delta, d, phi = params
        return {"delta_m": float(delta), "d_m": float(d), "phi_deg": float(phi)}

    @staticmethod
    def write(*, path: Path, payload: dict[str, Any]) -> None:
        temporary = path.with_suffix(".tmp")
        with temporary.open("w") as stream:
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)

    @staticmethod
    def source_digest() -> str:
        root = Path(__file__).resolve().parents[1]
        digest = hashlib.sha256()
        paths = [Path(__file__).resolve()]
        for domain in ("racktoserving3d", "sweepoffshelf3d"):
            paths.extend(sorted((root / "src/hitl_pmp/environments" / domain).glob("*.py")))
        for path in paths:
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
        return digest.hexdigest()

    @staticmethod
    def same_initial(*, left: dict[str, Any], right: dict[str, Any]) -> bool:
        import numpy as np

        if left["locations"] != right["locations"] or left["supports"] != right["supports"]:
            return False
        if left["poses"].keys() != right["poses"].keys():
            return False
        for before, after in [
            (left["qpos"], right["qpos"]),
            (left["qvel"], right["qvel"]),
            *((left["poses"][name], right["poses"][name]) for name in left["poses"]),
        ]:
            if np.shape(before) != np.shape(after) or not np.allclose(
                before, after, atol=1e-8, rtol=0, equal_nan=False
            ):
                return False
        return True

    @staticmethod
    def compare(*, summary: dict[str, Any]) -> None:
        """Unknown cells remain unknown; existential coverage is not learning evidence."""
        states = summary["states"]
        valid = summary["stop_reason"] != "invalidating_issue"
        policies = []
        for params in summary["param_sets"]:
            attempts = [row for row in summary["attempts"] if row["params"] == params]
            successful = {row["state_id"] for row in attempts if row["success"]}
            failed = {
                row["state_id"]
                for row in attempts
                if row["status"] == "complete"
                and not row["success"]
                and not row.get("invalidating_issue")
            }
            policies.append({
                "params": params,
                "covered": len(successful),
                "tested": len(successful | failed),
                "total": len(states),
                "failed_state_ids": sorted(failed),
                "unresolved_state_ids": [
                    state["id"] for state in states if state["id"] not in successful | failed
                ],
                "all_states_covered": valid and len(successful) == len(states),
            })
        witnesses = [policy["params"] for policy in policies if policy["all_states_covered"]]
        status = "inconclusive"
        if not valid:
            status = "invalidated"
        elif witnesses:
            status = "witness_found"
        elif all(policy["failed_state_ids"] for policy in policies):
            status = "none_in_tested_grid"
        covered = sum(state.get("witness") is not None for state in states)
        summary["comparison"] = {
            "evidence_valid": valid,
            "per_state_oracle": {"covered": covered, "total": len(states)},
            "constant_policy": next(
                policy for policy in policies if policy["params"] == summary["constant_params"]
            ),
            "candidate_constant_policies": policies,
            "common_param_sets": witnesses,
            "common_params_status": status,
            "measured_distinct_layouts": sum(
                state.get("initial") is not None and not state.get("duplicate_of")
                for state in states
            ),
            "interpretation": (
                "Finite sampled states/parameter sets only. Missing or interrupted trials are "
                "unresolved. "
                "A common witness permits a constant policy on these states; absence on this "
                "grid does not prove parameter learning necessary. Robot-reset reachability "
                "and hardware validity are not tested."
            ),
        }
        summary["claim"]["all_sampled_states_covered"] = valid and covered == len(states)

    @staticmethod
    def same_layout(*, left: dict[str, Any], right: dict[str, Any]) -> bool:
        """Compare food layouts without conflating them with robot reset reachability."""
        import numpy as np

        names = set(left["locations"])
        return names == set(right["locations"]) and all(
            np.shape(left["poses"][name]) == np.shape(right["poses"][name])
            and np.allclose(left["poses"][name], right["poses"][name], atol=1e-8, rtol=0)
            for name in names
        )

    @staticmethod
    def run(
        *,
        manifest: dict[str, Any],
        output: Path,
        trial_runner: Callable[..., dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        trial_runner = Coverage.run_process if trial_runner is None else trial_runner
        summary = deepcopy(manifest)
        summary.update(attempts=[], stop_reason=None, source_sha256=Coverage.source_digest())
        deadline = time.monotonic() + manifest["limits"]["wall_seconds_total"]
        limits = manifest["limits"]
        (output / "trials").mkdir()
        Coverage.compare(summary=summary)
        Coverage.write(path=output / "summary.json", payload=summary)
        try:
            for state in summary["states"]:
                for params in manifest["param_sets"]:
                    if time.monotonic() >= deadline:
                        summary["stop_reason"] = "timeout"
                        break
                    if len(summary["attempts"]) >= limits["max_trials_total"]:
                        summary["stop_reason"] = "max_trials"
                        break
                    if Coverage.source_digest() != summary["source_sha256"]:
                        summary["stop_reason"] = "invalidating_issue"
                        summary["invalidating_issue"] = "source changed during coverage"
                        break
                    trial_dir = output / "trials" / f"{len(summary['attempts']):04d}"
                    trial_dir.mkdir()
                    request = {
                        "state": deepcopy(state),
                        "params": params,
                        "config": manifest["config"],
                        "limits": limits,
                        "deadline": min(
                            deadline, time.monotonic() + limits["wall_seconds_per_trial"]
                        ),
                        "reference_initial": state.get("initial"),
                    }
                    Coverage.write(path=trial_dir / "request.json", payload=request)
                    attempt = {
                        "state_id": state["id"],
                        "params": params,
                        "directory": str(trial_dir),
                        "status": "running",
                        "success": False,
                    }
                    summary["attempts"].append(attempt)
                    summary["native_workers_started"] += int(trial_runner == Coverage.run_process)
                    state["status"] = "searching"
                    Coverage.write(path=output / "summary.json", payload=summary)
                    result = trial_runner(request=request, output=trial_dir)
                    attempt.update(result)
                    initial = result.get("initial")
                    if initial is not None:
                        reference = state.get("initial")
                        attempt["initial_matches_reference"] = (
                            reference is None
                            or Coverage.same_initial(left=reference, right=initial)
                        )
                        if not attempt["initial_matches_reference"]:
                            attempt["invalidating_issue"] = (
                                "initial state differs across param_sets"
                            )
                        if reference is None:
                            state["initial"] = initial
                            state["measured_locations"] = initial["locations"]
                            state["duplicate_of"] = [
                                other["id"]
                                for other in summary["states"]
                                if other is not state
                                and other.get("initial") is not None
                                and Coverage.same_layout(left=other["initial"], right=initial)
                            ]
                    if Coverage.source_digest() != summary["source_sha256"]:
                        attempt["invalidating_issue"] = "source changed during coverage"
                    attempt["success"] = bool(
                        result.get("status") == "complete"
                        and initial is not None
                        and all(
                            result.get(key) is True
                            for key in (
                                "success",
                                "task_succeeded",
                                "scraper_home",
                                "settled",
                                "return_completed",
                            )
                        )
                        and not result.get("errors")
                        and not attempt.get("invalidating_issue")
                    )
                    if attempt.get("invalidating_issue"):
                        state["status"] = "invalid"
                        summary["stop_reason"] = "invalidating_issue"
                    elif result["status"] == "interrupted" or (
                        result["status"] in ("timeout", "worker_crashed")
                        and manifest["search_mode"] != "grid"
                    ):
                        state["status"] = "incomplete"
                        summary["stop_reason"] = result["status"]
                    elif result["status"] in ("timeout", "worker_crashed"):
                        state["status"] = "covered" if state.get("witness") else "incomplete"
                    elif attempt["success"]:
                        state["status"] = "covered"
                        state.setdefault("witness", params)
                        state.setdefault("successful_param_sets", []).append(params)
                    else:
                        state["status"] = "covered" if state.get("witness") else "uncovered"
                    Coverage.compare(summary=summary)
                    Coverage.write(path=output / "summary.json", payload=summary)
                    if summary["stop_reason"] or (
                        attempt["success"] and manifest["search_mode"] == "existential"
                    ):
                        break
                if summary["stop_reason"]:
                    break
        except KeyboardInterrupt:
            summary["stop_reason"] = "interrupted"
        except Exception:
            summary["stop_reason"] = "invalidating_issue"
            summary["traceback"] = traceback.format_exc()
        finally:
            Coverage.compare(summary=summary)
            summary["stop_reason"] = summary["stop_reason"] or "search_complete"
            Coverage.write(path=output / "summary.json", payload=summary)
        return summary

    @staticmethod
    def run_process(*, request: dict[str, Any], output: Path) -> dict[str, Any]:
        command = [
            sys.executable,
            "-u",
            str(Path(__file__).resolve()),
            "--worker",
            str(output / "request.json"),
            "--physical-witness-confirmed",
        ]
        return Coverage.supervise(command=command, deadline=request["deadline"], output=output)

    @staticmethod
    def supervise(*, command: list[str], deadline: float, output: Path) -> dict[str, Any]:
        status = None
        with (output / "native.log").open("w") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            try:
                process.wait(timeout=max(0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                status = "timeout"
            except KeyboardInterrupt:
                status = "interrupted"
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=1)
        path = output / "trial.json"
        result = json.loads(path.read_text()) if path.exists() else {}
        if status or process.returncode != 0 or result.get("status") == "running" or not result:
            result.update(status=status or "worker_crashed", success=False)
        result["worker_returncode"] = process.returncode
        Coverage.write(path=path, payload=result)
        return result


class TrialLimit(BaseException):
    """A budget boundary must not be swallowed by a native skill's error handler."""


class NativeTrial:
    """Only initialization and the recorded human layout setup may write poses."""

    @staticmethod
    def make_env(*, request: dict[str, Any]) -> Any:
        root = Path(__file__).resolve().parents[1]
        for module, relative in (
            ("kinder", "reference/kindergarden"),
            ("kinder_models", "reference/kinder-baselines"),
        ):
            spec = importlib.util.find_spec(module)
            if (
                spec is None
                or spec.origin is None
                or not Path(spec.origin).resolve().is_relative_to((root / relative).resolve())
            ):
                raise RuntimeError(f"{module} must resolve inside this worktree's {relative}")
        from hitl_pmp.environments.racktoserving3d.config import RackServingConfig
        from hitl_pmp.environments.racktoserving3d.environment import RackServingEnvironment

        return RackServingEnvironment(
            config=RackServingConfig(**request["config"]),
            scene_bg=False,
            skill_step_limit=request["limits"]["control_steps_per_skill"],
        )

    @staticmethod
    def resolve_params(*, params: dict[str, Any]) -> tuple[float, float, float]:
        """The fixed (delta, d, phi) of one GatherAndSweep parameter set."""
        return float(params["delta_m"]), float(params["d_m"]), float(params["phi_deg"])

    @staticmethod
    def physical_category(*, location: str | None, supports: frozenset[str], settled: bool) -> str:
        """Where a food physically ended, independent of any controller error."""
        if location == "HoldingFood":
            return "held"
        if location == "OnTray":
            return "on_plate" if settled else "on_plate_unsettled"
        if "box_0" in supports:
            return "bridging" if "rack_0" in supports else "overhang"
        if location == "OnRack" or "rack_0" in supports:
            return "on_rack"
        if location == "SpilledCounter" or "cupboard_1" in supports:
            return "on_counter"
        if location == "SpilledFloor" or "floor" in supports:
            return "floor"
        return "unsupported"

    @staticmethod
    def snapshot(*, env: Any) -> dict[str, Any]:
        backend = env.backend()
        backend.sync_clock()
        model, data = backend.model(), backend.data()
        names = (*backend.cube_names, "wiper_0", "rack_0", "box_0")
        return {
            "qpos": data.qpos.tolist(),
            "qvel": data.qvel.tolist(),
            "poses": {
                name: [
                    *data.xpos[model.body(name).id].tolist(),
                    *data.xquat[model.body(name).id].tolist(),
                ]
                for name in names
            },
            "locations": {name: backend.location(name=name) for name in backend.cube_names},
            "supports": {
                name: sorted(backend.supporting_bodies(name=name)) for name in backend.cube_names
            },
            "physical": {
                name: NativeTrial.physical_category(
                    location=backend.location(name=name),
                    supports=frozenset(backend.supporting_bodies(name=name)),
                    settled=backend.settled_on_tray(name=name),
                )
                for name in backend.cube_names
            },
            "ledger": backend.ledger.model_dump(mode="json"),
            "control_steps": backend.control_steps(),
            "simulation_seconds": float(data.time),
        }

    @staticmethod
    def execute(
        *,
        request: dict[str, Any],
        output: Path,
        env_factory: Callable[..., Any] | None = None,
    ) -> dict[str, Any]:
        import numpy as np

        from hitl_pmp.environments.racktoserving3d.skills import SKILL_IDS

        result: dict[str, Any] = {
            "status": "running",
            "success": False,
            "initial": None,
            "final": None,
            "config": request["config"],
            "phase": "initialization",
            "trace": [],
            "errors": [],
            "task_succeeded": False,
            "scraper_home": False,
            "settled": False,
            "return_completed": False,
            "invalidating_issue": None,
        }
        env = None
        initialized = False
        phase_start = 0
        limits = request["limits"]

        def persist() -> None:
            Coverage.write(path=output / "trial.json", payload=result)

        def check_budget() -> None:
            if time.monotonic() >= request["deadline"]:
                raise TrialLimit("timeout")
            steps = env.backend().control_steps()
            if (
                steps - phase_start >= limits["control_steps_per_skill"]
                or steps >= limits["control_steps_per_trial"]
            ):
                raise TrialLimit("control_step_limit")

        def checkpoint() -> None:
            if initialized and env.backend().control_steps() % 100 == 0:
                result["latest"] = NativeTrial.snapshot(env=env)
                result["trace"].append({"phase": result["phase"], "state": result["latest"]})
                persist()

        def measure() -> None:
            measured = NativeTrial.snapshot(env=env)
            result["final"] = measured
            result["trace"].append({"phase": result["phase"], "state": measured})
            persist()

        persist()
        try:
            factory = NativeTrial.make_env if env_factory is None else env_factory
            env = factory(request=request)
            env.set_control_step_observers(before=check_budget, after=checkpoint)
            env.reset_to_seed(seed=request["state"]["seed"])
            initialized = True
            if request["state"]["layout"] == "human_batch_return":
                result["phase"] = "human_batch_return"
                env.reset_movables(destination="human_batch_return")
            result["initial"] = NativeTrial.snapshot(env=env)
            ledger = result["initial"]["ledger"]
            result["actual_setup_cost"] = ledger["human_cost"]
            expected_count = int(request["state"]["layout"] == "human_batch_return")
            if ledger["human_interventions"] != expected_count or not math.isclose(
                ledger["human_cost"], request["state"]["expected_human_setup_cost"], abs_tol=1e-8
            ):
                raise RuntimeError("human setup cost/intervention count differs from manifest")
            reference = request.get("reference_initial")
            if reference is not None and not Coverage.same_initial(
                left=reference, right=result["initial"]
            ):
                raise RuntimeError("initial state differs across param_sets")
            delta, d, phi = NativeTrial.resolve_params(params=request["params"])
            result["resolved_params"] = {"delta_m": delta, "d_m": d, "phi_deg": phi}
            measure()
            for skill in ("Wait", "PickScraper", "GatherAndSweep", "ReturnScraper"):
                phase_start = env.backend().control_steps()
                result["phase"] = skill
                persist()
                action = np.array([SKILL_IDS[skill], 0, delta, d, phi])
                env.take_action(action=action)
                error = env.last_skill_error()
                if error:
                    result["errors"].append({"phase": skill, "error": error})
                    if (
                        env.backend().control_steps() - phase_start
                        >= limits["control_steps_per_skill"]
                    ):
                        raise TrialLimit("control_step_limit")
                if skill == "ReturnScraper":
                    result["return_completed"] = error is None
                measure()
                if error and skill in ("Wait", "PickScraper"):
                    break
            phase_start = env.backend().control_steps()
            result["phase"] = "settling"
            persist()
            controllers = env.rack_controllers()
            with controllers.feedback():
                for _ in range(limits["settle_control_steps"]):
                    check_budget()
                    controllers.shelf.hold(count=1, label="coverage_settling")
            result["settled"] = True
            result["task_succeeded"] = bool(env.backend().task_succeeded())
            result["scraper_home"] = bool(controllers.tool.at_home())
            measure()
            result["success"] = bool(
                result["task_succeeded"]
                and result["scraper_home"]
                and result["return_completed"]
                and not result["errors"]
            )
            result["status"] = "complete"
        except TrialLimit as error:
            result["status"] = str(error)
            result["traceback"] = traceback.format_exc()
        except BaseException:
            result["status"] = "error"
            result["invalidating_issue"] = "unexpected native/setup/instrumentation error"
            result["traceback"] = traceback.format_exc()
        finally:
            if env is not None:
                try:
                    env.set_control_step_observers()
                    if initialized:
                        result["final"] = NativeTrial.snapshot(env=env)
                    persist()
                except BaseException:
                    result["success"] = False
                    result["invalidating_issue"] = "final measurement failed"
                    result["cleanup_traceback"] = traceback.format_exc()
                finally:
                    try:
                        env.close()
                    except BaseException:
                        result["success"] = False
                        result["invalidating_issue"] = "close failed"
                        result["close_traceback"] = traceback.format_exc()
            persist()
        return result


def build_manifest(
    *,
    seeds: list[int] | None = None,
    layouts: list[str] | None = None,
    param_sets: list[tuple[float, float, float]] | None = None,
    max_trials: int = 12,
    max_seconds: float = 150,
    step_limit: int = 3500,
    scraper_tilt_degrees: float = 45,
    states: list[str] | None = None,
    settle_steps: int = 20,
    trial_seconds: float = 180,
    search_mode: str = "existential",
    constant_params: tuple[float, float, float] | None = None,
) -> dict[str, Any]:
    """Describe requested trials and limits without importing a simulator.

    Parameter sets are GatherAndSweep(delta m, d m, phi deg); speed is the skill's constant.
    """
    from hitl_pmp.environments.racktoserving3d.config import RackServingConfig
    from hitl_pmp.environments.racktoserving3d.skills import (
        GATHER_AND_SWEEP_BOUNDS,
        SWEEP_SPEED_M_S,
    )

    bounds = GATHER_AND_SWEEP_BOUNDS

    seeds = [0, 1, 2] if seeds is None else seeds
    layouts = ["fixed", "human_batch_return"] if layouts is None else layouts
    param_sets = (
        [(0.005, 0.15, 0.0), (0.005, 0.15, -10.0), (0.005, 0.15, 10.0)]
        if param_sets is None
        else param_sets
    )
    if constant_params is not None:
        param_sets = [constant_params, *param_sets]
    if not seeds or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("seeds must be nonnegative integers")
    if not layouts or any(layout not in ("fixed", "human_batch_return") for layout in layouts):
        raise ValueError("layouts must be fixed or human_batch_return")
    if not param_sets or any(
        len(params) != 3
        or not all(math.isfinite(value) for value in params)
        or not all(low <= value <= high for value, (low, high) in zip(params, bounds, strict=True))
        for params in param_sets
    ):
        (delta_low, delta_high), (d_low, d_high), (phi_low, phi_high) = bounds
        raise ValueError(
            f"param_sets require finite delta [{delta_low},{delta_high}] m,"
            f" d [{d_low},{d_high}] m and phi [{phi_low},{phi_high}] deg"
        )
    if type(max_trials) is not int or not 1 <= max_trials <= 1000:
        raise ValueError("max_trials must be an integer in [1,1000]")
    if not math.isfinite(max_seconds) or not 0 < max_seconds <= 43200:
        raise ValueError("max_seconds must be finite and in (0,43200]")
    if not math.isfinite(trial_seconds) or not 0 < trial_seconds <= 900:
        raise ValueError("trial_seconds must be finite and in (0,900]")
    if search_mode not in ("existential", "grid"):
        raise ValueError("search_mode must be existential or grid")
    if type(step_limit) is not int or not 1 <= step_limit <= 12000:
        raise ValueError("step_limit must be an integer in [1,12000]")
    if type(settle_steps) is not int or not 1 <= settle_steps <= min(200, step_limit):
        raise ValueError("settle_steps must be in [1,min(200,step_limit)]")
    selected_states = [
        (layout, seed) for seed in dict.fromkeys(seeds) for layout in dict.fromkeys(layouts)
    ]
    if states is not None:
        selected_states = []
        for state in states:
            layout, separator, seed_text = state.partition(":")
            if (
                separator != ":"
                or layout not in ("fixed", "human_batch_return")
                or not seed_text.isdecimal()
            ):
                raise ValueError("--state must be fixed:SEED or human_batch_return:SEED")
            selected_states.append((layout, int(seed_text)))
        if not selected_states:
            raise ValueError("states cannot be empty")
    config = RackServingConfig(
        mode="training", cooling_seconds=5, scraper_tilt_degrees=scraper_tilt_degrees
    )
    return {
        "schema_version": 2,
        "execution_implemented": True,
        "native_workers_started": 0,
        "config": config.model_dump(mode="json"),
        "search_mode": search_mode,
        "constant_params": Coverage.params_entry(params=param_sets[0]),
        "sweep_speed_m_s": SWEEP_SPEED_M_S,
        "claim": {
            "full_space_proof": False,
            "common_parameter_required": False,
            "all_sampled_states_covered": False,
            "parameter_learning_required": False,
            "robot_reset_coverage": False,
        },
        "states": [
            {
                "id": f"{layout}:seed={seed}",
                "seed": seed,
                "layout": layout,
                "seed_changes_layout": layout != "fixed",
                "privileged_setup": True,
                "setup_interventions": ["human_batch_return"] if layout != "fixed" else [],
                "expected_human_setup_cost": config.human_interruption_cost
                if layout != "fixed"
                else 0,
                "status": "untested",
                "measured_locations": None,
            }
            for layout, seed in dict.fromkeys(selected_states)
        ],
        "param_sets": [
            Coverage.params_entry(params=params) for params in dict.fromkeys(param_sets)
        ],
        "limits": {
            "max_trials_total": max_trials,
            "wall_seconds_total": max_seconds,
            "wall_seconds_per_trial": trial_seconds,
            "control_steps_per_skill": step_limit,
            "control_steps_per_trial": 6 * step_limit + settle_steps,
            "settle_control_steps": settle_steps,
            "parallel_workers": 1,
        },
        "executor_requirements": [
            "Fresh environment per params; reset_to_seed with the state's seed.",
            "For human layouts, call training human_batch_return once after each seed reset.",
            "Log actual setup cost, food poses, support contacts, measured locations and config.",
            "Compare measured starting states across parameter sets; flag mismatches and"
            " duplicate layouts.",
            "Use actual Wait, PickScraper, GatherAndSweep(delta,d,phi), ReturnScraper controllers.",
            "Record each food's physical outcome separately from controller errors.",
            "Verify physical task success and scraper home; do not infer success from no error.",
            "Test the constant baseline first; existential mode stops at each state's success.",
            "Grid mode continues after success to compare all requested constant policies.",
            "Persist every attempt, exception traceback, measured final state and human cost.",
            "Enforce limits during skills, not just between trials; retain partial results.",
            "Use a memory-capped agents.slice scope and an external timeout; never parallelize.",
            "Finite sampled coverage is not a proof over all states or continuous parameters.",
        ],
    }


def main(*, argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--dry-run", action="store_true", help="manifest only (default)")
    modes.add_argument(
        "--run", action="store_true", help="native execution; external run-heavy required"
    )
    parser.add_argument("--physical-witness-confirmed", action="store_true")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--state", action="append", help="explicit layout:SEED; overrides seeds/layouts"
    )
    parser.add_argument("--scraper-tilt-degrees", type=float, default=45)
    parser.add_argument("--settle-steps", type=int, default=20)
    parser.add_argument("--search-mode", choices=("existential", "grid"), default="existential")
    parser.add_argument(
        "--constant-params",
        nargs=3,
        type=float,
        metavar=("DELTA", "D", "PHI"),
        help="baseline tested first for every state; defaults to first --params",
    )
    parser.add_argument("--trial-seconds", type=float, default=180)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument(
        "--layouts",
        nargs="+",
        choices=("fixed", "human_batch_return"),
        default=["fixed", "human_batch_return"],
    )
    parser.add_argument(
        "--params", nargs=3, action="append", type=float, metavar=("DELTA", "D", "PHI")
    )
    parser.add_argument("--max-trials", type=int, default=12)
    parser.add_argument("--max-seconds", type=float, default=150)
    parser.add_argument("--step-limit", type=int, default=3500)
    args = parser.parse_args(argv)
    if args.worker is not None:
        if not args.physical_witness_confirmed:
            parser.error("native worker requires main's physical witness confirmation")
        request = json.loads(args.worker.read_text())
        NativeTrial.execute(request=request, output=args.worker.parent)
        return 0
    if args.output is None:
        parser.error("--output is required")
    if args.run and not args.physical_witness_confirmed:
        parser.error("native coverage blocked until main confirms a physically valid witness")
    try:
        manifest = build_manifest(
            seeds=args.seeds,
            layouts=args.layouts,
            param_sets=None if args.params is None else [tuple(params) for params in args.params],
            max_trials=args.max_trials,
            max_seconds=args.max_seconds,
            step_limit=args.step_limit,
            scraper_tilt_degrees=args.scraper_tilt_degrees,
            states=args.state,
            settle_steps=args.settle_steps,
            trial_seconds=args.trial_seconds,
            search_mode=args.search_mode,
            constant_params=None if args.constant_params is None else tuple(args.constant_params),
        )
        args.output.mkdir(parents=True, exist_ok=False)
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))
    with (args.output / "manifest.json").open("x") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write("\n")
    if not args.run:
        print(
            f"Manifest only: {args.output / 'manifest.json'}; no native execution or coverage claim"
        )
        return 0

    def interrupted(*_args: Any) -> None:
        raise KeyboardInterrupt

    previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        summary = Coverage.run(manifest=manifest, output=args.output)
    finally:
        signal.signal(signal.SIGTERM, previous)
    covered = sum(state["status"] == "covered" for state in summary["states"])
    print(f"Covered {covered}/{len(summary['states'])}; {args.output / 'summary.json'}")
    return (
        0
        if (
            summary["claim"]["all_sampled_states_covered"]
            and summary["stop_reason"] == "search_complete"
        )
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
