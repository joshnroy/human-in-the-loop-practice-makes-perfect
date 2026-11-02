"""Bounded production-table physics gates; planning never starts a simulator."""

import argparse
import hashlib
import importlib.util
import json
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field


class Preflight:
    ROOT = Path(__file__).resolve().parents[1]
    REV3_SHA256 = "ddf53e8ccaa6267649e50ccec73348182ef26c88dc99458969cd8b1f49b03fee"
    NAMES = ("centered", "left", "fixed", "full5", "failure", "human")
    SKILLS = {
        "PickWiper": 0,
        "SweepTable": 1,
        "ReturnWiper": 2,
        "ReturnCubeToTableFromBox": 6,
        "ReturnCubeToTableFromFloor": 7,
    }

    @staticmethod
    def main(*, argv: list[str] | None = None) -> int:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--mode", choices=("plan", "physics", "learner-plan"), default="plan")
        parser.add_argument("--output-dir", type=Path, required=True)
        parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
        parser.add_argument(
            "--subset", nargs="+", choices=Preflight.NAMES, default=list(Preflight.NAMES)
        )
        parser.add_argument("--fixed-bin-offset", type=float, default=-0.18)
        parser.add_argument(
            "--scene-bg",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Include the production scene's static colliders and floor (default: enabled).",
        )
        for name in (
            "max_cases",
            "max_actions",
            "max_control_steps",
            "skill_step_limit",
            "case_seconds",
            "record_stride",
            "practice_steps",
            "evaluation_steps",
        ):
            parser.add_argument(
                "--" + name.replace("_", "-"),
                type=int,
                default=PreflightOptions.model_fields[name].default,
            )
        parser.add_argument("--record-motion", action="store_true")
        parser.add_argument("--video", action="store_true")
        args = vars(parser.parse_args(argv))
        output = args.pop("output_dir")
        try:
            options = PreflightOptions(**args)
            cases = Preflight.cases(options=options)
        except ValueError as error:
            parser.error(str(error))
        output.mkdir(parents=True, exist_ok=True)
        target = output / ("report.json" if options.mode == "physics" else "plan.json")
        if target.exists() or (options.mode == "physics" and any(output.iterdir())):
            parser.error(
                "refusing to overwrite existing preflight artifacts; use a fresh directory"
            )
        source = Preflight.source_manifest()
        report: dict[str, Any] = {
            "environment": "sweeptable3d",
            "mode": options.mode,
            "options": options.model_dump(mode="json"),
            "source": source,
            "required_contract": {
                "balls_diameter_m": 0.02,
                "round_wiper_length_m": 0.20,
                "handle_lean_deg": 45,
                "tray_rim_above_bottom_m": 0.005,
                "parameters": ["angle_deg", "speed_m_s"],
                "bounds": [[-45, 45], [0.05, 0.30]],
                "controller_revision": 3,
                "unload_counterfactual": False,
            },
            "cases": [case.model_dump() for case in cases],
            "learner_commands": Preflight.learner_commands(options=options, output=output),
            "physics_validated": False,
            "launch_ready": False,
            "learner_gate": "not_run",
            "launch_ready_reason": (
                "Requires complete physics coverage and separate learner validation."
            ),
            "evidence_policy": (
                "Only this checkout's observations count; exploratory evidence does not transfer."
            ),
        }
        if options.mode != "physics":
            Preflight.write_json(path=target, value=report)
            print(f"Plan only; no simulation or learner started. {target}")
            return 0
        report["results"] = []
        report["passed"] = False
        Preflight.write_json(path=target, value=report)
        for case in cases:
            directory = output / f"{case.name}-seed-{case.seed}"
            directory.mkdir()
            env = None
            try:
                env = Preflight.build_environment(case=case, options=options)
                contract = Preflight.verify_contract(env=env)
                result = Preflight.run_case(env=env, case=case, options=options, output=directory)
                result["contract"] = contract
                result["scene_sha256"] = contract.get("scene_sha256")
            except (Exception, BoundedStop) as error:
                result = {
                    "case": case.model_dump(),
                    "passed": False,
                    "finding": f"{type(error).__name__}: {error}",
                    "actions": [],
                }
            finally:
                if env is not None:
                    env.close()
            if options.video and (directory / "recording.jsonl").exists():
                result["video"] = Preflight.render_video(output=directory, options=options)
                if not result["video"]["saved"]:
                    result["passed"] = False
            result["source_sha256"] = source["sha256"]
            result["source_manifest"] = "../report.json:source"
            Preflight.write_json(path=directory / "result.json", value=result)
            report["results"].append(result)
            Preflight.write_json(path=target, value=report)
        final_source = Preflight.source_manifest()
        report["source_after"] = final_source
        report["source_unchanged"] = all(
            source[key] == final_source[key] for key in ("sha256", "head", "submodules")
        )
        report["passed"] = (
            all(result["passed"] for result in report["results"]) and report["source_unchanged"]
        )
        report["physics_validated"] = report["passed"]
        report["coverage_complete"] = set(options.subset) == set(Preflight.NAMES)
        report["physics_gate"] = (
            "passed" if report["passed"] and report["coverage_complete"] else "incomplete_or_failed"
        )
        report["passed_cases"] = sum(result["passed"] for result in report["results"])
        report["total_cases"] = len(cases)
        Preflight.write_json(path=target, value=report)
        print(f"Physics gate: {report['passed_cases']}/{len(cases)} cases; report: {target}")
        return 0 if report["passed"] else 1

    @staticmethod
    def render_video(*, output: Path, options: "PreflightOptions") -> dict[str, Any]:
        video = output / "episode.mp4"
        command = [
            sys.executable,
            "-m",
            "analysis.render_recording",
            "--run",
            str(output),
            "--output",
            str(video),
            "--plain-background",
        ]
        with (output / "replay.log").open("w") as log:
            try:
                result = subprocess.run(
                    command,
                    cwd=Preflight.ROOT,
                    stdout=log,
                    stderr=log,
                    timeout=options.case_seconds,
                    check=False,
                )
                return {
                    "saved": result.returncode == 0
                    and video.is_file()
                    and video.stat().st_size > 0,
                    "path": str(video),
                    "returncode": result.returncode,
                    "source": "saved recording only; separate replay scene, not robot execution",
                }
            except subprocess.TimeoutExpired:
                return {"saved": False, "finding": "saved-state replay exceeded wall-time cap"}

    @staticmethod
    def cases(*, options: "PreflightOptions") -> list["PreflightCase"]:
        if len(set(options.seeds)) != len(options.seeds) or len(set(options.subset)) != len(
            options.subset
        ):
            raise ValueError("duplicate seeds/subset entries would overwrite case evidence")
        if len(options.seeds) * len(options.subset) > options.max_cases:
            raise ValueError("selected cases exceed max_cases")
        cases = []
        for name in options.subset:
            for seed in options.seeds:
                angle, speed, offset = -27.0, 0.15, options.fixed_bin_offset
                if name == "centered":
                    angle, speed, offset = 0.0, 0.05, 0.0
                elif name == "left":
                    offset = -0.18
                elif name in {"failure", "human"}:
                    angle, speed = 45.0, 0.30
                cases.append(
                    PreflightCase(
                        name=name, seed=seed, angle_deg=angle, speed_m_s=speed, bin_offset=offset
                    )
                )
        return cases

    @staticmethod
    def learner_commands(*, options: "PreflightOptions", output: Path) -> list[list[str]]:
        commands = []
        for method in ("ees", "pomdp"):
            commands.append([
                "run-heavy",
                "systemd-run",
                "--user",
                "--scope",
                "-p",
                "MemoryMax=8G",
                "-p",
                "MemorySwapMax=0",
                "-p",
                "OOMPolicy=continue",
                "timeout",
                "--signal=TERM",
                "--kill-after=30s",
                f"{options.case_seconds}s",
                "scripts/with_env.sh",
                "python",
                "-m",
                "hitl_pmp.cli",
                "--env",
                "sweeptable3d",
                "--method",
                method,
                "--seed",
                str(options.seeds[0]),
                "--num-test-tasks",
                "1",
                "--fixed-bin-offset",
                str(options.fixed_bin_offset),
                "--skill-step-limit",
                str(options.skill_step_limit),
                "--practice-step-budget",
                str(options.practice_steps),
                "--measurement-interval-steps",
                str(options.practice_steps),
                "--no-initial-measurement",
                "--evaluation-control-steps",
                str(options.evaluation_steps),
                "--practice-reset-policy",
                "never",
                *([] if options.scene_bg else ["--no-scene-bg"]),
                "--defer-rendering",
                "--practice-cost-config",
                "configs/sweepoffshelf3d_practice_costs.json",
                "--record-sampler-draws",
                "--max-steps-per-interaction",
                "4",
                "--sampler-max-train-iters",
                "5",
                "--exploration-epsilon",
                "0",
                "--human-reset",
                *(
                    [
                        "--pomdp-solver",
                        "determinized_astar",
                        "--pomdp-max-search-iterations",
                        "20",
                        "--pomdp-search-depth",
                        "3",
                        "--pomdp-num-samples",
                        "8",
                        "--pomdp-num-particles",
                        "16",
                        "--pomdp-observation-probability-weight",
                        "0",
                    ]
                    if method == "pomdp"
                    else [
                        "--ees-prior-alpha",
                        "1",
                        "--ees-prior-beta",
                        "3",
                        "--goal-pursuit-horizon",
                        "0",
                    ]
                ),
                "--record-motion",
                "all",
                "--output-dir",
                str(output / f"learner-{method}"),
            ])
        return commands

    @staticmethod
    def build_environment(*, case: "PreflightCase", options: "PreflightOptions") -> Any:
        from hitl_pmp.environments.sweeptable3d.environment import SweepTable3DEnvironment

        return SweepTable3DEnvironment(
            scene_bg=options.scene_bg,
            canonical_seed=case.seed,
            human_seed=case.seed,
            fixed_bin_offset=case.bin_offset,
            skill_step_limit=options.skill_step_limit,
            reset_cubes_success_probability=1.0,
        )

    @staticmethod
    def verify_schema(*, bounds: Any) -> None:
        if not np.array_equal(np.asarray(bounds), [[-45, 45], [0.05, 0.30]]):
            raise ValueError(f"production [angle, speed] bounds mismatch: {bounds}")

    @staticmethod
    def verify_contract(*, env: Any) -> dict[str, Any]:
        import ast

        from hitl_pmp.environments.sweeptable3d.skills import SKILL_IDS, SweepTable3DSkills

        Preflight.verify_schema(bounds=SweepTable3DSkills.SWEEP_BOUNDS)
        if any(SKILL_IDS.get(name) != number for name, number in Preflight.SKILLS.items()):
            raise ValueError("production action IDs differ from the preflight contract")
        for package, folder in (("kinder", "kindergarden"), ("kinder_models", "kinder-baselines")):
            spec = importlib.util.find_spec(package)
            expected = (Preflight.ROOT / "reference" / folder).resolve()
            if (
                spec is None
                or spec.origin is None
                or not Path(spec.origin).resolve().is_relative_to(expected)
            ):
                raise ValueError(f"{package} must resolve inside this worktree's {expected}")
        scene_path = env.backend().scene_path
        scene = json.loads(scene_path.read_text())
        objects = scene["objects"]
        balls = [
            settings
            for kind, names in objects.items()
            if "ball" in kind or "sphere" in kind
            for settings in names.values()
        ]
        if len(balls) != 5 or any(
            not np.isclose(ball.get("radius", ball.get("size", -1)), 0.01) for ball in balls
        ):
            raise ValueError(
                "scene is not five 20 mm balls; cube investigation evidence is not transferable"
            )
        wiper = objects["table_wiper"]["wiper_0"]
        if wiper.get("head_shape") != "capsule" or not np.isclose(wiper["head_length"], 0.20):
            raise ValueError("scene requires a round 20 cm wiper")
        tray = objects["bin"]["box_0"]
        if not np.isclose(tray["height"] - tray["wall_thickness"], 0.005):
            raise ValueError("tray rim must be 5 mm above the bottom panel, not 5 mm overall")
        contact = scene_path.with_name("contact.py")
        tree = ast.parse(contact.read_text())
        revisions = [
            value.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Dict)
            for key, value in zip(node.keys, node.values, strict=True)
            if isinstance(key, ast.Constant)
            and key.value == "controller_revision"
            and isinstance(value, ast.Constant)
        ]
        if revisions != [3]:
            raise ValueError(f"expected frozen contact controller rev3, found {revisions}")
        if hashlib.sha256(contact.read_bytes()).hexdigest() != Preflight.REV3_SHA256:
            raise ValueError("contact rev3 source changed; no unload counterfactual is authorized")
        return {
            "scene_sha256": hashlib.sha256(scene_path.read_bytes()).hexdigest(),
            "controller_revision": 3,
            "controller_sha256": hashlib.sha256(contact.read_bytes()).hexdigest(),
            "geometry_source_checked": True,
            "live_geometry_checked": False,
        }

    @staticmethod
    def verify_live(*, env: Any, case: "PreflightCase") -> dict[str, Any]:
        from hitl_pmp.environments.sweeptable3d.wiper import TableWiper

        backend = env.backend()
        model = backend.model()
        diameters = []
        for name in backend.cube_names:
            body = model.body(name).id
            geoms = np.flatnonzero(model.geom_bodyid == body)
            if len(geoms) != 1 or int(model.geom_type[geoms[0]]) != 2:
                raise ValueError(f"live {name} is not a single sphere")
            diameters.append(float(2 * model.geom_size[geoms[0], 0]))
        if len(diameters) != 5 or not np.allclose(diameters, 0.02):
            raise ValueError(f"live ball diameter mismatch: {diameters}")
        wiper = env.controllers()._wiper
        head = wiper.head_geom
        length = float(2 * sum(model.geom_size[head, :2]))
        if int(model.geom_type[head]) != 3 or not np.isclose(length, 0.20):
            raise ValueError("live wiper is not a 20 cm capsule")
        handle = TableWiper.sweep_body_rotation(yaw_deg=0) @ wiper.handle_axis
        lean = float(np.rad2deg(np.arcsin(abs(handle[2]) / np.linalg.norm(handle))))
        if not np.isclose(lean, 45):
            raise ValueError(f"wiper commanded handle lean differs from 45 degrees: {lean}")
        position, _ = backend.core().get_object(backend.box_name).get_pose()
        offset = float(position[1] - backend.table_center_y())
        if not np.isclose(offset, case.bin_offset, atol=1e-4):
            raise ValueError(f"live bin offset {offset} differs from requested {case.bin_offset}")
        return {
            "ball_diameters_m": diameters,
            "round_wiper_length_m": length,
            "commanded_handle_lean_deg": lean,
            "bin_offset_m": offset,
        }

    @staticmethod
    def run_case(
        *, env: Any, case: "PreflightCase", options: "PreflightOptions", output: Path
    ) -> dict[str, Any]:
        report: dict[str, Any] = {
            "case": case.model_dump(),
            "passed": False,
            "actions": [],
            "checks": {},
            "setup_resets": 0,
        }
        report["config"] = {"case": case.model_dump(), "options": options.model_dump()}
        report["config_sha256"] = hashlib.sha256(
            json.dumps(report["config"], sort_keys=True).encode()
        ).hexdigest()
        writer = None
        audit = None
        started = time.monotonic()
        previous_handler = signal.getsignal(signal.SIGALRM)
        previous_timer = signal.getitimer(signal.ITIMER_REAL)
        signal.signal(signal.SIGALRM, lambda *_: Preflight.deadline())
        signal.setitimer(signal.ITIMER_REAL, options.case_seconds)
        try:
            env.reset_to_seed(seed=case.seed)
            report["setup_resets"] = 1
            report["live_contract"] = Preflight.verify_live(env=env, case=case)
            report["initial_qpos_sha256"] = hashlib.sha256(
                env.backend().data().qpos.tobytes()
            ).hexdigest()
            Preflight.save_state(env=env, path=output / "initial_state.npz")
            if options.record_motion or options.video:
                writer = Preflight.recording(env=env, case=case, options=options, output=output)
            with MotionAudit(env=env, options=options) as audit:
                runner = CaseRunner(
                    env=env, case=case, options=options, audit=audit, report=report, writer=writer
                )
                runner.run()
                audit.check()
            report["passed"] = True
        except (Exception, BoundedStop) as error:
            report["finding"] = f"{type(error).__name__}: {error}"
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_handler)
            if previous_timer[0] > 0:
                signal.setitimer(
                    signal.ITIMER_REAL,
                    max(0.001, previous_timer[0] - (time.monotonic() - started)),
                    previous_timer[1],
                )
            report["elapsed_seconds"] = time.monotonic() - started
            report["audit"] = audit.summary() if audit else {"not_started": True}
            if report["setup_resets"]:
                Preflight.save_state(env=env, path=output / "final_state.npz")
                report["final_observation"] = Preflight.observe(env=env)
                report["final_qpos_sha256"] = hashlib.sha256(
                    env.backend().data().qpos.tobytes()
                ).hexdigest()
            if writer is not None:
                writer.end_segment(success=report["passed"], outcome=report.get("finding"))
                writer.close()
        return report

    @staticmethod
    def deadline() -> None:
        raise BoundedStop("case wall-time cap reached")

    @staticmethod
    def save_state(*, env: Any, path: Path) -> None:
        data = env.backend().data()
        np.savez_compressed(
            path, qpos=data.qpos.copy(), qvel=data.qvel.copy(), time=float(data.time)
        )

    @staticmethod
    def observe(*, env: Any) -> dict[str, Any]:
        backend = env.backend()
        model, data, core = backend.model(), backend.data(), backend.core()
        balls = {}
        for name in backend.cube_names:
            position, quaternion = core.get_object(name).get_pose()
            joint = model.body_jntadr[model.body(name).id]
            address = model.jnt_dofadr[joint]
            balls[name] = {
                "xyz": np.asarray(position).tolist(),
                "quaternion_wxyz": np.asarray(quaternion).tolist(),
                "in_tray": bool(core.cube_in_box(name)),
                "qvel_linear_angular": data.qvel[address : address + 6].tolist(),
            }
        return {
            "atoms": [[name, list(objects)] for name, objects in sorted(backend.abstract_atoms())],
            "task_failed": bool(backend.task_failed()),
            "task_succeeded": bool(backend.task_succeeded()),
            "ground_cubes": list(backend.ground_cubes()),
            "balls": balls,
            "qpos": data.qpos.tolist(),
            "qvel": data.qvel.tolist(),
            "contact_metrics": dict(env.controllers().contact_metrics),
        }

    @staticmethod
    def recording(
        *, env: Any, case: "PreflightCase", options: "PreflightOptions", output: Path
    ) -> Any:
        from hitl_pmp.recording.run_record.types import Phase, RecordingHeader
        from hitl_pmp.recording.run_record.writer import RunRecordWriter

        backend = env.backend()
        header = RecordingHeader(
            environment="sweeptable3d",
            method="physics-preflight",
            run_seed=case.seed,
            scene={
                "scene_bg": options.scene_bg,
                "canonical_seed": case.seed,
                "fixed_bin_offset": case.bin_offset,
                "motion": {
                    "fps": max(1, round(backend.CONTROL_FREQUENCY / options.record_stride)),
                    "robot_joints": list(backend.ROBOT_JOINTS),
                },
            },
            feature_names={
                obj.name: list(obj.type.feature_names) for obj in env.get_current_state().data
            },
        )
        writer = RunRecordWriter(
            output_path=output / "recording.jsonl",
            header=header,
            state_provider=backend.motion_state,
        )
        env.attach_motion_sink(sink=writer, stride=options.record_stride)
        writer.begin_segment(phase=Phase.PRACTICE, scene_seed=case.seed)
        return writer

    @staticmethod
    def hash_files(*, paths: list[Path]) -> dict[str, Any]:
        files = {
            str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(set(paths))
            if path.is_file()
        }
        return {
            "files": files,
            "sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
        }

    @staticmethod
    def source_manifest() -> dict[str, Any]:
        root = Preflight.ROOT
        paths = list((root / "src").rglob("*.py")) + list((root / "src").rglob("*.json"))
        paths += [Path(__file__), root / "pyproject.toml", root / "scripts/with_env.sh"]
        dependencies = {}
        for package in ("kinder", "kinder_models", "mujoco", "numpy"):
            spec = importlib.util.find_spec(package)
            origin = Path(spec.origin).resolve() if spec and spec.origin else None
            dependencies[package] = str(origin) if origin else None
            if origin:
                paths.append(origin)
                if package in {"kinder", "kinder_models"}:
                    for suffix in ("*.py", "*.json", "*.xml"):
                        paths.extend(origin.parent.rglob(suffix))
        manifest = Preflight.hash_files(paths=paths)
        manifest["dependency_origins"] = dependencies
        for name, command in {
            "head": ["rev-parse", "HEAD"],
            "submodules": ["submodule", "status"],
            "dirty": ["status", "--porcelain", "--untracked-files=all"],
        }.items():
            result = subprocess.run(
                ["git", "-C", str(root), *command],
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )
            manifest[name] = (
                result.stdout.strip() if result.returncode == 0 else result.stderr.strip()
            )
        return manifest

    @staticmethod
    def write_json(*, path: Path, value: dict[str, Any]) -> None:
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        temporary.replace(path)


class CaseRunner:
    def __init__(
        self,
        *,
        env: Any,
        case: "PreflightCase",
        options: "PreflightOptions",
        audit: "MotionAudit",
        report: dict[str, Any],
        writer: Any = None,
    ) -> None:
        self.env, self.case, self.options = env, case, options
        self.audit, self.report, self.writer = audit, report, writer
        self.backend = env.backend()

    def require(self, *, name: str, condition: bool, message: str) -> None:
        self.report["checks"][name] = bool(condition)
        if not condition:
            raise ValueError(message)

    def atoms(self) -> Any:
        return self.backend.abstract_atoms()

    def all_at(self, *, predicate: str) -> bool:
        atoms = self.atoms()
        return all(
            (predicate, (name, self.backend.box_name) if predicate == "InBox" else (name,)) in atoms
            for name in self.backend.cube_names
        )

    def has(self, *, predicate: str) -> bool:
        return any(name == predicate for name, _ in self.atoms())

    def run(self) -> None:
        self.require(
            name="initial_all5_table",
            condition=len(self.backend.cube_names) == 5 and self.all_at(predicate="OnTable"),
            message="setup must contain all five balls on the table",
        )
        self.physical_action(skill="PickWiper")
        self.require(
            name="pickup_held",
            condition=self.has(predicate="HoldingWiper"),
            message="physical pickup did not hold the wiper",
        )
        sweep = self.action(skill="SweepTable")
        if self.case.name in {"failure", "human"}:
            self.failure()
        else:
            self.require(
                name="binary_all5_delivery",
                condition=sweep["binary_delivery_success"],
                message="sweep did not deliver all five balls without ground failure",
            )
            self.return_wiper()
            if self.case.name == "full5":
                self.restore_all()
                self.next_sweep()

    def return_wiper(self) -> None:
        self.physical_action(skill="ReturnWiper")
        self.require(
            name="physical_tool_return",
            condition=self.has(predicate="WiperHome")
            and self.has(predicate="HandEmpty")
            and not self.has(predicate="HoldingWiper"),
            message="wiper was not physically returned home with an empty hand",
        )

    def failure(self) -> None:
        self.require(
            name="failed_attempt_observed",
            condition=self.backend.task_failed(),
            message="failure candidate did not fail; reset coverage unproven",
        )
        ground = list(self.backend.ground_cubes())
        self.report["ground_contact_cubes"] = ground
        self.report["ground_contact_observed"] = bool(ground)
        if self.case.name == "failure":
            self.require(
                name="ground_failure_observed",
                condition=bool(ground),
                message="task_failed without ground_cubes is not ground-contact evidence",
            )
        self.require(
            name="held_after_failure",
            condition=self.has(predicate="HoldingWiper"),
            message="failed sweep no longer holds tool; held-return coverage unproven",
        )
        blocked = self.action(skill="SweepTable")
        self.require(
            name="failure_blocks_next_sweep",
            condition=bool(blocked["controller_error"])
            and blocked["actual_control_steps"] == 0
            and self.backend.task_failed(),
            message="latched failure did not refuse the next sweep without stepping",
        )
        self.return_wiper()
        self.require(
            name="held_tool_return_after_failure",
            condition=self.backend.task_failed(),
            message="returning only the held tool cleared ball failure",
        )
        if self.case.name == "human":
            self.human_reset()
        else:
            self.restore_all()
        self.require(
            name="failure_latch_cleared_after_full_reset",
            condition=self.all_at(predicate="OnTable") and not self.backend.task_failed(),
            message="failure did not clear after all five balls were back on table",
        )
        self.next_sweep()

    def restore_all(self) -> None:
        was_failed = self.backend.task_failed()
        for index, name in enumerate(self.backend.cube_names):
            if ("OnTable", (name,)) in self.atoms():
                continue
            skill = (
                "ReturnCubeToTableFromBox"
                if ("InBox", (name, self.backend.box_name)) in self.atoms()
                else "ReturnCubeToTableFromFloor"
            )
            self.physical_action(skill=skill, index=index)
            self.require(
                name=f"returned_{name}",
                condition=("OnTable", (name,)) in self.atoms(),
                message=f"physical return failed for {name}; no human fallback or teleport allowed",
            )
            if was_failed and not self.all_at(predicate="OnTable"):
                self.require(
                    name=f"latch_after_{name}",
                    condition=self.backend.task_failed(),
                    message="failure latch cleared before the full reset",
                )
        self.require(
            name="full5_table_restored",
            condition=self.all_at(predicate="OnTable"),
            message="not all five balls returned to the table",
        )

    def next_sweep(self) -> None:
        self.physical_action(skill="PickWiper")
        next_action = self.action(skill="SweepTable", angle=-27, speed=0.15)
        self.require(
            name="next_sweep_executed",
            condition=next_action["actual_control_steps"] > 0,
            message="reset did not permit a subsequent physical sweep",
        )
        self.require(
            name="next_sweep_binary_success",
            condition=next_action["binary_delivery_success"],
            message="next sweep failed binary delivery after physical recovery",
        )
        self.return_wiper()

    def physical_action(self, *, skill: str, index: int = 0) -> dict[str, Any]:
        row = self.action(skill=skill, index=index)
        self.require(
            name=f"{skill}_{len(self.report['actions'])}_executed",
            condition=row["actual_control_steps"] > 0 and row["controller_error"] is None,
            message=f"{skill} refused or failed; no physical recovery evidence",
        )
        return row

    def action(
        self, *, skill: str, index: int = 0, angle: float | None = None, speed: float | None = None
    ) -> dict[str, Any]:
        if len(self.report["actions"]) >= self.options.max_actions:
            raise BoundedStop("case action cap reached")
        self.audit.check()
        self.audit.action_start = self.audit.control_steps
        start, physics_start = self.audit.control_steps, self.audit.physics_steps
        action = np.array(
            [
                Preflight.SKILLS[skill],
                index,
                self.case.angle_deg if angle is None else angle,
                self.case.speed_m_s if speed is None else speed,
            ],
            dtype=float,
        )
        row: dict[str, Any] = {
            "actor": "robot",
            "skill": skill,
            "action": action.tolist(),
            "interrupted": False,
        }
        self.report["actions"].append(row)
        try:
            self.env.take_action(action=action)
            self.audit.check()
        except BoundedStop:
            row["interrupted"] = True
            raise
        finally:
            actual = self.audit.control_steps - start
            row.update(
                actual_control_steps=actual,
                cost_control_steps=actual,
                actual_physics_steps=self.audit.physics_steps - physics_start,
                reported_controller_steps=self.env.last_controller_steps(),
                controller_error=self.env.last_skill_error(),
                task_failed=bool(self.backend.task_failed()),
                binary_delivery_success=bool(
                    self.all_at(predicate="InBox")
                    and self.backend.task_succeeded()
                    and not self.backend.task_failed()
                    and self.env.last_skill_error() is None
                    and not row["interrupted"]
                ),
            )
            row["observation"] = Preflight.observe(env=self.env)
            if self.writer is not None:
                self.writer.robot_action(
                    label=skill,
                    start_step=start,
                    end_step=self.audit.control_steps,
                    success=not row["controller_error"] and not row["interrupted"],
                    outcome=row["controller_error"],
                    cost=float(actual),
                    goal_reached=row["binary_delivery_success"],
                )
        return row

    def human_reset(self) -> None:
        if len(self.report["actions"]) >= self.options.max_actions:
            raise BoundedStop("case action cap reached")
        self.audit.check()
        start = self.audit.control_steps
        self.audit.human = True
        try:
            self.env.reset_movables(destination="reset_cubes")
        finally:
            self.audit.human = False
            self.audit.expected = self.backend.data().qpos.copy()
            row = {
                "actor": "human",
                "skill": "reset_cubes",
                "privileged_pose_reset": True,
                "actual_control_steps": self.audit.control_steps - start,
                "cost_control_steps": None,
                "cost_note": "human intervention, not robot work or free robot reset",
            }
            self.report["actions"].append(row)
            if self.writer is not None:
                self.writer.human_action(
                    skill="reset_cubes",
                    start_step=start,
                    end_step=self.audit.control_steps,
                    success=self.all_at(predicate="OnTable") and not self.backend.task_failed(),
                    cost=None,
                )


class MotionAudit:
    """Read-only qpos continuity checks at native physics-step and action boundaries."""

    def __init__(self, *, env: Any, options: "PreflightOptions") -> None:
        self.backend = env.backend()
        self.gym, self.sim = self.backend.env(), self.backend.sim()
        self.options = options
        self.control_steps = self.physics_steps = self.outside_changes = self.action_start = 0
        self.human = False
        self.expected = self.backend.data().qpos.copy()
        self.original_control, self.original_physics = self.gym.step, self.sim.step

    def __enter__(self) -> "MotionAudit":
        def physics(*args: Any, **kwargs: Any) -> Any:
            self.check()
            try:
                return self.original_physics(*args, **kwargs)
            finally:
                self.physics_steps += 1
                self.expected = self.backend.data().qpos.copy()

        def control(*args: Any, **kwargs: Any) -> Any:
            self.check()
            if self.control_steps >= self.options.max_control_steps:
                raise BoundedStop("case control-step cap reached")
            if self.control_steps - self.action_start >= self.options.skill_step_limit:
                raise BoundedStop("per-action control-step cap reached")
            before = self.physics_steps
            try:
                return self.original_control(*args, **kwargs)
            finally:
                if self.physics_steps > before:
                    self.control_steps += 1
                self.check()

        self.gym.step, self.sim.step = control, physics
        return self

    def __exit__(self, *args: Any) -> None:
        self.gym.step, self.sim.step = self.original_control, self.original_physics

    def check(self) -> None:
        if not self.human and not np.array_equal(self.expected, self.backend.data().qpos):
            self.outside_changes += 1
            raise BoundedStop(
                "live qpos changed outside native physics step (setup/human resets excluded)"
            )

    def summary(self) -> dict[str, Any]:
        return {
            "actual_control_steps": self.control_steps,
            "actual_physics_steps": self.physics_steps,
            "outside_step_qpos_changes": self.outside_changes,
            "scope": (
                "net qpos changes between native sim.step calls and action boundaries; "
                "transient write-and-restore not detectable"
            ),
        }


class BoundedStop(BaseException):
    """Escape controller exception handlers without executing further robot steps."""


class PreflightCase(BaseModel):
    name: Literal["centered", "left", "fixed", "full5", "failure", "human"]
    seed: int = Field(default=0, ge=0)
    bin_offset: float = Field(default=-0.18, ge=-0.4, le=0.4, allow_inf_nan=False)
    angle_deg: float = Field(default=-27, ge=-45, le=45, allow_inf_nan=False)
    speed_m_s: float = Field(default=0.15, ge=0.05, le=0.30, allow_inf_nan=False)


class PreflightOptions(BaseModel):
    scene_bg: bool = True
    mode: Literal["plan", "physics", "learner-plan"] = "plan"
    seeds: list[int] = Field(default_factory=lambda: [0, 1, 2], min_length=1)
    subset: list[Literal["centered", "left", "fixed", "full5", "failure", "human"]] = Field(
        default_factory=lambda: list(Preflight.NAMES), min_length=1
    )
    fixed_bin_offset: float = Field(default=-0.18, ge=-0.4, le=0.4, allow_inf_nan=False)
    max_cases: int = Field(default=18, gt=0)
    max_actions: int = Field(default=16, gt=0)
    max_control_steps: int = Field(default=60000, gt=0)
    skill_step_limit: int = Field(default=12000, gt=0)
    case_seconds: int = Field(default=600, gt=0)
    record_stride: int = Field(default=1, gt=0)
    practice_steps: int = Field(default=2000, gt=0)
    evaluation_steps: int = Field(default=12000, gt=0)
    record_motion: bool = False
    video: bool = False


if __name__ == "__main__":
    raise SystemExit(Preflight.main())
