"""Fail-closed preflight and a fixed-seed, memory-capped agentic Tossing3D sweep."""

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

from hitl_pmp.agentic_runtime.sandbox import DockerContainer, DockerPolicyExecutor
from hitl_pmp.methods.agentic_options.cli import RuntimeConfig


class AgenticExperimentProtocol:
    """The preflight runs trusted code only and makes no model-provider calls."""

    @staticmethod
    def main() -> int:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--runtime-config", type=Path, required=True)
        parser.add_argument("--inputs", type=Path, required=True)
        parser.add_argument("--results-root", type=Path, required=True)
        parser.add_argument("--max-workers", type=int, required=True)
        parser.add_argument("--num-seeds", type=int, default=1)
        parser.add_argument("--num-cycles", type=int, default=2)
        parser.add_argument("--max-steps-per-interaction", type=int, default=10)
        parser.add_argument("--num-test-tasks", type=int, default=2)
        parser.add_argument("--human-reset", action=argparse.BooleanOptionalAction, default=True)
        parser.add_argument("--human-reset-practice-cost", type=float, default=5.0)
        parser.add_argument("--memory-max", default="8G")
        parser.add_argument("--preflight-only", action="store_true")
        parser.add_argument("--library", type=Path)
        args = parser.parse_args()
        if (
            min(args.max_workers, args.num_seeds, args.num_cycles, args.max_steps_per_interaction)
            < 1
        ):
            parser.error("workers, seeds, cycles and session action limit must be positive")
        root = Path(__file__).resolve().parents[1]
        args.results_root = args.results_root.resolve()
        args.inputs = args.inputs.resolve()
        args.runtime_config = args.runtime_config.resolve()
        raw_config = json.loads(args.runtime_config.read_text())
        if "api_key" in raw_config.get("vision", {}):
            raise ValueError(
                "Use vision.api_key_env; runtime snapshots must not contain credentials"
            )
        runtime = RuntimeConfig.model_validate(raw_config)
        runtime.coding = runtime.coding.model_copy(
            update={
                "sandbox": runtime.coding.sandbox.model_copy(
                    update={
                        "image": DockerContainer(settings=runtime.coding.sandbox).image_id(),
                    }
                ),
            }
        )
        output = args.results_root / "protocol"
        output.mkdir(parents=True, exist_ok=False)
        snapshot = output / "runtime.json"
        snapshot.write_text(runtime.model_dump_json(indent=2))
        (output / "protocol.json").write_text(json.dumps(vars(args), default=str, indent=2) + "\n")
        shutil.copytree(args.inputs, output / "inputs")
        digests = {
            str(path.relative_to(output)): AgenticExperimentProtocol.digest(path=path)
            for path in output.rglob("*")
            if path.is_file()
        }
        if args.library is not None:
            shutil.copyfile(args.library, output / "library.json")
            digests["library.json"] = AgenticExperimentProtocol.digest(path=output / "library.json")
        transport = runtime.coding.sandbox.transport_source()
        digests["robocode_transport.py"] = AgenticExperimentProtocol.digest(path=transport)
        digests["container_image"] = runtime.coding.sandbox.image
        (output / "digests.json").write_text(json.dumps(digests, indent=2) + "\n")
        preflight = AgenticExperimentProtocol.preflight(runtime=runtime)
        (output / "preflight.json").write_text(json.dumps(preflight, indent=2) + "\n")
        if not preflight["passed"]:
            print(f"Disconnected namespace preflight FAILED. Details: {output / 'preflight.json'}")
            print("No generation, model calls, or sweep was started. No network-enabled fallback.")
            return 2
        print("Disconnected namespace preflight passed; no model call was needed.")
        if args.preflight_only:
            return 0
        if any(
            not value.strip() or value.startswith("CHOOSE_")
            for value in (runtime.coding.model, str(runtime.vision.get("model", "")))
        ):
            raise ValueError("Choose explicit coding and vision models before running a sweep")
        if runtime.vision.get("provider") != "openai_compatible":
            raise ValueError("The VLM integration requires an openai_compatible vision provider")
        key_variable = runtime.vision.get("api_key_env")
        if key_variable and not os.environ.get(key_variable):
            raise ValueError("The configured vision credential environment variable is missing")
        dependency_paths = [
            root / "src",
            root / "reference/kindergarden/src",
            root / "reference/kinder-baselines/kinder-models/src",
            runtime.coding.sandbox.robocode_checkout.resolve() / "src",
        ]
        if not all(path.is_dir() for path in dependency_paths):
            raise RuntimeError(
                "Populate local simulator submodules and the configured Robocode checkout"
            )
        environment = dict(os.environ)
        environment["PYTHONPATH"] = os.pathsep.join(str(path) for path in dependency_paths)
        environment["DISABLE_AUTO_DYNAMIC3D_SCENES_DOWNLOAD"] = "1"
        shared = [
            "--practice-reset-policy",
            "never",
            "--num-test-tasks",
            str(args.num_test_tasks),
            "--human-reset" if args.human_reset else "--no-human-reset",
            "--human-reset-practice-cost",
            str(args.human_reset_practice_cost),
        ]
        method = [
            "--agentic-inputs",
            str(output / "inputs"),
            "--agentic-runtime-config",
            str(snapshot),
            "--num-cycles",
            str(args.num_cycles),
            "--max-steps-per-interaction",
            str(args.max_steps_per_interaction),
        ]
        if args.library is not None:
            method += ["--agentic-library", str(output / "library.json")]
        command = [
            "systemd-run",
            "--user",
            "--scope",
            "-p",
            f"MemoryMax={args.memory_max}",
            "-p",
            "OOMPolicy=continue",
            str(root / "scripts/with_env.sh"),
            sys.executable,
            str(root / "scripts/run_sweep.py"),
            "--env",
            "tossing3d",
            "--methods",
            "agentic-options",
            "--num-seeds",
            str(args.num_seeds),
            "--max-workers",
            str(args.max_workers),
            "--results-root",
            str(args.results_root),
            "--shared-args",
            shlex.join(shared),
            "--method-args",
            "agentic-options=" + shlex.join(method),
        ]
        plan = {
            "seeds": list(range(args.num_seeds)),
            "max_workers": args.max_workers,
            "memory_max": args.memory_max,
            "command": command,
            "note": (
                "Seeds fix simulation randomness; provider completions are archived, "
                "not assumed deterministic."
            ),
        }
        (output / "sweep_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
        print(shlex.join(command), flush=True)
        return subprocess.run(command, cwd=root, env=environment, check=False).returncode

    @staticmethod
    def preflight(*, runtime: RuntimeConfig) -> dict[str, Any]:
        settings = runtime.coding.sandbox
        executor = DockerPolicyExecutor(settings=settings)
        with tempfile.TemporaryDirectory(prefix="hitl-agentic-preflight-") as tmp:
            work = Path(tmp)
            shutil.copyfile(settings.transport_source(), work / "transport.py")
            (work / "transport.json").write_text(
                json.dumps({"listeners": [], "strict_blackbox": True})
            )
            (work / "policy_worker.py").write_text(
                'print("HITL_AGENTIC_ISOLATION_OK", flush=True)\n'
            )
            name = f"hitl-preflight-{uuid.uuid4().hex}"
            command = executor.command(work_dir=work, name=name)
            try:
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=settings.timeout_seconds,
                    env={"PATH": os.defpath, "LANG": "C.UTF-8"},
                )
                return {
                    "passed": result.returncode == 0
                    and "HITL_AGENTIC_ISOLATION_OK" in result.stdout,
                    "returncode": result.returncode,
                    "stdout": result.stdout[-8000:],
                    "stderr": result.stderr[-8000:],
                    "command": command,
                    "generated_code_executed": False,
                    "model_calls": 0,
                }
            except subprocess.TimeoutExpired:
                return {
                    "passed": False,
                    "error": "isolated preflight timed out",
                    "command": command,
                    "generated_code_executed": False,
                    "model_calls": 0,
                }
            finally:
                executor.container.remove(name=name)

    @staticmethod
    def digest(*, path: Path) -> str:
        value = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                value.update(chunk)
        return value.hexdigest()


if __name__ == "__main__":
    raise SystemExit(AgenticExperimentProtocol.main())
