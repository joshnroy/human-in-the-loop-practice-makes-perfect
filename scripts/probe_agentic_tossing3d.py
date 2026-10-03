"""Reproduce the native bridge check; this is not a generated-policy benchmark.

Run from a worktree with its pinned KINDER submodules populated:
    scripts/with_env.sh systemd-run --user --scope -p MemoryMax=6G \
        -p OOMPolicy=continue python scripts/probe_agentic_tossing3d.py

The probe runs one trusted numeric option containing two control periods. It
does not call an LLM, run generated code, learn, or claim task success.
"""

import argparse
import base64
import json
import os
import socket
import sys
from pathlib import Path
from typing import Any

import numpy as np

from hitl_pmp.agentic_runtime.sandbox import RobotRelay
from hitl_pmp.environments.tossing3d.agentic_bridge import (
    AgenticTossing3DEnvironment,
    Tossing3DAgenticBridge,
)
from hitl_pmp.environments.tossing3d.kinder_backend import ControllerRun


class NativeBridgeProbe:
    """Explicitly separate interface verification from scientific performance."""

    @staticmethod
    def main() -> None:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--seed", type=int, default=125)
        parser.add_argument(
            "--output-dir", type=Path, default=Path("artifacts/agentic-bridge-smoke")
        )
        parser.add_argument("--through-relay", action="store_true")
        args = parser.parse_args()
        root = Path(__file__).resolve().parents[1]
        dependencies = [
            root / "reference/kindergarden/src",
            root / "reference/kinder-baselines/kinder-models/src",
        ]
        if not all(path.is_dir() for path in dependencies):
            raise RuntimeError("Populate this worktree's KINDER submodules before the probe")
        sys.path[:0] = [str(path) for path in dependencies]
        os.environ["DISABLE_AUTO_DYNAMIC3D_SCENES_DOWNLOAD"] = "1"
        args.output_dir.mkdir(parents=True, exist_ok=True)
        env = AgenticTossing3DEnvironment()
        try:
            env.reset_to_seed(seed=args.seed)
            bridge = Tossing3DAgenticBridge(env=env, step_limit=2)
            before = bridge.observe()
            NativeBridgeProbe.save_image(observation=before, path=args.output_dir / "initial.png")
            executor = TrustedProbeOption(
                bridge=bridge,
                relay_path=args.output_dir.resolve() / "probe.sock" if args.through_relay else None,
            )
            env.register_policy(skill_id=100, name="trusted_host_probe", executor=executor)
            state = env.take_action(action=np.array([100, 0, 0, 0, 0], dtype=float))
            after = bridge.observe()
            NativeBridgeProbe.save_image(observation=after, path=args.output_dir / "after.png")
            imported = {
                name: str(Path(sys.modules[name].__file__).resolve())
                for name in ("kinder", "kinder_models")
            }
            if not all(Path(path).is_relative_to(root / "reference") for path in imported.values()):
                raise RuntimeError("Native probe imported a dependency from another checkout")
            result = {
                "probe_kind": "trusted numeric host bridge, not generated-policy success",
                "seed": args.seed,
                "through_relay": args.through_relay,
                "dependency_paths": imported,
                "action_spec": bridge.action_spec(),
                "observation_keys": list(after),
                "proprioception_keys": list(after["proprioception"]),
                "control_steps": bridge.control_steps,
                "option_steps": state.get(obj=env.scene, feature_name="steps_taken"),
                "base_x_before": before["proprioception"]["pos_base_x"],
                "base_x_after": after["proprioception"]["pos_base_x"],
                "last_controller_steps": env.last_controller_steps(),
            }
            if result["control_steps"] != 2 or result["option_steps"] != 1:
                raise AssertionError("Two control periods must consume exactly one option slot")
            (args.output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result, indent=2))
        finally:
            env.close()

    @staticmethod
    def save_image(*, observation: dict[str, Any], path: Path) -> None:
        for index, image in enumerate(observation["images"]):
            destination = path if index == 0 else path.with_stem(path.stem + "-interaction")
            destination.write_bytes(base64.b64decode(image.split(",", 1)[1]))


class TrustedProbeOption:
    """A transparent actuator probe, deliberately not represented as a learned skill."""

    def __init__(self, *, bridge: Tossing3DAgenticBridge, relay_path: Path | None) -> None:
        self.bridge = bridge
        self.relay_path = relay_path

    def __call__(self, *, params: np.ndarray) -> ControllerRun:
        del params
        action = np.zeros(18)
        action[0] = 0.005
        schedule = {
            "schedule": [np.zeros(18).tolist()] * self.bridge.action_spec()["schedule_rows"]
        }
        if self.relay_path is None:
            self.bridge.step(action=action.tolist())
            self.bridge.step(action=schedule)
        else:
            with RobotRelay(path=self.relay_path, bridge=self.bridge, max_steps=2):
                self.request(payload={"operation": "observe"})
                self.request(payload={"operation": "step", "action": action.tolist()})
                self.request(payload={"operation": "step", "action": schedule})
                self.request(payload={"operation": "finish", "done": True})
            self.relay_path.unlink(missing_ok=True)
        return ControllerRun(steps=self.bridge.control_steps, terminated=True)

    def request(self, *, payload: dict[str, Any]) -> dict[str, Any]:
        assert self.relay_path is not None
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(30)
            client.connect(str(self.relay_path))
            client.sendall(json.dumps(payload).encode() + b"\n")
            with client.makefile("rb") as stream:
                reply = json.loads(stream.readline())
        if "error" in reply:
            raise RuntimeError(reply["error"])
        return dict(reply["result"])


if __name__ == "__main__":
    NativeBridgeProbe.main()
