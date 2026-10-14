"""Stdlib-only entrypoint copied into the disconnected container, never host-run."""

import importlib.util
import json
import random
import socket
from pathlib import Path
from typing import Any


class ContainerPolicyWorker:
    @staticmethod
    def request(*, payload: dict[str, Any]) -> dict[str, Any]:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(120)
            client.connect("/sandbox/robot.sock")
            client.sendall(json.dumps(payload, allow_nan=False).encode() + b"\n")
            with client.makefile("rb") as stream:
                result = json.loads(stream.readline())
        if "error" in result:
            raise RuntimeError(result["error"])
        return dict(result["result"])

    @staticmethod
    def main() -> None:
        observation = ContainerPolicyWorker.request(payload={"operation": "observe"})
        try:
            done = ContainerPolicyWorker.run(observation=observation)
            ContainerPolicyWorker.request(payload={"operation": "finish", "done": done})
        except BaseException as exc:
            ContainerPolicyWorker.request(
                payload={
                    "operation": "finish",
                    "done": False,
                    "error": f"{type(exc).__name__}: {exc}"[:4000],
                }
            )

    @staticmethod
    def run(*, observation: dict[str, Any]) -> bool:
        request = json.loads(Path("/sandbox/request.json").read_text())
        random.seed(request["seed"])
        path = Path("/sandbox/controllers") / request["controller"]
        spec = importlib.util.spec_from_file_location("generated_policy", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        memory: dict[str, Any] = {}
        done = False
        for _ in range(request["max_steps"]):
            observation = {**observation, "execution_seed": request["seed"]}
            command = module.policy(observation, memory, request["parameters"])
            if not isinstance(command, dict) or set(command) != {"action", "memory", "done"}:
                raise ValueError("Policy must return action, memory, and done")
            if not isinstance(command["done"], bool) or not isinstance(command["memory"], dict):
                raise ValueError("Invalid policy memory or done marker")
            memory = command["memory"]
            if command["action"] is not None:
                observation = ContainerPolicyWorker.request(
                    payload={"operation": "step", "action": command["action"]}
                )
            if command["done"]:
                done = True
                break
        return done


if __name__ == "__main__":
    ContainerPolicyWorker.main()
