"""Standalone stdio MCP for the local PyBullet planning scene, never MuJoCo.

Installed beside planning_scene.py in the source-clean planning image.
"""

import contextlib
import ctypes
import importlib.util
import json
import os
import sys
from collections.abc import Generator
from pathlib import Path
from typing import Any


def tools() -> list[dict[str, Any]]:
    common = {
        "observation_path": {"type": "string", "description": "Recorded object-state JSON file"},
        "robot_spec_path": {"type": "string", "default": "robot_spec.json"},
        "output_path": {"type": "string"},
    }
    return [
        {
            "name": "render_state",
            "description": (
                "Render a recorded state in the PyBullet planning scene. Not real-world evidence."
            ),
            "inputSchema": {
                "type": "object",
                "properties": common,
                "required": ["observation_path", "output_path"],
            },
        },
        {
            "name": "render_policy",
            "description": (
                "Preview a Python policy kinematically in PyBullet. "
                "Fixed objects; no grasp/toss dynamics or MuJoCo access."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    **common,
                    "policy_path": {"type": "string"},
                    "parameters": {"type": "object"},
                    "max_steps": {"type": "integer", "minimum": 1},
                },
                "required": ["observation_path", "output_path", "policy_path", "max_steps"],
            },
        },
    ]


def call(*, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    from planning_scene import PlanningScene

    observation = json.loads(Path(arguments["observation_path"]).read_text())
    robot_spec = json.loads(Path(arguments.get("robot_spec_path", "robot_spec.json")).read_text())
    with PlanningScene(
        observation=observation, specification=robot_spec["planning_scene"]
    ) as scene:
        if name == "render_state":
            result = {
                "planning_only": True,
                "image": scene.render_state(path=arguments["output_path"]),
            }
        elif name == "render_policy":
            spec = importlib.util.spec_from_file_location(
                "candidate_policy", arguments["policy_path"]
            )
            if spec is None or spec.loader is None:
                raise ValueError("Cannot load candidate policy")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            result = scene.render_policy(
                policy=module.policy,
                parameters=arguments.get("parameters", {}),
                max_steps=arguments["max_steps"],
                output_dir=arguments["output_path"],
            )
        else:
            raise ValueError("Unknown planning tool")
    return {"content": [{"type": "text", "text": json.dumps(result)}]}


@contextlib.contextmanager
def silence_tool_stdout() -> Generator[None, None, None]:
    """Protect JSON-RPC from both Python prints and extension-library stdout."""
    sys.stdout.flush()
    saved = os.dup(1)
    try:
        os.dup2(2, 1)
        with contextlib.redirect_stdout(sys.stderr):
            yield
    finally:
        sys.stderr.flush()
        ctypes.CDLL(None).fflush(None)
        os.dup2(saved, 1)
        os.close(saved)


def dispatch(*, request: dict[str, Any]) -> dict[str, Any] | None:
    if "id" not in request:
        return None
    method = request.get("method")
    try:
        if method == "initialize":
            result = {
                "protocolVersion": request.get("params", {}).get("protocolVersion", "2024-11-05"),
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "pybullet-planning", "version": "1"},
            }
        elif method == "tools/list":
            result = {"tools": tools()}
        elif method == "tools/call":
            params = request["params"]
            # Candidate policy logging must never corrupt the stdio protocol.
            with silence_tool_stdout():
                result = call(name=params["name"], arguments=params.get("arguments", {}))
        elif method == "ping":
            result = {}
        else:
            return {
                "jsonrpc": "2.0",
                "id": request["id"],
                "error": {"code": -32601, "message": "Unknown method"},
            }
    except Exception as exc:
        result = {"isError": True, "content": [{"type": "text", "text": str(exc)}]}
    return {"jsonrpc": "2.0", "id": request["id"], "result": result}


def main() -> None:
    for line in sys.stdin:
        response = dispatch(request=json.loads(line))
        if response is not None:
            print(json.dumps(response), flush=True)


if __name__ == "__main__":
    main()
