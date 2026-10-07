"""The planning MCP protocol is local and distinct from evaluator evidence tools."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

from hitl_pmp.agentic_runtime import planning_mcp


def test_stdio_handshake_and_tools_without_simulator_imports(*, tmp_path):
    # The image installs these standalone files, without the package types.py.
    script = tmp_path / "planning_mcp.py"
    shutil.copyfile(planning_mcp.__file__, script)
    shutil.copyfile(
        Path(planning_mcp.__file__).with_name("planning_scene.py"), tmp_path / "planning_scene.py"
    )
    requests = [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2024-11-05"},
        },
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {"jsonrpc": "2.0", "id": 3, "method": "ping"},
    ]
    process = subprocess.run(
        [sys.executable, str(script)],
        input="\n".join(json.dumps(item) for item in requests) + "\n",
        capture_output=True,
        text=True,
        check=True,
    )
    responses = [json.loads(line) for line in process.stdout.splitlines()]
    assert [response["id"] for response in responses] == [1, 2, 3]
    tools = responses[1]["result"]["tools"]
    assert {tool["name"] for tool in tools} == {"render_state", "render_policy"}
    assert "max_steps" in tools[1]["inputSchema"]["required"]
    assert "no grasp/toss dynamics" in tools[1]["description"]
