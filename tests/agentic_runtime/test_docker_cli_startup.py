"""Fresh writable CLI state and retained budget logs without operator credentials."""

import os
from pathlib import Path

import pytest

from hitl_pmp.agentic_runtime.robocode import RobocodeCodingAgent


class TestCLIState:
    @pytest.mark.parametrize("backend", ["codex", "claude"])
    def test_writable_cli_parent_retains_existing_accounting_path(
        self, *, tmp_path: Path, backend: str
    ) -> None:
        sandbox = tmp_path / "run" / "sandbox"
        sandbox.mkdir(parents=True)
        logs = sandbox / ".agent_sessions" / backend
        logs.mkdir(parents=True)
        (logs / "budget_status.json").write_text('{"remaining": 2}')

        state_mount, session_mount = RobocodeCodingAgent.cli_state_mounts(
            sandbox_dir=sandbox, backend=backend
        )

        state, root, readonly = state_mount
        assert state.is_relative_to(sandbox.parent) and not state.is_relative_to(sandbox)
        assert state.stat().st_uid == os.getuid()
        assert list(state.iterdir()) == []
        assert root == f"/home/node/.{backend}" and not readonly
        assert session_mount == (
            logs,
            f"{root}/sessions" if backend == "codex" else f"{root}/projects",
            False,
        )
        assert (logs / "budget_status.json").read_text() == '{"remaining": 2}'
        (state / "write-probe").write_text("CLI can initialize its own state")

    def test_backend_error_cannot_choose_a_host_mount(self, *, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="supports claude or codex"):
            RobocodeCodingAgent.cli_state_mounts(sandbox_dir=tmp_path, backend="../../etc")
