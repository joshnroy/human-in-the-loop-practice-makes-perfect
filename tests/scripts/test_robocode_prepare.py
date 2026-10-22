"""Freezing a RoboCode launch, and the token loader it runs through."""

import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from hitl_pmp.agentic_runtime.sandbox import SandboxSettings
from scripts.robocode_prepare import LAUNCH_HOURS, RoboCodePrepare
from scripts.sweepoffshelf3d_launch import ROBOCODE, ShelfLaunch
from scripts.with_claude_token import ClaudeTokenLoader

REPO = Path(__file__).resolve().parents[2]
PROVENANCE = REPO / "src/hitl_pmp/full_agentic/bootstrap_provenance.json"


def _robocode(*, root: Path, files: dict[str, bytes]) -> dict[str, str]:
    for path, content in files.items():
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_bytes(content)
    return {path: hashlib.sha256(content).hexdigest() for path, content in files.items()}


class TestRobocodeProvenance:
    def test_the_shipped_provenance_pins_135_files(self) -> None:
        assert len(json.loads(PROVENANCE.read_text())["robocode_source"]) == 135

    def test_every_pinned_file_must_match(self, *, tmp_path: Path) -> None:
        pinned = _robocode(root=tmp_path, files={"a.py": b"a", "src/b.py": b"b"})
        assert RoboCodePrepare.matching(robocode=tmp_path, pinned=pinned) == (2, 2)
        RoboCodePrepare.verify_robocode(robocode=tmp_path, pinned=pinned)
        (tmp_path / "src/b.py").write_bytes(b"tampered")
        assert RoboCodePrepare.matching(robocode=tmp_path, pinned=pinned) == (1, 2)
        with pytest.raises(SystemExit, match="1/2"):
            RoboCodePrepare.verify_robocode(robocode=tmp_path, pinned=pinned)

    def test_a_missing_file_does_not_match(self, *, tmp_path: Path) -> None:
        pinned = _robocode(root=tmp_path, files={"a.py": b"a"})
        pinned["gone.py"] = hashlib.sha256(b"x").hexdigest()
        assert RoboCodePrepare.matching(robocode=tmp_path, pinned=pinned) == (1, 2)


class TestFrozenLaunch:
    def test_sandbox_settings_are_valid_and_point_into_the_root(self, *, tmp_path: Path) -> None:
        settings = SandboxSettings.model_validate(
            RoboCodePrepare.sandbox_settings(root=tmp_path, image="sha256:abc")
        )
        assert settings.robocode_checkout == tmp_path / "source/robocode"
        assert settings.resource_pool_dir == tmp_path / "resource-pool"
        assert settings.artifact_dir == tmp_path / "execution"
        assert settings.memory_mb == 4096 and settings.coder_memory_mb == 16384
        assert settings.image == "sha256:abc"

    def test_the_launch_script_passes_six_hours_explicitly(self, *, tmp_path: Path) -> None:
        # Josh, 2026-10-22: the RoboCode launch runs for at most 6 h.
        assert LAUNCH_HOURS == 6
        text = RoboCodePrepare.launch_script(root=tmp_path, token_file=Path("/t"))
        assert "--hours 6 " in text
        line = next(line for line in text.splitlines() if "sweepoffshelf3d_launch" in line)
        words = shlex.split(line.replace('"$@"', ""))
        args = ShelfLaunch.parse(argv=words[words.index("scripts.sweepoffshelf3d_launch") + 1 :])
        assert args.mode == "launch" and args.arms == [ROBOCODE]
        assert args.hours == 6
        assert args.results_root == tmp_path / "results"
        assert args.sandbox_settings == tmp_path / "sandbox.json"
        assert args.token_file == Path("/t")
        assert not args.go, "the go is Josh's to add"
        assert words[0] == str(tmp_path / "source/repo/scripts/with_env.sh")

    def test_the_launch_script_is_valid_bash(self, *, tmp_path: Path) -> None:
        script = tmp_path / "launch.sh"
        script.write_text(RoboCodePrepare.launch_script(root=tmp_path, token_file=Path("/t")))
        subprocess.run(["bash", "-n", str(script)], check=True)

    def test_a_pin_must_be_the_checkout_runs_import(self) -> None:
        RoboCodePrepare.require_pin(name="kindergarden", pinned="abc", checked_out="abc")
        with pytest.raises(SystemExit, match="kindergarden"):
            RoboCodePrepare.require_pin(name="kindergarden", pinned="abc", checked_out="def")

    def test_prepare_refuses_an_existing_source(self, *, tmp_path: Path) -> None:
        (tmp_path / "source").mkdir()
        with pytest.raises(SystemExit, match="exists"):
            RoboCodePrepare.main(argv=["--root", str(tmp_path), "--ref", "HEAD"])


class TestTokenLoader:
    def test_a_literal_assignment_is_read(self, *, tmp_path: Path) -> None:
        source = tmp_path / "rc"
        source.write_text("x=1\nexport CLAUDE_CODE_OAUTH_TOKEN=old\nCLAUDE_CODE_OAUTH_TOKEN=tok\n")
        assert ClaudeTokenLoader.token(source=source) == "tok"

    @pytest.mark.parametrize(
        "line",
        [
            "CLAUDE_CODE_OAUTH_TOKEN=$OTHER",
            "CLAUDE_CODE_OAUTH_TOKEN=`cat f`",
            "CLAUDE_CODE_OAUTH_TOKEN=",
        ],
    )
    def test_anything_but_a_literal_is_refused(self, *, tmp_path: Path, line: str) -> None:
        source = tmp_path / "rc"
        source.write_text(line + "\n")
        with pytest.raises(RuntimeError, match="literal"):
            ClaudeTokenLoader.token(source=source)

    def test_no_assignment_is_refused(self, *, tmp_path: Path) -> None:
        source = tmp_path / "rc"
        source.write_text("x=1\n")
        with pytest.raises(RuntimeError, match="no token"):
            ClaudeTokenLoader.token(source=source)

    def test_a_module_runs_as_main_with_the_token_and_its_argv(self, *, tmp_path: Path) -> None:
        """`-m` keeps the module's own `__main__` guard: the runner's spawn-pool children
        re-import it by name instead of re-running a launcher script (2026-10-14)."""
        (tmp_path / "rc").write_text("CLAUDE_CODE_OAUTH_TOKEN=tok\n")
        (tmp_path / "probe_mod.py").write_text(
            "import json, os, sys\n"
            "if __name__ == '__main__':\n"
            "    print(json.dumps([os.environ['CLAUDE_CODE_OAUTH_TOKEN'], sys.argv[1:]]))\n"
        )
        env = {**os.environ, "PYTHONPATH": f"{tmp_path}{os.pathsep}{REPO}"}
        env.pop("CLAUDE_CODE_OAUTH_TOKEN", None)
        out = subprocess.run(
            [
                sys.executable, str(REPO / "scripts/with_claude_token.py"),
                str(tmp_path / "rc"), "-m", "probe_mod", "--a", "1",
            ],
            env=env, capture_output=True, text=True, check=True,
        ).stdout  # fmt: skip
        assert json.loads(out) == ["tok", ["--a", "1"]]
