"""Load a literal `CLAUDE_CODE_OAUTH_TOKEN` assignment from a file, then run a program.

    python scripts/with_claude_token.py TOKEN_FILE -m MODULE [ARG ...]
    python scripts/with_claude_token.py TOKEN_FILE SCRIPT.py [ARG ...]

The RoboCode coder authenticates with that variable. Only a literal assignment is
accepted, never one that expands a variable or runs a command, so the file is read but
never executed. Run the Full-Agentic runner with `-m hitl_pmp.full_agentic.runner`: its
`__main__` guard is what keeps the evaluation pool's `spawn` children from re-running the
whole runner, which a launcher script run by path did on 2026-10-14.
"""

import os
import re
import runpy
import shlex
import sys
from pathlib import Path

NAME = "CLAUDE_CODE_OAUTH_TOKEN"


class ClaudeTokenLoader:
    """A static-method container, never instantiated."""

    @staticmethod
    def token(*, source: Path) -> str:
        """The value of the last `[export] CLAUDE_CODE_OAUTH_TOKEN=...` line in `source`."""
        lines = [
            line
            for line in source.read_text().splitlines()
            if re.match(r"\s*(?:export\s+)?" + NAME + r"=", line)
        ]
        if not lines:
            raise RuntimeError(f"{source} has no token assignment")
        parts = shlex.split(lines[-1], comments=True)
        if parts and parts[0] == "export":
            parts = parts[1:]
        if len(parts) != 1 or not parts[0].startswith(NAME + "="):
            raise RuntimeError("the token must be a literal assignment")
        value = parts[0].split("=", 1)[1]
        if not value or "$" in value or "`" in value:
            raise RuntimeError("the token must be a nonempty literal assignment")
        return value

    @staticmethod
    def main(*, argv: list[str]) -> None:
        source, *program = argv
        if not program:
            raise SystemExit(__doc__)
        os.environ[NAME] = ClaudeTokenLoader.token(source=Path(source))
        if program[0] == "-m":
            module, *arguments = program[1:]
            sys.argv = [module, *arguments]
            runpy.run_module(module, run_name="__main__", alter_sys=True)
        else:
            target, *arguments = program
            sys.argv = [target, *arguments]
            runpy.run_path(target, run_name="__main__")


if __name__ == "__main__":
    ClaudeTokenLoader.main(argv=sys.argv[1:])
