"""Calls one function of a frozen generated module, in a process the host does not trust.

`DockerCodeRunner` is the only runner a real run uses: the agentic runtime's
`DockerContainer` (no network, read-only root, all capabilities dropped, capped memory
and pids) behind Robocode's namespace-verifying supervisor. `SubprocessCodeRunner`
runs the same worker as a host subprocess and exists for tests, which have no Docker.
"""

import abc
import json
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict

from hitl_pmp.agentic_runtime.sandbox import DockerContainer, SandboxSettings

from .types import CodeCallOutcome, CodeVersion
from .worker import RESULT_PREFIX

WORKER = Path(__file__).with_name("worker.py")


class CodeRunner(BaseModel, abc.ABC):
    """One call, one fresh process: nothing a call does survives into the next."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    timeout_seconds: float = 60.0
    # Imports numpy, scipy and sklearn make of their own: allowed at run time so a
    # permitted library works, while the host's AST check still refuses them as direct
    # imports of generated code.
    ALLOWED_MODULES: ClassVar[tuple[str, ...]] = (
        "numpy",
        "scipy",
        "sklearn",
        "joblib",
        "threadpoolctl",
    )

    def call(
        self,
        *,
        version: CodeVersion,
        skill_name: str,
        function: str,
        payload: dict[str, Any],
    ) -> CodeCallOutcome:
        started = time.monotonic()
        module = version.modules.get(skill_name)
        if module is None or version.snapshot_dir is None:
            return CodeCallOutcome(
                ok=False, error=f"no module for {skill_name} in v{version.index}"
            )
        with tempfile.TemporaryDirectory(prefix="hitl-coded-sampler-") as tmp:
            work = Path(tmp)
            shutil.copyfile(WORKER, work / "worker.py")
            request = {
                "code_dir": self.code_dir_inside(),
                "module": Path(module).name,
                "skill": skill_name,
                "function": function,
                "payload": payload,
                "allowed_modules": list(self.ALLOWED_MODULES),
            }
            (work / "request.json").write_text(json.dumps(request, allow_nan=False))
            try:
                stdout, error = self.execute(
                    work=work, code_dir=version.snapshot_dir / Path(module).parent
                )
            except subprocess.TimeoutExpired:
                return CodeCallOutcome(
                    ok=False,
                    error=f"timed out after {self.timeout_seconds} s",
                    latency_seconds=time.monotonic() - started,
                )
        latency = time.monotonic() - started
        for line in stdout.splitlines():
            if line.startswith(RESULT_PREFIX):
                result = json.loads(line[len(RESULT_PREFIX) :])
                return CodeCallOutcome(
                    ok=bool(result["ok"]),
                    value=result.get("value"),
                    error=result.get("error"),
                    latency_seconds=latency,
                )
        return CodeCallOutcome(
            ok=False,
            error=f"no result line; stderr tail: {error[-2000:]}",
            latency_seconds=latency,
        )

    @abc.abstractmethod
    def code_dir_inside(self) -> str:
        """Where the frozen module directory appears to the worker."""

    @abc.abstractmethod
    def execute(self, *, work: Path, code_dir: Path) -> tuple[str, str]:
        """Run `work/worker.py` with `code_dir` visible; return (stdout, stderr)."""


class SubprocessCodeRunner(CodeRunner):
    """Host subprocess, isolated interpreter. Tests only: it is not a sandbox."""

    _code_dir: str = ""

    def code_dir_inside(self) -> str:
        return self._code_dir

    def call(
        self,
        *,
        version: CodeVersion,
        skill_name: str,
        function: str,
        payload: dict[str, Any],
    ) -> CodeCallOutcome:
        module = version.modules.get(skill_name)
        if module is not None and version.snapshot_dir is not None:
            self._code_dir = str((version.snapshot_dir / Path(module).parent).resolve())
        return super().call(
            version=version, skill_name=skill_name, function=function, payload=payload
        )

    def execute(self, *, work: Path, code_dir: Path) -> tuple[str, str]:
        del code_dir
        result = subprocess.run(
            [sys.executable, "-I", str(work / "worker.py")],
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
            check=False,
            cwd=work,
        )
        return result.stdout, result.stderr


class DockerCodeRunner(CodeRunner):
    """The disconnected execution sandbox the agentic methods already use."""

    settings: SandboxSettings
    memory_mb: int = 2048

    def code_dir_inside(self) -> str:
        return "/sandbox/code"

    def execute(self, *, work: Path, code_dir: Path) -> tuple[str, str]:
        # The bind target must exist in the read-only /sandbox mount beforehand.
        (work / "code").mkdir()
        shutil.copyfile(self.settings.transport_source(), work / "transport.py")
        (work / "transport.json").write_text(json.dumps({"listeners": [], "strict_blackbox": True}))
        # Its own container record per call; never the shared practice pool.
        settings = self.settings.model_copy(update={"resource_pool_dir": None})
        container = DockerContainer(settings=settings, memory_mb=self.memory_mb)
        name = f"hitl-coded-sampler-{uuid.uuid4().hex}"
        command = container.command(
            name=name,
            work_dir=work,
            mounts=((code_dir, "/sandbox/code", True),),
            argv=[
                "/sandbox/transport.py",
                "/sandbox/transport.json",
                settings.container_python,
                "/sandbox/worker.py",
            ],
        )
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
                stdin=subprocess.DEVNULL,
                env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"},
            )
        finally:
            container.remove(name=name)
        return result.stdout, result.stderr
