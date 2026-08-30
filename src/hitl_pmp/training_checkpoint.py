"""Atomic, trusted-local training checkpoints for scheduled Tossing3D POMDP runs.

Pickle stores fitted torch modules and private learner state. Never load a checkpoint
from an untrusted source: deserialization can execute code. These are same-code,
same-dependency recovery files, not a portable model interchange format.
"""

import argparse
import hashlib
import io
import json
import os
import pickle
import platform
import random
import tempfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, BinaryIO, Protocol, runtime_checkable

import numpy as np
import torch

from hitl_pmp.core.method.skill_provider import SkillProvider
from hitl_pmp.core.method.types import Skill
from hitl_pmp.core.metrics.metrics import Metrics
from hitl_pmp.core.problem.problem import Problem
from hitl_pmp.core.problem.tasks.types import Predicate
from hitl_pmp.loop_checkpoint import LoopCheckpoint


@runtime_checkable
class CheckpointMethod(Protocol):
    """Structural recovery capability; the generic runner imports no concrete method."""

    skill_provider: SkillProvider

    def skills(self) -> tuple[Skill, ...]: ...
    def predicates(self) -> tuple[Predicate, ...]: ...
    def checkpoint_learning_state(self) -> dict[str, Any]: ...
    def restore_learning_state(self, *, state: dict[str, Any]) -> None: ...


@runtime_checkable
class CheckpointTasks(Protocol):
    """The two independent streams needed by the currently supported task generator."""

    @property
    def train_rng(self) -> np.random.Generator: ...

    @property
    def test_rng(self) -> np.random.Generator: ...


class SymbolPickler(pickle.Pickler):
    """Save symbolic definitions by name, never pickle their classifier lambdas."""

    def persistent_id(self, obj: Any) -> tuple[str, str] | None:  # noqa: PLR0917
        if isinstance(obj, (Skill, Predicate)):
            return type(obj).__name__, obj.name
        return None


class SymbolUnpickler(pickle.Unpickler):
    def __init__(self, *, file: BinaryIO, method: CheckpointMethod) -> None:
        super().__init__(file)
        symbols: tuple[Skill | Predicate, ...] = (*method.skills(), *method.predicates())
        self.symbols = {(type(symbol).__name__, symbol.name): symbol for symbol in symbols}
        reset = method.skill_provider.human_cube_bin_reset_skill()
        if reset is not None:
            self.symbols[(type(reset.skill).__name__, reset.skill.name)] = reset.skill

    def persistent_load(self, pid: Any) -> Any:  # noqa: PLR0917
        try:
            return self.symbols[pid]
        except (KeyError, TypeError) as error:
            raise ValueError(f"Unknown checkpoint symbol: {pid!r}") from error


class TrainingCheckpoint:
    """One atomic recovery file. Sidecar recordings remain separate run segments."""

    def __init__(self, *, path: Path, args: argparse.Namespace) -> None:
        self.path = path
        self.args = args

    @staticmethod
    def configuration(*, args: argparse.Namespace) -> dict[str, Any]:
        ignored = {"output_dir", "checkpoint", "resume", "re_run", "wandb_tag"}
        return json.loads(
            json.dumps(
                {
                    key: value
                    for key, value in vars(args).items()
                    if key not in ignored and not key.startswith("record_")
                },
                default=str,
            )
        )

    @staticmethod
    def runtime_versions() -> dict[str, str]:
        versions = {"python": platform.python_version()}
        for name in ("numpy", "torch", "pydantic", "kindergarden", "kinder-models"):
            try:
                versions[name] = version(name)
            except PackageNotFoundError:
                versions[name] = "not-installed"
        digest = hashlib.sha256()
        source = Path(__file__).parent
        for path in sorted(source.rglob("*.py")):
            digest.update(str(path.relative_to(source)).encode())
            digest.update(path.read_bytes())
        versions["hitl_source_sha256"] = digest.hexdigest()
        return versions

    def save(
        self,
        *,
        progress: LoopCheckpoint,
        method: CheckpointMethod,
        problem: Problem,
        evaluation_problem: Problem,
        metrics: Metrics,
        runner_state: dict[str, Any],
    ) -> None:
        assert isinstance(problem.tasks, CheckpointTasks)
        assert isinstance(evaluation_problem.tasks, CheckpointTasks)
        buffer = io.BytesIO()
        SymbolPickler(buffer, protocol=pickle.HIGHEST_PROTOCOL).dump({
            "progress": progress,
            "method": method.checkpoint_learning_state(),
            "metrics": metrics,
            "runner": runner_state,
            "train_rng": problem.tasks.train_rng.bit_generator.state,
            "test_rng": problem.tasks.test_rng.bit_generator.state,
            "eval_train_rng": evaluation_problem.tasks.train_rng.bit_generator.state,
            "eval_test_rng": evaluation_problem.tasks.test_rng.bit_generator.state,
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
        })
        payload = buffer.getvalue()
        header = {
            "schema": 1,
            "config": self.configuration(args=self.args),
            "versions": self.runtime_versions(),
            "output_dir": str(self.args.output_dir.resolve()) if self.args.output_dir else None,
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(dir=self.path.parent, delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(json.dumps(header).encode() + b"\n")
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
            directory = os.open(self.path.parent, os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def load(self, *, path: Path, method: CheckpointMethod) -> dict[str, Any]:
        """Validate metadata/digest before deserializing a *trusted* local file."""
        with path.open("rb") as handle:
            header = json.loads(handle.readline())
            payload = handle.read()
        if header.get("schema") != 1:
            raise ValueError("Unsupported training checkpoint schema")
        if header["config"] != self.configuration(args=self.args):
            raise ValueError("Resume configuration differs from the checkpoint; repeat run flags")
        if header["versions"] != self.runtime_versions():
            raise ValueError("Resume dependency versions differ from the checkpoint")
        if self.args.output_dir and str(self.args.output_dir.resolve()) == header["output_dir"]:
            raise ValueError("Resume requires a new --output-dir to preserve existing recordings")
        if hashlib.sha256(payload).hexdigest() != header["sha256"]:
            raise ValueError("Training checkpoint is corrupt (checksum mismatch)")
        result: dict[str, Any] = SymbolUnpickler(file=io.BytesIO(payload), method=method).load()
        return result

    @staticmethod
    def restore(
        *,
        saved: dict[str, Any],
        method: CheckpointMethod,
        problem: Problem,
        evaluation_problem: Problem,
    ) -> None:
        # Scheduled periods and evaluation episodes rebuild their scene from task
        # seeds before taking any action. Do not pretend float32 replay snapshots
        # are sufficient for exact never-reset continuation.
        problem.hard_reset()
        evaluation_problem.hard_reset()
        method.restore_learning_state(state=saved["method"])
        for tasks, train_key, test_key in (
            (problem.tasks, "train_rng", "test_rng"),
            (evaluation_problem.tasks, "eval_train_rng", "eval_test_rng"),
        ):
            assert isinstance(tasks, CheckpointTasks)
            tasks.train_rng.bit_generator.state = saved[train_key]
            tasks.test_rng.bit_generator.state = saved[test_key]
        random.setstate(saved["python_rng"])
        np.random.set_state(saved["numpy_rng"])
        torch.set_rng_state(saved["torch_rng"])
