"""`--sampler claude_code`: the flags and the per-run composition of a coded sampler."""

import argparse
import json
from pathlib import Path

from hitl_pmp.agentic_runtime.sandbox import SandboxSettings
from hitl_pmp.core.method.skill_provider import SkillProvider

from .coder import RobocodeCoder
from .history import TrajectoryHistory
from .prompts import CodedSamplerPrompts
from .runner import DockerCodeRunner
from .sampler import CodedSamplerHub
from .types import CodedSamplerMode
from .workspace import CodingWorkspace


class CodedSamplerCli:
    """A static-method container, never instantiated."""

    REPRODUCIBILITY_NOTE = (
        "--sampler claude_code: the seed does not pin the generated code; a rerun at "
        "the same seed can write different samplers."
    )

    @staticmethod
    def add_arguments(*, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--sampler",
            choices=("mlp", "claude_code"),
            default="mlp",
            help="Skill-parameter sampler: the per-skill MLP classifier (default), or "
            "code a Claude Code agent writes and refits once per cycle.",
        )
        parser.add_argument(
            "--claude-sampler-mode",
            choices=[mode.value for mode in CodedSamplerMode],
            default=CodedSamplerMode.SCORER.value,
            help="scorer: generated score() replaces the MLP's scores under the same "
            "argmax wrapper. sampler: generated sample() proposes the parameters.",
        )
        parser.add_argument("--coded-sampler-sandbox-settings", type=Path, default=None)
        parser.add_argument("--coded-sampler-model", default="claude-opus-5-5")
        parser.add_argument("--coded-sampler-effort", default="high")
        parser.add_argument(
            "--coded-sampler-refit-timeout",
            type=float,
            default=3600.0,
            help="Wall-clock seconds one refit may take before it is abandoned.",
        )
        parser.add_argument("--coded-sampler-coder-memory-mb", type=int, default=8192)
        parser.add_argument("--coded-sampler-call-memory-mb", type=int, default=2048)

    @staticmethod
    def validate(*, args: argparse.Namespace) -> None:
        if args.sampler != "claude_code":
            return
        if args.output_dir is None:
            raise ValueError("--sampler claude_code needs --output-dir for its workspace")
        if args.coded_sampler_sandbox_settings is None:
            raise ValueError("--sampler claude_code needs --coded-sampler-sandbox-settings")
        args.coded_sampler_reproducibility = CodedSamplerCli.REPRODUCIBILITY_NOTE

    @staticmethod
    def build(*, args: argparse.Namespace, provider: SkillProvider) -> CodedSamplerHub | None:
        if args.sampler != "claude_code":
            return None
        output = Path(args.output_dir).resolve()
        coding = output / "coding"
        sandbox_dir = coding / "sandbox"
        sandbox_dir.mkdir(parents=True, exist_ok=True)
        base = SandboxSettings.model_validate(
            json.loads(Path(args.coded_sampler_sandbox_settings).read_text())
        )
        # A per-run pool: the coder container is capped, and it never queues behind
        # another run's containers.
        settings = base.model_copy(
            update={
                "artifact_dir": coding / "execution",
                "resource_pool_dir": coding / "resource-pool",
                "coder_memory_mb": args.coded_sampler_coder_memory_mb,
            }
        )
        mode = CodedSamplerMode(args.claude_sampler_mode)
        return CodedSamplerHub(
            mode=mode,
            provider=provider,
            workspace=CodingWorkspace(root=sandbox_dir, mode=mode),
            coder=RobocodeCoder(
                settings=settings,
                sandbox_dir=sandbox_dir,
                model=args.coded_sampler_model,
                effort=args.coded_sampler_effort,
                timeout_seconds=args.coded_sampler_refit_timeout,
            ),
            runner=DockerCodeRunner(
                settings=settings,
                memory_mb=args.coded_sampler_call_memory_mb,
                timeout_seconds=CodedSamplerPrompts.CALL_SECONDS,
            ),
            output_dir=output,
            history=TrajectoryHistory(),
        )
