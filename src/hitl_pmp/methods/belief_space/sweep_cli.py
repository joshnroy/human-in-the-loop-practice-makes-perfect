"""Construct Sweep Model B without changing the Tossing3D method factory."""

import argparse
import json
import math
from pathlib import Path
from typing import Protocol, cast

from hitl_pmp.cli_protocols import EnvironmentCli
from hitl_pmp.core.method.skill_provider import DomainContext
from hitl_pmp.core.problem.tasks.types import GroundAtom
from hitl_pmp.sampler_draws import SamplerDrawRecorder

from .sweep_method import SweepPomdpMethod


class SweepDeploymentProvider(Protocol):
    def deployment_initial_atoms(self) -> frozenset[GroundAtom]: ...
    def deployment_goal_atoms(self) -> frozenset[GroundAtom]: ...


class SweepPomdpCli:
    @staticmethod
    def run(*, args: argparse.Namespace, env_cli: type[EnvironmentCli]) -> None:
        manifest = json.loads(args.sweep_manifest.read_text())
        model_config = manifest["pomdp"]
        # The generic config snapshot records args. Keep its values identical to
        # those actually passed to the model, including manifest overrides.
        for name, value in model_config.items():
            if name not in SweepPomdpMethod.model_fields:
                raise ValueError(f"Unknown frozen Sweep model setting {name!r}")
            setattr(args, name, value)
        draw_recorder = SamplerDrawRecorder.open_if_requested(args=args)

        def factory(context: DomainContext) -> SweepPomdpMethod:  # noqa: PLR0917 (factory protocol)
            provider = context.skill_provider
            # Existing named algorithm settings are forwarded unchanged. Domain
            # calibrated fields below override defaults, never data-derived scores.
            values = {
                name: getattr(args, name)
                for name in SweepPomdpMethod.model_fields
                if hasattr(args, name)
                and name not in {"env", "skill_provider", "decision_log", "deployment_horizon"}
            }
            values.update(model_config)
            if getattr(args, "env", None) == "sweep_simple3d":
                names = tuple(skill.name for skill in provider.deployment_skills())
                if (
                    "trainable_skill_names" in model_config
                    and tuple(model_config["trainable_skill_names"]) != names
                ):
                    raise ValueError("Simple trainable skills must match the deployment provider")
                calibration = manifest.get("random_competences")
                if (
                    not isinstance(calibration, dict)
                    or set(calibration) != set(names)
                    or any(
                        isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not math.isfinite(value)
                        or not 0 <= value <= 1
                        for value in calibration.values()
                    )
                ):
                    raise ValueError(
                        "Simple random_competences requires explicit calibrated probabilities "
                        f"for exactly {names}; no Drawer calibration is reused"
                    )
                values["trainable_skill_names"] = names
                args.trainable_skill_names = names
            initial_method = cast(SweepDeploymentProvider, provider).deployment_initial_atoms
            goal_method = cast(SweepDeploymentProvider, provider).deployment_goal_atoms
            values.update(
                env=context.env,
                reset_cost_gate=args.ees_reset_gate,
                draw_recorder=draw_recorder,
                skill_provider=provider,
                random_competences=manifest["random_competences"],
                deployment_initial_atoms=initial_method(),
                deployment_goal_atoms=goal_method(),
                deployment_horizon=manifest["deployment_horizon"],
                human_skill_names=tuple(
                    s.skill.name for s in provider.human_cube_bin_reset_skills()
                ),
                decision_log=None
                if args.output_dir is None
                else Path(args.output_dir) / "sweep_decisions.jsonl",
            )
            method = SweepPomdpMethod(**values)
            if args.output_dir is not None:
                output = Path(args.output_dir)
                output.mkdir(parents=True, exist_ok=True)
                resolved = method.model_dump(
                    mode="json", exclude={"env", "skill_provider", "draw_recorder"}
                )
                (output / "sweep_resolved_method.json").write_text(
                    json.dumps(resolved, indent=2) + "\n"
                )
            return method

        env_cli.run_method(
            args=args,
            method_factory=factory,
            num_cycles=args.num_cycles,
            max_steps_per_interaction=args.max_steps_per_interaction,
        )
