import argparse

from hitl_pmp.cli_protocols import EnvironmentCli

from .skill_oracle_method import SkillOracleMethod


class SkillOracleCli:
    """Plugs SkillOracleMethod into the global CLI under --method skill-oracle.
    Domain-agnostic: it drives whatever `--env` was selected (via the `env_cli`
    handed to `run`), taking that domain's privileged solver from the DomainContext
    (`ctx.oracle`) rather than importing any specific environment. A static-method
    container, never instantiated, same as every other business-logic class in this
    project."""

    @staticmethod
    def add_arguments(*, parser: argparse.ArgumentParser) -> None:
        """No method-specific flags -- everything SkillOracleMethod needs is the
        selected env's DomainContext, built by that env's own run_method."""
        del parser

    @staticmethod
    def run(*, args: argparse.Namespace, env_cli: type[EnvironmentCli]) -> None:
        env_cli.run_method(
            args=args,
            method_factory=lambda ctx: SkillOracleMethod(env=ctx.env, oracle=ctx.oracle),
            num_cycles=0,  # an oracle never practices/learns -- one evaluation sweep only
            max_steps_per_interaction=0,  # unused: never reached with num_cycles=0
        )
