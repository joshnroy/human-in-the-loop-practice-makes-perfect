import argparse

from hitl_pmp.environments.lightswitch.cli import LightSwitchCli

from .skill_oracle_method import SkillOracleMethod


class SkillOracleCli:
    """Plugs SkillOracleMethod into the global CLI under --method skill-oracle.
    Lives alongside SkillOracleMethod under methods/oracle/ (not
    environments/lightswitch/), since a method-CLI is method-specific glue, not
    environment-specific -- see environments/lightswitch/README's cli.py
    convention and methods/README.md's own. A static-method container, never
    instantiated, same as every other business-logic class in this project."""

    @staticmethod
    def add_arguments(*, parser: argparse.ArgumentParser) -> None:
        """No method-specific flags -- SkillOracleMethod hardcodes Light Switch
        internals directly (TODO(scale): this is this codebase's only
        environment so far), so everything it needs comes from --env
        lightswitch's own add_arguments, already registered by then."""
        del parser

    @staticmethod
    def run(*, args: argparse.Namespace) -> None:
        LightSwitchCli.run_method(
            args=args,
            method_factory=lambda env: SkillOracleMethod(env=env),
            num_cycles=0,  # an oracle never practices/learns -- one evaluation sweep only
            max_steps_per_interaction=0,  # unused: never reached with num_cycles=0
        )
