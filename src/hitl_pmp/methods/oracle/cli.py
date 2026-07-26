import argparse

from hitl_pmp.environments.lightswitch.cli import LightSwitchCli
from hitl_pmp.environments.tossingroom.cli import TossingRoomCli

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
        """No method-specific flags -- SkillOracleMethod carries no hyperparameters of
        its own; everything it needs comes from the selected --env's own
        add_arguments (lightswitch or tossingroom), already registered by then. run()
        below dispatches to that env's composition root."""
        del parser

    @staticmethod
    def run(*, args: argparse.Namespace) -> None:
        """Dispatches on --env to the selected domain's own composition root
        (<Domain>Cli.run_method), so --method skill-oracle works for every registered
        environment. The two branches are spelled out (rather than a dict) because
        each domain's run_method declares its own concrete environment type for
        method_factory, which a heterogeneous dict would erase. This mapping is
        hardcoded here (not read from hitl_pmp/cli.py's ENVIRONMENTS) since
        hitl_pmp/cli.py is the top layer -- methods/ may not import it (import-linter).
        getattr default: the global CLI always sets args.env, but this domain's own
        isolated tests build a parser without it and expect the Light Switch path."""
        env_name = getattr(args, "env", "lightswitch")
        if env_name == "tossingroom":
            TossingRoomCli.run_method(
                args=args,
                method_factory=lambda env: SkillOracleMethod(env=env),
                num_cycles=0,
                max_steps_per_interaction=0,
            )
        else:
            LightSwitchCli.run_method(
                args=args,
                method_factory=lambda env: SkillOracleMethod(env=env),
                num_cycles=0,  # an oracle never practices/learns -- one evaluation sweep only
                max_steps_per_interaction=0,  # unused: never reached with num_cycles=0
            )
