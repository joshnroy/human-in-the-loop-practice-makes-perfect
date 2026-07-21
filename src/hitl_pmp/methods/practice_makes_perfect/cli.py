import argparse

from hitl_pmp.environments.lightswitch.cli import LightSwitchCli

from .random_skills_method import RandomSkillsMethod


class RandomSkillsCli:
    """Plugs RandomSkillsMethod into the global CLI under --method random-skills.
    Lives alongside RandomSkillsMethod under methods/practice_makes_perfect/ (not
    environments/lightswitch/), matching methods/oracle/cli.py's SkillOracleCli
    precedent: a method-CLI is method-specific glue, not environment-specific. A
    static-method container, never instantiated, same as every other
    business-logic class in this project."""

    @staticmethod
    def add_arguments(*, parser: argparse.ArgumentParser) -> None:
        """No method-specific flags -- RandomSkillsMethod's own RNG reuses the
        global --seed (already registered by hitl_pmp/cli.py; also drives Light
        Switch's own task-sampling RNG via --env lightswitch's add_arguments), so
        there's no separate seed flag to add here."""
        del parser

    @staticmethod
    def run(*, args: argparse.Namespace) -> None:
        LightSwitchCli.run_method(
            args=args,
            method_factory=lambda env: RandomSkillsMethod(env=env, seed=args.seed),
            num_cycles=0,  # this baseline never practices/learns -- one evaluation sweep only
            max_steps_per_interaction=0,  # unused: never reached with num_cycles=0
        )
