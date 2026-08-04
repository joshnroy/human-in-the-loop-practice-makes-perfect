import argparse

from hitl_pmp.cli import ENVIRONMENTS, Cli
from hitl_pmp.environments.tossing3d.cli import Tossing3DCli
from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment


def _parse(*, argv: list[str]) -> argparse.Namespace:
    return Cli.parse_args(argv=argv)


def test_tossing3d_is_registered_under_its_own_name() -> None:
    assert ENVIRONMENTS["tossing3d"] is Tossing3DCli


def test_domain_flags_default_to_the_environments_own_defaults() -> None:
    args = _parse(argv=["--env", "tossing3d", "--method", "skill-oracle"])
    fields = Tossing3DEnvironment.model_fields
    assert args.variant == fields["variant"].default
    assert args.swing_low == fields["swing_low"].default
    assert args.swing_high == fields["swing_high"].default
    assert args.canonical_seed == fields["canonical_seed"].default


def test_swing_prior_is_overridable() -> None:
    args = _parse(
        argv=[
            "--env",
            "tossing3d",
            "--method",
            "ees",
            "--swing-low",
            "0.4",
            "--swing-high",
            "0.8",
        ]
    )
    assert (args.swing_low, args.swing_high) == (0.4, 0.8)


def test_throw_standoff_is_deliberately_not_a_flag() -> None:
    """It is a ClassVar because a module-level Predicate reads it; exposing it as a
    flag would let a run configure a standoff `AtThrowPose` then denies."""
    args = _parse(argv=["--env", "tossing3d", "--method", "skill-oracle"])
    assert not hasattr(args, "throw_standoff")
