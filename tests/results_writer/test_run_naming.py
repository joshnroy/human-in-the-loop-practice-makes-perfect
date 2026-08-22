"""Covers `RunNamer`: the one place a run's name is built, for any tracker backend.

The load-bearing property is not any particular string -- it is that the namer is
**injective over the arms of a real sweep grid**. A name that does not distinguish two
arms puts two different experiments in one canonical slot, which is the defect this
whole feature exists to remove: the live project holds 121 runs under 10 distinct
names, 13/121 of them called `ees-seed0` across two different environments.

Nothing here imports `wandb`, so all of it runs on CI, which never installs the
optional extra.
"""

import argparse

import pytest

from hitl_pmp.results_writer.run_naming import RunNamer


class Namespaces:
    """Resolved argparse namespaces, built the way the real CLI would resolve them.

    A static-method container, never instantiated, same as every other business-logic
    class in this project. Deliberately *complete* namespaces rather than the two or
    three attributes a given assertion reads: the namer's contract is that it fails
    loudly on a namespace missing a field it names, so a half-built namespace would
    exercise the error path by accident in every test."""

    @staticmethod
    def ees_tossingroom(**overrides: object) -> argparse.Namespace:
        fields: dict[str, object] = {
            "env": "tossingroom",
            "method": "ees",
            "seed": 3,
            "practice_reset_policy": "never",
            "two_way_ledge": False,
            "unsplit_skills": False,
            "ask_for_reset_cube_bin_cost": None,
            "num_cycles": 100,
        }
        fields.update(overrides)
        return argparse.Namespace(**fields)

    @staticmethod
    def skill_oracle_lightswitch(**overrides: object) -> argparse.Namespace:
        """The oracle's namespace genuinely has **no** `num_cycles` and neither reset
        skill's cost flag: `SkillOracleCli.add_arguments` adds nothing and its `run`
        passes the literal 0 to `MethodRunner`. So absence is a real state the namer
        must handle, not a hypothetical."""
        fields: dict[str, object] = {
            "env": "lightswitch",
            "method": "skill-oracle",
            "seed": 7,
            "practice_reset_policy": "scheduled",
        }
        fields.update(overrides)
        return argparse.Namespace(**fields)


def test_the_name_carries_environment_method_arm_and_seed() -> None:
    """The whole point: a reader can tell what a run was from the run list, without
    opening it."""
    assert RunNamer.name(args=Namespaces.ees_tossingroom()) == (
        "tossingroom-ees-oneway-split-never-cube-bin-reset-cost-none-c100-seed3"
    )


def test_the_seed_sorts_last() -> None:
    """Seed is the innermost loop of every sweep, so it goes last: an alphabetical run
    list then groups an arm's seeds together instead of interleaving arms."""
    assert RunNamer.name(args=Namespaces.ees_tossingroom()).endswith("-seed3")


def test_a_boolean_arm_flag_names_both_of_its_states() -> None:
    """`--two-way-ledge` is a store_true, so the off state has no token of its own
    unless one is declared. Naming both states means a run says which ledge it ran on
    rather than leaving the reader to infer it from a missing word."""
    one_way = RunNamer.name(args=Namespaces.ees_tossingroom(two_way_ledge=False))
    two_way = RunNamer.name(args=Namespaces.ees_tossingroom(two_way_ledge=True))
    assert "-oneway-" in one_way
    assert "-twoway-" in two_way


def test_a_method_specific_field_is_omitted_when_the_method_does_not_have_it() -> None:
    """No `num_cycles`, and no invented default standing in for it.

    `getattr(args, "num_cycles", 0)` is the trap: the absent-attribute default happens
    to equal the literal the oracle passes today, so it looks right and would silently
    put a genuinely different run under an existing name the first time a method
    computes its own cycle count."""
    name = RunNamer.name(args=Namespaces.skill_oracle_lightswitch())
    assert name == "lightswitch-skill-oracle-scheduled-seed7"
    assert "c0" not in name


def test_a_missing_required_field_raises_instead_of_being_defaulted() -> None:
    """`--seed` is global, so its absence means the namespace is not a resolved one --
    or a flag was renamed and every name silently lost an axis. Either way, loudly."""
    namespace = Namespaces.ees_tossingroom()
    del namespace.seed
    with pytest.raises(ValueError, match="seed"):
        RunNamer.name(args=namespace)


def test_names_are_url_and_path_safe() -> None:
    """A run name ends up in a URL and in offline directory names, so it stays
    lowercase `[a-z0-9-]` regardless of what a flag's value looked like -- a float cost
    is the case worth pinning here, since "0.134" contains a "." _slug must strip."""
    name = RunNamer.name(args=Namespaces.ees_tossingroom(ask_for_reset_cube_bin_cost=0.134))
    assert name == name.lower()
    assert set(name) <= set("abcdefghijklmnopqrstuvwxyz0123456789-")


def test_every_arm_of_a_realistic_grid_gets_its_own_name() -> None:
    """The property that matters, on the grid shape this project actually runs: two
    ledges x two reset policies x two cycle budgets x ten seeds.

    80/80 distinct. Under the previous `f"{method}-seed{seed}"` scheme the same grid
    produced 10/80 distinct names, which is the defect being fixed."""
    names = {
        RunNamer.name(
            args=Namespaces.ees_tossingroom(
                two_way_ledge=two_way,
                practice_reset_policy=policy,
                num_cycles=cycles,
                seed=seed,
            )
        )
        for two_way in (False, True)
        for policy in ("scheduled", "never")
        for cycles in (10, 100)
        for seed in range(10)
    }
    assert len(names) == 80


def test_two_methods_on_one_environment_do_not_collide() -> None:
    """The oracle's shorter name (no cycle count, no reset-skill cost flags) must not be a prefix
    collision with a learner's -- absence of an optional field is only safe because
    `method`, which determines that absence, is itself in the name."""
    assert RunNamer.name(args=Namespaces.skill_oracle_lightswitch()) != RunNamer.name(
        args=Namespaces.ees_tossingroom(env="lightswitch", method="skill-oracle")
    )
