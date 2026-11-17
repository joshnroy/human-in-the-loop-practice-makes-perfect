"""Ours' Serving value model with GatherAndSweep split into Gather then Sweep5."""

import numpy as np
import pytest

from hitl_pmp.environments.racktoserving3d.config import RackServingConfig
from hitl_pmp.environments.racktoserving3d.environment import RackServingEnvironment
from hitl_pmp.environments.racktoserving3d.skill_provider import RackServingSkillProvider
from hitl_pmp.methods.belief_space.practice_domain import PracticeDomainWithoutSkills
from hitl_pmp.methods.belief_space.rack_serving_domain import (
    RACK_CHAIN_SKILLS,
    RACK_ROBOT_SKILLS,
    RACK_SERVING_DOMAIN,
    RACK_SKILL_STEPS,
    RackServingDeployment,
)
from hitl_pmp.methods.belief_space.types.practice_example_source import PracticeExampleSource

TWO_ROUTE = {"bulk_transfer_only": True, "two_route_rewards": True}


def _competences():
    c = {n: np.array([0.3, 0.6, 0.9]) for n in RACK_ROBOT_SKILLS}
    c["Sweep"] = np.array([0.5, 0.2, 0.7])
    return c


@pytest.mark.parametrize(
    ("kwargs", "value", "steps"),
    [
        (
            {},
            [0.6794141665487089, 0.4474017421721601, 0.8170423758138591],
            [2863.4882638629097, 5267.865110118402, 1829.5523270514102],
        ),
        (
            {"one_bulk_attempt": True},
            [0.292140240530279, 0.16435977262899204, 0.557544444438869],
            [1979.7312575472097, 4356.15061547008, 1424.55555554131],
        ),
        (
            {"one_bulk_attempt": True, "bulk_rewards": {"Sweep": 3.0, "GatherAndSweep": 5.0}},
            [1.2723669731002787, 2.5642591093329914, 4.065244444403791],
            [1979.7312575472097, 4356.15061547008, 4347.55555551208],
        ),
    ],
)
def test_without_the_split_the_value_model_is_exactly_the_old_one(*, kwargs, value, steps):
    v, s = RackServingDeployment.solve(
        competences=_competences(), horizon=12, step_weight=0.0001, **kwargs
    )
    assert v.tolist() == value
    assert s.tolist() == steps


def test_without_the_split_the_plan_has_the_old_states():
    plan = RackServingDeployment.plan(
        competences={n: x[:1] for n, x in _competences().items()},
        horizon=12,
        step_weight=0.0001,
        one_bulk_attempt=True,
        bulk_rewards={"Sweep": 3.0, "GatherAndSweep": 5.0},
    )
    assert len(plan) == 336
    assert all(state[1] != "gathered" for state, _ in plan)


def test_sweep5_is_reached_only_through_gather_and_priced_as_the_five_route():
    competences = {
        "PickScraper": np.array([1.0]),
        "ReturnScraper": np.array([1.0]),
        "PickFood": np.array([1.0]),
        "ReturnFoodToRack": np.array([1.0]),
        "Sweep": np.array([0.5]),
        "Gather": np.array([0.5]),
        "Sweep5": np.array([0.9]),
    }
    rewards = {"Sweep": 3.0, "Sweep5": 5.0}
    value, _ = RackServingDeployment.solve(
        competences=competences,
        horizon=3,
        step_weight=0.0,
        one_bulk_attempt=True,
        bulk_rewards=rewards,
    )
    # PickScraper, then Gather: on success Sweep5 (0.9 x 5), on failure Sweep (0.5 x 3).
    assert value[0] == pytest.approx(0.5 * 4.5 + 0.5 * 1.5)
    no_chain, _ = RackServingDeployment.solve(
        competences={**competences, "Gather": np.array([0.0])},
        horizon=3,
        step_weight=0.0,
        one_bulk_attempt=True,
        bulk_rewards=rewards,
    )
    assert no_chain[0] == pytest.approx(1.5)


def test_the_gathered_bit_is_read_from_the_measured_atom():
    atoms = [("HoldingScraper", ("robot", "wiper_0"))] + [
        ("OnRack", (f"cube_{i}", "rack_0")) for i in range(5)
    ]
    assert RackServingDeployment.state_of(atoms=atoms) == (5, "scraper", False)
    assert RackServingDeployment.state_of(atoms=[*atoms, ("Gathered", ())]) == (
        5,
        "gathered",
        False,
    )


def test_the_default_domain_declares_nothing_new():
    names = [name for name, _ in RACK_SERVING_DOMAIN.belief_sources]
    assert not set(RACK_CHAIN_SKILLS) & set(names)
    assert not set(RACK_CHAIN_SKILLS) & RACK_SERVING_DOMAIN.failure_effect_skills
    assert {"Gather", "Sweep5"} <= RACK_SKILL_STEPS.keys()


def _domain(*, split: bool):
    provider = RackServingSkillProvider(
        env=RackServingEnvironment(
            config=RackServingConfig(**TWO_ROUTE, split_gather_and_sweep=split)
        )
    )
    offered = frozenset(s.name for s in provider.skills()) | frozenset(
        g.skill.name for g in provider.human_cube_bin_reset_skills()
    )
    return PracticeDomainWithoutSkills.keep_offered(
        domain=RACK_SERVING_DOMAIN.with_goal_rewards(provider=provider), offered=offered
    )


def test_the_split_provider_gets_beliefs_for_gather_and_sweep5_and_prices_sweep5():
    split = _domain(split=True)
    names = [name for name, _ in split.belief_sources]
    assert names[-2:] == list(RACK_CHAIN_SKILLS) == ["Gather", "Sweep5"]
    assert "GatherAndSweep" not in names
    assert all(dict(split.belief_sources)[n] == PracticeExampleSource.SAMPLER for n in names[-2:])
    assert {"Gather", "Sweep5"} <= split.failure_effect_skills
    # Every older belief keeps its prior seed slot.
    seeds = split.prior_seed_order
    plain_order = [name for name, _ in RACK_SERVING_DOMAIN.belief_sources]
    assert list(seeds[: len(plain_order)]) == plain_order
    assert dict(split.bulk_rewards) == {"Sweep5": 5.0, "Sweep": 3.0}
    plain = _domain(split=False)
    assert not {"Gather", "Sweep5"} & {name for name, _ in plain.belief_sources}
    assert dict(plain.bulk_rewards) == {"GatherAndSweep": 5.0, "Sweep": 3.0}
