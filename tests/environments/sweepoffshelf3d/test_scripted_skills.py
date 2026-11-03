"""`ShelfSkills` contact predicates against their per-contact reference definitions.

The predicates run every physics substep, so they are vectorised; these tests pin
the vectorised forms to the plain per-contact loops they replace, on randomised
contact sets drawn from a fake model with real MuJoCo-shaped arrays.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from hitl_pmp.environments.sweepoffshelf3d.scripted_skills import ShelfSkills

NGEOM = 40
WIPER_BODY = 3


class FakeContacts:
    """MuJoCo's struct-of-arrays contact view: field arrays sized to the active set."""

    def __init__(self, *, pairs: np.ndarray, dist: np.ndarray):
        self.geom1 = pairs[:, 0].astype(np.int32)
        self.geom2 = pairs[:, 1].astype(np.int32)
        self.dist = dist.astype(float)

    def __getitem__(self, index: slice) -> list[SimpleNamespace]:
        return [
            SimpleNamespace(geom1=g1, geom2=g2, dist=d)
            for g1, g2, d in zip(self.geom1, self.geom2, self.dist, strict=True)
        ][index]


def fake_shelf(*, rng: np.random.Generator, ncon: int) -> SimpleNamespace:
    bodies = rng.integers(0, 8, size=NGEOM)
    bodies[[0, 1]] = WIPER_BODY
    bodies[[10, 11, 12, 13]] = [5, 5, 6, 6]
    pairs = rng.integers(0, NGEOM, size=(ncon, 2))
    # Bias towards the interesting pairs: pads against the wiper, either order.
    for row in range(ncon):
        if rng.random() < 0.3:
            pad, wiper = int(rng.choice([10, 11, 12, 13])), int(rng.choice([0, 1]))
            pairs[row] = (pad, wiper) if rng.random() < 0.5 else (wiper, pad)
    shelf = SimpleNamespace(
        model=SimpleNamespace(geom_bodyid=bodies, ngeom=NGEOM),
        data=SimpleNamespace(
            ncon=ncon, contact=FakeContacts(pairs=pairs, dist=rng.normal(0, 0.002, ncon))
        ),
        wiper_body=WIPER_BODY,
        pad_sides=[frozenset({10, 11}), frozenset({12, 13})],
    )
    return shelf


def reference_pads_hold_wiper(*, shelf: SimpleNamespace) -> bool:
    """The per-contact loop `pads_hold_wiper` replaced, verbatim in behaviour."""
    touched = [False] * len(shelf.pad_sides)
    for contact in shelf.data.contact[: shelf.data.ncon]:
        for mine, other in ((contact.geom1, contact.geom2), (contact.geom2, contact.geom1)):
            if shelf.model.geom_bodyid[other] != shelf.wiper_body:
                continue
            for side, pads in enumerate(shelf.pad_sides):
                touched[side] |= int(mine) in pads
    return all(touched)


@pytest.mark.parametrize("seed", range(400))
def test_pads_hold_wiper_matches_the_per_contact_loop(*, seed):
    rng = np.random.default_rng(seed)
    shelf = fake_shelf(rng=rng, ncon=int(rng.integers(0, 12)))
    assert ShelfSkills.pads_hold_wiper(shelf) == reference_pads_hold_wiper(shelf=shelf)


def test_pads_hold_wiper_needs_both_sides_and_ignores_non_wiper_contacts():
    shelf = fake_shelf(rng=np.random.default_rng(0), ncon=0)
    shelf.model.geom_bodyid[20] = 7

    def set_pairs(pairs):
        array = np.array(pairs, dtype=int).reshape(-1, 2)
        shelf.data.contact = FakeContacts(pairs=array, dist=np.zeros(len(array)))
        shelf.data.ncon = len(array)

    set_pairs([])
    assert not ShelfSkills.pads_hold_wiper(shelf)
    set_pairs([(10, 0)])
    assert not ShelfSkills.pads_hold_wiper(shelf)
    set_pairs([(10, 0), (1, 13)])
    assert ShelfSkills.pads_hold_wiper(shelf)
    set_pairs([(10, 0), (13, 20)])
    assert not ShelfSkills.pads_hold_wiper(shelf)


def test_pads_hold_wiper_follows_a_reassigned_pad_partition():
    shelf = fake_shelf(rng=np.random.default_rng(1), ncon=0)
    shelf.data.contact = FakeContacts(pairs=np.array([[10, 0], [12, 0]]), dist=np.zeros(2))
    shelf.data.ncon = 2
    assert ShelfSkills.pads_hold_wiper(shelf)
    shelf.pad_sides = [frozenset({10}), frozenset({11})]
    assert not ShelfSkills.pads_hold_wiper(shelf)


def cost_shelf(*, rng: np.random.Generator, ncon: int) -> SimpleNamespace:
    """Robot 10-19 (pads 10-13), cupboard 20-27, box 28-31, wiper 0-1, others free."""
    pairs = rng.integers(0, NGEOM, size=(ncon, 2))
    dist = rng.normal(-0.001, 0.003, ncon)
    dist[rng.random(ncon) < 0.1] = 0.0
    dist[rng.random(ncon) < 0.05] = -0.001
    contacts = FakeContacts(pairs=pairs, dist=dist)
    return SimpleNamespace(
        model=SimpleNamespace(ngeom=NGEOM),
        mujoco=SimpleNamespace(mj_collision=lambda model, data: None),
        robot_geoms=set(range(10, 20)),
        pad_geoms={10, 11, 12, 13},
        cupboard_geoms=set(range(20, 28)),
        box_geoms=set(range(28, 32)),
        wiper_geoms=[0, 1],
        scratch=SimpleNamespace(ncon=ncon, contact=contacts),
    )


def reference_collision_cost(*, shelf: SimpleNamespace, d: SimpleNamespace, holding: bool) -> float:
    """The per-contact loop `collision_cost` replaced, verbatim in behaviour."""
    cost = 0.0
    wiper = set(shelf.wiper_geoms)
    fixed = shelf.cupboard_geoms | shelf.box_geoms
    robot = shelf.robot_geoms
    for contact in d.contact[: d.ncon]:
        a, b = contact.geom1, contact.geom2
        depth = max(0.0, -contact.dist)
        if a in robot and b in robot:
            cost += max(0.0, depth - 0.001)
        elif (a in robot and b in fixed) or (b in robot and a in fixed):
            cost += depth + 0.001
        elif not holding and (
            (a in robot and b in wiper and a not in shelf.pad_geoms)
            or (b in robot and a in wiper and b not in shelf.pad_geoms)
        ):
            cost += depth
    return cost


@pytest.mark.parametrize("seed", range(300))
@pytest.mark.parametrize("holding", [False, True])
def test_collision_cost_is_bit_identical_to_the_per_contact_loop(*, seed, holding):
    rng = np.random.default_rng(seed)
    shelf = cost_shelf(rng=rng, ncon=int(rng.integers(0, 40)))
    got = ShelfSkills.collision_cost(shelf, d=shelf.scratch, holding=holding)
    want = reference_collision_cost(shelf=shelf, d=shelf.scratch, holding=holding)
    assert float(got) == float(want)
