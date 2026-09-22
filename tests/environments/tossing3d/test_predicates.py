"""Offline tests for Tossing3D's five predicate *wrappers*.

The classifiers themselves are upstream's now, and their semantics are
`test_kb_predicate_parity.py`'s subject. What is left here is the wiring, which is
entirely ours and entirely capable of being wrong in ways upstream cannot catch: which
`Type`s each predicate is declared over, which object goes in which slot, and what
happens when a state carries no abstraction at all.

**This file used to be 450 lines of boundary probes**, most of them for `InBin` and for
the throw-pose band this domain derived from live scene geometry. The throw-pose band is
gone outright -- upstream deleted `RobotAtThrowPose` when it composed the base move into
the toss, so there is no pose between the two skills for a predicate to name. What is left
moved:

- The three *pure* upstream classifiers (`HandEmpty`, `OnGround`, `MovableIsDownX`) are
  `@staticmethod`s over an `ObjectCentricState`, which is constructible with no MuJoCo,
  so their probes live in `test_kb_predicate_parity.py` and still run offline.
- `Holding`'s forward-kinematics conjunct and `MovableInGoalRegion`'s ground-fixture
  read genuinely need a simulator, so their probes moved to `test_kinder_fidelity.py`.
  That is a real reduction in defence-in-depth for `InBin` -- this domain's success
  criterion, and the one a wrong goal box has already shipped against once -- and it is
  the price of the classifier no longer being ours to test in isolation.
"""

import importlib.util

import numpy as np
import pytest

from hitl_pmp.core.problem.tasks.types import Predicate
from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.predicates import (
    CLOSED_EMPTY,
    HAND_EMPTY,
    HOLDING,
    IN_BIN,
    KB_HAND_EMPTY,
    KB_HOLDING,
    KB_IN_GOAL_REGION,
    KB_IS_DOWN_X,
    KB_ON_GROUND,
    ON_GROUND,
    REACHABLE,
)
from hitl_pmp.environments.tossing3d.types import Tossing3DState

from .observations import HOLDING_ATOMS, INITIAL_ATOMS, LANDED_IN_REGION_ATOMS, state

_ENV = Tossing3DEnvironment()

_ALL = (IN_BIN, HAND_EMPTY, HOLDING, ON_GROUND, REACHABLE)


def test_closed_empty_requires_a_closed_gripper_without_a_held_cube() -> None:
    closed_empty = state(abstract_atoms=frozenset())
    opened = state(abstract_atoms=frozenset({(KB_HAND_EMPTY, ("robot",))}))
    holding = state(abstract_atoms=HOLDING_ATOMS)
    assert CLOSED_EMPTY.holds(closed_empty, (_ENV.robot, _ENV.cube))
    assert not CLOSED_EMPTY.holds(opened, (_ENV.robot, _ENV.cube))
    assert not CLOSED_EMPTY.holds(holding, (_ENV.robot, _ENV.cube))


def test_every_predicate_declares_the_types_it_is_actually_applied_to() -> None:
    """A `Predicate`'s `types` is what `SkillGrounder` enumerates over, so a wrong entry
    silently grounds a predicate onto objects it was never written for."""
    expected = {
        IN_BIN: (Tossing3DEnvironment.cube_type, Tossing3DEnvironment.bin_type),
        HAND_EMPTY: (Tossing3DEnvironment.robot_type,),
        HOLDING: (Tossing3DEnvironment.robot_type, Tossing3DEnvironment.cube_type),
        ON_GROUND: (Tossing3DEnvironment.cube_type,),
        REACHABLE: (Tossing3DEnvironment.cube_type, Tossing3DEnvironment.barrier_type),
    }
    for predicate, types in expected.items():
        assert predicate.types == types, predicate.name


def test_this_domains_predicate_names_are_unchanged_by_the_swap() -> None:
    """Upstream's classifiers, this domain's vocabulary. The names appear in the PDDL
    `skills.py` emits, in the operator model and in `Tossing3DTasks`' goal, so renaming
    them to upstream's would have been an operator-model change rather than a predicate
    one -- deliberately out of scope here."""
    assert [p.name for p in _ALL] == [
        "InBin",
        "HandEmpty",
        "Holding",
        "OnGround",
        "Reachable",
    ]


def test_each_predicate_reads_the_upstream_atom_it_is_backed_by() -> None:
    """The mapping from this domain's vocabulary to upstream's, one predicate at a time.

    Each case supplies *only* the atom that predicate should be reading, so a wrapper
    wired to the wrong upstream name fails here rather than agreeing by coincidence with
    whatever else happened to be true.
    """
    cases = [
        (IN_BIN, (_ENV.cube, _ENV.bin), (KB_IN_GOAL_REGION, ("cube_0",))),
        (HAND_EMPTY, (_ENV.robot,), (KB_HAND_EMPTY, ("robot",))),
        (HOLDING, (_ENV.robot, _ENV.cube), (KB_HOLDING, ("robot", "cube_0"))),
        (ON_GROUND, (_ENV.cube,), (KB_ON_GROUND, ("cube_0",))),
        (REACHABLE, (_ENV.cube, _ENV.barrier), (KB_IS_DOWN_X, ("cube_0", "cuboid_barrier"))),
    ]
    for predicate, objects, atom in cases:
        assert predicate.holds(state(abstract_atoms=frozenset({atom})), objects), predicate.name
        assert not predicate.holds(state(abstract_atoms=frozenset()), objects), predicate.name


def test_the_lambda_adapters_pass_objects_through_in_declaration_order() -> None:
    """`Predicate.holds` is a positional `(state, objects)` callable, and each predicate
    here adapts that to a keyword-only lookup. A transposed pair would typecheck and then
    silently ask whether the barrier is left of the cube."""
    reachable = state(abstract_atoms=frozenset({(KB_IS_DOWN_X, ("cube_0", "cuboid_barrier"))}))
    assert REACHABLE.holds(reachable, (_ENV.cube, _ENV.barrier))
    assert not REACHABLE.holds(reachable, (_ENV.barrier, _ENV.cube))

    holding = state(abstract_atoms=HOLDING_ATOMS)
    assert HOLDING.holds(holding, (_ENV.robot, _ENV.cube))
    assert not HOLDING.holds(holding, (_ENV.cube, _ENV.robot))


def test_in_bin_drops_its_second_argument_because_upstreams_is_unary() -> None:
    """`MovableInGoalRegion` takes only the movable: it reads the scored region off the
    live scene rather than off a target object. This domain keeps the binary shape --
    it is what the goal and the operators are written against -- so the bin argument is
    accepted and ignored, which means a second bin would be indistinguishable from the
    first. Pinned here so that is a documented property rather than a surprise."""
    in_region = state(abstract_atoms=frozenset({(KB_IN_GOAL_REGION, ("cube_0",))}))
    assert IN_BIN.holds(in_region, (_ENV.cube, _ENV.bin))
    # The barrier is not a bin, and the predicate cannot tell.
    assert IN_BIN.holds(in_region, (_ENV.cube, _ENV.barrier))


def test_a_state_with_no_abstraction_raises_rather_than_answering_false() -> None:
    """The quiet-wrongness guard.

    A `core.State` built for a translation test carries no atoms, and every predicate
    would then be vacuously `False` -- which reads exactly like "the cube is not in the
    bin" and would let a broken pipeline score a plausible zero. Raising makes the
    missing abstraction a loud failure instead.
    """
    bare = state()
    assert bare.abstract_atoms is None
    for predicate in _ALL:
        with pytest.raises(ValueError, match="no abstraction attached"):
            predicate.holds(bare, (_ENV.robot, _ENV.cube, _ENV.bin, _ENV.barrier))


def test_an_empty_abstraction_is_not_the_same_as_a_missing_one() -> None:
    """`frozenset()` means "upstream said nothing holds here", which is a real answer and
    must not raise. `None` means "nobody asked upstream", which must."""
    empty = state(abstract_atoms=frozenset())
    assert not HAND_EMPTY.holds(empty, (_ENV.robot,))


def test_the_state_subclass_is_still_a_core_state() -> None:
    """`Tossing3DState` has to satisfy everything a `core.State` does -- the feature-dim
    validator included -- or every consumer above `environments/` breaks."""
    built = state(abstract_atoms=INITIAL_ATOMS)
    assert isinstance(built, Tossing3DState)
    assert built.get(obj=_ENV.cube, feature_name="x") == pytest.approx(0.7129)
    with pytest.raises(ValueError, match="declares"):
        Tossing3DState(data={_ENV.cube: np.zeros(2)})


def test_the_abstraction_survives_the_deep_copy_the_environment_does() -> None:
    """`Tossing3DEnvironment.snapshot`/`restore` round-trip a state through
    `model_copy(deep=True)`. An abstraction that did not survive that would leave a
    restored state unable to answer its own predicates."""
    original = state(abstract_atoms=LANDED_IN_REGION_ATOMS)
    copied = original.model_copy(deep=True)
    assert copied.abstract_atoms == LANDED_IN_REGION_ATOMS
    assert IN_BIN.holds(copied, (_ENV.cube, _ENV.bin))


def test_all_five_predicates_are_predicates() -> None:
    """Cheap, but it is what stops a refactor leaving a bare lambda where a `Predicate`
    is expected -- `SkillGrounder` and `PddlWriter` both read `.name` and `.types`."""
    assert all(isinstance(predicate, Predicate) for predicate in _ALL)
    assert len({predicate.name for predicate in _ALL}) == len(_ALL)


# --------------------------------------------------------------------------- GraspClear
# The measured planner boundary these brackets pin: at the live controller, a pick with
# the cube's face 0.040 m from the bin's outer footprint plane is REFUSED ("No
# collision-free cube grasp", 0 steps) and one at 0.050 m is planned and succeeds.
# See `predicates.py`'s GraspClear docstring for the constants that boundary decomposes
# into. Offsets below are cube-center offsets from the bin center: o = 0.15 - 0.025 - g
# for an inside face gap g against the 0.30 m footprint and the 0.05 m cube.


def _grasp_clear_at(*, cube_x: float, cube_y: float = 0.0) -> bool:
    from hitl_pmp.environments.tossing3d.predicates import GRASP_CLEAR

    observed = state(cube_x=cube_x, cube_y=cube_y)
    return GRASP_CLEAR.holds(observed, (_ENV.cube, _ENV.bin))


def test_grasp_clear_brackets_the_measured_refusal_boundary() -> None:
    bin_x = 2.0
    assert not _grasp_clear_at(cube_x=bin_x - 0.105)  # face gap 0.020: refused live
    assert not _grasp_clear_at(cube_x=bin_x - 0.085)  # face gap 0.040: refused live
    assert _grasp_clear_at(cube_x=bin_x - 0.075)  # face gap 0.050: planned and held
    assert _grasp_clear_at(cube_x=bin_x)  # centered: planned and held


def test_grasp_clear_boundary_is_inclusive_at_the_margin() -> None:
    # Exactly the margin counts as clear: the measured 0.050 case was accepted live.
    assert _grasp_clear_at(cube_x=2.0 - (0.15 - 0.025 - 0.05))


def test_grasp_clear_applies_on_both_axes() -> None:
    assert not _grasp_clear_at(cube_x=2.0, cube_y=0.105)
    assert _grasp_clear_at(cube_x=2.0, cube_y=0.075)


def test_a_cube_far_outside_the_footprint_is_clear_by_default() -> None:
    assert _grasp_clear_at(cube_x=0.7129)  # the initial scene: 1.1 m from the bin


def test_a_cube_hugging_the_outside_of_the_wall_is_not_clear() -> None:
    # Face 0.020 m from the outer footprint plane, outside the bin.
    assert not _grasp_clear_at(cube_x=2.0 - 0.15 - 0.025 - 0.02)
    # Face 0.050 m out: clear.
    assert _grasp_clear_at(cube_x=2.0 - 0.15 - 0.025 - 0.05)


def test_a_cube_overlapping_the_wall_band_is_not_clear() -> None:
    assert not _grasp_clear_at(cube_x=2.0 - 0.15)  # centered on the footprint plane


def test_a_corner_approach_is_measured_in_two_dimensions() -> None:
    # 0.015 m beyond the footprint on each axis: diagonal distance ~0.021 < margin.
    assert not _grasp_clear_at(cube_x=2.0 - 0.19, cube_y=0.19)


@pytest.mark.skipif(
    importlib.util.find_spec("kinder") is None, reason="KINDER simulator dependency"
)
def test_grasp_clear_matches_the_live_grasp_planners_verdict() -> None:
    """Predicate <=> planner, on the three diagnosed states: a centered in-bin cube is
    clear and picked; an edge-lodged cube (face gap 0.020) is not-clear and REFUSED;
    the lip-equivalent gap (0.094) is clear and the planner accepts and picks it --
    the trap was the refusals, not the lip (see the 2026-09-22 trap diagnosis)."""
    import json as jsonlib

    from hitl_pmp.environments.tossing3d.kinder_backend import KinderBackend
    from hitl_pmp.environments.tossing3d.predicates import GRASP_CLEAR

    env = Tossing3DEnvironment()
    try:
        env.reset_to_seed(seed=125)
        base = KinderBackend.snapshot_to_plain(snapshot=env.backend().snapshot())
        for gap, expect_clear, expect_refused, expect_held in (
            (0.125, True, False, True),
            (0.020, False, True, False),
            (0.094, True, False, True),
        ):
            plain = jsonlib.loads(jsonlib.dumps(base))
            plain["bin_0"][0] = -0.5
            plain["bin_0"][1] = 0.0
            plain["cube_0"][0] = -0.5 - (0.15 - 0.025 - gap)
            plain["cube_0"][1] = 0.0
            # Resting on the bin floor, as landed cubes do (the trap-run prestates
            # all show z = 0.044); leaving the reset's ground z embeds the cube in
            # the bin floor and every grasp is refused for the wrong reason.
            plain["cube_0"][2] = 0.0444
            restored = env.restore_plain_snapshot(plain=plain)
            assert GRASP_CLEAR.holds(restored, (env.cube, env.bin)) == expect_clear, gap
            landed = env.take_action(action=np.array([0, 0, 0, 0, 0], dtype=float))
            error = env.last_skill_error()
            refused = bool(error and "No collision-free" in error)
            assert refused == expect_refused, (gap, error)
            assert HOLDING.holds(landed, (env.robot, env.cube)) == expect_held, gap
    finally:
        env.close()
