"""`BinOnGround(bin)`: the bin rests upright on the floor, and every robot skill needs it.

In EXP-21 seed 0 a toss tipped the robot-side bin onto its side with the cube inside it.
Every precondition of `PickCube` still held -- `GraspClear` is a 2-D test against an
upright 0.30 m footprint -- so the planner kept dispatching a top-down pick that the
grasp planner refused every time (the palm meets the bin wall that is now on top).
`BinOnGround` makes that state one where only a reset applies.

The bin poses below are taken from EXP-21 state logs (`tossing3d_state_log.jsonl`),
in KINDER's own feature names; see `bin_on_ground.py` for the threshold study.
"""

import importlib.util
import json
import math
from pathlib import Path

import pytest

from hitl_pmp.core.method.types import LiftedAtom
from hitl_pmp.core.problem.tasks.types import GroundAtom
from hitl_pmp.environments.tossing3d.bin_on_ground import (
    BIN_ON_GROUND_MAX_HEIGHT_M,
    BIN_ON_GROUND_MAX_TILT_DEG,
    BinOnGroundGeometry,
)
from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.layout import Tossing3DLayout
from hitl_pmp.environments.tossing3d.predicates import (
    BIN_AT_SIDE,
    BIN_ON_GROUND,
    CUBE_AT_SIDE,
    GRASP_CLEAR,
    HAND_EMPTY,
    HOLDING,
    IN_BIN,
    NOT_HOLDING,
    ON_GROUND,
    PICKUP_UNBLOCKED,
    ROBOT_AT_SIDE,
)
from hitl_pmp.environments.tossing3d.sides import Tossing3DSides
from hitl_pmp.environments.tossing3d.skill_provider import Tossing3DSkillProvider
from hitl_pmp.environments.tossing3d.skills import Tossing3DSkills
from hitl_pmp.planning.grounding import SkillGrounder

from .observations import BIN_ON_GROUND_ATOM, INITIAL_ATOMS, state

_TIPPED_FIXTURE = Path(__file__).parent / "fixtures" / "exp21_tipped_bin_state.json"
_RESET_NAMES = {"ask_for_reset_cube_bin_only", "non_human_reset_cube_bin_only"}


def _bin(*, z: float, tilt_deg: float = 0.0, yaw_deg: float = 0.0) -> dict[str, float]:
    """A bin pose tilted about its own x axis after a yaw, in KINDER feature names."""
    half_tilt = math.radians(tilt_deg) / 2
    half_yaw = math.radians(yaw_deg) / 2
    # q = yaw(z) * tilt(x)
    qw = math.cos(half_yaw) * math.cos(half_tilt)
    qx = math.cos(half_yaw) * math.sin(half_tilt)
    qy = math.sin(half_yaw) * math.sin(half_tilt)
    qz = math.sin(half_yaw) * math.cos(half_tilt)
    return {"x": 0.0, "y": 0.0, "z": z, "qw": qw, "qx": qx, "qy": qy, "qz": qz}


# ------------------------------------------------------------------- the classifier


def test_upright_resting_bins_are_on_the_ground() -> None:
    # EXP-21's first logged tick (seed 0): the bin at rest where the scene put it.
    assert BinOnGroundGeometry.holds(
        bin_={
            "x": 1.8369,
            "y": 1.4474,
            "z": -0.00010775,
            "qw": 1.0,
            "qx": 0.0,
            "qy": 0.0,
            "qz": 0.0,
        }
    )
    # Yawed 180 degrees is still upright (the trap-2 fixture's bin).
    assert BinOnGroundGeometry.holds(bin_=_bin(z=-0.0002, yaw_deg=180.0))
    # The largest resting tilt and depth any floor-resting bin reached across the EXP-21
    # logs: 0.2 degrees, z = -0.001.
    assert BinOnGroundGeometry.holds(bin_=_bin(z=-0.001, tilt_deg=0.2, yaw_deg=37.0))


def test_the_exp21_tipped_bin_is_not_on_the_ground() -> None:
    plain = json.loads(_TIPPED_FIXTURE.read_text())
    x, y, z, qw, qx, qy, qz = plain["bin_0"][:7]
    assert (round(x, 3), round(y, 3), round(z, 3)) == (-0.264, -1.727, 0.150)
    assert BinOnGroundGeometry.tilt_deg(bin_={"qx": qx, "qy": qy}) == pytest.approx(90.0, abs=0.2)
    assert not BinOnGroundGeometry.holds(
        bin_={"x": x, "y": y, "z": z, "qw": qw, "qx": qx, "qy": qy, "qz": qz}
    )


@pytest.mark.parametrize(
    ("label", "z", "tilt_deg"),
    [
        # Leaning 14.7 degrees with the cube under one wall (EXP-21 poses at 14.6-14.8
        # degrees all rest at z = 0.038).
        ("leaning on the cube, 14.7 deg", 0.038, 14.7),
        # EXP-21 Model B lambda 3e-6's frozen practice states, where the cube passed
        # through the bin floor and every pick was refused: seed 1 rests on the cube
        # at 15.01 deg, seed 2 rests level on it at 0.82 deg -- inside the tilt gate,
        # so only the height gate rejects it.
        ("EXP-21 seed 1: resting on the cube", 0.039, 15.01),
        ("EXP-21 seed 2: level on the cube", 0.049, 0.82),
        # The shallowest leaning pose logged: one edge propped on the cube.
        ("propped on the cube, 2.7 deg", 0.007, 2.74),
        # Flat on top of the cube (EXP-21, z = 0.049 at tilt < 1 degree), and the
        # least-tilted pose on it (1.23 degrees).
        ("flat on top of the cube", 0.049, 0.5),
        ("nearly flat on top of the cube", 0.049, 1.23),
        # On its side, like the tipped fixture.
        ("on its side", 0.150, 90.0),
        # Upside down, or balanced on the 0.2 m barrier.
        ("upside down", 0.2, 180.0),
        ("on the barrier", 0.2, 0.0),
    ],
)
def test_tipped_leaning_or_propped_bins_are_not_on_the_ground(
    *, label: str, z: float, tilt_deg: float
) -> None:
    del label
    assert not BinOnGroundGeometry.holds(bin_=_bin(z=z, tilt_deg=tilt_deg, yaw_deg=20.0))


def test_each_gate_rejects_on_its_own() -> None:
    """Tilt and height are separate conjuncts: neither is implied by the other."""
    assert not BinOnGroundGeometry.holds(
        bin_=_bin(z=0.0, tilt_deg=BIN_ON_GROUND_MAX_TILT_DEG + 0.5)
    )
    assert not BinOnGroundGeometry.holds(bin_=_bin(z=BIN_ON_GROUND_MAX_HEIGHT_M + 0.001))
    assert not BinOnGroundGeometry.holds(bin_=_bin(z=-(BIN_ON_GROUND_MAX_HEIGHT_M + 0.001)))


# -------------------------------------------------------------------- the predicate


def test_the_predicate_is_a_lookup_of_the_boundary_atom() -> None:
    env = Tossing3DEnvironment()
    upright = state(abstract_atoms=INITIAL_ATOMS)
    tipped = state(abstract_atoms=INITIAL_ATOMS - {BIN_ON_GROUND_ATOM})
    assert BIN_ON_GROUND.holds(upright, (env.bin,))
    assert not BIN_ON_GROUND.holds(tipped, (env.bin,))
    with pytest.raises(ValueError, match="no abstraction"):
        BIN_ON_GROUND.holds(state(), (env.bin,))


# ------------------------------------------------------------- the operator models


def _provider(*, layout: Tossing3DLayout) -> Tossing3DSkillProvider:
    env = Tossing3DEnvironment(layout=layout, scene_bg=False)
    return Tossing3DSkillProvider(env=env, offer_non_human_reset=True)


@pytest.mark.parametrize("layout", list(Tossing3DLayout))
def test_every_robot_skill_requires_bin_on_ground_and_declares_the_predicate(
    *, layout: Tossing3DLayout
) -> None:
    provider = _provider(layout=layout)
    assert BIN_ON_GROUND in provider.predicates()
    for skill in provider.skills():
        bins = [v for v in skill.parameters if v.type == Tossing3DEnvironment.bin_type]
        assert len(bins) == 1, skill.name
        assert LiftedAtom(predicate=BIN_ON_GROUND, variables=(bins[0],)) in skill.preconditions, (
            skill.name
        )


def test_appending_the_bin_preserves_upstream_prefix_order() -> None:
    """OpenGripper gains the bin last, as #346 appended side/bin parameters."""
    skills = Tossing3DSkills
    assert skills.OPEN_GRIPPER.parameters == (skills._robot, skills._cube, skills._bin)


def test_the_toss_does_not_model_tipping() -> None:
    """The planners assume the bin stays upright and replan from the observed state."""
    toss = Tossing3DSkills.MOVE_TO_TOSS_LOCATION_AND_TOSS
    assert BIN_ON_GROUND not in toss.ignore_effects
    assert not any(atom.predicate == BIN_ON_GROUND for atom in toss.delete_effects)


@pytest.mark.parametrize("layout", list(Tossing3DLayout))
def test_every_reset_re_places_the_bin_upright_without_requiring_it(
    *, layout: Tossing3DLayout
) -> None:
    provider = _provider(layout=layout)
    resets = provider.movables_reset_skills()
    assert {reset.skill.name for reset in resets} == _RESET_NAMES
    for reset in resets:
        assert any(atom.predicate == BIN_ON_GROUND for atom in reset.skill.add_effects)
        assert not any(atom.predicate == BIN_ON_GROUND for atom in reset.skill.preconditions)


def _pickable_atoms(*, provider: Tossing3DSkillProvider) -> frozenset[GroundAtom]:
    """A hand-empty cube on the robot's side of the barrier, bin across it."""
    env = provider.env
    robot, cube, bin_, barrier = env.robot, env.cube, env.bin, env.barrier
    return frozenset({
        GroundAtom(predicate=HAND_EMPTY, objects=(robot,)),
        GroundAtom(predicate=ON_GROUND, objects=(cube,)),
        GroundAtom(predicate=NOT_HOLDING, objects=(robot, cube)),
        GroundAtom(predicate=ROBOT_AT_SIDE, objects=(robot, barrier, Tossing3DSides.robot)),
        GroundAtom(predicate=CUBE_AT_SIDE, objects=(cube, barrier, Tossing3DSides.robot)),
        GroundAtom(predicate=BIN_AT_SIDE, objects=(bin_, barrier, Tossing3DSides.opposite)),
        GroundAtom(predicate=GRASP_CLEAR, objects=(cube, bin_)),
        GroundAtom(predicate=PICKUP_UNBLOCKED, objects=(cube,)),
    })


def _bin_on_ground(*, provider: Tossing3DSkillProvider) -> GroundAtom:
    return GroundAtom(predicate=BIN_ON_GROUND, objects=(provider.env.bin,))


def _closed_empty_atoms(*, provider: Tossing3DSkillProvider) -> frozenset[GroundAtom]:
    """Gripper closed on nothing: OpenGripper's own precondition, minus the bin."""
    from hitl_pmp.environments.tossing3d.predicates import CLOSED_EMPTY

    env = provider.env
    atoms = _pickable_atoms(provider=provider) - {
        GroundAtom(predicate=HAND_EMPTY, objects=(env.robot,))
    }
    return atoms | {GroundAtom(predicate=CLOSED_EMPTY, objects=(env.robot, env.cube))}


def _holding_atoms(*, provider: Tossing3DSkillProvider) -> frozenset[GroundAtom]:
    env = provider.env
    atoms = _pickable_atoms(provider=provider) - {
        GroundAtom(predicate=HAND_EMPTY, objects=(env.robot,)),
        GroundAtom(predicate=ON_GROUND, objects=(env.cube,)),
        GroundAtom(predicate=NOT_HOLDING, objects=(env.robot, env.cube)),
    }
    return atoms | {GroundAtom(predicate=HOLDING, objects=(env.robot, env.cube))}


def test_a_tipped_bin_leaves_only_the_resets_applicable() -> None:
    provider = _provider(layout=Tossing3DLayout.BARRIER)
    upright_marker = _bin_on_ground(provider=provider)
    for atoms in (
        _pickable_atoms(provider=provider),
        _closed_empty_atoms(provider=provider),
        _holding_atoms(provider=provider),
    ):
        upright = SkillGrounder.applicable_ground_skills(
            skills=provider.skills(),
            objects=provider.objects(),
            true_atoms=atoms | {upright_marker},
        )
        assert upright, "sanity: some robot skill applies with the bin upright"
        tipped = SkillGrounder.applicable_ground_skills(
            skills=provider.skills(), objects=provider.objects(), true_atoms=atoms
        )
        assert tipped == []
        for reset in provider.movables_reset_skills():
            assert reset.preconditions <= atoms


def test_bin_on_ground_holds_after_a_reset() -> None:
    from hitl_pmp.methods.belief_space.tossing3d_transition_model import apply_success_effects

    provider = _provider(layout=Tossing3DLayout.BARRIER)
    tipped = _pickable_atoms(provider=provider)
    for reset in provider.movables_reset_skills():
        effects = {reset: (reset.add_effects, reset.delete_effects, reset.ignore_effects)}
        after = apply_success_effects(true_atoms=tipped, ground_skill=reset, effects=effects)
        assert _bin_on_ground(provider=provider) in after


# ---------------------------------------------------------------- both planners


def _pomdp_method():
    from hitl_pmp.methods.belief_space.tossing3d_method import Tossing3DPomdpMethod

    env = Tossing3DEnvironment(scene_bg=False)
    return Tossing3DPomdpMethod(
        env=env,
        skill_provider=Tossing3DSkillProvider(env=env, offer_non_human_reset=True),
        seed=0,
        pomdp_num_particles=32,
    )


def test_the_belief_space_planner_offers_only_resets_with_a_tipped_bin() -> None:
    from hitl_pmp.methods.belief_space.tossing3d_transition_model import (
        make_tossing3d_search_state,
    )

    method = _pomdp_method()
    model = method._pomdp_model  # noqa: SLF001
    provider = method.skill_provider
    tipped = _pickable_atoms(provider=provider)
    upright = tipped | {_bin_on_ground(provider=provider)}
    valid_upright = model.get_valid_actions(
        environment_state=make_tossing3d_search_state(state=method.pomdp_state, true_atoms=upright)
    )
    assert "PickCube" in {ground.skill.name for ground in valid_upright}
    valid_tipped = model.get_valid_actions(
        environment_state=make_tossing3d_search_state(state=method.pomdp_state, true_atoms=tipped)
    )
    assert valid_tipped
    assert {ground.skill.name for ground in valid_tipped} <= _RESET_NAMES


def test_imagined_toss_and_pick_outcomes_keep_the_bin_upright() -> None:
    """The toss does not model tipping, so no imagined branch may drop BinOnGround --
    if one did, search would silently see every post-toss plan blocked."""
    from hitl_pmp.methods.belief_space.tossing3d_transition_model import (
        make_tossing3d_search_state,
    )

    method = _pomdp_method()
    model = method._pomdp_model  # noqa: SLF001
    provider = method.skill_provider
    marker = _bin_on_ground(provider=provider)
    for atoms in (_pickable_atoms(provider=provider), _holding_atoms(provider=provider)):
        search_state = make_tossing3d_search_state(
            state=method.pomdp_state, true_atoms=atoms | {marker}
        )
        actions = [
            ground
            for ground in model.get_valid_actions(environment_state=search_state)
            if ground.skill.name not in _RESET_NAMES
        ]
        assert actions
        for action in actions:
            for _probability, _next_state, after in model.outcomes(
                environment_state=search_state, state=method.pomdp_state, action=action
            ):
                assert marker in after, action.skill.name


def test_ees_plans_through_the_reset_from_a_tipped_bin() -> None:
    from hitl_pmp.methods.practice_makes_perfect.ees_method import EesMethod

    env = Tossing3DEnvironment(scene_bg=False)
    provider = Tossing3DSkillProvider(env=env, human_reset_practice_cost=5.0)
    method = EesMethod(env=env, skill_provider=provider, seed=0, reset_cost_gate=False)
    goal = frozenset({GroundAtom(predicate=IN_BIN, objects=(env.cube, env.bin))})
    tipped = _pickable_atoms(provider=provider)

    plan = method.plan_to(init_atoms=tipped, goal=goal, costs={}, practicing=True)

    assert [ground.skill.name for ground in plan] == [
        "ask_for_reset_cube_bin_only",
        "PickCube",
        "MoveToTossLocationAndToss",
    ]
    upright_plan = method.plan_to(
        init_atoms=tipped | {_bin_on_ground(provider=provider)}, goal=goal, costs={}
    )
    assert [ground.skill.name for ground in upright_plan] == [
        "PickCube",
        "MoveToTossLocationAndToss",
    ]


# ------------------------------------------------------------------- live KINDER


@pytest.mark.skipif(importlib.util.find_spec("kinder") is None, reason="needs KINDER")
def test_the_live_exp21_tipped_state_leaves_only_resets_to_both_planners() -> None:
    """Restore the EXP-21 seed-0 practice state (the tick after the tipping toss) in the
    live scene: the boundary must classify the bin off the ground, the belief-space
    planner must offer only resets, and Fast Downward must plan through one."""
    from hitl_pmp.methods.belief_space.tossing3d_method import Tossing3DPomdpMethod
    from hitl_pmp.methods.belief_space.tossing3d_transition_model import (
        make_tossing3d_search_state,
    )
    from hitl_pmp.methods.practice_makes_perfect.ees_method import EesMethod

    plain = json.loads(_TIPPED_FIXTURE.read_text())
    env = Tossing3DEnvironment()
    try:
        env.reset_to_seed(seed=125)
        upright_start = env.get_current_state()
        assert BIN_ON_GROUND.holds(upright_start, (env.bin,))
        tipped = env.restore_plain_snapshot(plain=plain)
        assert not BIN_ON_GROUND.holds(tipped, (env.bin,))

        provider = Tossing3DSkillProvider(env=env, offer_non_human_reset=True)
        atoms = SkillGrounder.abstract_state(
            state=tipped, objects=provider.objects(), predicates=provider.predicates()
        )
        # The rest of PickCube's preconditions still hold -- the EXP-21 trap.
        pick_preconditions_but_bin = {atom.predicate.name for atom in atoms} >= {
            "HandEmpty",
            "OnGround",
            "RobotAtSide",
            "CubeAtSide",
            "GraspClear",
        }
        assert pick_preconditions_but_bin
        assert (
            SkillGrounder.applicable_ground_skills(
                skills=provider.skills(), objects=provider.objects(), true_atoms=atoms
            )
            == []
        )

        pomdp = Tossing3DPomdpMethod(
            env=env, skill_provider=provider, seed=0, pomdp_num_particles=32
        )
        valid = pomdp._pomdp_model.get_valid_actions(  # noqa: SLF001
            environment_state=make_tossing3d_search_state(state=pomdp.pomdp_state, true_atoms=atoms)
        )
        assert valid
        assert {ground.skill.name for ground in valid} <= _RESET_NAMES

        ees = EesMethod(
            env=env,
            skill_provider=Tossing3DSkillProvider(env=env),
            seed=0,
            reset_cost_gate=False,
        )
        # The tipped state already scores: the cube lies inside the bin's region and
        # KINDER's own `_check_goals` agrees. Scoring is deliberately unchanged here, so
        # InBin is not a goal FD has to plan for; practising the pick is.
        assert IN_BIN.holds(tipped, (env.cube, env.bin))
        assert env.backend().check_goals()
        goal = frozenset({GroundAtom(predicate=HOLDING, objects=(env.robot, env.cube))})
        plan = ees.plan_to(init_atoms=atoms, goal=goal, costs={}, practicing=True)
        assert [ground.skill.name for ground in plan] == [
            "ask_for_reset_cube_bin_only",
            "PickCube",
        ]

        # The real reset re-places the bin upright, as the operator model promises.
        assert env.reset_movables(destination="robot_side")
        assert BIN_ON_GROUND.holds(env.get_current_state(), (env.bin,))
    finally:
        env.close()
