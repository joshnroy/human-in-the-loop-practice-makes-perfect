"""Bin pouring is available for an in-bin planning refusal, and performs real dynamics."""

import importlib.util

import numpy as np
import pytest

from hitl_pmp.core.method.types import GroundSkill
from hitl_pmp.core.problem.tasks.types import GroundAtom
from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.layout import Tossing3DLayout
from hitl_pmp.environments.tossing3d.predicates import HOLDING, IN_BIN, PICK_PLANNABLE
from hitl_pmp.environments.tossing3d.skill_provider import Tossing3DSkillProvider
from hitl_pmp.planning.fast_downward import FastDownwardPlanner
from hitl_pmp.planning.grounding import SkillGrounder

from .observations import INITIAL_ATOMS, state


def _scene(*, in_bin: bool = True, plannable: bool = False, same_side: bool = True):
    atoms = set(INITIAL_ATOMS)
    atoms.discard(("PickPlannable", ("robot", "cube_0")))
    if plannable:
        atoms.add(("PickPlannable", ("robot", "cube_0")))
    if in_bin:
        atoms.add(("MovableInGoalRegion", ("cube_0",)))
    if not same_side:
        # Typed side predicates classify the positions carried in the observation.
        return state(cube_x=2.0, abstract_atoms=frozenset(atoms))
    return state(cube_x=0.7, bin_x=0.7, abstract_atoms=frozenset(atoms))


@pytest.mark.parametrize("layout", list(Tossing3DLayout))
@pytest.mark.parametrize(
    ("in_bin", "plannable", "same_side", "expected"),
    [
        (True, False, True, True),
        (True, True, True, False),
        (False, False, True, False),
        (True, False, False, False),
    ],
)
def test_pouring_requires_in_bin_and_no_pick_plan(
    *, layout, in_bin, plannable, same_side, expected
):
    env = Tossing3DEnvironment(layout=layout)
    provider = Tossing3DSkillProvider(env=env)
    skills = provider.skills()
    assert "PourCubeOutOfBin" in {s.name for s in skills}
    observed = _scene(in_bin=in_bin, plannable=plannable, same_side=same_side)
    atoms = SkillGrounder.abstract_state(
        state=observed, objects=provider.objects(), predicates=provider.predicates()
    )
    applicable = SkillGrounder.applicable_ground_skills(
        skills=skills, objects=provider.objects(), true_atoms=atoms
    )
    assert any(s.skill.name == "PourCubeOutOfBin" for s in applicable) is expected


@pytest.mark.parametrize("layout", list(Tossing3DLayout))
def test_a_refused_in_bin_pick_can_be_planned_without_a_human(*, layout):
    env = Tossing3DEnvironment(layout=layout)
    provider = Tossing3DSkillProvider(env=env, offer_human_reset=False)
    observed = _scene()
    atoms = SkillGrounder.abstract_state(
        state=observed, objects=provider.objects(), predicates=provider.predicates()
    )
    plan = FastDownwardPlanner.plan(
        skills=provider.skills(),
        predicates=provider.predicates(),
        types=provider.types(),
        objects=provider.objects(),
        init_atoms=atoms,
        goal=frozenset({GroundAtom(predicate=HOLDING, objects=(env.robot, env.cube))}),
    )
    assert [step.skill.name for step in plan] == [
        "PourCubeOutOfBin",
        "PickCubeFromFloor" if layout == Tossing3DLayout.SAME_SIDE else "PickCube",
    ]
    skill = next(s for s in provider.skills() if s.name == "PourCubeOutOfBin")
    grounded = GroundSkill(
        skill=skill, objects=(env.robot, env.cube, env.bin, env.barrier, provider.objects()[-2])
    )
    assert GroundAtom(predicate=IN_BIN, objects=(env.cube, env.bin)) in grounded.delete_effects
    assert (
        GroundAtom(predicate=PICK_PLANNABLE, objects=(env.robot, env.cube)) in grounded.add_effects
    )
    assert grounded.skill.param_dim == 0
    np.testing.assert_array_equal(
        provider.compute_action(ground_skill=grounded, params=np.zeros(0), state=observed),
        [4, 0, 0, 0, 0],
    )


@pytest.mark.skipif(importlib.util.find_spec("kinder") is None, reason="KINDER is not installed")
@pytest.mark.parametrize("layout", list(Tossing3DLayout))
@pytest.mark.parametrize("fixture", ["exp22_A.json", "exp22_C.json", "exp22_D.json"])
def test_newly_plannable_historical_picks_do_not_allow_pouring(*, layout, fixture):
    import json
    from pathlib import Path

    env = Tossing3DEnvironment(layout=layout, scene_bg=False)
    try:
        env.hard_reset()
        backend = env.kinder_backend()
        plain = json.loads(
            (Path(__file__).parent / "fixtures/pick_plannable" / fixture).read_text()
        )
        plain["cuboid_barrier"] = backend.snapshot_to_plain(snapshot=backend.live_state())[
            "cuboid_barrier"
        ]
        state = env.restore_plain_snapshot(plain=plain)
        assert PICK_PLANNABLE.holds(state, (env.robot, env.cube))
        before = backend.snapshot_to_plain(snapshot=backend.live_state())
        env.take_action(action=np.array([env.pour_bin_id, 0, 0, 0, 0]))
        assert "unplannable in-bin pick" in env.last_skill_error()
        assert sum(env.last_controller_steps()) == 0
        assert backend.snapshot_to_plain(snapshot=backend.live_state()) == before
    finally:
        env.close()


@pytest.mark.skipif(importlib.util.find_spec("kinder") is None, reason="KINDER is not installed")
@pytest.mark.parametrize("layout", list(Tossing3DLayout))
@pytest.mark.parametrize("fixture", ["exp22_F.json"])
def test_physical_pour_recovers_a_saved_refusal(*, layout, fixture):
    import json
    from pathlib import Path

    from scipy.spatial.transform import Rotation

    from hitl_pmp.environments.tossing3d.predicates import BIN_ON_GROUND, ON_GROUND

    env = Tossing3DEnvironment(layout=layout, scene_bg=False)
    observations = []
    try:
        env.hard_reset()
        backend = env.kinder_backend()
        plain = json.loads(
            (Path(__file__).parent / "fixtures" / "pick_plannable" / fixture).read_text()
        )
        # These recordings predate the same-side layout's movable barrier. Keep the
        # current barrier geometry; replay the robot, bin and cube from the refusal.
        plain["cuboid_barrier"] = backend.snapshot_to_plain(snapshot=backend.live_state())[
            "cuboid_barrier"
        ]
        before = env.restore_plain_snapshot(plain=plain)
        assert IN_BIN.holds(before, (env.cube, env.bin))
        assert not PICK_PLANNABLE.holds(before, (env.robot, env.cube))
        original = backend.snapshot_to_plain(snapshot=backend.live_state())["bin_0"][:7]

        def capture():
            current = backend.live_state()
            bin_ = current.get_object_from_name("bin_0")
            robot = current.get_object_from_name("robot")
            observations.append(
                tuple(current.get(bin_, key) for key in ("x", "y", "z", "qw", "qx", "qy", "qz"))
                + (current.get(robot, "pos_base_rot"),)
            )

        backend.set_control_step_observers(before=None, after=capture)
        after = env.take_action(action=np.array([env.pour_bin_id, 0, 0, 0, 0]))
        backend.set_control_step_observers(before=None, after=None)
        assert env.last_skill_error() is None, env.last_skill_error()
        assert sum(env.last_controller_steps()) > 300
        assert max(pose[2] for pose in observations) > 0.35
        axes = Rotation.from_quat(np.asarray(observations)[:, [4, 5, 6, 3]]).apply([0, 0, 1])
        assert np.min(axes[:, 2]) < -0.9, "bin never turned upside down"
        samples = np.asarray(observations)
        mid_roll = (np.abs(axes[:, 2]) < 0.3) & (samples[:, 2] > 0.3)
        assert np.count_nonzero(mid_roll) > 5
        forward = np.column_stack((
            np.cos(samples[:, 7]),
            np.sin(samples[:, 7]),
            np.zeros(len(samples)),
        ))
        assert np.mean(np.abs(np.sum(axes[mid_roll] * forward[mid_roll], axis=1))) < 0.2, (
            "bin tipped fore/aft instead of sideways"
        )
        first_lift = np.flatnonzero((samples[:, 2] > 0.3) & (axes[:, 2] > 0.9))[0]
        yaw_delta = np.angle(np.exp(1j * (samples[mid_roll, 7] - samples[first_lift, 7])))
        assert np.mean(np.abs(yaw_delta)) == pytest.approx(np.pi / 2, abs=0.08)
        final = backend.snapshot_to_plain(snapshot=backend.live_state())["bin_0"][:7]
        np.testing.assert_allclose(final[:3], original[:3], atol=0.025)
        delta = (
            Rotation.from_quat(np.array(final)[[4, 5, 6, 3]])
            * Rotation.from_quat(np.array(original)[[4, 5, 6, 3]]).inv()
        )
        assert delta.magnitude() < np.deg2rad(5)
        assert not IN_BIN.holds(after, (env.cube, env.bin))
        assert ON_GROUND.holds(after, (env.cube,))
        assert BIN_ON_GROUND.holds(after, (env.bin,))
        assert PICK_PLANNABLE.holds(after, (env.robot, env.cube))
        # Execute the follow-up pick, rather than trusting only its dry-run verdict.
        picked = env.take_action(action=np.array([env.pick_cube_id, 0, 0, 0, 0]))
        assert env.last_skill_error() is None, env.last_skill_error()
        assert HOLDING.holds(picked, (env.robot, env.cube))
    finally:
        env.close()


@pytest.mark.skipif(importlib.util.find_spec("kinder") is None, reason="KINDER is not installed")
@pytest.mark.parametrize("layout", list(Tossing3DLayout))
def test_pour_refuses_an_inapplicable_raw_action_without_moving_the_world(*, layout):
    env = Tossing3DEnvironment(layout=layout, scene_bg=False)
    try:
        env.hard_reset()
        backend = env.kinder_backend()
        model = backend._object_centric()._robot_env.sim.model.mj_model  # noqa: SLF001
        assert model.body_subtreemass[model.body("bin_0").id] == pytest.approx(1.0)
        assert model.body_subtreemass[model.body("cube_0").id] == pytest.approx(0.1)
        before = backend.snapshot_to_plain(snapshot=backend.live_state())
        env.take_action(action=np.array([env.pour_bin_id, 0, 0, 0, 0]))
        assert env.last_controller_steps() == (0,)
        assert "unplannable in-bin pick" in env.last_skill_error()
        assert backend.snapshot_to_plain(snapshot=backend.live_state()) == before
    finally:
        env.close()


@pytest.mark.skipif(importlib.util.find_spec("kinder") is None, reason="KINDER is not installed")
def test_no_finger_force_causes_a_reported_grasp_failure(*, monkeypatch):
    import json
    from pathlib import Path

    from hitl_pmp.environments.tossing3d.kinder_backend import BinPourController

    original_step = BinPourController.step

    def no_grip(self):  # noqa: PLR0917 - controller callback
        action = original_step(self)
        action[10] = 0.0
        return action

    monkeypatch.setattr(BinPourController, "step", no_grip)
    env = Tossing3DEnvironment(scene_bg=False)
    try:
        env.hard_reset()
        fixture = Path(__file__).parent / "fixtures/pick_plannable/exp22_F.json"
        env.restore_plain_snapshot(plain=json.loads(fixture.read_text()))
        after = env.take_action(action=np.array([env.pour_bin_id, 0, 0, 0, 0]))
        assert "both fingers" in env.last_skill_error()
        assert sum(env.last_controller_steps()) > 0
        assert IN_BIN.holds(after, (env.cube, env.bin))
        assert (
            env.kinder_backend().snapshot_to_plain(snapshot=env.kinder_backend().live_state())[
                "bin_0"
            ][2]
            < 0.01
        )
    finally:
        env.close()


@pytest.mark.skipif(importlib.util.find_spec("kinder") is None, reason="KINDER unavailable")
@pytest.mark.parametrize("layout", list(Tossing3DLayout))
def test_disabling_pour_preserves_the_movable_one_kg_bin(*, layout):
    env = Tossing3DEnvironment(
        layout=layout, scene_bg=False, enable_bin_pouring=False, bin_mass_kg=1.0, cube_mass_kg=0.1
    )
    try:
        env.hard_reset()
        model = env.kinder_backend()._object_centric()._robot_env.sim.model.mj_model
        assert model.body_jntnum[model.body("bin_0").id] == 1
        assert model.body_subtreemass[model.body("bin_0").id] == pytest.approx(1.0)
        assert model.body_subtreemass[model.body("cube_0").id] == pytest.approx(0.1)
        assert "PourCubeOutOfBin" not in {s.name for s in Tossing3DSkillProvider(env=env).skills()}
    finally:
        env.close()
