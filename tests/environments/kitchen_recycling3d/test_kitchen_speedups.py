"""The kitchen's two pure-performance shortcuts, each held to changing no result.

A reused PyBullet planning scene must plan exactly what a fresh one plans, and carry
nothing from its last use; a scene built without the visual-only furniture meshes must
simulate exactly what the full scene simulates, and only a scene that will not render
may be built that way. Tossing3D keeps building its planning scene fresh per use.
"""

import importlib.util
import json
from argparse import Namespace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from hitl_pmp.environments.kitchen_recycling3d.cli import KitchenRecyclingCli
from hitl_pmp.environments.kitchen_recycling3d.domain import KitchenConfig
from hitl_pmp.environments.kitchen_recycling3d.environment import KitchenRecycling3DEnvironment
from hitl_pmp.environments.tossing3d.kinder_backend import KinderBackend
from hitl_pmp.environments.tossing3d.predicates import HOLDING

pytestmark = [
    pytest.mark.skipif(
        importlib.util.find_spec("kinder") is None
        or importlib.util.find_spec("kinder_models") is None,
        reason="needs KINDER",
    ),
    pytest.mark.xdist_group("kitchen_recycling3d_speedups"),
]

REPO = Path(__file__).resolve().parents[3]
CONFIG = REPO / "scripts/kitchen_recycling/experiment-config.json"
FIXTURES = sorted((Path(__file__).parent / "fixtures").glob("*.json"))
SEED = 3


def kitchen(*, visual_furniture: bool = True) -> KitchenRecycling3DEnvironment:
    return KitchenRecycling3DEnvironment(
        kitchen_config=KitchenConfig.model_validate_json(CONFIG.read_text()),
        scene_bg=False,
        visual_furniture=visual_furniture,
    )


@pytest.fixture(scope="module")
def env():
    environment = kitchen()
    yield environment
    environment.close()


class Canonical:
    """A plan as plain floats, so two plans compare exactly, element for element."""

    @staticmethod
    def of(value: Any) -> Any:  # noqa: PLR0917 (a map callback)
        if hasattr(value, "theta") and hasattr(value, "x") and hasattr(value, "y"):
            return ("SE2", float(value.x), float(value.y), float(value.theta()))
        if isinstance(value, np.ndarray):
            return ("array", value.dtype.str, value.tolist())
        if isinstance(value, dict):
            return {Canonical.of(key): Canonical.of(item) for key, item in value.items()}
        if isinstance(value, list | tuple):
            return tuple(Canonical.of(item) for item in value)
        if isinstance(value, float | int | np.floating | np.integer):
            return float(value)
        if hasattr(value, "name") and hasattr(value, "value"):
            return ("enum", value.name)
        return repr(value)


class PlanningScene:
    """What a planner can observe of a PyBullet planning scene: every body's base pose
    and joint state, the bin poses `bin_wall_clearance` reads, and the held-object
    attachment."""

    @staticmethod
    def observe(*, sim: Any) -> Any:
        import pybullet as p

        client = sim.physics_client_id
        bodies = []
        for index in range(p.getNumBodies(physicsClientId=client)):
            body = p.getBodyUniqueId(index, physicsClientId=client)
            joints = p.getNumJoints(body, physicsClientId=client)
            bodies.append((
                body,
                p.getBasePositionAndOrientation(body, physicsClientId=client),
                p.getBaseVelocity(body, physicsClientId=client),
                tuple(
                    p.getJointState(body, joint, physicsClientId=client)[:2]
                    for joint in range(joints)
                ),
            ))
        return (
            tuple(bodies),
            {name: Canonical.of(tuple(pose)) for name, pose in sim._bin_poses.items()},
            sim.base_link_to_held_obj,
        )

    @staticmethod
    def pick_plan(*, backend: Any, state: Any, sim: Any) -> Any:
        controller = backend._ground_controller(
            module="tossing",
            key="pick_cube",
            object_names=(backend.robot_name, backend.cube_name, backend.barrier_name),
            state=state,
            pybullet_sim=sim,
        )
        try:
            controller.reset(state, None)
            failure = None
        except BaseException as exc:  # noqa: BLE001 (TrajectorySamplingFailure included)
            failure = f"{type(exc).__name__}: {exc}"
        return failure, Canonical.of(controller.plans), Canonical.of(controller.grasp_candidate)


def recorded_states(*, env: KitchenRecycling3DEnvironment) -> list[tuple[str, Any]]:
    """Every fixture that is a whole logged tick (the rest hold a few features only)."""
    states = []
    for path in FIXTURES:
        recorded = json.loads(path.read_text())
        plain = {name: value for name, value in recorded.items() if isinstance(value, list)}
        if not isinstance(recorded.get("robot"), list):
            continue
        env.reset_to_seed(seed=SEED)
        env.restore_plain_snapshot(plain=plain, dtype=np.float64)
        states.append((path.stem, env.kinder_backend().live_state().copy()))
    return states


def test_a_reused_planning_scene_plans_what_a_fresh_one_plans(*, env):
    """Every recorded fixture state, planned in one reused scene (dirtied by every plan
    before it, in both orders) and in a scene built fresh for it."""
    backend = env.kinder_backend()
    states = recorded_states(env=env)
    fresh = {}
    for name, state in states:
        sim = backend._build_pick_collision_sim(state=state)
        try:
            fresh[name] = PlanningScene.pick_plan(backend=backend, state=state, sim=sim)
        finally:
            sim.close()
    assert len(fresh) >= 5
    assert any(failure is None for failure, _, _ in fresh.values())
    first = None
    for name, state in [*states, *reversed(states)]:
        sim = backend._pick_collision_sim(state=state)
        first = sim if first is None else first
        assert sim is first, "the planning scene was rebuilt rather than reused"
        try:
            assert PlanningScene.pick_plan(backend=backend, state=state, sim=sim) == fresh[name]
        finally:
            sim.close()


def test_a_reused_planning_scene_is_exactly_a_fresh_one_after_a_pick_held_the_cube(*, env):
    env.reset_to_seed(seed=SEED)
    backend = env.kinder_backend()
    held = env.take_action(action=np.array([env.pick_cube_id, 0, 0, 0, 0.0]))
    assert env.last_skill_error() is None
    assert HOLDING.holds(held, (env.robot, env.cube))
    for name, state in [("holding", backend.live_state()), *recorded_states(env=env)]:
        reused = backend._pick_collision_sim(state=state)
        fresh = backend._build_pick_collision_sim(state=state)
        try:
            assert reused.base_link_to_held_obj is None, name
            (bodies, *rest), (expected, *expected_rest) = (
                PlanningScene.observe(sim=reused),
                PlanningScene.observe(sim=fresh),
            )
            assert rest == expected_rest, name
            # PyBullet stores a pose set by `createMultiBody` and one set by
            # `resetBasePositionAndOrientation` to within an ulp of each other, so the
            # bin's orientation may differ in its last bit until the first `set_state`,
            # which every planner makes before its first query and which moves the bin
            # in a new scene exactly as in a reused one.
            bin_body = reused._bins[backend.bin_name]
            for body, want in zip(bodies, expected, strict=True):
                if body[0] == bin_body:
                    assert body[1][0] == want[1][0], name
                    np.testing.assert_allclose(body[1][1], want[1][1], rtol=0, atol=1e-15)
                    assert body[2:] == want[2:], name
                else:
                    assert body == want, name
            reused.set_state(state)
            fresh.set_state(state)
            assert PlanningScene.observe(sim=reused) == PlanningScene.observe(sim=fresh), name
        finally:
            reused.close()
            fresh.close()


def test_a_skill_returns_its_planning_scene_for_the_next_use(*, env):
    env.reset_to_seed(seed=SEED)
    backend = env.kinder_backend()
    sim = backend._pick_collision_sim(state=backend.live_state())
    sim.close()
    env.take_action(action=np.array([env.pick_cube_id, 0, 0, 0, 0.0]))
    again = backend._pick_collision_sim(state=backend.live_state())
    try:
        assert again is sim
    finally:
        again.close()


def test_tossing3d_still_builds_its_planning_scene_fresh_per_use():
    """The reuse is the kitchen's alone: Tossing3D's call path is the one it had, a stock
    `PyBulletSim` built per use, and its scene has no furniture switch to flip."""
    assert "visual_furniture" not in KinderBackend.model_fields
    backend = KinderBackend()
    try:
        backend.reset(seed=0)
        state = backend.live_state()
        first = backend._pick_collision_sim(state=state)
        first.close()
        second = backend._pick_collision_sim(state=state)
        try:
            assert second is not first
            assert type(second).__name__ == "PyBulletSim"
        finally:
            second.close()
    finally:
        backend.close()


class MujocoScene:
    @staticmethod
    def model(*, env: KitchenRecycling3DEnvironment) -> Any:
        return env.kinder_backend()._object_centric()._robot_env.sim.model.mj_model

    @staticmethod
    def furniture_meshes(*, env: KitchenRecycling3DEnvironment) -> list[str]:
        import mujoco

        model = MujocoScene.model(env=env)
        return [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, int(model.geom_dataid[geom]))
            for geom in range(model.ngeom)
            if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH
            and (
                mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_MESH, int(model.geom_dataid[geom])
                ).startswith("recycling_")
            )
        ]

    @staticmethod
    def rollout(*, env: KitchenRecycling3DEnvironment) -> list[bytes]:
        """The scene state, reset and then at every tick of a pick and a throw."""
        env.reset_to_seed(seed=SEED)
        backend = env.kinder_backend()
        ticks: list[bytes] = []

        data = backend._object_centric()._robot_env.sim.data.mj_data

        def record() -> None:
            ticks.append(data.qpos.tobytes() + data.qvel.tobytes() + data.act.tobytes())

        record()
        state = backend.live_state()
        ticks.append(
            b"".join(
                obj.name.encode() + b"\0" + np.asarray(state[obj]).tobytes()
                for obj in sorted(state, key=lambda obj: obj.name)
            )
        )
        backend.set_control_step_observers(before=None, after=record)
        try:
            env.take_action(action=np.array([env.pick_cube_id, 0, 0, 0, 0.0]))
            from hitl_pmp.environments.tossing3d.toss import Tossing3DToss

            state = env.get_current_state()
            env.take_action(
                action=Tossing3DToss.compute_action(
                    params=np.array((1.45, 160.0, 740.0)), state=state
                )
            )
        finally:
            backend.set_control_step_observers(before=None, after=None)
        return ticks


def test_a_scene_that_will_not_render_drops_only_the_visual_furniture_meshes(*, env):
    physics_only = kitchen(visual_furniture=False)
    try:
        assert MujocoScene.rollout(env=physics_only) == MujocoScene.rollout(env=env)
        assert MujocoScene.furniture_meshes(env=physics_only) == []
        assert len(MujocoScene.furniture_meshes(env=env)) == 8
        visual, plain = MujocoScene.model(env=env), MujocoScene.model(env=physics_only)
        assert (plain.ngeom, plain.nbody) == (visual.ngeom, visual.nbody)
        with pytest.raises(RuntimeError, match="visual furniture"):
            physics_only.kinder_backend().render()
        with pytest.raises(RuntimeError, match="visual furniture"):
            physics_only.kinder_backend().set_substep_recording(enabled=True)
    finally:
        physics_only.close()


def test_only_a_run_that_defers_rendering_drops_the_meshes():
    assert KitchenRecycling3DEnvironment.model_fields["visual_furniture"].default is True
    for defer, expected in ((True, False), (False, True), (None, True)):
        args = Namespace(
            kitchen_config=CONFIG,
            scene_bg=False,
            seed=0,
            test_env_seed_offset=10000,
            **({} if defer is None else {"defer_rendering": defer}),
        )
        problem = KitchenRecyclingCli.build_problem(args=args)
        assert problem.env.visual_furniture is expected, defer
        assert problem.env.build_backend().visual_furniture is expected, defer
