"""Physical goal parity, explicit restoration and continuous practice guard."""

import pytest

from hitl_pmp.environments.sweep_simple3d.environment import SweepSimpleEnvironment
from hitl_pmp.environments.sweep_simple3d.symbolic import SimpleSymbols


def test_native_goal_and_human_reset_contract() -> None:
    env = SweepSimpleEnvironment(canonical_seed=0)
    try:
        env.hard_reset()
        state = env.get_current_state()
        native = env.session().env.unwrapped._object_centric_env._check_goals()
        symbolic = all(a.predicate.holds(state, a.objects) for a in SimpleSymbols.goal_atoms())
        assert symbolic == native
        assert all(a.predicate.holds(state, a.objects) for a in SimpleSymbols.initial_atoms())
        edited = env.session().state.copy()
        for i, cube in enumerate(SimpleSymbols.CUBES):
            obj = edited.get_object_from_name(cube.name)
            for key, value in (("x", 0.25 + i * 0.1), ("y", 1.2), ("z", 0.011)):
                edited.set(obj, key, value)
        env.session().restore(state=edited)
        state = env.observe()
        assert env.session().env.unwrapped._object_centric_env._check_goals()
        assert all(a.predicate.holds(state, a.objects) for a in SimpleSymbols.goal_atoms())
        assert env.reset_movables()
        env.take_action(action=env.noop_action())
        with pytest.raises(RuntimeError, match="Automatic reset"):
            env.hard_reset()
        assert env._hard_reset_count == 1
        assert env._human_reset_count == 1
    finally:
        env.close()


def test_closed_gripper_proximity_does_not_imply_a_physical_grasp(
    *,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hitl_pmp.environments.sweep_simple3d.controllers import FloorPrimitives
    from hitl_pmp.environments.sweep_simple3d.session import SweepSimpleSession

    session = SweepSimpleSession(seed=0)
    primitive = FloorPrimitives.create(session=session, distance=0.7, heading_offset=0.0)
    try:
        monkeypatch.setattr(SweepSimpleSession, "gripper", lambda self: 1.0)
        handle, _ = primitive.wiper_handle_geometry()
        assert not primitive.wiper_in_hand(
            gripper=session.mj_data.geom_xpos[handle].copy(), wiper=session.position(name="wiper_0")
        )
    finally:
        primitive.scene._sim.close()
        session.close()
