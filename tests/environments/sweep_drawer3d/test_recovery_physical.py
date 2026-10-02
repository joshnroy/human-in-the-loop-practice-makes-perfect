"""Executed adapter recovery regressions in explicitly constructed diagnostic states."""

import json

import numpy as np

from hitl_pmp.core.method.types import GroundSkill
from hitl_pmp.environments.sweep_drawer3d.environment import SweepDrawerEnvironment
from hitl_pmp.environments.sweep_drawer3d.symbolic import SweepSymbols
from tests.environments.sweep_drawer3d.test_repositioning import _lay_out


def _available(*, env, name):
    skill = next(s for s in SweepSymbols.skills() if s.name == name)
    ground = GroundSkill(skill=skill, objects=(SweepSymbols.SCENE,))
    return all(a.predicate.holds(env.current_state, a.objects) for a in ground.preconditions)


def _act(*, env, name):
    state = env.take_action(action=np.array([env.ACTION_NAMES.index(name), 0.0, 0.0]))
    event = json.loads((env.output_dir / "sweep_events.jsonl").read_text().splitlines()[-1])
    assert not event["error"], event
    assert event["symbolic_success"], event
    return state


def test_partial_drawer_reopens_through_adapter(*, tmp_path):
    env = SweepDrawerEnvironment(canonical_seed=6, output_dir=tmp_path)
    try:
        env.hard_reset()
        _lay_out(session=env.session(), drawer=0.19, cubes={})
        env.current_state = env.observe()
        assert _available(env=env, name="OpenResetDrawer")
        _act(env=env, name="OpenResetDrawer")
        assert not _available(env=env, name="OpenResetDrawer")
        assert env.current_state.get(obj=SweepSymbols.SCENE, feature_name="ResetDrawerOpen")
    finally:
        env.close()


def test_blocked_pick_is_not_offered_and_wiggle_can_enable_pick(*, tmp_path):
    env = SweepDrawerEnvironment(canonical_seed=6, output_dir=tmp_path)
    try:
        env.hard_reset()
        wall = 0.872 + 0.25
        _lay_out(
            session=env.session(),
            drawer=0.25,
            cubes={
                "cube_3": (wall - 0.015, 0.045, 0.2375, 0.0),
                "cube_1": (wall - 0.019, 0.077, 0.2375, 0.0),
                "cube_4": (wall - 0.039, 0.105, 0.2375, 0.0),
            },
        )
        env.current_state = env.observe()
        assert not _available(env=env, name="PickCube3")
        assert _available(env=env, name="WiggleDrawerCube3")
        _act(env=env, name="WiggleDrawerCube3")
        assert _available(env=env, name="PickCube3")
        assert not _available(env=env, name="ParkWiper")
    finally:
        env.close()


def test_pair_pick_tracks_both_cubes_and_release_clears_them(*, tmp_path):
    env = SweepDrawerEnvironment(canonical_seed=6, output_dir=tmp_path)
    try:
        env.hard_reset()
        _lay_out(
            session=env.session(),
            cubes={
                "cube_1": (1.05, 0.10, 0.01, 0.0),
                "cube_2": (1.0705, 0.10, 0.01, 0.0),
            },
        )
        env.current_state = env.observe()
        assert _available(env=env, name="PickGroup12")
        _act(env=env, name="PickGroup12")
        assert _available(env=env, name="PlaceGroup12")
        assert not _available(env=env, name="ParkWiper")
        assert _available(env=env, name="OpenGripper")
        _act(env=env, name="OpenGripper")
        assert not env._held
        assert not _available(env=env, name="PlaceGroup12")
    finally:
        env.close()


def test_pair_placement_checks_every_released_member(*, tmp_path):
    env = SweepDrawerEnvironment(canonical_seed=6, output_dir=tmp_path)
    try:
        env.hard_reset()
        _lay_out(
            session=env.session(),
            cubes={
                "cube_1": (1.05, 0.10, 0.01, 0.0),
                "cube_2": (1.0705, 0.10, 0.01, 0.0),
            },
        )
        env.current_state = env.observe()
        _act(env=env, name="PickGroup12")
        assert _available(env=env, name="PlaceGroup12")
        _act(env=env, name="PlaceGroup12")
        assert all(
            env.current_state.get(obj=SweepSymbols.SCENE, feature_name=f"InPile{i}") for i in (1, 2)
        )
        assert not env._held
    finally:
        env.close()


def test_valid_start_dropped_wiper_can_be_recovered_and_parked(*, tmp_path):
    env = SweepDrawerEnvironment(canonical_seed=6, output_dir=tmp_path)
    try:
        env.hard_reset()
        state = env.session().state.copy()
        wiper = state.get_object_from_name("wiper_0")
        for feature, value in zip(("x", "y", "z"), (0.0, 0.0, 0.001), strict=True):
            state.set(wiper, feature, value)
        env.session().restore(state=state)
        env.current_state = env.observe()
        assert _available(env=env, name="RecoverWiper")
        _act(env=env, name="RecoverWiper")
        assert _available(env=env, name="ParkWiper")
        _act(env=env, name="ParkWiper")
        assert env.current_state.get(obj=SweepSymbols.SCENE, feature_name="WiperHome")
    finally:
        env.close()


def test_three_cube_row_tracks_all_members_and_places_them(*, tmp_path):
    env = SweepDrawerEnvironment(canonical_seed=6, output_dir=tmp_path)
    try:
        env.hard_reset()
        _lay_out(
            session=env.session(),
            cubes={
                "cube_1": (1.05, 0.10, 0.01, 0.0),
                "cube_2": (1.0705, 0.10, 0.01, 0.0),
                "cube_3": (1.091, 0.10, 0.01, 0.0),
            },
        )
        env.current_state = env.observe()
        assert _available(env=env, name="PickGroup123")
        assert not _available(env=env, name="PickGroup13")
        _act(env=env, name="PickGroup123")
        assert _available(env=env, name="PlaceGroup123")
        _act(env=env, name="PlaceGroup123")
        assert all(
            env.current_state.get(obj=SweepSymbols.SCENE, feature_name=f"InPile{i}")
            for i in (1, 2, 3)
        )
        assert not env._held
    finally:
        env.close()
