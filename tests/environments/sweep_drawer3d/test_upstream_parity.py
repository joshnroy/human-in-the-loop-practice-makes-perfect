"""Pinned predicate boundary parity and no hidden live reset during construction."""

import numpy as np
import pytest


def test_direct_upstream_boundary_parity_and_no_live_state_change():
    pytest.importorskip("shapely")
    from hitl_pmp.environments.sweep_drawer3d.session import SweepDrawerSession
    from hitl_pmp.environments.sweep_drawer3d.upstream import SweepUpstreamFacts

    session = SweepDrawerSession(seed=0)
    before = session.mj_data.qpos.copy()
    abstractor = SweepUpstreamFacts.create(state=session.state)
    try:
        assert np.array_equal(before, session.mj_data.qpos)
        for pos in (0.0, 0.039999, 0.04, 0.040001, 0.15):
            for grip in (0.0, 0.001, 0.001001, 0.1, 0.100001):
                state = session.state.copy()
                drawer = state.get_object_from_name("kitchen_island_drawer_s1c1")
                robot = state.get_object_from_name("robot")
                cube = state.get_object_from_name("cube_0")
                state.set(drawer, "pos", pos)
                state.set(robot, "pos_gripper", grip)
                # Deliberately outside the drawer: upstream InDrawer is a height
                # heuristic, not physical containment. Never strengthen its name.
                state.set(cube, "x", 3.0)
                state.set(cube, "z", 0.02)
                # ObjectCentricState stores float32; compare the exact stored
                # value, including rounding at nominal .001/.04/.1 boundaries.
                actual_pos = state.get(drawer, "pos")
                actual_grip = state.get(robot, "pos_gripper")
                expected = abstractor.state_abstractor(state).atoms
                mapped = SweepUpstreamFacts.observed(abstractor=abstractor, state=state)
                assert mapped == frozenset(SweepUpstreamFacts.feature(atom=a) for a in expected)
                assert ("DrawerOpen" in mapped) == (actual_pos > 0.04)
                assert ("DrawerClosed" in mapped) == (actual_pos <= 0.04)
                assert ("InDrawer0" in mapped) == (actual_pos > 0.04)
                assert ("HandEmpty" in mapped) == bool(np.isclose(actual_grip, 0.0, atol=0.001))
                assert ("HoldingWiper" in mapped) == (actual_grip > 0.1)
                assert ("OnTableWiper" in mapped) == (actual_grip <= 0.1)
        assert SweepUpstreamFacts.goal(abstractor=abstractor, state=session.state) == frozenset((
            "DrawerOpen",
            "HoldingWiper",
            *(f"InDrawer{i}" for i in range(5)),
        ))
        assert np.array_equal(before, session.mj_data.qpos)
    finally:
        SweepUpstreamFacts.close(abstractor=abstractor)
        session.close()


def test_stock_operators_and_goal_match_actual_native_factory():
    from kinder_bilevel_planning.env_models.dynamic3d.tidybot3d_sweep3D import (
        create_bilevel_planning_models,
    )

    from hitl_pmp.environments.sweep_drawer3d.environment import SweepDrawerEnvironment
    from hitl_pmp.environments.sweep_drawer3d.skill_provider import SweepDrawerSkillProvider
    from hitl_pmp.environments.sweep_drawer3d.upstream import SweepUpstreamFacts

    env = SweepDrawerEnvironment()
    env.hard_reset()
    session = env.session()
    models = create_bilevel_planning_models(
        session.env.observation_space, session.env.action_space, num_objects=5
    )
    try:
        names = {"open_drawer": "OpenDrawer", "pick_wiper": "PickWiper", "sweep": "Sweep"}
        own = {s.name: s for s in SweepDrawerSkillProvider(env=env).deployment_skills()}
        for native in models.ground_operators:
            ours = own[names[native.name]]
            for attribute in ("preconditions", "add_effects", "delete_effects"):
                assert {a.predicate.name for a in getattr(ours, attribute)} == {
                    SweepUpstreamFacts.feature(atom=a) for a in getattr(native, attribute)
                }
        assert {
            a.predicate.name for a in SweepDrawerSkillProvider.deployment_goal_atoms()
        } == SweepUpstreamFacts.goal(abstractor=env._upstream, state=session.state)
        observed = SweepUpstreamFacts.observed(abstractor=env._upstream, state=session.state)
        for name in observed:
            assert (
                env.get_current_state().get(
                    obj=__import__(
                        "hitl_pmp.environments.sweep_drawer3d.symbolic", fromlist=["SweepSymbols"]
                    ).SweepSymbols.SCENE,
                    feature_name=name,
                )
                == 1.0
            )
    finally:
        # The upstream factory owns a separate simulator; never reset practice.
        for cell in models.transition_fn.__closure__ or ():
            value = cell.cell_contents
            if hasattr(value, "close"):
                value.close()
        SweepUpstreamFacts.close(abstractor=models.state_abstractor.__self__)
        env.close()
