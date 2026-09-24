"""The wide proposal must contain both known-success and known-miss throws.

The witness tuples below are `(standoff m, speed deg/s, release ms)` rows (the yaw
column they were measured with was 0 for every one, and is gone from the
parameters: the stand direction is now chosen per state) copied from the 2026-09-22
nine-location coverage sweep
(`artifacts/tossing3d-learning-protocol-fix-20260922/coverage/summary.json`, KINDER
`8f600231`, controller `427ad6c`): each success tuple scored at 9/9 tested bin
locations and each miss tuple completed a physical non-scoring throw at 9/9. They are
the evidence that the support spans both outcome classes everywhere in the bin region.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from hitl_pmp.cli import Cli
from hitl_pmp.core.method.types import GroundSkill
from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.layout import Tossing3DLayout
from hitl_pmp.environments.tossing3d.recovery_skills import SameSideSkills
from hitl_pmp.environments.tossing3d.sides import (
    BIN_RESET_REGION_BY_SIDE,
    Tossing3DSide,
    Tossing3DSides,
)
from hitl_pmp.environments.tossing3d.skill_provider import Tossing3DSkillProvider
from hitl_pmp.environments.tossing3d.skills import (
    TOSS_DISTANCE_BOUNDS,
    TOSS_RELEASE_MS_BOUNDS,
    TOSS_SPEED_BOUNDS,
    Tossing3DSkills,
)
from hitl_pmp.environments.tossing3d.toss import Tossing3DToss
from hitl_pmp.environments.tossing3d.wide_long_range_proposal import (
    FAR_STAND_X_LIMIT_M,
    WIDE_TOSS_RELEASE_MS_BOUNDS,
    WIDE_TOSS_SPEED_BOUNDS,
    WIDE_TOSS_STANDOFF_BOUNDS,
    WideLongRangeTossProposal,
)

SUCCESS_WITNESSES = (
    (2.5, 360.0, 480.0),
    (2.5, 390.0, 460.0),
    (2.5, 390.0, 480.0),
    (2.5, 420.0, 440.0),
    (2.5, 420.0, 460.0),
)
MISS_WITNESSES = (
    (2.5, 330.0, 520.0),
    (2.5, 360.0, 460.0),
    (2.5, 360.0, 500.0),
    (2.5, 390.0, 440.0),
)

# In-band witnesses from the 2026-09-23 controller-wide coverage probes (6
# standoffs x 3 robot-side receiver positions x 10x8-ish (speed, release)
# grids, 2,340 completed physical trials, `results/overnight-coverage-probe-
# 20260923`): one scoring and one completed-miss tuple per proposal-supported
# standoff rung, every one re-validated 1/1 at this test's own pinned
# arrangement. The rungs span the coinciding same-side/far-side feasible band
# [1.25, 2.6]: slow-late lobs at the short end (150, 720 at 1.25), the mid-band
# ridge, and the far edge, where the scoring cells ((390, 450), (420, 450))
# coincide with the far calibrated regime -- the side-invariance the
# 2026-09-23 barrier-arc and near-barrier probes measured at 5/5 vs 5/5 and
# 12/12.
ROBOT_SIDE_WITNESSES = (
    # (standoff m, speed deg/s, release ms, scores?)
    (1.25, 150.0, 720.0, True),
    (1.25, 115.0, 400.0, False),
    (1.5, 185.0, 650.0, True),
    (1.5, 115.0, 400.0, False),
    (1.75, 220.0, 600.0, True),
    (1.75, 115.0, 400.0, False),
    (2.0, 290.0, 500.0, True),
    (2.0, 115.0, 400.0, False),
    (2.5, 390.0, 450.0, True),
    (2.5, 115.0, 400.0, False),
)
# The probe's arrangement: reset_to_seed then a robot-side movables reset, both
# deterministic per seed; 125 is the canonical practice seed and its arrangement
# picks cleanly (bin near (-0.331, 0.965), cube on the spawn strip).
ROBOT_SIDE_WITNESS_SEED = 125

# Near corner, center, far corner of the swept bin region x in [2.60, 3.42],
# y in [-2.30, 2.30]; all three are among the sweep's nine tested locations.
SPANNING_BIN_LOCATIONS = ((2.6, -2.3), (3.01, 0.0), (3.42, 2.3))
# The sweep's single fresh-pickup seed; one seed fully determines a run.
SWEEP_SEED = 2026092400
# 9/9 locations scored at (390, 460) and 9/9 missed at (390, 440) in the sweep.
SIM_SUCCESS_WITNESS = (2.5, 390.0, 460.0)
SIM_MISS_WITNESS = (2.5, 390.0, 440.0)


def _toss_ground_skill(*, env: Tossing3DEnvironment) -> GroundSkill:
    return GroundSkill(
        skill=Tossing3DSkills.MOVE_TO_TOSS_LOCATION_AND_TOSS,
        objects=(env.robot, env.bin, env.cube, env.barrier, Tossing3DSides.opposite),
    )


def test_samples_stay_inside_declared_and_controller_bounds(*, no_kinder_import) -> None:
    """One draw for every toss, both sides: all three parameters jointly over their
    full ranges, every marginal genuinely filling its band."""
    del no_kinder_import
    rng = np.random.default_rng(2026092200)
    repeat_rng = np.random.default_rng(2026092200)
    samples = np.stack([WideLongRangeTossProposal.sample(rng=rng) for _ in range(2000)])
    repeated = np.stack([WideLongRangeTossProposal.sample(rng=repeat_rng) for _ in range(2000)])
    assert np.array_equal(samples, repeated)
    assert np.all(np.isfinite(samples))
    declared_bounds = np.array([
        WIDE_TOSS_STANDOFF_BOUNDS,
        WIDE_TOSS_SPEED_BOUNDS,
        WIDE_TOSS_RELEASE_MS_BOUNDS,
    ])
    controller_bounds = np.array([
        TOSS_DISTANCE_BOUNDS,
        TOSS_SPEED_BOUNDS,
        TOSS_RELEASE_MS_BOUNDS,
    ])
    # Speed and release stay the controller's own ranges; the standoff
    # floor is derived as max(controller floor, nearest far bin minus the
    # measured legal standing line). Under the graded receiver region the far
    # term is 1.48 - 0.99 = 0.49, so the CONTROLLER floor binds and the
    # same-side and far-side feasible ranges coincide at [1.25, 2.6] exactly.
    assert np.array_equal(declared_bounds[1:], controller_bounds[1:])
    assert WIDE_TOSS_STANDOFF_BOUNDS[0] == pytest.approx(
        max(
            TOSS_DISTANCE_BOUNDS[0],
            BIN_RESET_REGION_BY_SIDE[Tossing3DSide.OPPOSITE].ranges[0][0] - FAR_STAND_X_LIMIT_M,
        )
    )
    assert WIDE_TOSS_STANDOFF_BOUNDS[0] == TOSS_DISTANCE_BOUNDS[0] == 1.25
    assert WIDE_TOSS_STANDOFF_BOUNDS[1] == TOSS_DISTANCE_BOUNDS[1]
    assert np.all(samples >= declared_bounds[:, 0])
    assert np.all(samples <= declared_bounds[:, 1])
    for column in range(3):
        assert np.ptp(samples[:, column]) > 0.9 * np.ptp(declared_bounds[column])
    for params in samples[:16]:
        assert WideLongRangeTossProposal.contains(params=params)


def test_draws_pin_values_and_rng_consumption_order(*, no_kinder_import) -> None:
    """Exact-vector regression: standoff, speed, release, in that consumption
    order. The yaw draw is gone, so from the second value on these are not the
    #362-era vectors; the first standoff, drawn first from the same stream, is."""
    del no_kinder_import
    rng = np.random.default_rng(2026092313)
    expected = (
        (1.4359164368538408, 162.80405694898116, 698.57189157958),
        (2.516924272521253, 268.73560302527744, 797.2560402741044),
        (2.1768931683408512, 165.1400112014693, 504.5428850899565),
        (1.8740167042746583, 399.5490334965635, 824.4832454611482),
    )
    for row in expected:
        assert tuple(WideLongRangeTossProposal.sample(rng=rng).tolist()) == row


def test_most_draws_land_off_the_calibrated_ridge(*, no_kinder_import) -> None:
    del no_kinder_import
    rng = np.random.default_rng(2026092201)
    samples = np.stack([WideLongRangeTossProposal.sample(rng=rng) for _ in range(4096)])
    ridge = np.array([
        WideLongRangeTossProposal.release_ridge(speed=speed) for speed in samples[:, 1]
    ])
    offsets = np.abs(samples[:, 2] - ridge)
    near_ridge = int(np.count_nonzero(offsets <= 5.0))
    far_off_ridge = int(np.count_nonzero(offsets >= 15.0))
    assert near_ridge < 4096 // 5
    assert far_off_ridge > 4096 // 2


def test_every_sweep_witness_lies_inside_the_support(*, no_kinder_import) -> None:
    del no_kinder_import
    for params in (*SUCCESS_WITNESSES, *MISS_WITNESSES):
        assert WideLongRangeTossProposal.contains(params=np.array(params)), params


def test_both_bin_sides_draw_from_the_one_unified_proposal(*, no_kinder_import) -> None:
    """The toss proposal interface is the same for both sides: the BinAtSide
    binding no longer routes to a different draw, so a robot-side toss and an
    opposite-side toss consume the identical sample stream."""
    del no_kinder_import
    env = Tossing3DEnvironment()
    provider = Tossing3DSkillProvider(env=env)
    for side in (Tossing3DSides.robot, Tossing3DSides.opposite):
        toss = GroundSkill(
            skill=Tossing3DSkills.MOVE_TO_TOSS_LOCATION_AND_TOSS,
            objects=(env.robot, env.bin, env.cube, env.barrier, side),
        )
        rng = np.random.default_rng(62)
        direct_rng = np.random.default_rng(62)
        for _ in range(10):
            assert np.array_equal(
                provider.sample_params(ground_skill=toss, rng=rng),
                WideLongRangeTossProposal.sample(rng=direct_rng),
            ), side


@pytest.mark.skipif(
    importlib.util.find_spec("kinder") is None, reason="KINDER simulator dependency"
)
def test_every_witness_rung_has_a_plannable_direction_at_both_sides_receivers() -> None:
    """The direction selector, with the real base planner, finds a stand at every
    witness standoff rung both at the natural far bin the scene resets with and at
    representative robot-side receivers (the corners and centre of the grasp-safe
    region) -- a `NoFeasibleTossDirectionError` here would be a pool-breaking hole."""
    import json as jsonlib
    from pathlib import Path

    from hitl_pmp.environments.tossing3d.toss_direction import TossDirectionSelector

    stuck_path = Path(__file__).parent / "fixtures" / "trap2_stuck_state.json"
    plain = jsonlib.loads(stuck_path.read_text())
    env = Tossing3DEnvironment()
    rungs = sorted({standoff for standoff, *_ in ROBOT_SIDE_WITNESSES})
    try:
        initial = env.reset_to_seed(seed=125)
        bin_x = float(initial.get(obj=env.bin, feature_name="x"))
        for standoff in rungs:
            choice = TossDirectionSelector.select_for_state(state=initial, standoff=standoff)
            assert choice.stand_xy[0] < bin_x, standoff
        for bin_xy in ((-0.9, -1.0), (-0.35, 0.0), (0.2, 1.5), (-0.65, -1.0)):
            moved = jsonlib.loads(jsonlib.dumps(plain))
            moved["bin_0"][0], moved["bin_0"][1] = bin_xy
            state = env.restore_plain_snapshot(plain=moved)
            for standoff in rungs:
                TossDirectionSelector.select_for_state(state=state, standoff=standoff)
    finally:
        env.close()


@pytest.mark.skipif(
    importlib.util.find_spec("kinder") is None, reason="KINDER simulator dependency"
)
def test_a_centre_robot_side_bin_has_no_direction_at_the_top_standoff() -> None:
    """A measured hole, pinned so it cannot change silently: at the robot-side bin
    (-0.35, 0.0) a 2.6 m stand is across the barrier to the east, into the 45-degree
    corner colliders to the north and south, and outside the west wall (the real
    planner fails). So the draw raises -- loudly, by design -- rather than being
    filtered out of the pool."""
    import json as jsonlib
    from pathlib import Path

    from hitl_pmp.environments.tossing3d.toss_direction import (
        NoFeasibleTossDirectionError,
        TossDirectionSelector,
    )

    plain = jsonlib.loads(
        (Path(__file__).parent / "fixtures" / "trap2_stuck_state.json").read_text()
    )
    plain["bin_0"][0], plain["bin_0"][1] = (-0.35, 0.0)
    env = Tossing3DEnvironment()
    try:
        env.reset_to_seed(seed=125)
        state = env.restore_plain_snapshot(plain=plain)
        with pytest.raises(NoFeasibleTossDirectionError) as caught:
            TossDirectionSelector.select_for_state(state=state, standoff=2.6)
    finally:
        env.close()
    reasons = caught.value.reasons
    assert reasons[0].startswith("separating_obstacle:collider:cuboid_barrier")
    assert reasons[90].startswith("target_collision:collider:tossing_room")
    assert reasons[180] == "base_motion_plan_failed"
    assert reasons[270].startswith("target_collision:collider:tossing_room")


def test_support_holds_a_scoring_and_a_missing_combo_per_in_band_standoff(
    *, no_kinder_import
) -> None:
    """Under the graded receiver region the same-side and far-side feasible
    standoff bands coincide at [1.25, 2.6], so every rung of the shared band
    must hold scoring support. Every witness -- scoring and miss alike -- must
    lie inside the support, with both classes present at each tested rung; and
    every witness standoff must respect the (clamped) floor, so a sub-floor
    witness can never pin dead support."""
    del no_kinder_import
    by_standoff: dict[float, dict[bool, int]] = {}
    for standoff, speed, release, scores in ROBOT_SIDE_WITNESSES:
        assert standoff >= WIDE_TOSS_STANDOFF_BOUNDS[0], (standoff, speed, release)
        assert WideLongRangeTossProposal.contains(params=np.array([standoff, speed, release])), (
            standoff,
            speed,
            release,
        )
        by_standoff.setdefault(standoff, {True: 0, False: 0})[scores] += 1
    assert set(by_standoff) == {1.25, 1.5, 1.75, 2.0, 2.5}
    for standoff, counts in by_standoff.items():
        assert counts[True] >= 1, standoff
        assert counts[False] >= 1, standoff


@pytest.mark.skipif(
    importlib.util.find_spec("kinder") is None, reason="KINDER simulator dependency"
)
def test_robot_side_witnesses_score_and_complete_misses_at_a_robot_side_receiver() -> None:
    """Ten physical trials at the probe's own arrangement: per in-band standoff
    rung, the scoring witness scores 1/1 and the miss witness completes a
    non-scoring throw 1/1. The stand direction is chosen exactly as a learner's
    would be -- `Tossing3DToss.compute_action` on the live held state -- so the
    action pinned here is the action that ran."""
    from hitl_pmp.environments.tossing3d.predicates import HOLDING

    env = Tossing3DEnvironment()
    scored = 0
    completed_misses = 0
    try:
        robot_side_toss = GroundSkill(
            skill=Tossing3DSkills.MOVE_TO_TOSS_LOCATION_AND_TOSS,
            objects=(env.robot, env.bin, env.cube, env.barrier, Tossing3DSides.robot),
        )
        for standoff, speed, release, expect_score in ROBOT_SIDE_WITNESSES:
            env.reset_to_seed(seed=ROBOT_SIDE_WITNESS_SEED)
            assert env.reset_movables(destination="robot_side")
            picked = env.take_action(action=np.array([0, 0, 0, 0, 0], dtype=float))
            assert HOLDING.holds(picked, (env.robot, env.cube)), env.last_skill_error()
            action = Tossing3DToss.compute_action(
                params=np.array([standoff, speed, release]), state=picked
            )
            post = env.take_action(action=action)
            assert env.last_skill_error() is None, (standoff, speed, release)
            assert sum(env.last_controller_steps()) > 0
            landed_in_bin = all(
                atom.predicate.holds(post, atom.objects) for atom in robot_side_toss.add_effects
            )
            if expect_score:
                assert landed_in_bin, (standoff, speed, release, action[2])
                scored += 1
            else:
                assert not landed_in_bin, (standoff, speed, release, action[2])
                completed_misses += 1
    finally:
        env.close()
    assert scored == 5
    assert completed_misses == 5


def test_every_barrier_toss_draw_comes_from_the_wide_proposal(*, no_kinder_import) -> None:
    del no_kinder_import
    env = Tossing3DEnvironment()
    provider = Tossing3DSkillProvider(env=env)
    toss = _toss_ground_skill(env=env)
    rng = np.random.default_rng(60)
    direct_rng = np.random.default_rng(60)
    for _ in range(10):
        assert np.array_equal(
            provider.sample_params(ground_skill=toss, rng=rng),
            WideLongRangeTossProposal.sample(rng=direct_rng),
        )
    pick = GroundSkill(
        skill=Tossing3DSkills.PICK_CUBE,
        objects=(env.robot, env.cube, env.barrier, Tossing3DSides.robot, env.bin),
    )
    assert np.array_equal(
        provider.sample_params(ground_skill=pick, rng=rng),
        Tossing3DSkills.sample_params(ground_skill=pick, rng=direct_rng),
    )


def test_same_side_tosses_draw_the_same_unified_proposal(*, no_kinder_import) -> None:
    del no_kinder_import
    env = Tossing3DEnvironment(layout=Tossing3DLayout.SAME_SIDE)
    provider = Tossing3DSkillProvider(env=env)
    toss = _toss_ground_skill(env=env)
    rng = np.random.default_rng(61)
    direct_rng = np.random.default_rng(61)
    same_side_rng = np.random.default_rng(61)
    for _ in range(10):
        drawn = provider.sample_params(ground_skill=toss, rng=rng)
        assert np.array_equal(drawn, WideLongRangeTossProposal.sample(rng=direct_rng))
        assert np.array_equal(
            drawn, SameSideSkills.sample_params(ground_skill=toss, rng=same_side_rng)
        )


def test_the_removed_proposal_flag_is_rejected() -> None:
    with pytest.raises(SystemExit):
        Cli.parse_args(
            argv=["--env", "tossing3d", "--method", "ees", "--toss-proposal", "long-range"]
        )


@pytest.mark.skipif(
    importlib.util.find_spec("kinder") is None, reason="KINDER simulator dependency"
)
def test_witnesses_score_and_complete_misses_at_spanning_bin_locations(*, tmp_path: Path) -> None:
    """Six physical trials: one scoring and one completed-miss witness per location.

    Mirrors the sweep's own probe: a fresh environment and physical pickup per trial,
    the bin pinned by shrinking the stock task's `bin_init_region` to a point.
    """
    import kinder

    from hitl_pmp.environments.tossing3d.predicates import HOLDING

    for params in (SIM_SUCCESS_WITNESS, SIM_MISS_WITNESS):
        assert WideLongRangeTossProposal.contains(params=np.array(params)), params
    stock_path = Path(kinder.__file__).parent / "envs/dynamic3d/tasks/Tossing3D/Tossing3D-o1.json"
    stock = json.loads(stock_path.read_text())
    assert stock["regions"]["bin_init_region"]["ranges"] == [[1.48, -2.3, 3.42, 2.3]]
    scored = 0
    completed_misses = 0
    for index, (x, y) in enumerate(SPANNING_BIN_LOCATIONS):
        scene = json.loads(stock_path.read_text())
        scene["regions"]["bin_init_region"]["ranges"] = [[x, y, x, y]]
        scene_path = tmp_path / f"bin_{index:02d}.json"
        scene_path.write_text(json.dumps(scene))
        for params, expect_score in ((SIM_SUCCESS_WITNESS, True), (SIM_MISS_WITNESS, False)):
            env = Tossing3DEnvironment()
            try:
                env.backend().task_config_path = scene_path
                initial = env.reset_to_seed(seed=SWEEP_SEED)
                initial_xy = [initial.get(obj=env.bin, feature_name=c) for c in "xy"]
                assert np.allclose(initial_xy, (x, y), atol=1e-6, rtol=0)
                picked = env.take_action(action=np.array([0, 0, 0, 0, 0], dtype=float))
                assert HOLDING.holds(picked, (env.robot, env.cube)), env.last_skill_error()
                action = Tossing3DToss.compute_action(params=np.array(params), state=picked)
                # A far bin is always thrown at from the west: rotation 0, as measured.
                assert action[2] == 0.0
                env.take_action(action=action)
                assert env.last_skill_error() is None
                assert sum(env.last_controller_steps()) > 0
                if expect_score:
                    assert env.is_solved(), (x, y, params)
                    scored += 1
                else:
                    assert not env.is_solved(), (x, y, params)
                    completed_misses += 1
            finally:
                env.close()
    assert scored == 3
    assert completed_misses == 3
