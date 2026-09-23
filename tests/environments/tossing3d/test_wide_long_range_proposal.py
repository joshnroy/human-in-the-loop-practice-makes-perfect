"""The wide proposal must contain both known-success and known-miss throws.

The witness tuples below are `(standoff m, yaw rad, speed deg/s, release ms)` rows
copied from the 2026-09-22 nine-location coverage sweep
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
from hitl_pmp.environments.tossing3d.sides import Tossing3DSides
from hitl_pmp.environments.tossing3d.skill_provider import Tossing3DSkillProvider
from hitl_pmp.environments.tossing3d.skills import (
    TOSS_DISTANCE_BOUNDS,
    TOSS_RELEASE_MS_BOUNDS,
    TOSS_ROTATION_BOUNDS,
    TOSS_SPEED_BOUNDS,
    Tossing3DSkills,
)
from hitl_pmp.environments.tossing3d.wide_long_range_proposal import (
    WIDE_TOSS_RELEASE_MS_BOUNDS,
    WIDE_TOSS_SPEED_BOUNDS,
    WIDE_TOSS_STANDOFF_BOUNDS,
    WIDE_TOSS_YAW_BOUNDS,
    WideLongRangeTossProposal,
)

SUCCESS_WITNESSES = (
    (2.5, 0.0, 360.0, 480.0),
    (2.5, 0.0, 390.0, 460.0),
    (2.5, 0.0, 390.0, 480.0),
    (2.5, 0.0, 420.0, 440.0),
    (2.5, 0.0, 420.0, 460.0),
)
MISS_WITNESSES = (
    (2.5, 0.0, 330.0, 520.0),
    (2.5, 0.0, 360.0, 460.0),
    (2.5, 0.0, 360.0, 500.0),
    (2.5, 0.0, 390.0, 440.0),
)

# Short-standoff witnesses from the 2026-09-23 controller-wide coverage probe
# (10 speeds in [115, 420] x 8 releases in [400, 840] at standoffs
# {1.25, 1.5, 1.75} x 3 robot-side receiver positions spanning the grasp-safe
# rectangle, 720 completed physical trials, `results/overnight-coverage-probe-
# 20260923` on the 2026-09-22 stack): every scoring tuple below scored at all 3
# tested receiver positions and was re-validated 1/1 at this test's own pinned
# arrangement; every miss tuple completed a physical non-scoring throw there.
# The pre-extension robot-side speed/release band (330-420 deg/s x 430-520 ms)
# contained 0/240 scoring cells at every tested short standoff, which is what
# fed short-standoff practice an all-miss diet.
ROBOT_SIDE_WITNESSES = (
    # (standoff m, speed deg/s, release ms, scores?)
    (1.25, 150.0, 720.0, True),
    (1.25, 115.0, 400.0, False),
    (1.5, 185.0, 650.0, True),
    (1.5, 115.0, 400.0, False),
    (1.75, 220.0, 600.0, True),
    (1.75, 115.0, 400.0, False),
)
PRE_EXTENSION_SPEED_BOUNDS = (330.0, 420.0)
PRE_EXTENSION_RELEASE_MS_BOUNDS = (430.0, 520.0)
# The probe's arrangement: reset_to_seed then a robot-side movables reset, both
# deterministic per seed; 125 is the canonical practice seed and its arrangement
# picks cleanly (bin near (-0.331, 0.965), cube on the spawn strip).
ROBOT_SIDE_WITNESS_SEED = 125
ROBOT_SIDE_YAW_SCAN = 41

# Near corner, center, far corner of the swept bin region x in [2.60, 3.42],
# y in [-2.30, 2.30]; all three are among the sweep's nine tested locations.
SPANNING_BIN_LOCATIONS = ((2.6, -2.3), (3.01, 0.0), (3.42, 2.3))
# The sweep's single fresh-pickup seed; one seed fully determines a run.
SWEEP_SEED = 2026092400
# 9/9 locations scored at (390, 460) and 9/9 missed at (390, 440) in the sweep.
SIM_SUCCESS_WITNESS = (2.5, 0.0, 390.0, 460.0)
SIM_MISS_WITNESS = (2.5, 0.0, 390.0, 440.0)


def _toss_ground_skill(*, env: Tossing3DEnvironment) -> GroundSkill:
    return GroundSkill(
        skill=Tossing3DSkills.MOVE_TO_TOSS_LOCATION_AND_TOSS,
        objects=(env.robot, env.bin, env.cube, env.barrier, Tossing3DSides.opposite),
    )


def test_samples_stay_inside_declared_and_controller_bounds(*, no_kinder_import) -> None:
    """One draw for every barrier toss, both sides: all four parameters jointly
    over the controller's full ranges, every marginal genuinely filling its band."""
    del no_kinder_import
    rng = np.random.default_rng(2026092200)
    repeat_rng = np.random.default_rng(2026092200)
    samples = np.stack([WideLongRangeTossProposal.sample(rng=rng) for _ in range(2000)])
    repeated = np.stack([WideLongRangeTossProposal.sample(rng=repeat_rng) for _ in range(2000)])
    assert np.array_equal(samples, repeated)
    assert np.all(np.isfinite(samples))
    declared_bounds = np.array([
        WIDE_TOSS_STANDOFF_BOUNDS,
        WIDE_TOSS_YAW_BOUNDS,
        WIDE_TOSS_SPEED_BOUNDS,
        WIDE_TOSS_RELEASE_MS_BOUNDS,
    ])
    controller_bounds = np.array([
        TOSS_DISTANCE_BOUNDS,
        TOSS_ROTATION_BOUNDS,
        TOSS_SPEED_BOUNDS,
        TOSS_RELEASE_MS_BOUNDS,
    ])
    assert np.array_equal(declared_bounds, controller_bounds)
    assert np.all(samples >= declared_bounds[:, 0])
    assert np.all(samples <= declared_bounds[:, 1])
    for column in range(4):
        assert np.ptp(samples[:, column]) > 0.9 * np.ptp(declared_bounds[column])
    for params in samples[:16]:
        assert WideLongRangeTossProposal.contains(params=params)


def test_draws_are_byte_identical_to_the_pre_unification_robot_side_variant(
    *, no_kinder_import
) -> None:
    """The unified draw is exactly what `sample_robot_side` produced before the
    far point mass was retired: these vectors were generated at the #361 code
    from that variant, pinning values and rng consumption order across the
    unification."""
    del no_kinder_import
    rng = np.random.default_rng(2026092313)
    expected = (
        (1.4359164368538408, -1.0784000181958564, 321.96460666311793, 812.9234665995194),
        (1.9304690625709, 1.265605168021688, 324.4091972918219, 472.333130913595),
        (1.5707565792532756, -0.11864351652335503, 399.5490334965635, 824.4832454611482),
        (1.402842106829608, 1.1004221233348783, 406.0396811191806, 756.691302643899),
    )
    for row in expected:
        assert tuple(WideLongRangeTossProposal.sample(rng=rng).tolist()) == row


def test_most_draws_land_off_the_calibrated_ridge(*, no_kinder_import) -> None:
    del no_kinder_import
    rng = np.random.default_rng(2026092201)
    samples = np.stack([WideLongRangeTossProposal.sample(rng=rng) for _ in range(4096)])
    ridge = np.array([
        WideLongRangeTossProposal.release_ridge(speed=speed) for speed in samples[:, 2]
    ])
    offsets = np.abs(samples[:, 3] - ridge)
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
def test_unified_proposals_pass_the_geometry_gate_at_both_sides_receivers() -> None:
    """The one draw serves every receiver: the accepted pool must be non-empty
    both at representative robot-side receiver positions (where the retired far
    point mass yielded 0 of 10,000 -- the 2026-09-22 validation starvation) and
    at the natural far bin the scene resets with."""
    import json as jsonlib
    from pathlib import Path

    from hitl_pmp.environments.tossing3d.kinder_backend import KinderBackend
    from hitl_pmp.environments.tossing3d.parameter_feasibility import TossParameterFeasibility

    stuck_path = Path(__file__).parent / "fixtures" / "trap2_stuck_state.json"
    plain = jsonlib.loads(stuck_path.read_text())
    env = Tossing3DEnvironment()
    draws = 512

    def accepted_of(*, geometry: object) -> int:
        rng = np.random.default_rng(2026092211)
        return sum(
            TossParameterFeasibility.rejection_reason_from_geometry(
                geometry=geometry,
                params=WideLongRangeTossProposal.sample(rng=rng),
            )
            is None
            for _ in range(draws)
        )

    try:
        initial = env.reset_to_seed(seed=125)
        far_geometry = KinderBackend.toss_feasibility_geometry(snapshot=initial.object_centric)
        assert far_geometry is not None
        far_accepted = accepted_of(geometry=far_geometry)
        assert far_accepted > 0, f"far bin: 0/{draws} accepted"
        for bin_xy in ((-0.9, -1.0), (-0.35, 0.0), (0.2, 1.5), (-0.65, -1.0)):
            moved = jsonlib.loads(jsonlib.dumps(plain))
            moved["bin_0"][0], moved["bin_0"][1] = bin_xy
            state = env.restore_plain_snapshot(plain=moved)
            geometry = KinderBackend.toss_feasibility_geometry(snapshot=state.object_centric)
            assert geometry is not None
            accepted = accepted_of(geometry=geometry)
            assert accepted > 0, (bin_xy, f"0/{draws} accepted")
    finally:
        env.close()


def test_robot_side_support_holds_a_scoring_and_a_missing_combo_per_short_standoff(
    *, no_kinder_import
) -> None:
    """The 2026-09-22 verification run starved short-standoff practice of scoring
    labels: 0/31 robot-side draws below 2.0 m scored, because the band the draws
    came from had no scoring support there. Every probe witness -- scoring and
    miss alike -- must lie inside the robot-side support, and every scoring one
    lies outside the pre-extension speed/release band, which is what the widening
    newly covers."""
    del no_kinder_import
    by_standoff: dict[float, dict[bool, int]] = {}
    for standoff, speed, release, scores in ROBOT_SIDE_WITNESSES:
        assert WideLongRangeTossProposal.contains(
            params=np.array([standoff, 0.0, speed, release])
        ), (standoff, speed, release)
        if scores:
            in_pre_extension_band = (
                PRE_EXTENSION_SPEED_BOUNDS[0] <= speed <= PRE_EXTENSION_SPEED_BOUNDS[1]
                and PRE_EXTENSION_RELEASE_MS_BOUNDS[0]
                <= release
                <= PRE_EXTENSION_RELEASE_MS_BOUNDS[1]
            )
            assert not in_pre_extension_band, (standoff, speed, release)
        by_standoff.setdefault(standoff, {True: 0, False: 0})[scores] += 1
    assert set(by_standoff) == {1.25, 1.5, 1.75}
    for standoff, counts in by_standoff.items():
        assert counts[True] >= 1, standoff
        assert counts[False] >= 1, standoff


@pytest.mark.skipif(
    importlib.util.find_spec("kinder") is None, reason="KINDER simulator dependency"
)
def test_robot_side_witnesses_score_and_complete_misses_at_a_robot_side_receiver() -> None:
    """Six physical trials at the probe's own arrangement: per short standoff, the
    scoring witness scores 1/1 and the miss witness completes a non-scoring throw
    1/1. Yaw is chosen exactly as the probe chose it -- the median gate-accepted
    yaw from a fixed scan -- so the tuple pinned here is the tuple that ran."""
    from hitl_pmp.environments.tossing3d.kinder_backend import KinderBackend
    from hitl_pmp.environments.tossing3d.parameter_feasibility import TossParameterFeasibility
    from hitl_pmp.environments.tossing3d.predicates import HOLDING

    env = Tossing3DEnvironment()
    scored = 0
    completed_misses = 0
    try:
        robot_side_toss = GroundSkill(
            skill=Tossing3DSkills.MOVE_TO_TOSS_LOCATION_AND_TOSS,
            objects=(env.robot, env.bin, env.cube, env.barrier, Tossing3DSides.robot),
        )
        yaw_grid = np.linspace(
            TOSS_ROTATION_BOUNDS[0], TOSS_ROTATION_BOUNDS[1], ROBOT_SIDE_YAW_SCAN
        )
        for standoff, speed, release, expect_score in ROBOT_SIDE_WITNESSES:
            env.reset_to_seed(seed=ROBOT_SIDE_WITNESS_SEED)
            assert env.reset_movables(destination="robot_side")
            state = env.get_current_state()
            geometry = KinderBackend.toss_feasibility_geometry(snapshot=state.object_centric)
            assert geometry is not None
            accepted = [
                float(yaw)
                for yaw in yaw_grid
                if TossParameterFeasibility.rejection_reason_from_geometry(
                    geometry=geometry,
                    params=np.array([standoff, yaw, speed, release]),
                )
                is None
            ]
            assert accepted, (standoff, speed, release)
            yaw = accepted[len(accepted) // 2]
            picked = env.take_action(action=np.array([0, 0, 0, 0, 0], dtype=float))
            assert HOLDING.holds(picked, (env.robot, env.cube)), env.last_skill_error()
            post = env.take_action(action=np.array([1, standoff, yaw, speed, release], dtype=float))
            assert env.last_skill_error() is None, (standoff, speed, release)
            assert sum(env.last_controller_steps()) > 0
            landed_in_bin = all(
                atom.predicate.holds(post, atom.objects) for atom in robot_side_toss.add_effects
            )
            if expect_score:
                assert landed_in_bin, (standoff, speed, release)
                scored += 1
            else:
                assert not landed_in_bin, (standoff, speed, release)
                completed_misses += 1
    finally:
        env.close()
    assert scored == 3
    assert completed_misses == 3


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


def test_same_side_tosses_keep_their_own_sampler(*, no_kinder_import) -> None:
    del no_kinder_import
    env = Tossing3DEnvironment(layout=Tossing3DLayout.SAME_SIDE)
    provider = Tossing3DSkillProvider(env=env)
    toss = _toss_ground_skill(env=env)
    rng = np.random.default_rng(61)
    direct_rng = np.random.default_rng(61)
    for _ in range(10):
        assert np.array_equal(
            provider.sample_params(ground_skill=toss, rng=rng),
            SameSideSkills.sample_params(ground_skill=toss, rng=direct_rng),
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
    assert stock["regions"]["bin_init_region"]["ranges"] == [[2.6, -2.3, 3.42, 2.3]]
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
                env.take_action(action=np.array([1, *params], dtype=float))
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
