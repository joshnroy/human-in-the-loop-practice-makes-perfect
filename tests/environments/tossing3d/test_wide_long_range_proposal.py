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
    del no_kinder_import
    rng = np.random.default_rng(2026092200)
    repeat_rng = np.random.default_rng(2026092200)
    samples = np.stack([WideLongRangeTossProposal.sample(rng=rng) for _ in range(2000)])
    repeated = np.stack([WideLongRangeTossProposal.sample(rng=repeat_rng) for _ in range(2000)])
    assert np.array_equal(samples, repeated)
    assert np.all(np.isfinite(samples))
    assert np.all(samples[:, 0] == 2.5)
    assert np.all(samples[:, 1] == 0.0)
    assert np.all(samples[:, 2] >= WIDE_TOSS_SPEED_BOUNDS[0])
    assert np.all(samples[:, 2] <= WIDE_TOSS_SPEED_BOUNDS[1])
    assert np.all(samples[:, 3] >= WIDE_TOSS_RELEASE_MS_BOUNDS[0])
    assert np.all(samples[:, 3] <= WIDE_TOSS_RELEASE_MS_BOUNDS[1])
    controller_bounds = np.array([
        TOSS_DISTANCE_BOUNDS,
        TOSS_ROTATION_BOUNDS,
        TOSS_SPEED_BOUNDS,
        TOSS_RELEASE_MS_BOUNDS,
    ])
    assert np.all(samples >= controller_bounds[:, 0])
    assert np.all(samples <= controller_bounds[:, 1])


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
