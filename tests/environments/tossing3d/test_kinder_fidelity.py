"""The tests that genuinely drive KINDER.

This domain integrates KINDER rather than reimplementing it, so these tests pin the
*integration boundary* -- that what this repo's adapter reads, dispatches and asserts
agrees with what upstream actually produces. They deliberately do NOT re-test anything
KINDER already guarantees: there is no reimplementation of its physics or its goal check
to check against. `test_in_goal_region_agrees_with_kinders_own_goal_check` is the shape
of the whole file -- run the real simulator, ask upstream's `_check_goals()`, and require
our predicate to return the same verdict.

Two of them are a different kind, and are labelled as such below:
`test_goal_region_bounds_match_kinders_own_region` is a version tripwire pinning the
goal box element-wise against upstream's own computed `Region.bbox`, and
`test_a_full_power_toss_overshoots_the_goal_region`
characterises upstream's physics. Both are only meaningful because `pyproject.toml` pins
KINDER to an exact commit -- they are what makes an upstream bump that moves the goal
region or the swing dynamics fail loudly instead of silently restating every number
measured here.

Everything else in this directory checks the adapter's arithmetic against itself, which
needs no simulator. These are skipped where `kindergarden` is not installed (CI
included) rather than deleted, because the alternative is asserting nothing about the
seam anywhere. Run them with the optional dependency installed; see
`src/hitl_pmp/environments/tossing3d/README.md`.
"""

import resource

import numpy as np
import pytest

from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.predicates import IN_GOAL_REGION, REACHABLE
from hitl_pmp.environments.tossing3d.skill_oracle_policy import SkillOraclePolicy

from .conftest import GOAL_REGION, build_state, kinder_available

_ENV = Tossing3DEnvironment

pytestmark = kinder_available

_SHARED: list[Tossing3DEnvironment] = []


def shared_env() -> Tossing3DEnvironment:
    """One simulator for the whole module -- opening it compiles a MuJoCo model and
    connects a PyBullet client, so it is built once and reused. A module-level cache
    rather than a pytest fixture: this project's lint bans positional parameters
    outright (ruff PLR0917, max-positional-args = 0), and a fixture argument is one."""
    if not _SHARED:
        _SHARED.append(Tossing3DEnvironment())
    return _SHARED[0]


def _act(*, env: Tossing3DEnvironment, skill: int, param0: float = 0.0, param1: float = 0.0):
    return env.take_action(action=np.array([float(skill), float(param0), float(param1)]))


def _oracle_solve(*, env: Tossing3DEnvironment, seed: int, swing: float):
    state = env.reset_to_seed(seed=seed)
    for skill, param in (
        (_ENV.SKILL_PICK, SkillOraclePolicy.ORACLE_PICK_DISTANCE),
        (_ENV.SKILL_MOVE_TO_THROW_POSE, 0.0),
        (_ENV.SKILL_TOSS, swing),
    ):
        state = _act(env=env, skill=skill, param0=param)
    return state


def test_goal_region_bounds_match_kinders_own_region() -> None:
    """The element-wise pin, and the *only* check here that can catch a wrong box.

    `goal_region_bounds()` must be upstream's own `Region.bbox` -- the exact list
    `Region.check_in_region` compares a cube position against -- not a re-derivation of
    it and not the task JSON's raw `ranges[0]`. Reading it back from the same attribute
    means the two cannot drift; asserting it here means an upstream bump that moves the
    region fires a test rather than silently moving every number measured against it.

    This test exists because the differential random walk below cannot do this job: the
    JSON-vs-bbox discrepancy is confined to two 5 cm shells at the region boundary, and
    a random walk of whole skills essentially never lands the cube in one. The bug this
    replaces asserted the *JSON* literal was correct, which pinned a real error in
    place.
    """
    backend = shared_env().backend()
    upstream = backend._ensure_env().unwrapped._object_centric_env._ground_fixture
    expected = upstream.region_objects["blocks_goal_region"][0].bbox
    assert backend.goal_region_bounds() == pytest.approx(tuple(float(v) for v in expected))
    # And, for readability, the value that pin currently resolves to: the task JSON range
    # inflated by `ground_placement_threshold` = 0.05 on every side, z clamped at 0.
    assert backend.goal_region_bounds() == pytest.approx(GOAL_REGION)


def test_in_goal_region_agrees_with_kinders_own_goal_check() -> None:
    """The fidelity property this domain's headline number rests on: this domain's
    `InGoalRegion` predicate must agree with KINDER's `_check_goals`, not merely
    resemble it.

    Two complementary sources of states, because neither alone is sufficient:

    * A random walk of whole skills in the real simulator, which lands the cube anywhere
      from its start pose to inside the bin. This exercises real physics but samples the
      boundary essentially never.
    * Cube positions placed **deliberately inside the two 5 cm inflation shells** -- just
      inside and just outside each face of the box -- compared against
      `MujocoGround.check_in_region`, which is precisely the call `_check_goals`
      delegates the containment decision to. These are the positions where a box built
      by the wrong rule disagrees, and the random walk above cannot reach them.

    The second half deliberately does not go through `_check_goals`: that reads the
    cube's pose out of live MuJoCo, so exercising it at a chosen position would mean
    teleporting the cube, and the containment logic under test is downstream of the pose
    either way. Comparing against `check_in_region` tests the same arithmetic against
    the same box with nothing mocked on either side.
    """
    env = shared_env()
    backend = env.backend()
    rng = np.random.default_rng(0)
    compared = 0
    for seed in (0, 1, 2):
        env.reset_to_seed(seed=seed)
        for _ in range(4):
            skill = int(
                rng.choice([_ENV.SKILL_PICK, _ENV.SKILL_MOVE_TO_THROW_POSE, _ENV.SKILL_TOSS])
            )
            state = _act(env=env, skill=skill, param0=float(rng.uniform(0.25, 1.25)))
            ours = IN_GOAL_REGION.holds(state, (_ENV.cube, _ENV.goal_region))
            assert ours == backend.check_goals(), (
                f"seed {seed}: InGoalRegion said {ours}, KINDER's _check_goals disagreed"
            )
            compared += 1
    assert compared == 12

    # The boundary shells the random walk cannot reach. Each pair straddles one face of
    # the true box by 2 cm, so every one of them lies strictly inside a 5 cm inflation
    # shell -- i.e. every "inside" case here would be scored a failure by the old,
    # un-inflated box.
    x_min, y_min, _z_min, x_max, y_max, z_max = GOAL_REGION
    boundary_positions = [
        (x_min + 0.02, 0.0, 0.05),  # inside the -x shell, outside the JSON range
        (x_min - 0.02, 0.0, 0.05),  # outside the true box entirely
        (x_max - 0.02, 0.0, 0.05),  # inside the +x shell, outside the JSON range
        (x_max + 0.02, 0.0, 0.05),  # outside the true box entirely
        (2.0, y_min + 0.02, 0.05),  # inside the -y shell
        (2.0, y_min - 0.02, 0.05),  # outside
        (2.0, y_max - 0.02, 0.05),  # inside the +y shell
        (2.0, y_max + 0.02, 0.05),  # outside
        (2.0, 0.0, z_max - 0.02),  # inside the +z shell
        (2.0, 0.0, z_max + 0.02),  # outside
    ]
    unwrapped = backend._ensure_env().unwrapped._object_centric_env
    ground = unwrapped._ground_fixture
    for position in boundary_positions:
        state = build_state(env=env, cube=position)
        ours = IN_GOAL_REGION.holds(state, (_ENV.cube, _ENV.goal_region))
        theirs = ground.check_in_region(
            np.array(position, dtype=np.float32), "blocks_goal_region", unwrapped._robot_env
        )
        assert ours == bool(theirs), (
            f"cube at {position}: InGoalRegion said {ours}, check_in_region said {theirs}"
        )
        compared += 1
    assert compared == 22

    # And the shells are genuinely discriminating: every "inside" case above is a
    # position the pre-fix, un-inflated box would have scored as a miss. If this ever
    # stops holding, the boundary list has drifted off the shells it is meant to probe.
    json_range = unwrapped.task_config["regions"]["blocks_goal_region"]["ranges"][0]
    json_low, json_high = np.array(json_range[:3]), np.array(json_range[3:])
    inside_shell = [
        position
        for position in boundary_positions
        if ground.check_in_region(
            np.array(position, dtype=np.float32), "blocks_goal_region", unwrapped._robot_env
        )
        and not (np.all(np.array(position) >= json_low) and np.all(np.array(position) <= json_high))
    ]
    assert len(inside_shell) == 5, inside_shell


def test_reset_to_the_same_seed_reproduces_the_same_initial_state() -> None:
    """`set_state` reinstalls a State by re-running its KINDER seed, which is only sound
    if a reset is deterministic. Checked after driving the simulator far away in
    between, which is the case the harness actually creates."""
    env = shared_env()
    first = env.reset_to_seed(seed=11)
    cube_before = np.array(first[_ENV.cube])
    robot_before = np.array(first[_ENV.robot])
    _oracle_solve(env=env, seed=11, swing=1.0)
    again = env.reset_to_seed(seed=11)
    assert np.array_equal(np.array(again[_ENV.cube]), cube_before)
    assert np.array_equal(np.array(again[_ENV.robot]), robot_before)


def test_the_oracle_swing_actually_reaches_the_goal_region() -> None:
    """The oracle's swing constant is the achievable ceiling every learning number here
    is read against, so it has to be a measured fact rather than a comment."""
    env = shared_env()
    solved = 0
    for seed in (0, 1, 2):
        state = _oracle_solve(env=env, seed=seed, swing=SkillOraclePolicy.ORACLE_SWING)
        solved += int(IN_GOAL_REGION.holds(state, (_ENV.cube, _ENV.goal_region)))
    assert solved >= 2, f"oracle solved only {solved}/3 -- ORACLE_SWING may need remeasuring"


def test_a_full_power_toss_overshoots_the_goal_region() -> None:
    """A characterisation of upstream's physics rather than of this adapter, and sound
    only because `pyproject.toml` pins KINDER exactly: swing=1.0 is KINDER's own demo
    toss and lands the cube deep in the bin at x ~ 2.22, past the region's 2.15 edge.
    This is what makes the swing dial worth learning: the obvious value is the wrong
    one, and a KINDER bump that changed it would invalidate the swing table this
    domain's README reports.

    Note the margin here is 7 cm, not the 12 cm it was when this domain mistakenly
    scored against the un-inflated JSON range. The region's far edge (2.15) is *inside*
    the bin's footprint (from 2.08), so what this test characterises is that a cube in
    the bin comes to rest past that edge rather than that the bin is out of bounds. In
    practice the two amount to the same thing --
    `test_no_swing_rests_the_cube_in_the_goal_regions_overlap_with_the_bin` measures that
    no swing rests a cube in the strip where they differ -- but they are different claims
    and only the first is what this test checks.

    Asserted on seeds 0 and 2, not seed 1. On seed 1 the grasp is marginal and the cube
    slips out of the gripper during `move_to_target`, landing at x ~ 1.58 without ever
    being tossed -- so seed 1 measures the Pick, not the swing. That is the same seed
    `test_the_oracle_swing_actually_reaches_the_goal_region` tolerates by asserting 2 of
    3 rather than 3 of 3.
    """
    for seed in (0, 2):
        state = _oracle_solve(env=shared_env(), seed=seed, swing=1.0)
        assert state.get(obj=_ENV.cube, feature_name="x") > GOAL_REGION[3], (
            f"seed {seed}: full-power toss did not clear the region's far edge"
        )
        assert not IN_GOAL_REGION.holds(state, (_ENV.cube, _ENV.goal_region))


def test_no_swing_rests_the_cube_in_the_goal_regions_overlap_with_the_bin() -> None:
    """The overlap on x in [2.08, 2.15] exists on paper and is empty in practice, and
    that is worth pinning because the arithmetic alone says the opposite.

    Scope, stated precisely: **no swing, at this domain's fixed `throw_standoff` (1.35 m)
    and `ORACLE_PICK_DISTANCE`,** rests the cube in the strip. Those are ClassVars
    exactly so that `swing` is this domain's only throw-shaping dial, so that is the
    space a learned sampler searches -- but it is not a claim that the strip is
    unreachable from every base pose.

    Two things squeeze it. The bin's near wall occupies x in [2.080, 2.100] (0.30 m bin
    centred at 2.2305, 0.02 m walls), so the part of the overlap a cube can physically
    *rest in* is only x in [2.126, 2.150] once its own 0.025 m half-extent is accounted
    for. And the swing dial does not move the landing point continuously: `TossController`
    both releases the gripper on the first control tick past a fixed 0.46 fraction of a
    trapezoidal profile *and* plans the swing through PyBullet motion planning, so a
    longer swing changes the release tick and the planned path in steps rather than
    smoothly. Which of the two dominates is not measured here; that it steps is.
    Measured on seed 0, the landing x is ~2.016 for every swing up to 0.959 and ~2.216
    from 0.960 -- a 20 cm step over a 0.001 change, straddling the whole strip -- and it
    is not even monotone: 0.962 lands short again.

    So the honest statement about this domain is that a cube inside the bin is a *scored
    failure*: it comes to rest at x ~ 2.22, and no swing puts it anywhere between. The
    demo clip is captioned accordingly, and `ORACLE_SWING` is not retuned toward the
    overlap, because there is nothing there to tune toward. A KINDER bump that changed
    the release fraction, the profile or the bin's placement could open the strip up --
    which is exactly when this test should fail and be re-measured.
    """
    # Two seeds, because the claim this backs is about the domain rather than about one
    # episode -- and because the step's location moves with the seed (0.960 on seed 0,
    # 0.959 on the demo's), so a single seed would not show that the strip stays empty
    # wherever the step happens to fall. Seeds 1 and 3 are excluded for the reason
    # `test_the_oracle_swing_actually_reaches_the_goal_region` tolerates 2 of 3: their
    # grasp fails, so they measure the Pick rather than the swing.
    bin_near_wall_x = 2.08
    for seed in (0, 2):
        landings = [
            _oracle_solve(env=shared_env(), seed=seed, swing=swing).get(
                obj=_ENV.cube, feature_name="x"
            )
            for swing in (0.95, 0.955, 0.958, 0.959, 0.96, 0.97, 1.0)
        ]
        # Not vacuous: the sampled swings really do straddle the step, so "nothing landed
        # in the strip" is a statement about a crossing rather than about one side of it.
        assert min(landings) < bin_near_wall_x < max(landings), (
            f"seed {seed}: swings {landings} never crossed the bin's near wall -- "
            "the bracket has moved"
        )
        in_strip = [x for x in landings if bin_near_wall_x <= x <= GOAL_REGION[3]]
        assert not in_strip, (
            f"seed {seed}: a swing rested the cube in the goal region's overlap with the "
            f"bin at x={in_strip}; the README's claim that the strip is unreachable needs "
            "re-measuring"
        )


def test_the_feature_sink_stays_in_lockstep_with_the_frame_sink() -> None:
    """`scripts/render_tossing3d_demo.py` captions frame `i` with features `i`, so the
    two sinks agreeing index-for-index is load-bearing for whether the caption on a
    published clip is true. Both are appended in the same place in `_step`; this pins
    that they still are, and that the last entry is the state the episode ended in."""
    env = shared_env()
    env.reset_to_seed(seed=0)
    backend = env.backend()
    frames: list[np.ndarray] = []
    features: list[dict[str, tuple[float, ...]]] = []
    backend.capture_frames_into(sink=frames)
    backend.capture_features_into(sink=features)
    try:
        _act(env=env, skill=_ENV.SKILL_PICK, param0=SkillOraclePolicy.ORACLE_PICK_DISTANCE)
    finally:
        backend.capture_frames_into(sink=None)
        backend.capture_features_into(sink=None)
    assert len(frames) > 1
    assert len(frames) == len(features)
    assert features[-1] == backend.read_features()


def test_a_tossed_cube_is_unreachable() -> None:
    """The domain's whole reason for existing: after a toss the cube is past the
    barrier, so no further Pick can ever apply."""
    env = shared_env()
    assert REACHABLE.holds(env.reset_to_seed(seed=2), (_ENV.cube, _ENV.barrier))
    state = _oracle_solve(env=env, seed=2, swing=SkillOraclePolicy.ORACLE_SWING)
    assert not REACHABLE.holds(state, (_ENV.cube, _ENV.barrier))


def test_a_kinder_planning_failure_is_a_no_op_not_a_crash() -> None:
    """Inverse kinematics genuinely has no solution for some base poses. That is a
    failure of the Pick skill, and `take_action` has to stay total over the Box."""
    env = shared_env()
    env.reset_to_seed(seed=0)
    for rot in np.linspace(_ENV().pick_rot_low, _ENV().pick_rot_high, 5):
        assert _act(env=env, skill=_ENV.SKILL_PICK, param0=0.5, param1=float(rot)) is not None


def test_every_swing_the_prior_can_draw_stays_executable() -> None:
    """Guards the seam between the prior and the simulator: a swing the sampler can draw
    but the controllers reject would show up as an unexplained plateau, not an error."""
    env = shared_env()
    for swing in (env.swing_low, 0.5, env.swing_high):
        assert _oracle_solve(env=env, seed=0, swing=swing) is not None


def test_skill_executions_do_not_leak_memory() -> None:
    """`KinderBackend._release` has to reclaim the PyBullet client each grounded
    controller opens, or a sweep OOMs the machine rather than finishing.

    This is a regression test for a real incident, not a hypothetical: grounding a
    controller per skill execution leaked one live `p.connect(p.DIRECT)` plus the Kinova
    URDF and meshes -- ~150 MB per Pick, ~315 MB per Toss -- which took a 40-step run to
    18.7 GB and put a 60 GB box on an OOM trajectory.

    Peak RSS via `resource`, so no dependency beyond the stdlib, and the threshold sits
    with enormous margin on both sides: leaking costs >6 GB over these 40 executions,
    while reclaiming holds growth to roughly nothing.
    """
    env = shared_env()
    env.reset_to_seed(seed=0)
    # Warm up first: the one-off cost of the first grounding of each controller is not
    # the leak, and folding it into the measurement would only add noise.
    for skill in (_ENV.SKILL_PICK, _ENV.SKILL_MOVE_TO_THROW_POSE, _ENV.SKILL_TOSS):
        _act(env=env, skill=skill, param0=0.5)
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    for index in range(40):
        env.reset_to_seed(seed=index % 3)
        _act(env=env, skill=_ENV.SKILL_PICK, param0=0.55)
        _act(env=env, skill=_ENV.SKILL_TOSS, param0=0.75)
    # ru_maxrss is in kilobytes on Linux.
    grew_mb = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - before) / 1024
    assert grew_mb < 500, f"peak RSS grew {grew_mb:.0f} MB over 40 skill executions"
