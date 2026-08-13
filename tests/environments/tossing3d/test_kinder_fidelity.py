"""Simulator-backed tests: does this integration actually agree with KINDER?

**Every test in this file skips cleanly without KINDER**, gated on
`importlib.util.find_spec("kinder")` -- the *import* package name, not the distribution
name `kindergarden`. CI never installs the optional extra, so on CI this whole file
skips and the offline files carry the suite. Locally, KINDER installs into `hitl-pmp`
itself (the `tossing3d` extra), so this runs in the ordinary gate:

    scripts/with_env.sh python -m pytest tests/environments/tossing3d/ -q

and under a memory cap, because these tests execute real controllers:

    systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0 -p OOMPolicy=continue -- ...

What these check that the offline tests structurally cannot:

1. `predicates.IN_BIN` agrees with KINDER's own `_check_goals()`. That is the
   whole basis for this domain trusting its own symbolic layer rather than the simulator.
2. The goal box in the `State` is the live `Region.bbox`, element for element -- the only
   check that can catch a wrong box, which has shipped once already.
3. The oracle's pick parameters really are what upstream's own sampler draws.
4. The oracle reproduces the rest positions and step counts recorded in
   `docs/kinder-environment-validation.md`.
5. The `Tossing3D-o1.json` the installed KINDER ships still puts the bin on the box that
   scores. That one is new. This domain no longer commits a scene of its own, so the
   scene moves with the `reference/kindergarden` pin -- and that check is what makes a
   pin bump a **loud** event rather than a silent change of geometry under every number
   already measured.

**Several tests here used to run twice, once per `Tossing3DTaskConfig` member.** That
enum is gone -- upstream's bin fix (`kindergarden` PR #126) landed on `Tossing3D-o1.json`
itself rather than as a new variant, so `STOCK` and `COINCIDENT` came to load the same
scene. See `Tossing3DEnvironment.backend` for the full history.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from hitl_pmp.core.method.types import LabeledAction
from hitl_pmp.environments.tossing3d import predicates
from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.predicates import (
    IN_BIN,
    THROW_OVERSHOOT_MARGIN,
    THROW_RANGE,
    THROW_RANGE_MAX,
    THROW_RANGE_MIN,
    THROW_SHORTFALL_MARGIN,
    THROW_STANDOFF_BOUNDS,
    TOSS_RELEASE_MS_BOUNDS,
    TOSS_SPEED_BOUNDS,
    InBinClassifier,
    RobotAtSuccessfulThrowPoseClassifier,
)
from hitl_pmp.environments.tossing3d.problem import Tossing3DProblem
from hitl_pmp.environments.tossing3d.skill_oracle_policy import (
    ORACLE_PICK_DISTANCE,
    ORACLE_PICK_ROTATION,
    SkillOraclePolicy,
)
from hitl_pmp.environments.tossing3d.tasks import Tossing3DTasks

# The DISTRIBUTION is `kindergarden`; the IMPORT package is `kinder`. Keying on the
# distribution name here would skip everything, always.
needs_kinder = pytest.mark.skipif(
    importlib.util.find_spec("kinder") is None or importlib.util.find_spec("kinder_models") is None,
    reason="KINDER is an optional extra (`kindergarden` + `kinder_models`); CI never installs it",
)

pytestmark = needs_kinder

# Upstream's own `test_pick_ground_toss` seed, and the seed every number in
# `docs/kinder-environment-validation.md` was measured at.
CANONICAL_SEED = 125

# The measured landing at standoff 1.35, from that same record, reproduced by
# `scripts/tossing3d_oracle_demo.py` on this machine.
#
# > **Superseded 2026-08-13 by the 1 kHz gripper release; left exactly as published.**
# > 1.9902 remains correct for the throw it measured -- the gripper opening on the first
# > *control step* past path fraction 0.46, at pins `kinder-baselines` `1b564a1` +
# > `kindergarden` `98ad2c0`, where it re-measured at 1.9901. Scheduling the release on an
# > absolute millisecond moves the landing +41.6 mm with the fraction held at 0.46 and no
# > parameter changed. A future pin bump that moves it again should add a third line here
# > rather than edit either.
REST_X_PRE_1KHZ_RELEASE = 1.9902

# What the oracle lands at under the scheduled 1 kHz release, at
# `ORACLE_RELEASE_SPEED_DEG_S` and `ORACLE_GRIPPER_RELEASE_MS`.
REST_X = 2.0318
BIN_FLOOR_Z = 0.0444

# `(release_speed, gripper_release_ms)` probes for "does *any* throw score from here".
# The oracle's own pair leads because it is verified to score at standoff 1.15 on
# `CANONICAL_SEED`; the others are the measured reach argmax (763 ms) and a slow throw.
_UNION_PROBE_PARAMS = ((140.0, 720.0), (140.0, 763.0), (60.0, 850.0))


def _env():
    return Tossing3DEnvironment()


def test_in_bin_agrees_with_kinders_own_goal_check_on_the_oracles_trajectory() -> None:
    """The differential test this domain's trust rests on, checked at every step of a real
    oracle episode.

    Both verdicts flip during the episode -- `False` while the cube is on the floor and in
    the gripper, `True` once it lands -- so agreeing at every step is a real constraint and
    not something a predicate hardwired to `True` would pass. The two standoffs come from
    `test_the_derived_band_agrees_with_whether_the_throw_actually_scores`: one solves the
    scene and one does not, so the pair also exercises agreement on a `False` final
    verdict, which is what the retired stock arm used to provide."""
    for standoff in (1.15, 1.45):
        env = _env()
        try:
            # Built ONCE, before the rollout. `build_task` rebuilds the scene (see
            # `Tossing3DTasks`' docstring), so building it inside the loop would silently
            # reset the episode between every step.
            goal = Tossing3DTasks(env=env, seed=0).build_task(scene_seed=CANONICAL_SEED).goal
            state = env.reset_to_seed(seed=CANONICAL_SEED)
            for _ in range(3):
                symbolic = InBinClassifier.holds(state=state, cube=env.cube, target=env.bin)
                assert symbolic == env.is_solved(), (
                    f"standoff {standoff}: InBin said {symbolic} while KINDER's "
                    f"own _check_goals() said {env.is_solved()}"
                )
                action = SkillOraclePolicy.get_labeled_action(
                    state=state, env=env, goal=goal, throw_standoff=standoff
                )
                state = env.take_action(action=action.action)
            assert (
                InBinClassifier.holds(state=state, cube=env.cube, target=env.bin) == env.is_solved()
            )
        finally:
            env.close()


def test_the_goal_box_in_the_state_is_the_live_region_bbox_element_for_element() -> None:
    """The only check that can catch a wrong goal box. Pinned against `Region.bbox` read
    back off the compiled model, never against the task JSON's literal -- the JSON range
    is inflated by 0.05 m per side before it becomes a region, and a predicate written
    against the literal is 2/3 of the true width on x."""
    env = _env()
    try:
        state = env.reset_to_seed(seed=CANONICAL_SEED)
        live = env.backend().goal_region_bbox()
        corners = ("x_min", "y_min", "z_min", "x_max", "y_max", "z_max")
        in_state = tuple(state.get(obj=env.bin, feature_name=name) for name in corners)
        assert in_state == pytest.approx(live)
        # And it is the inflated box, not the JSON's [1.90, 2.10].
        assert live[0] == pytest.approx(1.85, abs=1e-6)
        assert live[3] == pytest.approx(2.15, abs=1e-6)
    finally:
        env.close()


def test_oracle_pick_parameters_match_upstreams_own_sampler() -> None:
    """`skill_oracle_policy.py` writes these two out as literals so the oracle is
    deterministic without importing KINDER. This is the check that they are in fact what
    upstream's own `PickShelfController.sample_parameters` draws from the rng upstream's
    own test constructs."""
    env = _env()
    try:
        env.reset_to_seed(seed=CANONICAL_SEED)
        backend = env.backend()
        api = backend.api()
        lifted = api.shelf_controllers(backend._env.action_space)
        kinder_state = backend._state
        controller = lifted["pick_shelf"].ground((
            kinder_state.get_object_from_name(backend.robot_name),
            kinder_state.get_object_from_name(backend.cube_name),
        ))
        distance, rotation = controller.sample_parameters(kinder_state, np.random.default_rng(123))
        assert distance == pytest.approx(ORACLE_PICK_DISTANCE)
        assert rotation == pytest.approx(ORACLE_PICK_ROTATION)
    finally:
        env.close()


def test_the_oracle_reproduces_the_recorded_landing_and_step_counts() -> None:
    """The reference numbers: cube at rest x = 2.0318, z = 0.0444 (the bin's interior
    floor, i.e. the cube is *in* the bin), `_check_goals()` True, and the four controller
    executions terminating in 71 / 23 / 16 / 18 steps."""
    env = _env()
    try:
        # Built ONCE, before the rollout: `build_task` rebuilds the scene.
        goal = Tossing3DTasks(env=env, seed=0).build_task(scene_seed=CANONICAL_SEED).goal
        state = env.reset_to_seed(seed=CANONICAL_SEED)
        steps: list[int] = []
        for _ in range(3):
            action = SkillOraclePolicy.get_labeled_action(state=state, env=env, goal=goal)
            state = env.take_action(action=action.action)
            steps.extend(env.last_controller_steps())
            assert env.last_skill_error() is None, env.last_skill_error()

        assert steps == [71, 23, 16, 18]
        assert state.get(obj=env.cube, feature_name="x") == pytest.approx(REST_X, abs=1e-3)
        assert state.get(obj=env.cube, feature_name="z") == pytest.approx(BIN_FLOOR_Z, abs=1e-3)
        assert env.is_solved()
        assert IN_BIN.holds(state, (env.cube, env.bin))
    finally:
        env.close()


@pytest.mark.parametrize(("standoff", "expected"), [(1.15, True), (1.45, False)])
def test_the_derived_band_agrees_with_whether_the_throw_actually_scores(
    *, standoff: float, expected: bool
) -> None:
    """**The calibration guard for `THROW_RANGE`.**

    `RobotAtSuccessfulThrowPose` derives its acceptance band from live scene geometry plus
    one calibrated constant: the distance a throw displaces the cube. Everything else is
    read from the `State`, so this constant is the only thing that can go stale -- and it
    goes stale silently, because a wrong value still yields a plausible-looking band. If
    upstream changes the toss controller, the windup configuration, the cube's mass or the
    physics step, this is what fails.

    **The two standoffs are chosen to catch the specific wrong value.** `THROW_RANGE` is
    the *impact* range, 1.275 m. The tempting mismeasurement is the free-floor rest
    displacement, 1.3499 m, which is 0.075 m longer because it includes post-impact roll;
    the cube only rolls when it misses, since on this config the bin sits on the goal
    region and catches anything that lands inside. Under that wrong value the band shifts
    from `[1.150, 1.375]` -- the tightened band `THROW_OVERSHOOT_MARGIN`/
    `THROW_SHORTFALL_MARGIN` derive (see `predicates.py`) -- to `[1.225, 1.450]`, and
    **both** of these standoffs still flip: 1.15 would be predicted a miss under the wrong
    constant, and 1.45 a hit. Re-run live against a real KINDER install as part of the
    band-tightening change (both parametrised cases still pass: `predicted == expected`
    at both 1.15 and 1.45), so this is confirmed against the simulator, not only argued
    from the derivation above. Measured over three scene seeds, 1.15 solves
    3/3 and 1.45 solves 0/3, so the predicate must say exactly the opposite of the wrong
    value at both points.

    Asserted against real episode outcomes rather than a recorded number, so this is a
    check that the symbolic layer still describes the dynamics.

    `expected` means "*some* toss parameterisation scores from here", so the reality
    half probes `_UNION_PROBE_PARAMS` rather than the oracle, whose release millisecond
    is not an input to the predicate. The strong form of the `False` case is
    `test_no_toss_parameterisation_scores_from_beyond_the_accepted_band`.

    Resting x is not asserted: first contact here is the bin (`bin_0` in 17/18 committed
    grid rows at 140 deg/s), and one cell's three seeds rest 216 mm apart while their
    distances before first ground contact agree to 2.2 mm.
    """
    env = _env()
    try:
        goal = Tossing3DTasks(env=env, seed=0).build_task(scene_seed=CANONICAL_SEED).goal
        state = env.reset_to_seed(seed=CANONICAL_SEED)
        for _ in range(3):
            action = SkillOraclePolicy.get_labeled_action(
                state=state, env=env, goal=goal, throw_standoff=standoff
            )
            state = env.take_action(action=action.action)
            assert env.last_skill_error() is None, env.last_skill_error()
            if action.label.startswith("MoveToThrowPose("):
                predicted = RobotAtSuccessfulThrowPoseClassifier.holds(
                    state=state, robot=env.robot, target=env.bin
                )

        assert predicted == expected, (
            f"at standoff {standoff} the predicate says {predicted}, but it was measured "
            f"to be {expected}. THROW_RANGE = {THROW_RANGE} is no longer calibrated."
        )
    finally:
        env.close()

    # Short-circuits on the first scoring pair, so the `True` case costs one sequence.
    scored = next(
        (
            (speed, release_ms)
            for speed, release_ms in _UNION_PROBE_PARAMS
            if _throw_scores_from_standoff(standoff=standoff, speed=speed, release_ms=release_ms)
        ),
        None,
    )
    assert (scored is not None) == expected, (
        f"at standoff {standoff} the predicate says {expected}, but of the "
        f"{len(_UNION_PROBE_PARAMS)} toss parameterisations probed "
        f"{'none scored' if scored is None else f'{scored} scored'}. The band and the "
        "dynamics disagree about whether ANY parameterisation scores from this pose."
    )


def test_move_to_throw_pose_at_the_lower_standoff_bound_does_not_disturb_the_barrier() -> None:
    """**The regression guard for the bug `THROW_STANDOFF_BOUNDS`'s lower end exists to
    fix.** `move_to_target`'s base motion planner has collision-checking hardcoded off
    upstream, and `cuboid_barrier` is a real dynamic MuJoCo body -- not a static
    collision-free waypoint check -- so a `MoveToThrowPose` standoff that is too short
    drives the base straight through it and knocks it over.

    This would have caught the original defect: at the old lower bound (0.45 m) running
    this exact sequence visibly displaces the barrier. Three independent sweeps using the
    real oracle Pick parameters (`ORACLE_PICK_DISTANCE`, `ORACLE_PICK_ROTATION` -- a
    placeholder-param probe earlier gave a wrong, lower number and was caught and
    corrected) found the worst colliding standoff is 1.00 m and never exceeded it: 10
    seeds x 0.005 m resolution over [0.98, 1.03], 10 seeds x 0.02 m resolution over
    [0.90, 1.40], and all four corners of `Pick`'s own full sampling box
    (`PICK_DISTANCE_BOUNDS`, `PICK_ROTATION_BOUNDS`) plus the oracle point, since `Pick`
    also samples during practice, not just at its oracle default. The new lower bound,
    1.10 m, is `BARRIER_COLLISION_MARGIN` (0.10 m) clear of that worst measured point, and
    a confirming sweep of the new range at 0.05 m resolution over 10 seeds scored 140/140
    clear with the barrier bit-exact every time.

    Runs the oracle's own Pick, then `MoveToThrowPose` at exactly
    `THROW_STANDOFF_BOUNDS[0]` -- so this is red against the old bounds and green against
    the new ones without hardcoding either number."""
    env = _env()
    try:
        goal = Tossing3DTasks(env=env, seed=0).build_task(scene_seed=CANONICAL_SEED).goal
        state = env.reset_to_seed(seed=CANONICAL_SEED)
        barrier_before = tuple(
            state.get(obj=env.barrier, feature_name=name) for name in ("x", "y", "z")
        )

        # Pick, at the oracle's own parameters.
        action = SkillOraclePolicy.get_labeled_action(state=state, env=env, goal=goal)
        state = env.take_action(action=action.action)
        assert env.last_skill_error() is None, env.last_skill_error()

        # MoveToThrowPose at exactly the sampler's lower bound.
        action = SkillOraclePolicy.get_labeled_action(
            state=state, env=env, goal=goal, throw_standoff=THROW_STANDOFF_BOUNDS[0]
        )
        state = env.take_action(action=action.action)
        assert env.last_skill_error() is None, env.last_skill_error()

        barrier_after = tuple(
            state.get(obj=env.barrier, feature_name=name) for name in ("x", "y", "z")
        )
        assert barrier_after == pytest.approx(barrier_before, abs=1e-4), (
            f"MoveToThrowPose(standoff={THROW_STANDOFF_BOUNDS[0]}) moved cuboid_barrier "
            f"from {barrier_before} to {barrier_after} -- the base drove through it"
        )
    finally:
        env.close()


def test_the_shipped_scene_still_puts_the_bin_on_the_box_that_scores() -> None:
    """**The guard that makes a `reference/kindergarden` pin bump loud instead of silent.**

    This domain no longer commits a scene of its own: it runs whatever
    `Tossing3D-o1.json` the installed KINDER registers. That is the decision, and it has
    a cost worth naming rather than leaving implicit -- **the scene now moves with the
    pin.** That coupling is precisely the defect that produced this test's existence: the
    retired `Tossing3DTaskConfig.STOCK` meant "whatever the submodule ships", so its
    meaning moved when the pin moved, and two tests broke with nobody having edited them.

    Accepting the coupling therefore requires making it observable, which is this test.
    It reads the task JSON out of the KINDER that is actually **installed** -- located
    through the import system rather than a hardcoded path, so it is the tree that is
    *run*, not a second checkout that might sit at a different commit -- and pins the one
    property the domain cannot survive losing: `bin_init_region` is centred on
    `blocks_goal_region`'s centre, i.e. the bin sits on the box that scores.

    **Why that property and not the file's bytes.** A byte or hash comparison would fail
    on every pin bump, including ones that change nothing about the geometry, and a test
    that cries wolf gets deleted. The form this refuses is the one that broke the domain:
    upstream `1183de7` moved the bin to x = 2.23 and left the region behind, putting the
    two centres 230 mm apart.

    **Why the tolerance is 5 mm rather than exact, measured rather than chosen.** The two
    known-good forms are not identical, and the difference is instructive. The pin this
    repo currently records (`4113237`) declares `[[2.0, -0.0005, 2.001, 0.0005]]` -- a
    1 mm-wide sampling range running from 2.0 to 2.001, whose **mean is 2.0005**, i.e.
    0.5 mm past the goal region's centre. kg#126 as **merged** to the lab repo
    (`98ad2c0`) tightened it to `[[2.0, 0.0, 2.0, 0.0]]`, zero-width and exactly on
    centre. So bumping the pin onto the merged commit shifts the bin's mean position by
    0.5 mm: tiny, but a real dynamics change rather than a formatting one, and worth
    saying out loud because numbers in this file were measured under the jitter form.
    5 mm admits both, matches
    `test_the_bin_and_the_goal_region_coincide_in_the_shipped_scene`'s live tolerance,
    and excludes 230 mm by a factor of 46.

    The width bound is a second, independent thing worth pinning: it keeps the bin's
    placement effectively deterministic, which is what lets a single scene seed reproduce
    a landing position. Both known forms are <= 1 mm wide.

    Offline as far as the geometry goes -- it reads the JSON, not a compiled model --
    but it needs KINDER installed to find the file, so it lives here with the rest of the
    simulator-gated tests. `test_the_bin_and_the_goal_region_coincide_in_the_shipped_scene`
    is the live counterpart, measured off MuJoCo after inflation and sampling.
    """
    import json

    import kinder

    task_json = (
        Path(kinder.__file__).resolve().parent
        / "envs"
        / "dynamic3d"
        / "tasks"
        / "Tossing3D"
        / "Tossing3D-o1.json"
    )
    assert task_json.is_file(), f"the installed KINDER has no o1 task JSON at {task_json}"
    regions = json.loads(task_json.read_text())["regions"]

    (bin_range,) = regions["bin_init_region"]["ranges"]
    bin_x_start, bin_x_end = bin_range[0], bin_range[2]
    bin_centre = (bin_x_start + bin_x_end) / 2

    (goal_range,) = regions["blocks_goal_region"]["ranges"]
    goal_centre = (goal_range[0] + goal_range[3]) / 2

    assert bin_centre == pytest.approx(goal_centre, abs=0.005), (
        f"{task_json} declares bin_init_region centred at x = {bin_centre} against "
        f"blocks_goal_region centred at x = {goal_centre}. The bin has come off the box "
        "that scores, which is the pre-PR-126 defect: a cube landing IN the bin would "
        "score a FAILURE, and training on this scene would reward missing it. The pin "
        "moved the scene -- decide whether that is wanted before trusting any number "
        "measured after this point."
    )
    assert bin_x_end - bin_x_start <= 0.001, (
        f"{task_json} samples the bin over {bin_x_end - bin_x_start} m on x. Above about "
        "a millimetre the bin's position stops being effectively determined by the scene "
        "seed, and per-seed landing positions recorded in this file stop reproducing."
    )


def test_the_bin_and_the_goal_region_coincide_in_the_shipped_scene() -> None:
    """**The invariant the whole domain rests on, re-measured live rather than asserted
    about JSON:** the bin sits on the box that scores, so "the cube is in the bin" and
    "the cube is in the goal region" are the same event.

    Measured rather than read off the file because neither box is where the file says it
    is -- `blocks_goal_region` is inflated by 0.05 m per side before it becomes a region,
    and the bin's placement is sampled from a 1 mm-wide range.

    **This is the fixed state of a scene that used to be broken, and the gap is the whole
    story.** Upstream commit `1183de7` moved `bin_init_region` to x = 2.23 and left
    `blocks_goal_region` behind, putting the bin ~230 mm off the region: a cube landing
    *in* the bin scored a **failure**, and training against it would have rewarded
    missing. `kindergarden` PR #126 put the bin back to x = 2.0. The tolerance below is
    5 mm against a measured ~0.1 mm, and 230 mm is the value it exists to exclude -- so
    this test is red on the pre-fix scene without needing one to compare against.
    """
    env = _env()
    try:
        state = env.reset_to_seed(seed=CANONICAL_SEED)
        # The scored box rides on the bin object now that the goal region is not modelled
        # separately, but it is still `blocks_goal_region`'s live bbox rather than the
        # bin's own geometry -- so this stays a comparison of two independent sources.
        goal_min = state.get(obj=env.bin, feature_name="x_min")
        goal_max = state.get(obj=env.bin, feature_name="x_max")
        bin_x = state.get(obj=env.bin, feature_name="x")
        centre = (goal_min + goal_max) / 2
        gap = abs(bin_x - centre)
        assert gap < 0.005, (
            f"the bin is {gap} m off the goal region's centre. At ~0.23 m this is the "
            "pre-PR-126 scene, in which a cube landing in the bin scores a failure."
        )
    finally:
        env.close()


def test_a_full_episode_through_the_problem_solves_the_default_scene() -> None:
    """End to end through the harness's own path: `run_task_episode` with the oracle
    policy, on the canonical scene."""
    env = _env()
    try:
        tasks = Tossing3DTasks(env=env, seed=0)
        problem = Tossing3DProblem(env=env, tasks=tasks)
        task = tasks.build_task(scene_seed=CANONICAL_SEED)

        # core.method.types.Policy is a positional Callable[[State], LabeledAction].
        def policy(state) -> LabeledAction:  # noqa: PLR0917
            return SkillOraclePolicy.get_labeled_action(state=state, env=env, goal=task.goal)

        solved, frames, _ = problem.run_task_episode(task=task, policy=policy)
        assert solved
        assert frames == []
    finally:
        env.close()


def test_the_renderer_produces_a_frame_whose_dimensions_ffmpeg_accepts() -> None:
    from hitl_pmp.environments.tossing3d.renderer import Tossing3DRenderer

    env = _env()
    try:
        state = env.reset_to_seed(seed=CANONICAL_SEED)
        frame = Tossing3DRenderer.render_frame(state=state, env=env, label=None)
        assert frame.dtype.name == "uint8"
        assert frame.ndim == 3 and frame.shape[2] == 3
        assert frame.shape[0] % 16 == 0
        assert frame.shape[1] % 16 == 0
        assert frame.shape[0] == 480 + Tossing3DRenderer.caption_height
    finally:
        env.close()


def test_render_returns_a_copy_not_a_view_into_mujocos_reused_buffer() -> None:
    """`np.asarray` does not copy an already-uint8 array, so without an explicit copy this
    hands back a view into the render context's reused buffer -- harmless one frame at a
    time, silently wrong the moment frames accumulate into a clip."""
    env = _env()
    try:
        env.reset_to_seed(seed=CANONICAL_SEED)
        first = env.backend().render()
        before = first.copy()
        env.backend().render()
        assert np.array_equal(first, before)
    finally:
        env.close()


def test_task_sampling_draws_distinct_scenes_for_train_and_test() -> None:
    """A test task must be a scene the method never practiced on. The split is a split of
    scene seeds, drawn from two streams derived from `seed`."""
    env = _env()
    try:
        tasks = Tossing3DTasks(env=env, seed=0)
        train_seed = Tossing3DTasks.draw_scene_seed(rng=tasks.train_rng)
        test_seed = Tossing3DTasks.draw_scene_seed(rng=tasks.test_rng)
        assert train_seed != test_seed
        task = tasks.build_task(scene_seed=train_seed)
        assert task.goal.describe() == "InBin(cube_0, bin_0)"
        assert not task.goal.is_satisfied(state=task.initial_state)
    finally:
        env.close()


def test_set_state_rebuilds_the_scene_so_reset_to_task_really_rewinds() -> None:
    """`PracticeLoop` and `run_task_episode` both restore a task's initial state before
    running it. In this domain that is a genuine simulator rebuild, not a field
    assignment, so it is worth proving that the cube really goes back."""
    env = _env()
    try:
        tasks = Tossing3DTasks(env=env, seed=0)
        task = tasks.build_task(scene_seed=CANONICAL_SEED)
        start_x = task.initial_state.get(obj=env.cube, feature_name="x")

        state = env.reset_to_seed(seed=CANONICAL_SEED)
        action = SkillOraclePolicy.get_labeled_action(state=state, env=env, goal=task.goal)
        moved = env.take_action(action=action.action)
        assert moved.get(obj=env.cube, feature_name="z") > 0.1, "the grasp should have lifted it"

        env.set_state(state=task.initial_state)
        assert env.get_current_state().get(obj=env.cube, feature_name="x") == pytest.approx(
            start_x, abs=1e-6
        )
    finally:
        env.close()


def test_a_recorded_episode_is_physics_rate_rather_than_one_frame_per_skill() -> None:
    """The frame count is the whole claim. One `take_action` here is a whole controller
    execution -- tens of MuJoCo ticks -- so rendering once per decision gave a four-frame
    `episode.mp4` of a domain that exists to show a throw. With the gymnasium
    `RenderCollection` wrapper in place, every one of those ticks reaches the clip.

    Asserted as a floor rather than an exact number: the oracle's controllers terminate
    on their own conditions (71/23/16/18 at the canonical seed), and pinning the total
    would make this a second, fragile copy of the step-count test above.
    """
    from hitl_pmp.environments.tossing3d.renderer import Tossing3DRenderer

    env = _env()
    try:
        tasks = Tossing3DTasks(env=env, seed=0)
        problem = Tossing3DProblem(env=env, tasks=tasks)
        task = tasks.build_task(scene_seed=CANONICAL_SEED)

        def policy(state) -> LabeledAction:  # noqa: PLR0917
            return SkillOraclePolicy.get_labeled_action(state=state, env=env, goal=task.goal)

        solved, frames, _ = problem.run_task_episode(
            task=task, policy=policy, renderer=Tossing3DRenderer
        )

        assert solved
        assert len(frames) > 100, f"expected a physics-rate clip, got {len(frames)} frames"
        assert len({frame.shape for frame in frames}) == 1, "ffmpeg needs one frame size"
        # Recording is per-episode and must not leak into the next, unrendered one.
        assert env.backend().record_substeps is False
    finally:
        env.close()


def test_recording_does_not_change_where_the_cube_comes_to_rest() -> None:
    """Presentation-only, proved rather than asserted in prose: the same episode run with
    and without the recording wrapper has to land the cube in the same place. A wrapper
    that stepped the simulator, or a per-tick render that perturbed it, would show up
    here and nowhere else."""
    from hitl_pmp.environments.tossing3d.renderer import Tossing3DRenderer

    rest = {}
    for label, renderer in (("plain", None), ("recorded", Tossing3DRenderer)):
        env = _env()
        try:
            tasks = Tossing3DTasks(env=env, seed=0)
            problem = Tossing3DProblem(env=env, tasks=tasks)
            task = tasks.build_task(scene_seed=CANONICAL_SEED)

            def policy(state, task=task, env=env) -> LabeledAction:  # noqa: PLR0917
                return SkillOraclePolicy.get_labeled_action(state=state, env=env, goal=task.goal)

            problem.run_task_episode(task=task, policy=policy, renderer=renderer)
            final = env.get_current_state()
            rest[label] = (
                final.get(obj=env.cube, feature_name="x"),
                final.get(obj=env.cube, feature_name="z"),
            )
        finally:
            env.close()

    assert rest["plain"][0] == pytest.approx(REST_X, abs=1e-3)
    assert rest["recorded"] == pytest.approx(rest["plain"], abs=1e-6)


# How far outside the accepted band to stand. Clears `move_to_target`'s own stopping
# noise -- PR #196's closest miss at 1.375 was 0.1 mm.
BAND_EDGE_PROBE_OFFSET = 0.05

# A coarse sample of the toss parameter box, as (release_speed, gripper_release_ms)
# fractions of their bounds. `(1.0, 0.42)` is the measured reach argmax: at 140 deg/s
# the impact range peaks near 763 ms, not at either end of `TOSS_RELEASE_MS_BOUNDS`.
_PARAM_GRID = (
    (0.0, 0.5),
    (1.0, 0.5),
    (0.0, 0.0),
    (1.0, 1.0),
    (0.5, 0.5),
    (1.0, 0.42),
)


def _lerp(*, bounds: tuple[float, float], fraction: float) -> float:
    return bounds[0] + fraction * (bounds[1] - bounds[0])


def _throw_scores_from_standoff(*, standoff: float, speed: float, release_ms: float) -> bool:
    """Run one `Pick -> MoveToThrowPose(standoff) -> Toss(speed, release_ms)` and report
    whether KINDER's own `_check_goals()` scored it. Raw action interface rather than
    `SkillOraclePolicy`, which hard-codes the toss parameters callers choose here."""
    env = _env()
    try:
        env.reset_to_seed(seed=CANONICAL_SEED)
        env.take_action(action=np.array([env.pick_id, ORACLE_PICK_DISTANCE, ORACLE_PICK_ROTATION]))
        env.take_action(action=np.array([env.move_to_throw_pose_id, standoff, 0.0]))
        env.take_action(action=np.array([env.toss_id, speed, release_ms]))
        return bool(env.is_solved())
    finally:
        env.close()


def _accepted_standoff_band() -> tuple[float, float]:
    """The band `RobotAtSuccessfulThrowPose` accepts, read off a real scene rather than
    recomputed, so it cannot drift from the predicate."""
    env = _env()
    try:
        state = env.reset_to_seed(seed=CANONICAL_SEED)
        bin_x = state.get(obj=env.bin, feature_name="x")
        lo = (
            bin_x
            + THROW_RANGE_MIN
            - (state.get(obj=env.bin, feature_name="x_max") - THROW_OVERSHOOT_MARGIN)
        )
        hi = (
            bin_x
            + THROW_RANGE_MAX
            - (state.get(obj=env.bin, feature_name="x_min") + THROW_SHORTFALL_MARGIN)
        )
        return (lo, hi)
    finally:
        env.close()


def test_no_toss_parameterisation_scores_from_beyond_the_accepted_band() -> None:
    """A pose past the band's far edge is unthrowable, by every toss parameterisation in
    bounds. The converse of every other check on this predicate, which only ever visits
    poses the band accepts -- so without this the band could widen arbitrarily.

    Only the far edge is probed: the near edge sits at 0.21 m, below the sampler's
    1.10 m floor, so `BAND_EDGE_PROBE_OFFSET` below it is not a pose `MoveToThrowPose`
    can reach.

    The pass is not vacuous. At the probe standoff of 1.450 every cell executes with no
    skill error, the base arrives (`base_x` 0.5497 against the commanded 2.0 - 1.45),
    and the cube flies 0.377 to 1.206 m over `_PARAM_GRID`. A wider manual check at the
    same pose -- 6 release milliseconds bracketing the reach argmax, over 3 seeds --
    scored `0/18`, the furthest resting at x = 1.8323, 68 mm short of the trimmed box's
    1.900 near edge. `0/24` at this pose in total."""
    lo, hi = _accepted_standoff_band()
    standoff = hi + BAND_EDGE_PROBE_OFFSET

    scored = [
        (speed_fraction, ms_fraction)
        for speed_fraction, ms_fraction in _PARAM_GRID
        if _throw_scores_from_standoff(
            standoff=standoff,
            speed=_lerp(bounds=TOSS_SPEED_BOUNDS, fraction=speed_fraction),
            release_ms=_lerp(bounds=TOSS_RELEASE_MS_BOUNDS, fraction=ms_fraction),
        )
    ]

    assert not scored, (
        f"standoff {standoff:.3f} is {BAND_EDGE_PROBE_OFFSET} m beyond the accepted band "
        f"[{lo:.3f}, {hi:.3f}], so RobotAtSuccessfulThrowPose rejects it -- but "
        f"{len(scored)}/{len(_PARAM_GRID)} toss parameterisations scored from it: {scored}. "
        "The band is too NARROW: it rejects a pose that is genuinely throwable."
    )


def test_the_converse_guard_would_catch_a_widened_band(*, monkeypatch: pytest.MonkeyPatch) -> None:
    """Proves the guard above can fail: its negative assertion passes trivially if the
    band it reads is not the band the predicate applies. A pose one metre past the far
    edge is rejected at the shipped constants and accepted once `THROW_RANGE_MAX` is
    widened past it. Executes no throw."""
    _, hi_before = _accepted_standoff_band()

    env = _env()
    try:
        state = env.reset_to_seed(seed=CANONICAL_SEED)
        # On the bin's axis, so the standoff conjunct is the only thing under test.
        far_past_the_box = state.get(obj=env.bin, feature_name="x") - (hi_before + 1.0)
        state.set(obj=env.robot, feature_name="pos_base_x", feature_val=far_past_the_box)
        state.set(
            obj=env.robot,
            feature_name="pos_base_y",
            feature_val=state.get(obj=env.bin, feature_name="y"),
        )

        assert not RobotAtSuccessfulThrowPoseClassifier.holds(
            state=state, robot=env.robot, target=env.bin
        ), "a pose 1 m beyond the band's far edge must be rejected at the shipped constants"

        monkeypatch.setattr(predicates, "THROW_RANGE_MAX", THROW_RANGE_MAX + 1.5)
        assert RobotAtSuccessfulThrowPoseClassifier.holds(
            state=state, robot=env.robot, target=env.bin
        ), (
            "widening THROW_RANGE_MAX by 1.5 m must make the predicate accept that same "
            "pose -- if it does not, the accepted band is not derived from this constant "
            "and the converse guard is probing an edge the predicate does not have"
        )
    finally:
        env.close()
