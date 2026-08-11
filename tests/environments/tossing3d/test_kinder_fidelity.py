"""Simulator-backed tests: does this integration actually agree with KINDER?

**Every test in this file skips cleanly without KINDER**, gated on
`importlib.util.find_spec("kinder")` -- the *import* package name, not the distribution
name `kindergarden`. CI never installs the optional extra, so on CI this whole file
skips and the offline files carry the suite. Run it under the KINDER venv:

    /path/to/kinder-venv/bin/python -m pytest tests/environments/tossing3d/ -q

and under a memory cap, because these tests execute real controllers:

    systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0 -p OOMPolicy=continue -- ...

What these check that the offline tests structurally cannot:

1. `predicates.IN_GOAL_REGION` agrees with KINDER's own `_check_goals()`. That is the
   whole basis for this domain trusting its own symbolic layer rather than the simulator.
2. The goal box in the `State` is the live `Region.bbox`, element for element -- the only
   check that can catch a wrong box, which has shipped once already.
3. The oracle's pick parameters really are what upstream's own sampler draws.
4. The oracle reproduces the rest positions and step counts recorded in
   `docs/kinder-environment-validation.md`.
"""

import importlib.util

import numpy as np
import pytest

from hitl_pmp.core.method.types import LabeledAction
from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment, Tossing3DTaskConfig
from hitl_pmp.environments.tossing3d.predicates import (
    IN_GOAL_REGION,
    THROW_RANGE,
    THROW_STANDOFF_BOUNDS,
    InGoalRegionClassifier,
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

# The measured landings at standoff 1.35, from that same record, reproduced by
# `scripts/tossing3d_oracle_demo.py` on this machine.
COINCIDENT_REST_X = 1.9902
STOCK_REST_X = 2.2197
BIN_FLOOR_Z = 0.0444


def _env(*, task_config: Tossing3DTaskConfig = Tossing3DTaskConfig.COINCIDENT):
    return Tossing3DEnvironment(task_config=task_config)


def test_in_goal_region_agrees_with_kinders_own_goal_check_on_the_oracles_trajectory() -> None:
    """The differential test this domain's trust rests on. Checked at every step of a
    real oracle episode on both configs, because the two disagree about the verdict and a
    predicate that got the box wrong would be right on one and wrong on the other."""
    for task_config, standoff in (
        (Tossing3DTaskConfig.COINCIDENT, 1.35),
        (Tossing3DTaskConfig.STOCK, 1.35),
    ):
        env = _env(task_config=task_config)
        try:
            # Built ONCE, before the rollout. `build_task` rebuilds the scene (see
            # `Tossing3DTasks`' docstring), so building it inside the loop would silently
            # reset the episode between every step.
            goal = Tossing3DTasks(env=env, seed=0).build_task(scene_seed=CANONICAL_SEED).goal
            state = env.reset_to_seed(seed=CANONICAL_SEED)
            for _ in range(3):
                symbolic = InGoalRegionClassifier.holds(
                    state=state, cube=env.cube, goal_region=env.goal_region
                )
                assert symbolic == env.is_solved(), (
                    f"{task_config.value}: InGoalRegion said {symbolic} while KINDER's "
                    f"own _check_goals() said {env.is_solved()}"
                )
                action = SkillOraclePolicy.get_labeled_action(
                    state=state, env=env, goal=goal, throw_standoff=standoff
                )
                state = env.take_action(action=action.action)
            assert (
                InGoalRegionClassifier.holds(
                    state=state, cube=env.cube, goal_region=env.goal_region
                )
                == env.is_solved()
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
        in_state = tuple(state.get(obj=env.goal_region, feature_name=name) for name in corners)
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


def test_the_oracle_reproduces_the_recorded_coincident_landing_and_step_counts() -> None:
    """The reference numbers: cube at rest x = 1.9902, z = 0.0444 (the bin's interior
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
        assert state.get(obj=env.cube, feature_name="x") == pytest.approx(
            COINCIDENT_REST_X, abs=1e-3
        )
        assert state.get(obj=env.cube, feature_name="z") == pytest.approx(BIN_FLOOR_Z, abs=1e-3)
        assert env.is_solved()
        assert IN_GOAL_REGION.holds(state, (env.cube, env.goal_region))
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

    Asserted against the real episode outcome rather than against a recorded number, so
    this is a check that the symbolic layer still describes the dynamics.
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
                    state=state, robot=env.robot, target=env.bin, goal_region=env.goal_region
                )

        assert predicted == expected, (
            f"at standoff {standoff} the predicate says {predicted}, but it was measured "
            f"to be {expected}. THROW_RANGE = {THROW_RANGE} is no longer calibrated."
        )
        assert env.is_solved() == expected, (
            f"at standoff {standoff} the episode scored {env.is_solved()}, not {expected} "
            "as measured -- the dynamics moved, so THROW_RANGE needs re-measuring."
        )
    finally:
        env.close()


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


def test_the_same_standoff_on_stock_lands_in_the_bin_and_scores_a_failure() -> None:
    """The contrast that makes the default non-negotiable. Same seed, same skills, same
    standoff -- only the task JSON differs -- and the cube lands *in* the bin at
    x = 2.2197 with `_check_goals()` False, because on stock the bin sits 23 cm past the
    goal region. Training against this scene would reward missing the bin."""
    env = _env(task_config=Tossing3DTaskConfig.STOCK)
    try:
        state = env.reset_to_seed(seed=CANONICAL_SEED)
        goal = Tossing3DTasks(env=env, seed=0).build_task(scene_seed=CANONICAL_SEED).goal
        for _ in range(3):
            action = SkillOraclePolicy.get_labeled_action(state=state, env=env, goal=goal)
            state = env.take_action(action=action.action)
        assert state.get(obj=env.cube, feature_name="x") == pytest.approx(STOCK_REST_X, abs=1e-3)
        assert state.get(obj=env.cube, feature_name="z") == pytest.approx(BIN_FLOOR_Z, abs=1e-3)
        assert not env.is_solved()
        assert not IN_GOAL_REGION.holds(state, (env.cube, env.goal_region))
    finally:
        env.close()


def test_the_bin_and_the_goal_region_coincide_under_the_default_config() -> None:
    """The claim the default rests on, re-measured live rather than asserted about JSON:
    neither box is where the file says it is (the region is inflated; the bin's placement
    is sampled from a 1 mm-wide range). Worst-edge disagreement is ~0.1 mm on coincident
    against ~230 mm on stock."""
    for task_config, expected_gap in (
        (Tossing3DTaskConfig.COINCIDENT, 0.005),
        (Tossing3DTaskConfig.STOCK, None),
    ):
        env = _env(task_config=task_config)
        try:
            state = env.reset_to_seed(seed=CANONICAL_SEED)
            goal_min = state.get(obj=env.goal_region, feature_name="x_min")
            goal_max = state.get(obj=env.goal_region, feature_name="x_max")
            bin_x = state.get(obj=env.bin, feature_name="x")
            centre = (goal_min + goal_max) / 2
            gap = abs(bin_x - centre)
            if expected_gap is None:
                assert gap > 0.2, f"stock's bin should be ~23 cm off the region, got {gap}"
            else:
                assert gap < expected_gap, f"coincident's bin is {gap} m off the region"
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
        assert task.goal.describe() == "InGoalRegion(cube_0, blocks_goal_region)"
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

    assert rest["plain"][0] == pytest.approx(COINCIDENT_REST_X, abs=1e-3)
    assert rest["recorded"] == pytest.approx(rest["plain"], abs=1e-6)
