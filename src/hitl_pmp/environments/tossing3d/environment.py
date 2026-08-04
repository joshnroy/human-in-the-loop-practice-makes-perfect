from typing import ClassVar

import numpy as np
from gymnasium.spaces import Box
from pydantic import PrivateAttr

from hitl_pmp.core.problem.environment.environment import Environment
from hitl_pmp.core.problem.environment.types import Action, Object, State, Type

from .kinder_backend import KinderBackend


class Tossing3DEnvironment(Environment):
    """KINDER's `Tossing3D` (https://prpl-group.com/kinder-site/environments/tossing3d/):
    a TidyBot++ mobile manipulator must get a cube from the floor into a goal region on
    the far side of an immovable barrier. The barrier is 5 m wide and blocks the base,
    so the cube can only get there through the air -- the robot has to *toss* it.

    This class implements **no dynamics**. It is the adapter that presents KINDER's own
    simulator as a `core.Environment`: `take_action` decodes an action into one of
    KINDER's parameterized controllers and hands it to `KinderBackend`, and every state
    read is KINDER's own observation reshaped into this repo's flat `State`.

    **The cube cannot be retrieved.** Once it crosses the barrier -- whether it lands in
    the goal region or overshoots into the bin behind it -- the robot cannot reach it
    again. That is the property this domain was integrated for: it is the concrete case the
    project's V1 proposal names as EES's predicted failure mode ("cannot reset the
    environment under a suboptimal policy... when the goal is reached it can't reset").
    Read `docs/experiment-logs/`'s Tossing3D entry before drawing any conclusion from a
    learning curve measured here: `PracticeLoop` hands out a free reset every
    interaction period, which supplies exactly the resets that property is about
    removing.

    One `take_action` is one *skill* execution (a few hundred MuJoCo control ticks
    driven by a KINDER controller), not one 20 ms tick, matching every other domain in
    this repo. The raw action is `[skill_id, param0, param1]`:
      * `skill_id` -- 0 Pick, 1 MoveToThrowPose, 2 Toss.
      * `param0` -- Pick's base standoff distance; Toss's swing dial. Unused by
        MoveToThrowPose.
      * `param1` -- Pick's base rotation offset. Unused by the others.
    `take_action` is TOTAL over the whole Box: a non-finite or unknown value, a skill
    whose preconditions do not hold in the simulator, and a genuine KINDER planning
    failure (inverse kinematics with no solution, motion planning returning None) are
    all silent no-ops rather than crashes.

    KINDER reports `terminated=True` the moment its own goal check passes, and this
    wrapper deliberately ignores that flag: an interaction period runs for its full
    `--max-steps-per-interaction` regardless. Nothing is lost by continuing, because a
    solved state is absorbing here -- the cube sits beyond the barrier, so `Reachable`
    is false, `Pick` has no applicable grounding, and no sequence of skills can undo
    the success. Letting the flag end the period early would instead have made a
    *solved* period cheaper in transitions than a failed one, quietly biasing the
    x-axis of every learning curve.

    The `State` layout is deliberately small and flat, and every predicate in
    `predicates.py` reads only from it. `goal_region` carries KINDER's own
    `blocks_goal_region` box as features -- put in the state, like Tossing Room's pile
    room, precisely so a module-level `Predicate` (whose signature is only
    `(state, objects)`) can test against it without reaching for env config. `scene`
    carries the KINDER reset seed, which is what makes a `State` restorable at all: see
    `set_state`.
    """

    SKILL_PICK: ClassVar[int] = 0
    SKILL_MOVE_TO_THROW_POSE: ClassVar[int] = 1
    SKILL_TOSS: ClassVar[int] = 2

    robot_type: ClassVar[Type] = Type(
        name="robot", feature_names=("base_x", "base_y", "base_rot", "holding")
    )
    cube_type: ClassVar[Type] = Type(name="cube", feature_names=("x", "y", "z"))
    bin_type: ClassVar[Type] = Type(name="bin", feature_names=("x", "y", "z"))
    barrier_type: ClassVar[Type] = Type(name="barrier", feature_names=("x", "y", "z"))
    region_type: ClassVar[Type] = Type(
        name="region", feature_names=("x_min", "y_min", "z_min", "x_max", "y_max", "z_max")
    )
    # The KINDER reset seed that produced this state. A structural part of the state
    # rather than bookkeeping on the side: MuJoCo state is thousands of floats that no
    # flat feature vector reconstructs, so "which episode is this" is the only handle
    # by which a State can be reinstalled -- see set_state.
    scene_type: ClassVar[Type] = Type(name="scene", feature_names=("seed",))

    robot: ClassVar[Object] = Object(name="robot", type=robot_type)
    cube: ClassVar[Object] = Object(name="cube_0", type=cube_type)
    bin_object: ClassVar[Object] = Object(name="bin_0", type=bin_type)
    barrier: ClassVar[Object] = Object(name="cuboid_barrier", type=barrier_type)
    goal_region: ClassVar[Object] = Object(name="blocks_goal_region", type=region_type)
    scene: ClassVar[Object] = Object(name="scene", type=scene_type)

    action_space: ClassVar[Box] = Box(-np.inf, np.inf, (3,))

    # Base standoff for MoveToThrowPose, in metres from the bin, and how far off it the
    # base may stop and still count as at the throw pose. ClassVars, not constructor
    # fields, and deliberately not CLI flags: `AtThrowPose` is a module-level Predicate
    # whose signature is only (state, objects), so it cannot read per-instance config,
    # and these are properties of the KINDER scene's fixed geometry rather than
    # per-run configuration. This is the same rule Light Switch follows for the values
    # its predicates need. The standoff is fixed rather than sampled so that this
    # domain has exactly one throw-shaping dial (Toss's swing), the same shape as
    # Tossing Room's single `force` -- a learning curve then attributes improvement to
    # one sampler rather than to an interaction between two.
    throw_standoff: ClassVar[float] = 1.35
    throw_pose_tolerance: ClassVar[float] = 0.15

    variant: str = "o1"
    # Bounds the uniform prior over Pick's (distance, rot) draws. These are KINDER's
    # own `MOVE_TO_TARGET_DISTANCE_BOUNDS`/`MOVE_TO_TARGET_ROT_BOUNDS`
    # (kinder_models/dynamic3d/utils.py), i.e. the range its own `pick_shelf` sampler
    # randomizes over, so this port learns the quantity KINDER randomizes rather than
    # one invented here.
    pick_distance_low: float = 0.5
    pick_distance_high: float = 0.6
    pick_rot_low: float = -np.pi / 4
    pick_rot_high: float = np.pi / 4
    # Bounds the uniform prior over Toss's swing dial. 1.0 is KINDER's own demo toss,
    # which overshoots the goal region into the bin behind it; the region is reached
    # around 0.6-0.9. The prior is deliberately wider than that band so an unpracticed
    # sampler does not already succeed most of the time -- Tossing Room's log records
    # what happens when it does (an unpracticed EES scoring 94.7% left a learned
    # sampler no headroom to demonstrate anything).
    swing_low: float = 0.25
    swing_high: float = 1.25
    # The seed hard_reset() resets to. Only ever used before a run starts; Tasks draws
    # its own per-task seeds.
    canonical_seed: int = 0

    _backend: KinderBackend | None = PrivateAttr(default=None)
    _goal_region_bounds: tuple[float, ...] | None = PrivateAttr(default=None)

    def backend(self) -> KinderBackend:
        """The one live simulator this environment drives, opened on first use. Lazy
        because opening it compiles a MuJoCo model and connects a PyBullet client,
        which is several seconds -- too expensive to pay in a constructor that tests
        call to check a feature layout."""
        if self._backend is None:
            self._backend = KinderBackend(variant=self.variant)
        return self._backend

    def goal_region_bounds(self) -> tuple[float, ...]:
        """KINDER's `blocks_goal_region` box, cached after the first read since the
        variant's task JSON cannot change mid-run."""
        if self._goal_region_bounds is None:
            self._goal_region_bounds = self.backend().goal_region_bounds()
        return self._goal_region_bounds

    def build_state(
        self, *, features: dict[str, tuple[float, ...]], seed: int, region: tuple[float, ...]
    ) -> State:
        """Assemble a `State` from the backend's flat readings. Separate from the
        backend call so the whole predicate/skill layer can be tested against
        hand-written feature dicts with no simulator present."""
        return State(
            data={
                self.robot: np.array(features["robot"], dtype=float),
                self.cube: np.array(features["cube"], dtype=float),
                self.bin_object: np.array(features["bin"], dtype=float),
                self.barrier: np.array(features["barrier"], dtype=float),
                self.goal_region: np.array(region, dtype=float),
                self.scene: np.array([float(seed)], dtype=float),
            }
        )

    def reset_to_seed(self, *, seed: int) -> State:
        """Reset the simulator to a KINDER episode seed and read the resulting state
        back. Read back rather than predicted: the initial cube pose comes out of
        KINDER's own region sampler, and a `State` that merely *claims* to be the
        initial state would disagree with the one `reset_to_task` later produces."""
        features = self.backend().reset(seed=seed)
        state = self.build_state(features=features, seed=seed, region=self.goal_region_bounds())
        self.current_state = state
        return state

    def set_state(self, *, state: State) -> None:
        """Reinstall a state by re-running the KINDER reset that produced it.

        This is the one place this domain departs from the usual `set_state`
        contract, and it departs because MuJoCo leaves no alternative: the real state
        is thousands of floats of positions, velocities and contact data, and the six
        feature vectors above are a lossy view of it. So `set_state` honours the
        `scene.seed` feature and reproduces *that episode's initial state* -- verified
        deterministic, including after the simulator has been driven far away in
        between. It is therefore only meaningful for a state this environment itself
        produced at a reset, which is exactly how the harness uses it
        (`Problem.reset_to_task`). No `HumanOracle` is wired for this domain
        (`Problem.human` stays None, as in Tossing Room and Light Switch), so the
        privileged mid-episode override the base contract describes has no caller.
        """
        seed = int(round(state.get(obj=self.scene, feature_name="seed")))
        self.reset_to_seed(seed=seed)

    def take_action(self, *, action: Action) -> State:
        raw_skill = float(action[0])
        if not np.isfinite(raw_skill):
            return self.get_current_state()
        skill_id = int(round(raw_skill))
        param0, param1 = float(action[1]), float(action[2])
        if not (np.isfinite(param0) and np.isfinite(param1)):
            return self.get_current_state()

        if skill_id == self.SKILL_PICK:
            self.backend().execute_pick(distance=param0, rot=param1)
        elif skill_id == self.SKILL_MOVE_TO_THROW_POSE:
            self.backend().execute_move_to_throw_pose(distance=self.throw_standoff)
        elif skill_id == self.SKILL_TOSS:
            self.backend().execute_toss(swing=param0)
        else:
            # Unknown skill id -> no-op, no simulator call at all.
            return self.get_current_state()

        seed = int(round(self.get_current_state().get(obj=self.scene, feature_name="seed")))
        state = self.build_state(
            features=self.backend().read_features(),
            seed=seed,
            region=self.goal_region_bounds(),
        )
        self.current_state = state
        return state

    def get_valid_actions(self) -> list[Action]:
        # Two of the three skills carry continuous parameters, so there is no
        # finite/enumerable action list to return (same as Light Switch and Tossing
        # Room, whose dials are continuous too).
        return []

    def hard_reset(self) -> None:
        self.reset_to_seed(seed=self.canonical_seed)
