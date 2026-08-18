"""Hand-built `KinderObservation`s, so the offline half of this domain can be tested.

`KinderObservation` is deliberately a plain-floats boundary type (see
`kinder_backend.py`), which means everything above it -- the state translation, all five
predicates, the skills, the oracle -- is testable with no MuJoCo. CI installs the
`tossing3d` extra now, but that is recent; the offline half is still what makes most of
this package fast.

The numbers here are not invented: they are the scene as measured at upstream's own seed
125, recorded in `docs/kinder-environment-validation.md` and
reproduced by `scripts/tossing3d_oracle_demo.py`. Using the real geometry means an
offline test and a simulator-backed one are talking about the same scene.
"""

from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.kinder_backend import KinderObservation

# The live `Region.bbox` of `blocks_goal_region`: the task JSON's [2.00, 2.05] x
# [-0.07, 0.07] x [0, 0.10] inflated by ground_placement_threshold = 0.05 per side, z
# clamped at 0.
#
# **This moved with the `reference/kindergarden` pin bump on this branch**, and the earlier
# value is not restated because it is no longer live: upstream `270fdb6` narrowed
# `blocks_goal_region` from [1.90, 2.10] to [2.00, 2.05] on x, so the inflated box went
# from [1.85, 2.15] -- exactly the bin's own 0.30 m footprint -- to [1.95, 2.10], strictly
# inside it. The consequence is stated wherever it matters: **"the cube is in the bin" and
# "the cube scores" are no longer the same event.**
# `test_the_goal_box_in_the_state_is_the_live_region_bbox_element_for_element` measures
# this against the compiled model, so it cannot drift from the scene unnoticed.
GOAL_REGION_BBOX = (1.95, -0.12, 0.0, 2.10, 0.12, 0.15)

# The cube's own half-extent (`size: 0.025` in the task JSON), doubled. All three
# extents are equal, which is what makes `OnGround`'s face-interchangeable branch the one
# this scene takes -- upstream's branch, in `_check_on_ground`.
CUBE_BB_Z = 0.05
CUBE_BB_X = 0.05
CUBE_BB_Y = 0.05

# Where the bin sits in the shipped scene, measured live at seeds 0 and 125 and found
# bit-identical at both: `bin_init_region` is a zero-width range at x = 2.0, so the bin's
# 0.30 m footprint (x in [1.85, 2.15]) contains the box that scores rather than sitting
# past it. It sat at x = 2.2305 before upstream's fix (`kindergarden` PR #126), 23 cm off
# the box that scores -- which is why a cube landing IN the bin used to be a scored
# failure.
#
# **This was 2.0001, and the 0.1 mm was real rather than a rounding choice**: the range
# used to be 1 mm wide, so the sampled centre moved per seed. It does not any more.
BIN_X = 2.0

# `barrier_init_region` places the barrier at x = 1.3.
BARRIER_X = 1.3


# Where the cube actually starts at seed 125, measured live rather than guessed:
# `blocks_init_region` samples x in [0.5, 0.75], y in [-0.25, 0.25].
CUBE_START_X = 0.7129


def observation(
    *,
    cube_x: float = CUBE_START_X,
    cube_y: float = 0.0,
    cube_z: float = 0.025,
    cube_qx: float = 0.0,
    cube_qy: float = 0.0,
    cube_qz: float = 0.0,
    cube_qw: float = 1.0,
    gripper: float = 0.0,
    base_x: float = 0.0,
    base_y: float = 0.0,
    base_rot: float = 0.0,
    bin_x: float = BIN_X,
    goal_region: tuple[float, float, float, float, float, float] = GOAL_REGION_BBOX,
    solved: bool = False,
) -> KinderObservation:
    """One scene snapshot. Defaults are the initial state: cube flat on the floor at the
    near side, gripper open, robot at the origin."""
    return KinderObservation(
        features={
            "robot": {
                "pos_base_x": base_x,
                "pos_base_y": base_y,
                "pos_base_rot": base_rot,
                "pos_gripper": gripper,
            },
            "cube_0": {
                "x": cube_x,
                "y": cube_y,
                "z": cube_z,
                "qx": cube_qx,
                "qy": cube_qy,
                "qz": cube_qz,
                "qw": cube_qw,
                "bb_x": CUBE_BB_X,
                "bb_y": CUBE_BB_Y,
                "bb_z": CUBE_BB_Z,
            },
            "bin_0": {"x": bin_x, "y": 0.0, "z": 0.1},
            "cuboid_barrier": {"x": BARRIER_X, "y": 0.0, "z": 0.05},
        },
        goal_region=goal_region,
        solved=solved,
    )


# The abstraction of the initial scene: gripper open, cube flat on the floor on the
# robot's side of the barrier. Written out rather than computed because the predicates
# are upstream's now and two of the five need a live simulator to evaluate (see
# `predicates.py`), while what a symbolic-layer test wants is simply *an* abstract state
# to plan from. Anything asserting the classifiers' own semantics belongs in
# `test_kb_predicate_parity.py`, which calls upstream's classifiers directly.
INITIAL_ATOMS = frozenset({
    ("HandEmpty", ("robot",)),
    ("OnGround", ("cube_0",)),
    ("MovableIsDownX", ("cube_0", "cuboid_barrier")),
})

# Mid-episode: the cube is grasped and lifted, still on the robot's side of the barrier.
HOLDING_ATOMS = frozenset({
    ("Holding", ("robot", "cube_0")),
    ("MovableIsDownX", ("cube_0", "cuboid_barrier")),
})

# After a scoring throw: the cube is at rest inside the region and the hand is empty.
LANDED_IN_REGION_ATOMS = frozenset({
    ("HandEmpty", ("robot",)),
    ("OnGround", ("cube_0",)),
    ("MovableInGoalRegion", ("cube_0",)),
})

# After a throw that *missed*: the cube is at rest past the one-way barrier and outside
# the goal region, and the hand is empty. The domain's dead end -- no `MovableIsDownX`
# means `PickCube` is inapplicable, and an empty hand means the composed toss is too.
#
# `RobotAtThrowPose` is not in this set, and has no counterpart anywhere in this file:
# upstream deleted the classifier when it composed the base move into the toss, so there
# is no longer a state between them for any atom to name.
MISSED_TOSS_ATOMS = frozenset({
    ("HandEmpty", ("robot",)),
    ("OnGround", ("cube_0",)),
})


def state(
    *,
    env: Tossing3DEnvironment | None = None,
    seed: int = 125,
    steps_taken: int = 0,
    abstract_atoms=None,
    **kwargs,
):
    """The translated `core.State` for one scene snapshot.

    `abstract_atoms` defaults to absent, which is what a *translation* test wants: the
    resulting state carries the flat features and refuses to answer a predicate, rather
    than silently answering `False` to all five. Pass `INITIAL_ATOMS` (or a set of your
    own) when the test needs a symbolic layer.
    """
    environment = env if env is not None else Tossing3DEnvironment()
    return environment.build_state(
        observation=observation(**kwargs),
        seed=seed,
        steps_taken=steps_taken,
        abstract_atoms=abstract_atoms,
    )
