"""Hand-built `KinderObservation`s, so the offline half of this domain can be tested.

`KinderObservation` is deliberately a plain-floats boundary type (see
`kinder_backend.py`), which means everything above it -- the state translation, all six
predicates, the skills, the oracle -- is testable with no MuJoCo, and therefore on CI,
which never installs the optional `tossing3d` extra.

The numbers here are not invented: they are the scene as measured at upstream's own seed
125, recorded in `docs/kinder-environment-validation.md` and
reproduced by `scripts/tossing3d_oracle_demo.py`. Using the real geometry means an
offline test and a simulator-backed one are talking about the same scene.
"""

from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.kinder_backend import KinderObservation

# The live `Region.bbox` of `blocks_goal_region`: the task JSON's [1.90, 2.10] x
# [-0.10, 0.10] x [0, 0.10] inflated by ground_placement_threshold = 0.05 per side, z
# clamped at 0.
GOAL_REGION_BBOX = (1.85, -0.15, 0.0, 2.15, 0.15, 0.15)

# The cube's own half-extent (`size: 0.025` in the task JSON), doubled: `bb_z`.
CUBE_BB_Z = 0.05

# Where the bin sits in the shipped scene, measured live: `bin_init_region` is a 1 mm
# sampling range about x = 2.0, so the bin lands on the goal region rather than past it.
# It sat at x = 2.2305 before upstream's fix (`kindergarden` PR #126), 23 cm off the box
# that scores -- which is why a cube landing IN the bin used to be a scored failure.
BIN_X = 2.0001

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
                "bb_z": CUBE_BB_Z,
            },
            "bin_0": {"x": bin_x, "y": 0.0, "z": 0.1},
            "cuboid_barrier": {"x": BARRIER_X, "y": 0.0, "z": 0.05},
        },
        goal_region=goal_region,
        solved=solved,
    )


def state(
    *,
    env: Tossing3DEnvironment | None = None,
    seed: int = 125,
    steps_taken: int = 0,
    **kwargs,
):
    """The translated `core.State` for one scene snapshot."""
    environment = env if env is not None else Tossing3DEnvironment()
    return environment.build_state(
        observation=observation(**kwargs), seed=seed, steps_taken=steps_taken
    )
