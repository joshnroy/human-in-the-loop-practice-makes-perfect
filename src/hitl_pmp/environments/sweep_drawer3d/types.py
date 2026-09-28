"""Data for the SweepIntoDrawer3D self-reset: scene constants, step records, outcomes."""

from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict

CubeLocation = Literal["drawer", "counter", "floor", "other"]


class SweepDrawerScene(BaseModel):
    """Geometry of SweepIntoDrawer3D-o5 at the pinned kindergarden commit.

    Everything here is read off the compiled MuJoCo model (lab2 kitchen island, drawer
    `s1c1`) or the task JSON; `PlanningScene` re-reads the drawer boxes from the live model
    rather than trusting these numbers, and the tests pin the ones used as thresholds.
    """

    ENV_ID: ClassVar[str] = "kinder/SweepIntoDrawer3D-o5-v0"
    CUBES: ClassVar[tuple[str, ...]] = tuple(f"cube_{i}" for i in range(5))
    WIPER: ClassVar[str] = "wiper_0"
    DRAWER: ClassVar[str] = "kitchen_island_drawer_s1c1"
    ROBOT: ClassVar[str] = "robot"
    ISLAND_SLAB: ClassVar[str] = "collider:kitchen_island:7"

    CUBE_HALF: ClassVar[float] = 0.01
    COUNTER_TOP: ClassVar[float] = 0.46
    # The island's front edge (countertop) and the drawer face at pos = 0.
    COUNTER_EDGE_X: ClassVar[float] = 0.875
    DRAWER_FACE_X: ClassVar[float] = 0.895
    # The handle bar stands 4 cm proud of the face, below the chassis top: it, not the
    # face, is what a base parked in front of the open drawer touches first.
    DRAWER_HANDLE_FRONT_X: ClassVar[float] = 0.938
    # Inner face of the drawer's front wall at pos = 0, and its side walls.
    DRAWER_FRONT_INNER_X: ClassVar[float] = 0.872
    DRAWER_SIDE_INNER_Y: ClassVar[float] = 0.3085
    DRAWER_HALF_WIDTH: ClassVar[float] = 0.333
    DRAWER_FLOOR: ClassVar[float] = 0.2275
    DRAWER_WALL_TOP: ClassVar[float] = 0.444
    # Task JSON `blocks_init_region` resolved to world coordinates on the countertop.
    PILE_X: ClassVar[tuple[float, float]] = (0.625, 0.825)
    PILE_Y: ClassVar[tuple[float, float]] = (-0.2, 0.0)
    # Mobile base chassis (compiled mesh AABB) and the arm mount offset.
    CHASSIS_HALF: ClassVar[tuple[float, float]] = (0.276, 0.254)
    CHASSIS_TOP: ClassVar[float] = 0.383
    HOME: ClassVar[tuple[float, ...]] = (
        0.0,
        -0.3490658503988659,
        3.141592653589793,
        -2.548180707911721,
        0.0,
        -0.8726646259971648,
        1.5707963267948966,
    )


class GripperGeometry(BaseModel):
    """The TidyBot's Robotiq 2F-85 as the MuJoCo model has it, in the PyBullet EE frame.

    z is the approach axis and x the closing axis. Measured by driving the MuJoCo gripper
    to each command and reading the pad geoms (see the self-reset PR's methods).
    """

    # Command -> inner pad gap: 0.0 -> 8.5 cm, 0.635 -> 3.2 cm, 1.0 -> closed.
    CUBE_OPEN_CMD: ClassVar[float] = 0.635
    # PyBullet finger state with the same gap as CUBE_OPEN_CMD (open 0, closed 0.8).
    CUBE_OPEN_PB: ClassVar[float] = 0.52
    CLOSED_PB: ClassVar[float] = 0.8
    OPEN_HALF_GAP: ClassVar[float] = 0.0426
    # Pad centre and fingertip depth below the EE frame along the approach axis.
    PAD_CENTER_OPEN: ClassVar[float] = 0.016
    PAD_CENTER_PARTIAL: ClassVar[float] = 0.0375
    TIP_OPEN: ClassVar[float] = 0.035
    TIP_PARTIAL: ClassVar[float] = 0.047
    TIP_CLOSED: ClassVar[float] = 0.048
    # Finger footprint used by the 2D pre-filters.
    FINGER_THICK: ClassVar[float] = 0.010
    FINGER_HALF_WIDTH: ClassVar[float] = 0.012
    # Palm (gripper base): ~9.4 x 9 cm, bottom ~6.5 cm above the pads.
    PALM_HALF: ClassVar[tuple[float, float]] = (0.047, 0.045)
    PALM_LEVER: ClassVar[float] = 0.075


class ResetStep(BaseModel):
    """One executed (or refused) primitive of the reset, as written to the step log."""

    model_config = ConfigDict(extra="allow")

    index: int
    name: str
    phase: str
    kind: str
    success: bool
    ticks: int
    wall_s: float
    locations: dict[str, CubeLocation]
    drawer_pos: float
    note: str = ""


class ResetOutcome(BaseModel):
    """Where everything ended up after a reset, against the four reset criteria."""

    locations: dict[str, CubeLocation]
    in_pile: dict[str, bool]
    drawer_pos: float
    wiper_xy_error: float
    wiper_yaw_error_deg: float
    wiper_on_counter: bool
    robot_actions: int
    ticks: int
    wall_s: float

    @property
    def n_in_pile(self) -> int:
        return sum(self.in_pile.values())

    @property
    def drawer_closed(self) -> bool:
        return self.drawer_pos < 0.01

    @property
    def wiper_home(self) -> bool:
        return self.wiper_on_counter and self.wiper_xy_error < 0.02 and self.wiper_yaw_error_deg < 5

    @property
    def success(self) -> bool:
        return self.n_in_pile == len(self.in_pile) and self.drawer_closed and self.wiper_home
