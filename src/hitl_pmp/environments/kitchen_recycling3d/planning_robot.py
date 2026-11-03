"""The kitchen's PyBullet planning robot: upstream's Kinova Gen3 with the camera bracelet.

`PyBulletSim` loads pybullet_helpers' `gen3_7dof.urdf`, whose bracelet link is
`bracelet_no_vision_link`. KINDER's MuJoCo TidyBot and the real robot carry
`bracelet_with_vision_link`, which reaches ~3 cm further. In Della v7 that gap let the
pick plan hover paths that cleared the bin's robot-facing rim by 2.8 mm in PyBullet and
pushed the bracelet 3-4 mm into it in MuJoCo, so the pick aborted on contact.

The URDF here is that same file with only the bracelet's two mesh references swapped
(pybullet_helpers' own `gen3_7dof_vision.urdf` is a different robot: other links, a
camera and a forque, no `finger_tip`). It is derived from the installed pybullet_helpers
at first use, with mesh paths made absolute, so it follows whatever version is
installed and needs no workstation path.
"""

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Any, ClassVar

import pybullet as p
from kinder_models.dynamic3d.utils import PyBulletSim
from pybullet_helpers.motion_planning import create_joint_distance_fn
from pybullet_helpers.robots.kinova import KinovaGen3RobotiqGripperPyBulletRobot
from pybullet_helpers.utils import get_assets_path


class VisionBraceletPyBulletSim(PyBulletSim):
    """Upstream's planning scene with the vision-bracelet robot swapped in.

    Upstream builds its robot inside `__init__`, so the stock one is loaded, replaced by
    the same arm from the vision-bracelet URDF, and removed. Every link and joint keeps
    its name and index; only the bracelet's mesh differs.
    """

    def __init__(self, initial_state: Any) -> None:  # noqa: PLR0917 -- upstream signature
        super().__init__(initial_state)
        stock = self._robot
        self._robot = VisionBraceletKinovaGen3(
            self.physics_client_id, fixed_base=False, control_mode="reset"
        )
        p.removeBody(stock.robot_id, physicsClientId=self.physics_client_id)
        self._joint_distance_fn = create_joint_distance_fn(self._robot)


class VisionBraceletKinovaGen3(KinovaGen3RobotiqGripperPyBulletRobot):
    """pybullet_helpers' Kinova Gen3 + Robotiq 2F-85, wearing the vision bracelet."""

    @property
    def default_urdf_path(self) -> Path:
        return VisionBraceletUrdf.path()


class VisionBraceletUrdf:
    """`gen3_7dof.urdf` with the bracelet's visual and collision mesh swapped."""

    STOCK_MESH: ClassVar[str] = "bracelet_no_vision_link.STL"
    VISION_MESH: ClassVar[str] = "bracelet_with_vision_link.STL"
    _cached: ClassVar[Path | None] = None

    @classmethod
    def path(cls) -> Path:
        if cls._cached is None or not cls._cached.exists():
            cls._cached = cls._write(text=cls.text())
        return cls._cached

    @classmethod
    def text(cls) -> str:
        stock = get_assets_path() / "urdf" / "kortex_description" / "gen3_7dof.urdf"
        text = stock.read_text().replace("package://kortex_description/", f"{stock.parent}/")
        if text.count(cls.STOCK_MESH) != 2:
            raise ValueError(f"expected a visual and a collision {cls.STOCK_MESH} in {stock}")
        return text.replace(cls.STOCK_MESH, cls.VISION_MESH)

    @staticmethod
    def _write(*, text: str) -> Path:
        """Content-addressed, written atomically: concurrent workers agree on the file."""
        digest = hashlib.sha256(text.encode()).hexdigest()[:16]
        target = Path(tempfile.gettempdir()) / f"hitl-pmp-gen3-7dof-vision-bracelet-{digest}.urdf"
        if not target.exists():
            descriptor, partial = tempfile.mkstemp(dir=target.parent, suffix=".urdf")
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(text)
            Path(partial).replace(target)
        return target
