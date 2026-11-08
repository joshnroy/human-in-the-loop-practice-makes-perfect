"""Tossing3D's simulator backend, on the kitchen scene.

Everything symbolic -- the abstractor, `BinOnGround`, `PickPlannable`, the goal check --
is `KinderBackend`'s. What differs is geometry and the pick: the kitchen scene and its
furniture, the drive zone every base plan stays inside (the mats are the no-drive
line), the kitchen's own pick controller, and where a human puts the cube and bin.
"""

import json
import os
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from pydantic import Field, PrivateAttr

from hitl_pmp.environments.tossing3d.kinder_backend import KinderBackend, TossSettlingController
from hitl_pmp.environments.tossing3d.sides import Tossing3DSide
from hitl_pmp.environments.tossing3d.skill_timing import SkillTiming
from hitl_pmp.environments.tossing3d.types import ControllerRun, Tossing3DObservation

from .bin_geometry import TaperedBin, TaperedBinCollision
from .bin_regions import KitchenBinRegions
from .domain import KitchenConfig
from .drive_zone import KitchenDriveZone
from .placement import KitchenHumanPlacement
from .planning_scene import PlanningScenePool
from .scene import initial_positions, register_simulation_objects, scene_config

ROBOT_SIDE = "robot_side"
OPPOSITE_SIDE = "opposite_side"


class KitchenKinderBackend(KinderBackend):
    """One live kitchen scene. The cube's start is part of the scene, so a new seed
    rebuilds it, as the kitchen always did."""

    # The kitchen pick controller converges more slowly (damped endpoints, tighter
    # tolerances), so it keeps the step budget it was measured under.
    pick_step_limit: ClassVar[int] = 550
    # The pick planner's view of the mat edge: a 16 mm line, not the visual box.
    BARRIER_PLANNING_BOX: ClassVar[tuple[float, float, float]] = (0.016, 5.0, 0.001)
    # Seconds a human placement is left to settle before it is observed.
    PLACEMENT_SETTLE_S: ClassVar[float] = 0.8

    kitchen_config: KitchenConfig = Field(default_factory=KitchenConfig)
    env_id: str = "KitchenRecycling3D-v0"
    # The desk's and chairs' visual-only meshes (`furniture.VisualFurniture`). Only a
    # scene that will never be rendered may go without them, so the default keeps them;
    # rendering a scene built without them raises rather than drawing a bare room.
    visual_furniture: bool = True

    # Seeded cube spots a human reset tries before giving up on a pickable one.
    RESET_CANDIDATES: ClassVar[int] = 50

    _scene_seed: int | None = PrivateAttr(default=None)
    _planning_scenes: PlanningScenePool | None = PrivateAttr(default=None)
    _placements: int = PrivateAttr(default=0)
    # Per human reset, how many unpickable seeded spots were rejected before one passed.
    reset_rejections: list[int] = Field(default_factory=list)

    def drive_zone(self) -> tuple[float, float, float, float]:
        lower, upper = KitchenDriveZone.bounds(config=self.kitchen_config)
        return (float(lower[0]), float(lower[1]), float(upper[0]), float(upper[1]))

    def _reset(self, *, seed: int) -> Tossing3DObservation:
        if self._raw_env is not None and self._scene_seed != seed:
            self.close()
        self._scene_seed = seed
        self._placements = 0
        return super()._reset(seed=seed)

    def _make_scene(self, *, api: Any) -> Any:
        from gymnasium.envs.registration import EnvSpec
        from kinder.envs.dynamic3d.envs import TidyBot3DConfig

        register_simulation_objects()
        config = scene_config(config=self.kitchen_config, seed=self._scene_seed or 0)

        def make(*, path: Path) -> Any:
            spec = EnvSpec(
                id=self.env_id,
                entry_point="hitl_pmp.environments.kitchen_recycling3d.furniture:"
                + ("RecyclingEnv" if self.visual_furniture else "PhysicsOnlyRecyclingEnv"),
                kwargs={
                    "task_config_path": str(path),
                    "scene_render_camera": "task_view",
                    "config": TidyBot3DConfig(use_arm_velocities=True),
                },
            )
            return api.kinder.make(
                spec,
                render_mode=self.render_mode,
                scene_bg=self.scene_bg,
                allow_state_access=self.allow_state_access,
            )

        if hasattr(os, "memfd_create"):
            descriptor = os.memfd_create("kitchen-recycling-scene", os.MFD_CLOEXEC)
            with os.fdopen(descriptor, "w+", encoding="utf-8") as stream:
                json.dump(config, stream)
                stream.flush()
                return make(path=Path(f"/proc/self/fd/{stream.fileno()}"))
        with tempfile.TemporaryDirectory(prefix="kitchen-recycling-") as directory:
            generated = Path(directory) / "scene.json"
            generated.write_text(json.dumps(config))
            return make(path=generated)

    def _build_pick_collision_sim(self, *, state: Any) -> Any:
        """Tossing3D's planning scene with the kitchen's tapered bin, wall for wall the
        boxes MuJoCo simulates, the mat edge as a 16 mm line, and the robot wearing
        MuJoCo's camera bracelet. Every kitchen controller that holds a robot plans in
        this sim: the pick (and its contact monitor and PickPlannable dry run), the toss
        and the open-gripper clearance; the base approach plans in 2-D only.

        Always a new scene. The pick, its dry run and the toss take theirs from
        `_pick_collision_sim`, which reuses one; the open-gripper clearance check adds
        bodies to its scene, so it builds its own here."""
        from hitl_pmp.environments.vision_bracelet_robot import VisionBraceletPyBulletSim

        sim = VisionBraceletPyBulletSim(state)
        self._furnish_planning_scene(sim, state)
        return sim

    def _furnish_planning_scene(self, sim: Any, state: Any) -> None:  # noqa: PLR0917
        TaperedBinCollision.add(
            sim=sim,
            name=self.bin_name,
            pose=self._bin_pose(state=state),
            geometry=TaperedBin.of(config=self.kitchen_config),
        )
        sim._bounding_boxes[self.barrier_name] = self.BARRIER_PLANNING_BOX  # noqa: SLF001

    def _bin_pose(self, *, state: Any) -> Any:
        from pybullet_helpers.geometry import Pose

        obj = state.get_object_from_name(self.bin_name)
        return Pose(
            tuple(state.get(obj, f) for f in ("x", "y", "z")),
            tuple(state.get(obj, f) for f in ("qx", "qy", "qz", "qw")),
        )

    def _pick_collision_sim(self, *, state: Any) -> Any:
        """`_build_pick_collision_sim`'s scene, reused (`planning_scene.py`): handed
        back exactly as a new one would be, so every plan is the one a new scene gives.
        A use ends when its user closes the scene (the dry run) or when the skill that
        took it returns (`run_controller`)."""
        with SkillTiming.phase_of(timing=self._timing, name="pick_collision_sim"):
            if self._planning_scenes is None:
                self._planning_scenes = PlanningScenePool(
                    furnish=self._furnish_planning_scene,
                    bin_name=self.bin_name,
                    pose_of=self._bin_pose,
                )
            return self._planning_scenes.checkout(state=state)

    def run_controller(
        self,
        *,
        module: str,
        key: str,
        object_names: Sequence[str],
        params: np.ndarray | None,
        limit: int,
        pybullet_sim: Any = None,
    ) -> ControllerRun:
        """Tossing3D's, returning every planning scene the skill took once it is done."""
        pool = self._planning_scenes
        mark = [] if pool is None else list(pool.lent)
        try:
            return super().run_controller(
                module=module,
                key=key,
                object_names=object_names,
                params=params,
                limit=limit,
                pybullet_sim=pybullet_sim,
            )
        finally:
            if self._planning_scenes is not None:
                self._planning_scenes.release_since(mark=mark)

    def close(self) -> None:
        if self._planning_scenes is not None:
            self._planning_scenes.dispose()
            self._planning_scenes = None
        super().close()

    def render(self, *, follow_robot: bool = False) -> np.ndarray:
        self._require_visual_furniture()
        return super().render(follow_robot=follow_robot)

    def set_substep_recording(self, *, enabled: bool) -> None:
        if enabled:
            self._require_visual_furniture()
        super().set_substep_recording(enabled=enabled)

    def _require_visual_furniture(self) -> None:
        if not self.visual_furniture:
            raise RuntimeError(
                "This kitchen scene was built without its visual furniture "
                "(visual_furniture=False) and must not be rendered; build one with it."
            )

    def _ground_controller(
        self,
        *,
        module: str,
        key: str,
        object_names: Sequence[str],
        state: Any,
        pybullet_sim: Any = None,
    ) -> Any:
        """Tossing3D's grounding, with the kitchen's pick and its drive-zone toss and
        pour approach."""
        from .controllers import (
            RecyclingMoveToTargetController,
            RecyclingPickController,
            RecyclingTossController,
        )

        action_dim = int(self._raw_env.action_space.shape[0])
        if module == "tossing" and key == "pick_cube":
            sim = (
                pybullet_sim if pybullet_sim is not None else self._pick_collision_sim(state=state)
            )
            pick_names = (self.robot_name, self.cube_name, self.barrier_name, self.bin_name)
            pick = RecyclingPickController(
                [state.get_object_from_name(name) for name in pick_names], pybullet_sim=sim
            )
            vars(pick)["ACTION_DIM"] = action_dim
            spec = self.kitchen_config.objects[0]
            pick.configure(
                config=self.kitchen_config,
                spec=spec,
                calibration=self.kitchen_config.calibration(name=spec.name, context="far"),
            )
            return pick
        if module == "tossing" and key == "move_to_toss_location_and_toss":
            toss_names = (self.robot_name, self.cube_name, self.barrier_name)
            toss = RecyclingTossController(
                tuple(state.get_object_from_name(name) for name in toss_names),
                pybullet_sim=self._pick_collision_sim(state=state),
            )
            vars(toss)["ACTION_DIM"] = action_dim
            toss.configure(config=self.kitchen_config)
            return TossSettlingController(backend=self, controller=toss)
        if module == "tossing" and key == "move_to_target":
            move = RecyclingMoveToTargetController(
                tuple(state.get_object_from_name(name) for name in object_names)
            )
            vars(move)["ACTION_DIM"] = action_dim
            move.configure(
                config=self.kitchen_config,
                barrier_box=(self.barrier_name, self.BARRIER_PLANNING_BOX),
            )
            return move
        return super()._ground_controller(
            module=module,
            key=key,
            object_names=object_names,
            state=state,
            pybullet_sim=pybullet_sim,
        )

    def pick_known_unplannable(self, *, features: Mapping[str, Mapping[str, float]]) -> bool:
        """Beyond the drive zone's pick reach. Not the barrier plane: the kitchen pick
        reaches cubes a little way onto the mats, and refusing those would mask it."""
        cube = features[self.cube_name]
        return not KitchenDriveZone.pick_reachable(
            config=self.kitchen_config, position=(cube["x"], cube["y"])
        )

    def base_stand_allowed(self, *, xy: Sequence[float]) -> bool:
        return KitchenDriveZone.allows(config=self.kitchen_config, position=xy)

    def reset_cube_and_bin(
        self, *, bin_region: dict[str, object] | None = None
    ) -> Tossing3DObservation:
        """A human puts the bin in `bin_region["kitchen_side"]`'s region and the cube on
        the kitchen floor; the robot stays where it is.

        The bin's spot is drawn from that side's probe-certified region
        (`bin_regions.py`) by scene seed and reset count, clear of the robot and the
        furniture. The cube goes to the first seeded spot where, with the bin already
        placed, `PickPlannable` holds -- the same reach gate and pick dry run the planner
        reads -- so a reset never hands the planner a cube it must reset again.
        """
        with SkillTiming.phase_of(timing=self._timing, name="reset_cube_and_bin"):
            side = OPPOSITE_SIDE if bin_region is None else str(bin_region["kitchen_side"])
            return self._place_cube_and_bin(side=side)

    def _bin_destination(self, *, side: str, state: Any) -> tuple[float, float, float]:
        if side not in (ROBOT_SIDE, OPPOSITE_SIDE):
            raise ValueError(f"unknown kitchen reset side {side!r}")
        robot = state.get_object_from_name(self.robot_name)
        robot_xy = (state.get(robot, "pos_base_x"), state.get(robot, "pos_base_y"))
        furniture = [
            tuple(float(state.get(obj, key)) for key in ("x", "y", "bb_x", "bb_y"))
            for obj in state
            if obj.type.name == "mujoco_static_collider"
        ]
        return KitchenBinRegions.human_destination(
            config=self.kitchen_config,
            side=Tossing3DSide(side),
            robot=robot_xy,
            furniture=furniture,
            seed=self._scene_seed or 0,
            index=self._placements,
        )

    def _place_cube_and_bin(self, *, side: str) -> Tossing3DObservation:
        config = self.kitchen_config
        state = self._require_state().copy()
        try:
            bin_xyz = self._bin_destination(side=side, state=state)
        except ValueError:
            if side not in (ROBOT_SIDE, OPPOSITE_SIDE):
                raise
            # The robot covers every robot-side spot: it steps back to its scene-start
            # pose, as it would for a person resetting the scene (Tossing3D's rule).
            self._move_robot_to_start_pose()
            state = self._require_state().copy()
            bin_xyz = self._bin_destination(side=side, state=state)
        robot = state.get_object_from_name(self.robot_name)
        robot_xy = np.array([state.get(robot, "pos_base_x"), state.get(robot, "pos_base_y")])
        spec = config.objects[0]
        for attempt, cube_xy in enumerate(
            self._recovery_positions(robot_xy=robot_xy, bin_xyz=bin_xyz)
        ):
            cube_xyz = (*cube_xy, spec.dimensions[2] / 2 + 0.002)
            self._write_placement(
                state=state, placements=((self.bin_name, bin_xyz), (self.cube_name, cube_xyz))
            )
            if self._pick_plannable_here():
                self.reset_rejections.append(attempt)
                self._placements += 1
                return self.observe()
            if self._timing is not None:
                self._timing.count(name="reset_cube_unpickable_rejected")
            if attempt + 1 >= self.RESET_CANDIDATES:
                break
        raise ValueError(
            f"No {side} reset placement among {self.RESET_CANDIDATES} seeded candidates "
            "where the pick dry run succeeds"
        )

    def _write_placement(
        self, *, state: Any, placements: Sequence[tuple[str, Sequence[float]]]
    ) -> None:
        state = state.copy()
        for name, xyz in placements:
            obj = state.get_object_from_name(name)
            for feature, value in zip(
                ("x", "y", "z", "qx", "qy", "qz", "qw"), (*xyz, 0, 0, 0, 1), strict=True
            ):
                state.set(obj, feature, float(value))
            for feature in state.type_features[obj.type]:
                if feature in ("vx", "vy", "vz", "wx", "wy", "wz"):
                    state.set(obj, feature, 0.0)
        self._object_centric().set_state(state)
        self._state = state
        self._settle(seconds=self.PLACEMENT_SETTLE_S)

    def _pick_plannable_here(self) -> bool:
        """`PickPlannable` exactly as the abstraction will evaluate it on the settled
        placement: the drive-zone reach gate, then the kitchen pick's dry run."""
        live = self._require_state()
        features = {
            obj.name: {name: float(live.get(obj, name)) for name in live.type_features[obj.type]}
            for obj in live
        }
        if self.pick_known_unplannable(features=features):
            return False
        return self.pick_cube_plan_failure(state=live) is None

    def _recovery_positions(
        self, *, robot_xy: np.ndarray, bin_xyz: Sequence[float]
    ) -> Iterator[tuple[float, float]]:
        """Seeded kitchen-floor spots clear of the robot and the bin, drawn the way the
        scene scatters its start positions, in a fixed order per seed and reset."""
        config = self.kitchen_config
        spec = config.objects[0]
        base = (self._scene_seed or 0) + 100000 + self._placements * 1000
        for offset in range(1000):
            (position,) = initial_positions(
                config=config.model_copy(update={"scatter_objects": True}), seed=base + offset
            )[:1]
            xy = np.array(position)
            if np.linalg.norm(xy - robot_xy) < config.robot_radius + 0.2:
                continue
            if not KitchenHumanPlacement.recovery_clear_of_bin(
                config=config,
                position=xy,
                item_radius=float(np.linalg.norm(spec.dimensions)) / 2,
                bin_pose=(bin_xyz, (0.0, 0.0, 0.0, 1.0)),
            ):
                continue
            yield float(xy[0]), float(xy[1])

    def _settle(self, *, seconds: float) -> None:
        """Hold the robot still while a placement settles, stepping the raw scene so the
        settling is charged to no skill's control steps."""
        object_centric = self._object_centric()
        action = np.zeros(self._raw_env.action_space.shape, dtype=np.float32)
        robot = self._require_state().get_object_from_name(self.robot_name)
        # Keep the gripper as it was: a reset leaves the hand alone.
        action[10] = float(self._require_state().get(robot, "pos_gripper") > 0.2)
        ticks = max(1, int(np.ceil(seconds * object_centric._robot_env.control_frequency)))  # noqa: SLF001
        for _ in range(ticks):
            observation, _, _, _, _ = self._raw_env.step(action)
            self._state = self._raw_env.observation_space.devectorize(observation)
