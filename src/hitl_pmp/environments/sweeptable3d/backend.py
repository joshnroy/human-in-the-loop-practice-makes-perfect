"""Seeded ball layouts, a fixed tray, and one-attempt sweep task semantics."""

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from pydantic import Field, PrivateAttr
from scipy.spatial.transform import Rotation

from hitl_pmp.environments.sweepoffshelf3d.backend import (
    TASK_ID,
    OrientedBox,
    SweepOffShelf3DBackend,
)
from hitl_pmp.environments.sweepoffshelf3d.scripted_skills import ShelfSkills, SkillFailed

from .objects import SweepTable3DObjects as O
from .predicates import SweepTable3DAtoms


class GroundContactFailure(SkillFailed):
    """A ball contacted the ground during a task."""


class SweepTable3DBackend(SweepOffShelf3DBackend):
    scene_path: Path = Field(default_factory=lambda: Path(__file__).with_name("scene.json"))
    fixed_bin_offset: float = Field(default=-0.18, ge=-0.40, le=0.40)
    ball_names: ClassVar[tuple[str, ...]] = tuple(ball.name for ball in O.balls)
    cube_names: ClassVar[tuple[str, ...]] = ball_names

    _ground_failed: bool = PrivateAttr(default=False)
    _sweep_attempted: bool = PrivateAttr(default=False)
    _sweep_running: bool = PrivateAttr(default=False)
    _sweep_succeeded: bool = PrivateAttr(default=False)
    _balls_restored: bool = PrivateAttr(default=False)
    _reset_had_off_table: bool = PrivateAttr(default=False)
    _monitor_ground: bool = PrivateAttr(default=False)
    _resetting: bool = PrivateAttr(default=False)
    _ground_geoms: frozenset[int] = PrivateAttr(default=frozenset())
    _cube_geoms: dict[int, str] = PrivateAttr(default_factory=dict)
    _ground_cubes: set[str] = PrivateAttr(default_factory=set)
    _home_check: Callable[[], bool] | None = PrivateAttr(default=None)
    _cube_rng: np.random.Generator = PrivateAttr()
    _observed_sim: Any = PrivateAttr(default=None)
    _original_sim_step: Callable[..., Any] | None = PrivateAttr(default=None)
    _physics_observer: Callable[..., Any] | None = PrivateAttr(default=None)

    def env(self) -> Any:
        if self._env is None:
            import kinder
            from kinder.envs.dynamic3d.envs import TidyBot3DConfig

            rendering = {
                key: os.environ.get(key, "egl") for key in ("MUJOCO_GL", "PYOPENGL_PLATFORM")
            }
            kinder.register_all_environments()
            os.environ.update(rendering)
            from .balls import BallScene
            from .geometry import CapsuleWiperGeometry

            BallScene.install()
            CapsuleWiperGeometry.install()
            self._env = kinder.make(
                TASK_ID,
                task_config_path=str(self.scene_path),
                render_mode="rgb_array",
                allow_state_access=True,
                scene_bg=self.scene_bg,
                scene_render_camera="task_view",
                config=TidyBot3DConfig(use_arm_velocities=False),
            )
            OrientedBox.install(core=self.core())
            self._observe_steps(env=self._env)
        return self._env

    def _observe_steps(self, *, env: Any) -> None:
        step = env.step

        def observed(*args: Any, **kwargs: Any) -> Any:
            if self._before_step is not None:
                self._before_step()
            result = step(*args, **kwargs)
            if self._monitor_ground:
                self.observe_ground_contacts()
            if self._on_tick is not None:
                self._on_tick()
            if self._after_step is not None:
                self._after_step()
            if self._monitor_ground and self._ground_failed and not self._resetting:
                raise GroundContactFailure(f"ball touched ground: {sorted(self._ground_cubes)}")
            return result

        env.step = observed

    def _observe_physics_steps(self) -> None:
        """Chain the scene's sim.step; each native call advances one physics tick."""
        sim = self.sim()
        if sim is self._observed_sim:
            return
        self._detach_physics_observer()
        step = sim.step

        def observed(*args: Any, **kwargs: Any) -> Any:
            result = step(*args, **kwargs)
            if self._monitor_ground:
                self.observe_ground_contacts()
            return result

        self._observed_sim = sim
        self._original_sim_step = step
        self._physics_observer = observed
        sim.step = observed

    def _detach_physics_observer(self) -> None:
        if self._observed_sim is not None and self._observed_sim.step is self._physics_observer:
            self._observed_sim.step = self._original_sim_step
        self._observed_sim = None
        self._original_sim_step = None
        self._physics_observer = None

    def close(self) -> None:
        self._monitor_ground = False
        self._detach_physics_observer()
        super().close()

    def reset(self, *, seed: int) -> None:
        from .balls import BallScene

        self._monitor_ground = False
        self._resetting = False
        self._ground_failed = False
        self._ground_cubes = set()
        self._sweep_attempted = False
        self._sweep_running = False
        self._sweep_succeeded = False
        self._balls_restored = False
        self._reset_had_off_table = False
        self._home_check = None
        super().reset(seed=seed)
        self._observe_physics_steps()
        self._cube_rng = np.random.default_rng(np.random.SeedSequence([seed, 1049]))
        box = self.core().get_object(self.box_name)
        position, quaternion = box.get_pose()
        position = np.asarray(position, dtype=float).copy()
        position[1] = self.table_center_y() + self.fixed_bin_offset
        self._commit(placements={self.box_name: (position, quaternion)})
        self._commit(placements=self._cube_placements(names=list(self.cube_names)))
        self._initialize_wiper()
        BallScene.configure(core=self.core())
        self._cache_contact_geoms()
        self.grip = 0.0
        self._monitor_ground = True

    def _initialize_wiper(self) -> None:
        from .wiper import TableWiper

        wiper = TableWiper(shelf=ShelfSkills(env=self.env()))
        pose = wiper.rest_pose()
        self._check_placement(
            name=self.wiper_name,
            position=pose[:3, 3],
            quaternion=Rotation.from_matrix(pose[:3, :3]).as_quat()[[3, 0, 1, 2]],
        )
        wiper.initialize_rest()
        self.set_home_check(check=wiper.at_home)

    def _cache_contact_geoms(self) -> None:
        model = self.model()
        owners = {
            geom: self._root_name(model=model, body=int(model.geom_bodyid[geom]))
            for geom in range(model.ngeom)
        }
        self._cube_geoms = {
            geom: owner for geom, owner in owners.items() if owner in self.cube_names
        }
        self._ground_geoms = frozenset(
            geom
            for geom in range(model.ngeom)
            if self._is_ground_geom(model=model, geom=geom, owner=owners[geom])
        )
        if not self._ground_geoms:
            raise RuntimeError("No physical ground collision geometry found")

    @staticmethod
    def _root_name(*, model: Any, body: int) -> str:
        while int(model.body_parentid[body]) != 0:
            body = int(model.body_parentid[body])
        return str(model.body(body).name)

    def _is_ground_geom(self, *, model: Any, geom: int, owner: str) -> bool:
        if owner in (*self.cube_names, self.wiper_name, self.box_name):
            return False
        if owner.startswith(("robot", "cupboard", "table")):
            return False
        body = int(model.geom_bodyid[geom])
        if int(model.body_weldid[body]) != 0:
            return False
        if not (model.geom_contype[geom] or model.geom_conaffinity[geom]):
            return False
        name = (model.geom(geom).name or "").lower()
        return (
            int(model.geom_type[geom]) == 0
            or name == "floor"
            or name.startswith(("floor_", "ground"))
        )

    def table_center_y(self) -> float:
        board = self.core()._shelf_geoms()[1]
        return float(self.data().geom_xpos[board][1])

    def observe_ground_contacts(self) -> None:
        self._ground_cubes.update(self._live_ground_cubes())
        self._ground_failed |= bool(self._ground_cubes)

    def _live_ground_cubes(self) -> set[str]:
        cubes = set()
        data = self.data()
        for contact in data.contact[: data.ncon]:
            if not np.isfinite(contact.dist) or contact.dist > 0:
                continue
            for cube_geom, support in (
                (int(contact.geom1), int(contact.geom2)),
                (int(contact.geom2), int(contact.geom1)),
            ):
                if cube_geom in self._cube_geoms and support in self._ground_geoms:
                    cubes.add(self._cube_geoms[cube_geom])
        return cubes

    def task_failed(self) -> bool:
        return self._ground_failed or (
            self._sweep_attempted and not self._sweep_running and not self._sweep_succeeded
        )

    def task_succeeded(self) -> bool:
        return self._sweep_succeeded and not self._ground_failed

    def begin_sweep(self) -> None:
        """Consume the attempt before motion, including strokes that abort."""
        if self._ground_failed or self._sweep_attempted:
            raise SkillFailed("sweep attempt consumed; fully restore the practice scene")
        if not self._all_on_table(atoms=self._physical_atoms()):
            raise SkillFailed("sweep requires all five balls on the table")
        self._sweep_attempted = True
        self._sweep_running = True
        self._sweep_succeeded = False
        self._balls_restored = False

    def finish_sweep(self, *, completed: bool) -> None:
        """Only the completed stroke can earn settled, all-five capture."""
        if not self._sweep_running:
            return
        self._sweep_running = False
        self.observe_ground_contacts()
        self._sweep_succeeded = completed and not self._ground_failed and self._capture_settled()

    def _capture_settled(self) -> bool:
        from .balls import BallScene

        core = self.core()
        return all(
            not self.held(name=name) and BallScene.in_tray(core=core, name=name)
            for name in self.ball_names
        ) and BallScene.settled(core=core, names=self.ball_names)

    def ground_cubes(self) -> tuple[str, ...]:
        return tuple(sorted(self._ground_cubes))

    def set_home_check(self, *, check: Callable[[], bool]) -> None:
        self._home_check = check

    def wiper_at_home(self, *, front: float, wiper_board: float) -> bool:
        del front, wiper_board
        return self._home_check is not None and self._home_check()

    def abstract_atoms(self) -> frozenset[tuple[str, tuple[str, ...]]]:
        atoms = set(self._physical_atoms())
        if not self._ground_failed and not self._sweep_attempted:
            atoms.add(("TaskActive", ()))
        if self._balls_restored and self._all_on_table(atoms=frozenset(atoms)):
            atoms.add(("BallsRestored", ()))
        if self.task_succeeded():
            atoms.add(("SweepSucceeded", ()))
        return SweepTable3DAtoms.with_summaries(atoms=frozenset(atoms))

    def _physical_atoms(self) -> frozenset[tuple[str, tuple[str, ...]]]:
        from .balls import BallScene

        core = self.core()
        top = core.shelf_top_z(shelf=1)
        front = core.shelf_front_x()
        atoms: set[tuple[str, tuple[str, ...]]] = set()
        grounded = self._live_ground_cubes()
        holding = False
        for name in self.ball_names:
            if self.held(name=name):
                atoms.add(("HoldingCube", (O.robot.name, name)))
                holding = True
                continue
            if name in grounded:
                atoms.add(("OnFloor", (name,)))
            elif BallScene.in_tray(core=core, name=name):
                atoms.add(("InBox", (name, self.box_name)))
            elif self._on_table(name=name):
                atoms.add(("OnTable", (name,)))
        if self.held(name=self.wiper_name):
            atoms.add(("HoldingWiper", (O.robot.name, self.wiper_name)))
            holding = True
        else:
            home = self.wiper_at_home(front=front, wiper_board=top)
            atoms.add(("WiperHome" if home else "WiperOnFloor", (self.wiper_name,)))
        if not holding:
            atoms.add(("HandEmpty", (O.robot.name,)))
        return frozenset(atoms)

    def _on_table(self, *, name: str) -> bool:
        board = self.core()._shelf_geoms()[1]
        data = self.data()
        position = np.asarray(self.core().get_object(name).get_pose()[0], dtype=float)
        local = data.geom_xmat[board].reshape(3, 3).T @ (position - data.geom_xpos[board])
        half_size = self.model().geom_size[board]
        return bool(
            np.isfinite(local).all()
            and np.all(np.abs(local[:2]) + O.ball_radius <= half_size[:2])
            and abs(local[2] - half_size[2] - O.ball_radius) < 0.003
        )

    def _cube_placements(self, *, names: list[str]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        import mujoco

        if not set(names) <= set(self.cube_names):
            raise ValueError("compact placements require ball names from this scene")
        top = self.core().shelf_top_z(shelf=1)
        front = self.core().shelf_front_x()
        center = self.table_center_y()
        model, data, addresses, support = self._scratch(names=names)
        parked = [
            np.asarray(self.core().get_object(name).get_pose()[0][:2])
            for name in self.cube_names
            if name not in names
        ]
        placements = {}
        for name in names:
            body = model.body(name).id
            address = addresses[name]
            for _ in range(400):
                position = np.array([
                    self._cube_rng.uniform(front + 0.065, 1.555),
                    self._cube_rng.uniform(center - 0.07, center + 0.07),
                    top + 0.0105,
                ])
                if any(np.linalg.norm(position[:2] - other) < 0.035 for other in parked):
                    continue
                quaternion = np.array([1.0, 0.0, 0.0, 0.0])
                data.qpos[address : address + 7] = np.r_[position, quaternion]
                mujoco.mj_forward(model, data)
                if self._blocked(model=model, data=data, body=body, support=support):
                    continue
                placements[name] = (position, quaternion)
                parked.append(position[:2])
                break
            else:
                raise RuntimeError("No collision-free compact ball layout found; reset not applied")
        return placements

    def _check_placement(self, *, name: str, position: np.ndarray, quaternion: np.ndarray) -> None:
        import mujoco

        model, data, addresses, support = self._scratch(names=[name])
        address = addresses[name]
        data.qpos[address : address + 7] = np.r_[position, quaternion]
        mujoco.mj_forward(model, data)
        if self._blocked(model=model, data=data, body=model.body(name).id, support=support):
            raise SkillFailed(f"No collision-free rest placement for {name}; reset not applied")

    def robot_reset_xy(self, *, cube: str) -> tuple[float, float]:
        """Sample a safe compact target without changing any live object pose."""
        position, _ = self._cube_placements(names=[cube])[cube]
        return float(position[0]), float(position[1])

    def reset_cubes(self) -> list[str]:
        moved = super().reset_cubes()
        if moved:
            self._sweep_succeeded = False
            self._balls_restored = True
            self._clear_failure_if_all_on_table()
        return moved

    def reset_wiper(self) -> None:
        if any(self.held(name=name) for name in self.ball_names):
            raise SkillFailed("cannot reset the wiper while holding a ball")
        resetting = self._resetting
        self._resetting = True
        try:
            self._initialize_wiper()
        finally:
            self._resetting = resetting
        if not resetting:
            self._clear_failure_if_all_on_table()

    def begin_robot_reset(self, *, restores_balls: bool = False) -> None:
        if not self._resetting:
            self._sweep_succeeded = False
            atoms = self._physical_atoms()
            self._reset_had_off_table = restores_balls and any(
                ("OnTable", (name,)) not in atoms for name in self.ball_names
            )
        self._resetting = True

    def finish_robot_reset(self) -> None:
        if not self._resetting:
            return
        self._resetting = False
        self.observe_ground_contacts()
        if self._reset_had_off_table and self._all_on_table(atoms=self._physical_atoms()):
            self._balls_restored = True
        self._reset_had_off_table = False
        self._clear_failure_if_all_on_table()

    def _all_on_table(self, *, atoms: frozenset[tuple[str, tuple[str, ...]]]) -> bool:
        return all(("OnTable", (name,)) in atoms for name in self.ball_names)

    def _clear_failure_if_all_on_table(self) -> None:
        atoms = self._physical_atoms()
        if (
            self._balls_restored
            and self._all_on_table(atoms=atoms)
            and ("HandEmpty", (O.robot.name,)) in atoms
            and ("WiperHome", (self.wiper_name,)) in atoms
        ):
            self._ground_failed = False
            self._ground_cubes = set()
            self._sweep_attempted = False
            self._sweep_running = False
            self._sweep_succeeded = False
