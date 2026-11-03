"""Native KINDER scene with measured supports, movable rack and explicit human costs."""

import json
import os
import tempfile
from collections.abc import Callable
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import Field, PrivateAttr
from scipy.spatial.transform import Rotation

from hitl_pmp.core.control_steps import ControlStepObserverFailed
from hitl_pmp.environments.sweepoffshelf3d.backend import TASK_ID, SweepOffShelf3DBackend

from .config import RackServingConfig
from .geometry import install_geometry
from .layout import RackLayout
from .objects import RackServing3DObjects as Objects
from .predicates import AbstractAtom, RackServing3DAtoms
from .scene import build_scene
from .semantics import (
    CircularSupport,
    FoodLocation,
    FoodMeasurement,
    RackServingLedger,
    RectangularSupport,
    ServingRegion,
    SupportContact,
    SupportRegion,
    WellSupport,
    classify_food,
)

FOOTPRINT_SAMPLES = 64
MAX_RETURN_LAYOUTS = 300


class RackServingBackend(SweepOffShelf3DBackend):
    config: RackServingConfig = Field(default_factory=RackServingConfig)
    ledger: RackServingLedger = Field(default_factory=RackServingLedger)
    _scene_file: Path | None = PrivateAttr(default=None)
    _time_origin: float = PrivateAttr(default=0)
    _tick_observer: Callable[[], None] | None = PrivateAttr(default=None)
    _control_steps: int = PrivateAttr(default=0)
    _tool_home: Callable[[], bool] | None = PrivateAttr(default=None)
    _rng: Any = PrivateAttr(default=None)

    def env(self) -> Any:
        if self._env is None:
            import kinder
            from kinder.envs.dynamic3d.envs import TidyBot3DConfig

            rendering = {
                name: os.environ.get(name, "egl") for name in ("MUJOCO_GL", "PYOPENGL_PLATFORM")
            }
            kinder.register_all_environments()
            os.environ.update(rendering)
            install_geometry()
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as stream:
                json.dump(build_scene(config=self.config), stream)
                self._scene_file = Path(stream.name)
            self._env = kinder.make(
                TASK_ID,
                task_config_path=str(self._scene_file),
                render_mode="rgb_array",
                allow_state_access=True,
                scene_bg=self.scene_bg,
                scene_render_camera="task_view",
                config=TidyBot3DConfig(use_arm_velocities=False),
            )
            original_step = self._env.step

            def observed(*args: Any, **kwargs: Any) -> Any:
                self._notify_step(observer=self._before_step)
                result = original_step(*args, **kwargs)
                self._control_steps += 1
                self._notify_step(observer=self._after_step)
                self._notify_step(observer=self._tick_observer)
                return result

            self._env.step = observed
        return self._env

    def reset(self, *, seed: int) -> None:
        super().reset(seed=seed)
        for geom in self._pads[0] | self._pads[1]:
            self.model().geom_condim[geom] = self.config.gripper_contact_dimension
        self._rng = np.random.default_rng(seed)
        self.ledger = RackServingLedger(
            mode=self.config.mode,
            cooling_seconds=self.config.cooling_seconds,
            elapsed_seconds=self.config.initial_elapsed_seconds,
            wait_seconds=self.config.wait_seconds,
            human_interruption_cost=self.config.human_interruption_cost,
        )
        self._time_origin = float(self.data().time)
        core = self.core()
        core.shelf_top_z = lambda *, shelf: (
            self.support(name="rack_0").top_z if shelf == 1 else self.config.counter_height
        )
        core.shelf_front_x = lambda: self.support(name="rack_0").x_min
        core.box_interior = self.box_interior
        core.cube_in_box = lambda name: self.settled_on_tray(name=name)
        self._tool_home = None
        # Every episode starts from its own seed's draw of the shared rack layout.
        self._commit(placements=self.food_placements(names=list(self.cube_names)))

    def sync_clock(self) -> None:
        elapsed = float(self.data().time) - self._time_origin
        if elapsed < -1e-8:
            raise RuntimeError("simulation clock moved backwards without an episode reset")
        self.ledger = self.ledger.advance(seconds=max(0, elapsed))
        self._time_origin = float(self.data().time)

    def set_tick_observer(self, *, observer: Callable[[], None] | None) -> None:
        self._tick_observer = observer

    def control_steps(self) -> int:
        """Completed calls to physics, even if an after-step observer interrupts."""
        return self._control_steps

    @staticmethod
    def _notify_step(*, observer: Callable[[], None] | None) -> None:
        if observer is not None:
            try:
                observer()
            except Exception as error:
                raise ControlStepObserverFailed() from error

    def body_geoms(self, *, name: str) -> set[int]:
        model = self.model()
        body = model.body(name).id
        return {
            int(geom)
            for geom in np.flatnonzero(model.geom_bodyid == body)
            if model.geom_contype[geom] or model.geom_conaffinity[geom]
        }

    def corners(self, *, name: str) -> np.ndarray:
        model, data = self.model(), self.data()
        points: list[np.ndarray] = []
        for geom in self.body_geoms(name=name):
            rotation = data.geom_xmat[geom].reshape(3, 3)
            size = model.geom_size[geom].copy()
            if int(model.geom_type[geom]) == 3:
                size = np.array([size[0], size[0], size[0] + size[1]])
            elif int(model.geom_type[geom]) == 5:
                size = np.array([size[0], size[0], size[1]])
            elif int(model.geom_type[geom]) == 2:
                size[:] = size[0]
            elif int(model.geom_type[geom]) == 4:
                size = np.sqrt(rotation**2 @ size**2)
                rotation = np.eye(3)
            points.extend(
                data.geom_xpos[geom] + np.array(sign) * size @ rotation.T
                for sign in product((-1, 1), repeat=3)
            )
        return np.asarray(points)

    def footprint(self, *, name: str) -> np.ndarray:
        """Points whose convex hull encloses the body's horizontal shadow.

        An ellipsoid's shadow is an exact ellipse; its sampled polygon is scaled
        out by sec(pi/N) so the polygon circumscribes, never undercuts, it.
        Other geoms contribute their (conservative) bounding-box corners.
        """
        model, data = self.model(), self.data()
        angles = np.linspace(0, 2 * np.pi, FOOTPRINT_SAMPLES, endpoint=False)
        circle = np.stack([np.cos(angles), np.sin(angles)]) / np.cos(np.pi / FOOTPRINT_SAMPLES)
        points: list[np.ndarray] = []
        for geom in sorted(self.body_geoms(name=name)):
            if int(model.geom_type[geom]) != 4:
                continue
            rotation = data.geom_xmat[geom].reshape(3, 3)
            shadow = (rotation @ np.diag(np.square(model.geom_size[geom])) @ rotation.T)[:2, :2]
            points.extend((data.geom_xpos[geom][:2, None] + np.linalg.cholesky(shadow) @ circle).T)
        others = [geom for geom in self.body_geoms(name=name) if int(model.geom_type[geom]) != 4]
        if others or not points:
            points.extend(self.corners(name=name)[:, :2])
        return np.asarray(points)

    def support(self, *, name: str) -> ServingRegion:
        if name == "box_0" and self.config.serving_target == "round_plate":
            return self.plate_support()
        if name == "box_0" and self.config.serving_target == "lipped_platter":
            return self.platter_support()
        if name == "box_0" and self.config.serving_target == "corelle_plate":
            return self.corelle_support()
        if name == "counter":
            config = self.config
            return SupportRegion(
                x_min=config.counter_x_min,
                x_max=config.counter_x_max,
                y_min=config.counter_y_min,
                y_max=config.counter_y_max,
                top_z=config.counter_height,
            )
        corners = self.corners(name=name)
        lower, upper = corners.min(axis=0), corners.max(axis=0)
        return SupportRegion(
            x_min=lower[0], x_max=upper[0], y_min=lower[1], y_max=upper[1], top_z=upper[2]
        )

    def plate_support(self) -> CircularSupport:
        """Live circle of the plate's single cylinder geom.

        A tilted plate's shadow is an ellipse whose minor semi-axis is
        radius * |cos tilt|; using that as the radius keeps containment strict.
        """
        model, data = self.model(), self.data()
        geoms = [geom for geom in self.body_geoms(name="box_0") if int(model.geom_type[geom]) == 5]
        if len(geoms) != 1:
            raise RuntimeError("the round serving plate must be exactly one cylinder geom")
        geom = geoms[0]
        rotation = data.geom_xmat[geom].reshape(3, 3)
        return CircularSupport(
            center_x=float(data.geom_xpos[geom][0]),
            center_y=float(data.geom_xpos[geom][1]),
            radius=float(model.geom_size[geom][0] * abs(rotation[2, 2])),
            top_z=float(self.corners(name="box_0")[:, 2].max()),
        )

    def corelle_support(self) -> CircularSupport:
        """Live circle of the rim's inner edge, at the top face of the plate's well.

        Only the well cylinder is read: food scores inside the rim's inner edge and
        resting on the well, never on the rim. A tilted well's shadow is an ellipse
        whose minor semi-axis is radius * |cos tilt|, which keeps containment strict.
        """
        model, data = self.model(), self.data()
        well = model.geom("box_0_well").id
        rotation = data.geom_xmat[well].reshape(3, 3)
        radius, half_height = model.geom_size[well][:2]
        return WellSupport(
            center_x=float(data.geom_xpos[well][0]),
            center_y=float(data.geom_xpos[well][1]),
            radius=float(radius * abs(rotation[2, 2])),
            top_z=float(data.geom_xpos[well][2] + rotation[2, 2] * half_height),
        )

    def serving_near_edge_x(self) -> float:
        """World x of the serving object's edge nearest the rack, where a sweep's d is measured.

        For the Corelle plate it is the rim's outer edge on the live well centre line;
        otherwise the live footprint's maximum x.
        """
        if self.config.serving_target == "corelle_plate":
            return self.corelle_support().center_x + self.config.corelle_diameter / 2
        return float(self.corners(name="box_0")[:, 0].max())

    def platter_support(self) -> RectangularSupport:
        """Live inner rectangle of the platter's lip, at the top of its floor slab.

        A tilted platter's interior shadow shrinks by at most |cos tilt| on each
        axis; scaling both inner half extents by that keeps containment strict.
        """
        model, data = self.model(), self.data()
        floor = model.geom("box_0_platter_floor").id
        rotation = data.geom_xmat[floor].reshape(3, 3)
        half_x, half_y, _ = model.geom_size[floor]
        wall = self.config.platter_lip_wall
        shrink = abs(rotation[2, 2])
        return RectangularSupport(
            center_x=float(data.geom_xpos[floor][0]),
            center_y=float(data.geom_xpos[floor][1]),
            yaw=float(np.arctan2(rotation[1, 0], rotation[0, 0])),
            half_x=float((half_x - wall) * shrink),
            half_y=float((half_y - wall) * shrink),
            top_z=float(data.geom_xpos[floor][2] + np.abs(rotation[2]) @ model.geom_size[floor]),
        )

    def box_interior(self) -> tuple[tuple[float, float, float, float], float]:
        tray = self.support(name="box_0")
        return (tray.x_min, tray.y_min, tray.x_max, tray.y_max), tray.top_z

    def supporting_bodies(self, *, name: str) -> set[str]:
        model, data = self.model(), self.data()
        body = model.body(name).id
        supports: set[str] = set()
        for contact in data.contact[: data.ncon]:
            if contact.dist > 0.001:
                continue
            for mine, other in ((contact.geom1, contact.geom2), (contact.geom2, contact.geom1)):
                if model.geom_bodyid[mine] == body and contact.pos[2] < data.xpos[body, 2]:
                    other_body = int(model.geom_bodyid[other])
                    supports.add(model.body(other_body).name or "world")
                    if int(model.geom_type[other]) == 0 and abs(data.geom_xmat[other, 8]) > 0.999:
                        supports.add("floor")
        return supports

    def location(
        self, *, name: str, _visiting: frozenset[str] = frozenset()
    ) -> FoodLocation | None:
        """Raw support/grasp category, independent of velocity or readiness.

        Raw OnTray includes sliding/rolling contact; delivery scoring uses
        settled_on_tray instead. Unsupported food is not relabeled as spilled.
        Food resting on other food counts as resting on served food when that
        food is itself raw OnTray; `_visiting` breaks mutual-support cycles.
        """
        corners = self.corners(name=name)
        lower, upper = corners.min(axis=0), corners.max(axis=0)
        tray = self.support(name="box_0")
        measured = FoodMeasurement(
            position=tuple((lower + upper) / 2),
            half_extents=tuple((upper - lower) / 2),
            held=self.held(name=name),
            footprint=(
                tuple((float(x), float(y)) for x, y in self.footprint(name=name))
                if isinstance(tray, CircularSupport | RectangularSupport)
                else None
            ),
        )
        supports = self.supporting_bodies(name=name)
        visiting = _visiting | {name}
        on_served_food = any(
            self.location(name=other, _visiting=visiting) == "OnTray"
            for other in sorted(supports & set(self.cube_names) - visiting)
        )
        contact_names: dict[str, SupportContact] = {
            "rack_0": "rack",
            "box_0": "tray",
            "cupboard_1": "counter",
            "world": "floor",
            "floor": "floor",
            "ground": "floor",
        }
        location = classify_food(
            food=measured,
            rack=self.support(name="rack_0"),
            tray=tray,
            counter=self.support(name="counter"),
            support_contacts=frozenset(
                contact_names[name] for name in supports if name in contact_names
            )
            | ({"served_food"} if on_served_food else set()),
            support_tolerance=0.009,
        )
        if location == "HoldingFood" or location is None:
            return location
        expected = {"OnRack": "rack_0", "OnTray": "box_0", "SpilledCounter": "cupboard_1"}
        if location == "OnTray" and on_served_food and isinstance(tray, WellSupport):
            return location
        if location in expected and expected[location] not in supports:
            return None
        if location == "SpilledFloor" and not supports:
            return None
        return location

    def features(self) -> dict[str, np.ndarray]:
        model, data = self.model(), self.data()
        base = [
            data.qpos[model.jnt_qposadr[model.joint(f"robot_joint_{axis}").id]]
            for axis in ("x", "y", "th")
        ]
        features = {"robot": np.array([*base, self.grip])}
        for name in self.cube_names:
            position, quaternion = self.core().get_object(name).get_pose()
            features[name] = np.r_[position, quaternion]
        position, quaternion = self.core().get_object("wiper_0").get_pose()
        yaw = Rotation.from_quat(np.asarray(quaternion)[[1, 2, 3, 0]]).as_euler("xyz")[2]
        features["wiper_0"] = np.r_[position, yaw]
        for name in ("rack_0", "box_0"):
            region = self.support(name=name)
            features[name] = np.array([
                region.x_min,
                region.y_min,
                region.x_max,
                region.y_max,
                region.top_z,
            ])
        return features

    def abstract_atoms(self) -> frozenset[AbstractAtom]:
        """OnTray certifies settled delivery for practice effects and goal tests."""
        self.sync_clock()
        atoms: set[AbstractAtom] = set()
        held = False
        for food in Objects.food:
            location = self.location(name=food.name)
            if location == "HoldingFood":
                atoms.add((location, ("robot", food.name)))
                held = True
            elif location == "OnTray":
                if self.settled_on_tray(name=food.name):
                    atoms.add((location, (food.name, "box_0")))
            elif location == "OnRack":
                atoms.add((location, (food.name, "rack_0")))
            elif location is not None:
                atoms.add((location, (food.name,)))
        if self.held(name="wiper_0"):
            atoms.add(("HoldingScraper", ("robot", "wiper_0")))
            held = True
        elif self._tool_home is not None and self._tool_home():
            atoms.add(("ScraperHome", ("wiper_0",)))
        else:
            atoms.add(("ScraperAway", ("wiper_0",)))
            if self.data().xpos[self.model().body("wiper_0").id, 2] < 0.15:
                atoms.add(("ScraperOnFloor", ("wiper_0",)))
        if not held:
            atoms.add(("HandEmpty", ("robot",)))
        return RackServing3DAtoms.with_summaries(
            atoms=frozenset(atoms),
            ledger=self.ledger,
            available_food=frozenset(
                name for name in self.cube_names if self.food_available(name=name)
            ),
        )

    def food_available(self, *, name: str) -> bool:
        """Physical support permits a pick even when strict delivery containment fails."""
        supports = {"rack_0", "box_0", "cupboard_1", "world", "floor", "ground", *self.cube_names}
        return not self.held(name=name) and bool(self.supporting_bodies(name=name) & supports)

    def task_succeeded(self) -> bool:
        self.sync_clock()
        return self.ledger.ready and all(
            self.settled_on_tray(name=name) for name in self.cube_names
        )

    def settled_on_tray(self, *, name: str) -> bool:
        """Unheld, supported delivery with finite low linear and angular speed.

        Retain the rack's 0.02 m/s limit and the kitchen prototype's 0.5 rad/s
        angular limit. This is a current-state test, not a dwell-time guarantee.
        """
        if self.location(name=name) != "OnTray":
            return False
        model, data = self.model(), self.data()
        body = model.body(name).id
        joint = model.body_jntadr[body]
        address = model.jnt_dofadr[joint]
        velocity = data.qvel[address : address + 6]
        return bool(
            np.isfinite(velocity).all()
            and np.linalg.norm(velocity[:3]) <= 0.02
            and np.linalg.norm(velocity[3:]) <= 0.5
        )

    def batch_return(self) -> None:
        self.sync_clock()
        self.ledger = self.ledger.charge_human(intervention="human_batch_return")
        self._commit(placements=self.food_placements(names=list(self.cube_names)))

    def food_placements(self, *, names: list[str]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Place `names` on the rack as one draw of the shared layout distribution,
        conditioned on every other dumpling's current centre; a layout that collides
        with anything else in the scene is redrawn whole."""
        import mujoco

        model, scratch, addresses, _ = self._scratch(names=names)
        rack = self.support(name="rack_0")
        support = self.body_geoms(name="rack_0")
        others = [self.food_centre(name=other) for other in self.cube_names if other not in names]
        height = rack.top_z + self.config.dumpling_height / 2 + 0.002
        for _ in range(MAX_RETURN_LAYOUTS):
            layout = RackLayout.sample(rng=self._rng, rack=rack, count=len(names), others=others)
            placed = {
                name: (np.array([x, y, height]), RackLayout.quaternion(yaw=yaw))
                for name, (x, y, yaw) in zip(names, layout, strict=True)
            }
            for name, (position, quaternion) in placed.items():
                scratch.qpos[addresses[name] : addresses[name] + 7] = np.r_[position, quaternion]
            mujoco.mj_forward(model, scratch)
            if not any(
                self._blocked(model=model, data=scratch, body=model.body(name).id, support=support)
                for name in names
            ):
                return placed
        raise RuntimeError("no collision-free food return placement")

    def rack_put_back_draws(self, *, name: str, count: int) -> list[tuple[float, float]]:
        """`count` independent draws of where the robot puts `name` back: the shared
        layout distribution conditioned on every other dumpling's current centre,
        tried in order until one is reachable."""
        rack = self.support(name="rack_0")
        others = [self.food_centre(name=other) for other in self.cube_names if other != name]
        draws = []
        for _ in range(count):
            ((x, y, _),) = RackLayout.sample(rng=self._rng, rack=rack, count=1, others=others)
            draws.append((x, y))
        return draws

    def food_centre(self, *, name: str) -> tuple[float, float]:
        corners = self.corners(name=name)
        centre = (corners.min(axis=0) + corners.max(axis=0)) / 2
        return float(centre[0]), float(centre[1])

    def close(self) -> None:
        super().close()
        if self._scene_file is not None:
            self._scene_file.unlink(missing_ok=True)
            self._scene_file = None
