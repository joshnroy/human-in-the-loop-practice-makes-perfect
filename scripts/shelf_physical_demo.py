"""Scripted native Shelf physical feasibility proof, separate from baseline scores."""

import argparse
import json
import pickle
import traceback
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
from pybullet_helpers.geometry import multiply_poses

# Import the EGL module before KINDER registration.
from kinder.envs.dynamic3d.envs import ObjectCentricTidyBot3DEnv
from kinder_models.dynamic3d.shelf.parameterized_skills import (
    PyBulletSim,
    create_lifted_controllers,
)
from kinder_models.dynamic3d.shelf.state_abstractions import (
    CupboardRealStateAbstractor,
    HandEmpty,
    Holding,
    OnFixture,
    OnGround,
)
from relational_structs import GroundAtom
import kinder


class Demo:
    def __init__(
        self, *, output: Path, seed: int, max_actions: int, spread_placement: bool = False
    ) -> None:
        self.output = output
        output.mkdir(parents=True, exist_ok=True)
        task = Path(kinder.__file__).parent / "envs/dynamic3d/tasks/Shelf3D/Shelf3D-o8.json"
        self.env = ObjectCentricTidyBot3DEnv(
            task_config_path=str(task),
            num_objects=8,
            allow_state_access=True,
            render_mode="rgb_array",
        )
        self.abstractor = CupboardRealStateAbstractor(self.env)
        self.state, _ = self.env.reset(seed=seed)
        self.initial = self.state.copy()
        self.sim = PyBulletSim(self.state, rendering=False)
        self.controllers = create_lifted_controllers(self.env.action_space, pybullet_sim=self.sim)
        self.rng = np.random.default_rng(seed)
        self.robot = self.state.get_object_from_name(self.env.robot_name)
        self.cupboard = self.state.get_object_from_name("cupboard_1")
        self.records = []
        self.planning = []
        self.ticks = 0
        self.max_actions = max_actions
        self.spread_placement = spread_placement
        self.writer = imageio.get_writer(
            str(output / "physical-demo.mp4"), fps=10, codec="libx264", quality=7
        )
        self.snapshot("initial")
        self.frame("Initial native Shelf3D-o8 state")

    def snapshot(self, name):
        with (self.output / f"{name}.pkl").open("wb") as stream:
            pickle.dump(
                {
                    "state": self.state,
                    "initial": self.initial,
                    "ticks": self.ticks,
                    "actions": self.records,
                },
                stream,
            )

    def frame(self, label):
        frames = self.env.render_all_cameras()
        keys = [k for k in frames if "overview" in k or "task_view" in k]
        if not keys:
            keys = list(frames)[:1]
        pixels = np.concatenate([frames[k] for k in keys], axis=1)
        image = Image.fromarray(pixels)
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, image.width, 36), fill="black")
        draw.text(
            (8, 5), f"{label} | action {len(self.records)} | tick {self.ticks} | 1x", fill="white"
        )
        self.writer.append_data(np.asarray(image))

    def atoms(self):
        return self.abstractor.state_abstractor(self.state).atoms

    def prepare(self, skill, cube):
        objects = [self.robot, cube] + ([self.cupboard] if skill == "place_shelf" else [])
        choices = []
        for trial in range(20 if self.spread_placement and skill == "place_shelf" else 3):
            controller = self.controllers[skill].ground(objects)
            try:
                params = controller.sample_parameters(self.state, self.rng)
                controller.reset(self.state, params)
                self.planning.append({
                    "skill": skill,
                    "cube": cube.name,
                    "trial": trial,
                    "params": params.tolist(),
                    "prepared": True,
                })
                if not self.spread_placement or skill != "place_shelf":
                    return controller, params
                self.sim.robot.set_joints(controller._current_arm_joint_plan[-1])
                predicted = multiply_poses(self.sim.get_ee_pose(), self.sim.base_link_to_held_obj)
                distances = []
                for i in range(1, 9):
                    other = self.state.get_object_from_name(f"cube{i}")
                    if other != cube and self.state.get(other, "z") > 0.3:
                        distances.append(
                            np.linalg.norm(
                                np.array(predicted.position[:2])
                                - np.array([self.state.get(other, "x"), self.state.get(other, "y")])
                            )
                        )
                score = min(distances) if distances else 1.0
                choices.append((score, controller, params))
            except BaseException as error:
                if isinstance(error, (KeyboardInterrupt, SystemExit)):
                    raise
                self.planning.append({
                    "skill": skill,
                    "cube": cube.name,
                    "trial": trial,
                    "prepared": False,
                    "error": type(error).__name__ + ": " + str(error),
                })
        if choices:
            _, controller, params = max(choices, key=lambda choice: choice[0])
            return controller, params
        return None

    def execute(self, skill, cube, controller, params):
        row = {
            "action": len(self.records) + 1,
            "skill": skill,
            "cube": cube.name,
            "params": params.tolist(),
            "start_tick": self.ticks,
            "success": False,
        }
        self.records.append(row)
        cap = 3000 if skill in ("retrieve_shelf", "return_cube_to_start") else 400
        row["tick_cap"] = cap
        for _ in range(cap):  # Diagnostic caps; no Shelf learning horizon is frozen.
            action = controller.step()
            self.state, _, _, _, _ = self.env.step(action)
            controller.observe(self.state)
            self.ticks += 1
            self.frame(f"{skill}: {cube.name}")
            if controller.terminated():
                break
        row["end_tick"] = self.ticks
        row["terminated"] = controller.terminated()
        row["phases"] = {
            name: bool(getattr(controller, name, False))
            for name in ("_navigated", "_pre_grasp", "_closed_gripper", "_lifted")
        }
        if skill in ("retrieve_shelf", "return_cube_to_start"):
            row["actual_joints"] = controller._get_current_robot_arm_conf()[:7]
            row["approach_remaining"] = len(controller._checked_approach_waypoints)
            row["retract_remaining"] = len(controller._checked_retract_waypoints)
            row["next_approach"] = controller._checked_approach_waypoints[:1]
            row["next_retract"] = controller._checked_retract_waypoints[:1]
        atoms = self.atoms()
        expected = (
            GroundAtom(Holding, [self.robot, cube])
            if skill in ("pick_shelf", "retrieve_shelf")
            else GroundAtom(OnFixture, [cube, self.cupboard])
        )
        if skill == "return_cube_to_start":
            row["release_observations"] = controller.release_observations
            expected = GroundAtom(OnGround, [cube])
            position = np.array([self.state.get(cube, f) for f in ("x", "y", "z")])
            row["native_initial_region"] = bool(
                self.env._ground_fixture.check_in_region(
                    position, f"ground_{cube.name[4:]}_object_init_region", self.env._robot_env
                )
            )
            row["quaternion"] = [self.state.get(cube, f) for f in ("qx", "qy", "qz", "qw")]
        row["symbolic_success"] = expected in atoms
        row["success"] = expected in atoms and row["terminated"]
        if skill == "return_cube_to_start":
            row["success"] = row["success"] and row["native_initial_region"]
        row["atoms"] = sorted(str(atom) for atom in atoms)
        print(json.dumps(row), flush=True)
        self.snapshot(f"action-{len(self.records):02d}")
        self.save()
        return row["success"]

    def save(self):
        atoms = self.atoms()
        regions = {}
        for i in range(1, 9):
            cube = self.state.get_object_from_name(f"cube{i}")
            position = np.array([self.state.get(cube, f) for f in ("x", "y", "z")])
            region = f"cupboard_1_cube_{i}_goal_region"
            regions[cube.name] = {
                "position": position.tolist(),
                "in_goal_region": bool(
                    self.env._fixtures_dict["cupboard_1"].check_in_region(
                        position, region, self.env._robot_env
                    )
                ),
            }
        result = {
            "kind": "scripted physical proof, not bilevel benchmark",
            "actions": self.records,
            "planning_candidates": self.planning,
            "ticks": self.ticks,
            "native_task_goal": bool(self.env._check_goals()),
            "on_fixture": sum(
                GroundAtom(OnFixture, [self.state.get_object_from_name(f"cube{i}"), self.cupboard])
                in atoms
                for i in range(1, 9)
            ),
            "hand_empty": GroundAtom(HandEmpty, [self.robot]) in atoms,
            "controller_tick_cap": 400,
            "reset_complete": False,
            "physical_regions": regions,
            "placement_selection": "best predicted clearance among 20 native samples"
            if self.spread_placement
            else "first feasible native sample",
        }
        (self.output / "result.json").write_text(json.dumps(result, indent=2))

    def run(self):
        try:
            for _ in range(8):
                atoms = self.atoms()
                chosen = None
                for i in range(1, 9):
                    cube = self.state.get_object_from_name(f"cube{i}")
                    if GroundAtom(OnFixture, [cube, self.cupboard]) in atoms:
                        continue
                    prepared = self.prepare("pick_shelf", cube)
                    if prepared:
                        chosen = cube, prepared
                        break
                if chosen is None:
                    break
                cube, (controller, params) = chosen
                if not self.execute("pick_shelf", cube, controller, params):
                    break
                if len(self.records) >= self.max_actions:
                    break
                prepared = self.prepare("place_shelf", cube)
                if not prepared:
                    break
                if not self.execute("place_shelf", cube, *prepared):
                    break
                if len(self.records) >= self.max_actions:
                    break
        except BaseException:
            (self.output / "error.txt").write_text(traceback.format_exc())
            raise
        finally:
            self.save()
            self.snapshot("final")
            self.writer.close()
            self.env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1016164991)
    parser.add_argument("--max-actions", type=int, default=16)
    parser.add_argument("--spread-placement", action="store_true")
    args = parser.parse_args()
    Demo(
        output=args.output,
        seed=args.seed,
        max_actions=args.max_actions,
        spread_placement=args.spread_placement,
    ).run()
