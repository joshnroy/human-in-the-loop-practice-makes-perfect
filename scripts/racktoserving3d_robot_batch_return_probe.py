"""Measure RobotBatchReturn after a Sweep or a GatherAndSweep in the two-route setup.

    scripts/with_env.sh python scripts/racktoserving3d_robot_batch_return_probe.py \
        --bulk Sweep --seeds 0 1 2 --output probe.jsonl [--video-dir <dir>]

One JSON line per seed: where the five dumplings lay after the bulk try, what the robot
return moved (with each returned dumpling's distance from its slot and whether the
set-down wrote its yaw), how many dumplings were on their slots afterwards, its control
steps and error, and what the same bulk try then delivered. The bulk parameters are fixed
(Sweep d = 0.08; GatherAndSweep s, delta, d = 0.07, 0.03, 0.10), so the starting
states are those of tries that mostly deliver, not of a learner's early tries.
`--video-dir` records the robot return alone, full motion, in the lab2 scene.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from hitl_pmp.core.method.types import GroundSkill
from hitl_pmp.environments.racktoserving3d.config import RackServingConfig
from hitl_pmp.environments.racktoserving3d.environment import RackServingEnvironment
from hitl_pmp.environments.racktoserving3d.objects import RackServing3DObjects as Objects
from hitl_pmp.environments.racktoserving3d.recording import RackServingReplayRenderer
from hitl_pmp.environments.racktoserving3d.robot_batch_return import RackRobotBatchReturn
from hitl_pmp.environments.racktoserving3d.skill_provider import (
    RackServingActions,
    RackServingSkillProvider,
)

SETUP_A = RackServingConfig(
    start_layout="middle_three",
    side_offset=0.13,
    plate_tuck=0.035,
    surface_kind="wire_solid",
    scraper_attachment="grasp_weld",
    two_route_rewards=True,
    bulk_transfer_only=True,
    robot_batch_return=True,
)
BULK_PARAMS = {"Sweep": (0.08,), "GatherAndSweep": (0.07, 0.03, 0.10)}
BULK_OBJECTS = (Objects.robot, Objects.scraper, Objects.rack, Objects.tray, *Objects.food)
OBJECTS = {
    "PickScraper": (Objects.robot, Objects.scraper),
    "Sweep": BULK_OBJECTS,
    "GatherAndSweep": BULK_OBJECTS,
    "RobotBatchReturn": (Objects.robot, Objects.rack, Objects.tray, *Objects.food),
}
# Every control step is a frame. The simulator runs 10 control steps a second, so 30
# frames a second plays at three times real time.
VIDEO_FPS = 30
VIDEO_SIZE = (1280, 720)


class RobotBatchReturnProbe:
    @staticmethod
    def act(*, provider: RackServingSkillProvider, name: str) -> dict[str, Any]:
        env = provider.env
        skill = next(skill for skill in provider.skills() if skill.name == name)
        ground = GroundSkill(skill=skill, objects=OBJECTS[name])
        params = np.array(BULK_PARAMS.get(name, ()))
        env.take_action(action=RackServingActions.encode(ground_skill=ground, params=params))
        return {"error": env.last_skill_error(), "steps": env.last_controller_steps()}

    @staticmethod
    def where(*, env: RackServingEnvironment) -> dict[str, str]:
        """Each dumpling's measured location; on the rack, whether it is on its slot.
        `none` is the classifier's no-location (overhanging an edge, or in flight)."""
        backend = env.backend()
        check = RackRobotBatchReturn(controllers=env.rack_controllers())
        out = {}
        for name in backend.cube_names:
            location = backend.location(name=name)
            if location == "OnRack":
                location = "OnRackInPlace" if check.in_place(name=name) else "OnRackDisplaced"
            if location is None:
                x, _ = backend.food_centre(name=name)
                front = backend.support(name="rack_0").x_min
                location = "none(rack front edge)" if abs(x - front) < 0.04 else "none"
            out[name] = location
        return out

    @staticmethod
    def run_seed(
        *, provider: RackServingSkillProvider, bulk: str, seed: int, video_dir: Path | None
    ) -> dict[str, Any]:
        env = provider.env
        env.reset_to_seed(seed=seed)
        record: dict[str, Any] = {
            "bulk": bulk,
            "seed": seed,
            "yaw": env.config.robot_batch_return_yaw,
        }
        record["pick_scraper"] = RobotBatchReturnProbe.act(provider=provider, name="PickScraper")
        record["bulk_try"] = RobotBatchReturnProbe.act(provider=provider, name=bulk)
        backend = env.backend()
        record["plate_count_after_bulk"] = backend.plate_count()
        record["before"] = RobotBatchReturnProbe.where(env=env)
        writer = None
        if video_dir is not None:
            import imageio.v2 as imageio
            import mujoco

            model, data = backend.model(), backend.data()
            model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), VIDEO_SIZE[0])
            model.vis.global_.offheight = max(int(model.vis.global_.offheight), VIDEO_SIZE[1])
            renderer = mujoco.Renderer(model, height=VIDEO_SIZE[1], width=VIDEO_SIZE[0])
            camera = mujoco.MjvCamera()
            *lookat, camera.distance, camera.azimuth, camera.elevation = (
                RackServingReplayRenderer.CAMERA
            )
            camera.lookat[:] = lookat
            path = video_dir / f"robot-batch-return-after-{bulk}-seed{seed}.mp4"
            writer = imageio.get_writer(path, fps=VIDEO_FPS, macro_block_size=None)
            frames = [0]

            def capture() -> None:
                renderer.update_scene(data, camera=camera)
                writer.append_data(np.asarray(renderer.render(), dtype=np.uint8))
                frames[0] += 1

            backend.set_tick_observer(observer=capture)
            record["video"] = str(path)
        try:
            record["robot_return"] = RobotBatchReturnProbe.act(
                provider=provider, name="RobotBatchReturn"
            )
        finally:
            backend.set_tick_observer(observer=None)
            if writer is not None:
                writer.close()
                record["video_frames"] = frames[0]
        metrics = env.rack_controllers().metrics
        record["report"] = metrics.get("robot_batch_return")
        record["descents_stopped_short"] = metrics.get("descents_stopped_short", [])
        record["after"] = RobotBatchReturnProbe.where(env=env)
        record["in_place_after"] = sum(
            location == "OnRackInPlace" for location in record["after"].values()
        )
        record["scraper_held_after"] = bool(backend.held(name="wiper_0"))
        if record["robot_return"]["error"] is not None:
            # What a planner's next choice of the same skill finds: the scene as the
            # failed return left it.
            record["retry"] = RobotBatchReturnProbe.act(provider=provider, name="RobotBatchReturn")
            record["retry_report"] = env.rack_controllers().metrics.get("robot_batch_return")
            record["after_retry"] = RobotBatchReturnProbe.where(env=env)
            record["scraper_held_after_retry"] = bool(backend.held(name="wiper_0"))
        if record["robot_return"]["error"] is None:
            record["bulk_again"] = RobotBatchReturnProbe.act(provider=provider, name=bulk)
            record["plate_count_after_bulk_again"] = backend.plate_count()
        return record

    @staticmethod
    def main() -> None:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--bulk", choices=sorted(BULK_PARAMS), required=True)
        parser.add_argument("--seeds", type=int, nargs="+", required=True)
        parser.add_argument("--output", type=Path, required=True)
        parser.add_argument("--video-dir", type=Path, default=None)
        parser.add_argument(
            "--yaw",
            choices=["snap", "hand"],
            default="snap",
            help="RackServingConfig.robot_batch_return_yaw: the set-down writes the slot's "
            "yaw (the default), or the hand alone must bring the dumpling to it",
        )
        args = parser.parse_args()
        config = SETUP_A.model_copy(update={"robot_batch_return_yaw": args.yaw})
        env = RackServingEnvironment(config=config, scene_bg=args.video_dir is not None)
        provider = RackServingSkillProvider(env=env)
        try:
            with args.output.open("a") as stream:
                for seed in args.seeds:
                    record = RobotBatchReturnProbe.run_seed(
                        provider=provider, bulk=args.bulk, seed=seed, video_dir=args.video_dir
                    )
                    stream.write(json.dumps(record) + "\n")
                    stream.flush()
                    print(
                        f"{args.bulk} seed {seed}: in place {record['in_place_after']}/5,"
                        f" steps {record['robot_return']['steps']},"
                        f" error {record['robot_return']['error']}",
                        flush=True,
                    )
        finally:
            env.close()


if __name__ == "__main__":
    RobotBatchReturnProbe.main()
