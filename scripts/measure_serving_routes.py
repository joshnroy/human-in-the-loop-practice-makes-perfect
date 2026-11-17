"""Run each `split_gather_and_sweep` route once per episode seed: count what it serves,
how many control steps each skill took, and film the episodes asked for.

A route is PickScraper then Sweep (the three route), or PickScraper, Gather, Sweep5
(the five route). The scene is the Sweeping launch configuration (`LAUNCH_SCENE`):
`two_route_rewards`, `bulk_transfer_only`, the `middle_three` start. Episode seed 0
uses the rack oracle's nominal (s, delta, d); every other seed draws each skill's
parameters uniformly from the provider's own bounds, as an untrained sampler does.
`--render-seeds` films those episodes in full motion in the lab2 room, one frame per
`motion_stride()` control steps at `MOTION_FPS`::

    scripts/with_env.sh python scripts/measure_serving_routes.py \\
        --route five --seeds 0 1 2 --render-seeds 0 --output-dir <dir>
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from hitl_pmp.core.method.types import GroundSkill
from hitl_pmp.core.renderer.renderer import VideoStream
from hitl_pmp.environments.racktoserving3d.cli import RackServingCli
from hitl_pmp.environments.racktoserving3d.objects import RackServing3DObjects as Objects
from hitl_pmp.environments.racktoserving3d.recording import RackServingReplayRenderer
from hitl_pmp.environments.racktoserving3d.skill_provider import (
    RackServingOracle,
    RackServingSkillProvider,
)

# The environment flags of the 2026-11-14 Sweeping launch rows, plus the split.
LAUNCH_SCENE = [
    "--two-route-rewards",
    "1",
    "--start-layout",
    "middle_three",
    "--side-offset",
    "0.13",
    "--plate-tuck",
    "0.035",
    "--sampler-features",
    "parameters",
    "--bulk-transfer-only",
    "1",
    "--surface-kind",
    "wire_solid",
    "--scraper-attachment",
    "grasp_weld",
    "--skill-step-limit",
    "12000",
    "--physics-timestep",
    "0.002",
    "--lean-control-step",
    "1",
    "--base-step-limit",
    "0.04",
    "--planned-joint-step",
    "0.1",
    "--settle-until-still",
    "1",
    "--gather-delta-high",
    "0.2",
    "--robot-batch-return",
    "1",
    "--split-gather-and-sweep",
    "1",
]
ROUTES = {"three": ("PickScraper", "Sweep"), "five": ("PickScraper", "Gather", "Sweep5")}


class ServingRouteEpisode:
    @staticmethod
    def run(*, route: str, seed: int, video: Path | None) -> dict[str, object]:
        parser = argparse.ArgumentParser()
        RackServingCli.add_arguments(parser=parser)
        args = parser.parse_args(LAUNCH_SCENE)
        args.seed = seed
        problem = RackServingCli.build_problem(args=args, mode="training")
        env = problem.env
        provider = RackServingSkillProvider(env=env)
        env.reset_to_seed(seed=seed)
        backend = env.backend()
        skills = {skill.name: skill for skill in provider.skills()}
        s, delta, d = RackServingOracle.NOMINAL_GATHER_AND_SWEEP
        nominal = {"Gather": np.array([s, delta]), "Sweep": np.array([d]), "Sweep5": np.array([d])}
        rng = np.random.default_rng(seed)
        stream: VideoStream | None = None
        if video is not None:
            renderer = RackServingReplayRenderer(
                scene={**env.config.model_dump(mode="json"), "canonical_seed": seed},
                scene_bg=True,
            )
            stream = VideoStream(output_path=video, fps=RackServingReplayRenderer.MOTION_FPS)
            stride, ticks = RackServingReplayRenderer.motion_stride(), [0]

            def capture() -> None:
                if ticks[0] % stride == 0:
                    stream.append(frame=renderer.show(state=backend.motion_state()))
                ticks[0] += 1

            stream.append(frame=renderer.show(state=backend.motion_state()))
            backend.set_tick_observer(observer=capture)
        start = time.monotonic()
        steps = []
        try:
            for name in ROUTES[route]:
                ground = GroundSkill(
                    skill=skills[name],
                    objects=(Objects.robot, Objects.scraper)
                    if name == "PickScraper"
                    else (
                        Objects.robot,
                        Objects.scraper,
                        Objects.rack,
                        Objects.tray,
                        *Objects.food,
                    ),
                )
                params = (
                    nominal.get(name, np.zeros(0))
                    if seed == 0
                    else provider.sample_params(ground_skill=ground, rng=rng)
                )
                state = env.take_action(
                    action=provider.compute_action(
                        ground_skill=ground, params=params, state=env.get_current_state()
                    )
                )
                atoms = state.abstract_atoms
                steps.append({
                    "skill": name,
                    "params": [round(float(v), 4) for v in params],
                    "error": env.last_skill_error(),
                    "control_steps": int(env._last_controller_steps),
                    "gathered": ("Gathered", ()) in atoms,
                    "plate_count": int(backend.plate_count()),
                })
                if env.last_skill_error() is not None:
                    break
        finally:
            backend.set_tick_observer(observer=None)
            if stream is not None:
                stream.close()
        atoms = env.get_current_state().abstract_atoms
        result = {
            "route": route,
            "seed": seed,
            "steps": steps,
            "plate_count": int(backend.plate_count()),
            "locations": [backend.location(name=name) for name in backend.cube_names],
            "verdict": sorted(name for name, _ in atoms if name.startswith("Served")),
            "seconds": time.monotonic() - start,
            "frames": None if stream is None else stream.frames_written,
        }
        env.close()
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route", choices=sorted(ROUTES), required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--render-seeds", type=int, nargs="*", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        video = (
            args.output_dir / f"serving-split-{args.route}-seed{seed}.mp4"
            if seed in args.render_seeds
            else None
        )
        result = ServingRouteEpisode.run(route=args.route, seed=seed, video=video)
        print(json.dumps(result), flush=True)
        with (args.output_dir / f"{args.route}.jsonl").open("a") as handle:
            handle.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
