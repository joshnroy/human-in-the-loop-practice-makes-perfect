"""Run `--env racktoserving3d`'s skills once per episode seed, as the toy's
interface offers them, and report what physics decided for each.

An episode starts from the full rack and is one route, then the robot's reset:
`three` is sweep3 then robot_reset; `five` is gather, then sweep5 when gather left
the food Gathered, then robot_reset; `human` is sweep3 then the human's reset. With
`--params nominal` each skill takes the rack oracle's nominal values; with `uniform`
each draws uniformly from the provider's own bounds, as an untrained sampler does.
`--render-seeds` films those episodes in full motion in the lab2 room, one video per
skill, one frame per `motion_stride()` control steps at `MOTION_FPS`::

    scripts/with_env.sh python scripts/measure_serving.py \\
        --route five --params nominal --seeds 0 1 2 --render-seeds 0 --output-dir <dir>
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from hitl_pmp.core.renderer.renderer import VideoStream
from hitl_pmp.environments.racktoserving3d.cli import RackServingCli
from hitl_pmp.environments.racktoserving3d.recording import RackServingReplayRenderer
from hitl_pmp.environments.racktoserving3d.skill_provider import (
    RackServingOracle,
    RackServingSkillProvider,
)
from hitl_pmp.environments.racktoserving3d.toy_environment import RackServingToyEnvironment
from hitl_pmp.environments.racktoserving3d.toy_interface import (
    GATHER,
    HUMAN_RESET,
    ROBOT_RESET,
    SWEEP3,
    SWEEP5,
)

# The physical flags of the 2026-11-14 Serving launch rows (`serving-3d-20261114`).
LAUNCH_SCENE = [
    "--start-layout",
    "middle_three",
    "--side-offset",
    "0.13",
    "--plate-tuck",
    "0.035",
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
]
ROUTES = {
    "three": (SWEEP3, ROBOT_RESET),
    "five": (GATHER, SWEEP5, ROBOT_RESET),
    "human": (SWEEP3, HUMAN_RESET),
}


class ServingToyEpisode:
    @staticmethod
    def nominal(*, name: str) -> np.ndarray:
        d = np.array([RackServingOracle.NOMINAL_D])
        return {GATHER: np.array(RackServingOracle.NOMINAL_GATHER), SWEEP3: d, SWEEP5: d}.get(
            name, np.zeros(0)
        )

    @staticmethod
    def run(*, route: str, params: str, seed: int, videos: Path | None) -> dict[str, object]:
        parser = argparse.ArgumentParser()
        RackServingCli.add_arguments(parser=parser)
        args = parser.parse_args(LAUNCH_SCENE)
        args.seed = seed
        env = RackServingCli.build_problem(args=args, mode="training").env
        assert isinstance(env, RackServingToyEnvironment)
        provider = RackServingSkillProvider(env=env)
        env.reset_to_seed(seed=seed)
        backend = env.backend()
        rng = np.random.default_rng(seed)
        renderer = (
            RackServingReplayRenderer(
                scene={**env.config.model_dump(mode="json"), "canonical_seed": seed},
                scene_bg=True,
            )
            if videos is not None
            else None
        )
        stride = RackServingReplayRenderer.motion_stride()
        steps = []
        start = time.monotonic()
        for name in ROUTES[route]:
            if name == SWEEP5 and "Gathered" not in env.toy_flags():
                break
            stream: VideoStream | None = None
            if renderer is not None and videos is not None:
                path = videos / f"serving-toy-{route}-{params}-seed{seed}-{name}.mp4"
                stream = VideoStream(output_path=path, fps=RackServingReplayRenderer.MOTION_FPS)
                ticks = [0]

                def capture(*, stream: VideoStream = stream, ticks: list[int] = ticks) -> None:
                    if ticks[0] % stride == 0:
                        stream.append(frame=renderer.show(state=backend.motion_state()))
                    ticks[0] += 1

                stream.append(frame=renderer.show(state=backend.motion_state()))
                backend.set_tick_observer(observer=capture)
            ground = provider.ground(name=name)
            values = (
                ServingToyEpisode.nominal(name=name)
                if params == "nominal"
                else provider.sample_params(ground_skill=ground, rng=rng)
            )
            began, first = time.monotonic(), backend.control_steps()
            charged: list[bool] = []

            def price(*, charged: list[bool] = charged) -> None:
                charged.append(env.cost_observation()["resetting"])

            backend.set_step_observers(before=price, after=None)
            try:
                if name == HUMAN_RESET:
                    env.reset_movables(destination=HUMAN_RESET)
                else:
                    env.take_action(
                        action=provider.compute_action(
                            ground_skill=ground, params=values, state=env.get_current_state()
                        )
                    )
            finally:
                backend.set_step_observers(before=None, after=None)
                backend.set_tick_observer(observer=None)
                if stream is not None:
                    stream.append(frame=renderer.show(state=backend.motion_state()))
                    stream.close()
            measured = env.toy_measurement()
            steps.append({
                "skill": name,
                "params": [round(float(v), 4) for v in values],
                "error": env.last_skill_error(),
                "control_steps": int(backend.control_steps() - first),
                "priced_steps": int(sum(charged)),
                "seconds": round(time.monotonic() - began, 1),
                "measured": None if measured is None else measured.model_dump(),
                "toy_state": sorted(env.toy_flags()),
                "frames": None if stream is None else stream.frames_written,
            })
        result = {
            "route": route,
            "params": params,
            "seed": seed,
            "steps": steps,
            "seconds": round(time.monotonic() - start, 1),
        }
        env.close()
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route", choices=sorted(ROUTES), required=True)
    parser.add_argument("--params", choices=("nominal", "uniform"), required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--render-seeds", type=int, nargs="*", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        result = ServingToyEpisode.run(
            route=args.route,
            params=args.params,
            seed=seed,
            videos=args.output_dir if seed in args.render_seeds else None,
        )
        print(json.dumps(result), flush=True)
        with (args.output_dir / f"{args.route}-{args.params}.jsonl").open("a") as handle:
            handle.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
