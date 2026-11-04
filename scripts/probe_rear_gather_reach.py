"""Scratch-kinematic reach of every GatherAndSweep stroke from rear base poses.

After a real PickScraper (measured grasp), each stroke is checked the way
``RackControllers.push_stroke`` drives it: hover IK at carry height with the full
robot and held-tool clearance check, six lowering waypoints, then 1 cm push waypoints
and the release lift, each through ``RackScraper.ik`` (same IK, 8 mm / 2 deg
tolerance, joint-segment collision check) seeded from the previous waypoint. Two base
modes: ``fixed`` (the base holds one pose for the stroke, as today) and ``slide``
(the base translates with the blade, heading fixed; the arm holds one relative pose).
No MuJoCo stepping. Probe only.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from hitl_pmp.environments.racktoserving3d.config import RackServingConfig
from hitl_pmp.environments.racktoserving3d.controllers import RackControllers
from hitl_pmp.environments.racktoserving3d.environment import RackServingEnvironment
from hitl_pmp.environments.racktoserving3d.skills import SKILL_IDS
from hitl_pmp.environments.racktoserving3d.stroke import GatherAndSweepPlanner, Stroke
from hitl_pmp.environments.sweepoffshelf3d.scripted_skills import SkillFailed


class Reach:
    def __init__(
        self, *, controllers: RackControllers, tilt: float, push_step: float, restarts: int
    ) -> None:
        self.c, self.tilt, self.push_step = controllers, tilt, push_step
        self.restarts = restarts
        self.shelf, self.tool = controllers.shelf, controllers.tool

    def body(self, *, stroke: Stroke) -> np.ndarray:
        body = np.eye(4)
        body[:3, :3] = RackControllers.stroke_rotation(stroke=stroke, tilt=self.tilt)
        body[:3, 3] = stroke.start - self.tool.blade_edge(pose=body).mean(axis=0)
        return body

    def hand(self, *, body: np.ndarray) -> np.ndarray:
        return body @ np.linalg.inv(self.shelf.in_hand)

    def check(self, *, stroke: Stroke, base: np.ndarray, mode: str) -> tuple[bool, str]:
        """True if every waypoint of hover, lower, push and lift is reachable and clear."""
        shelf, tool = self.shelf, self.tool
        body = self.body(stroke=stroke)
        carry = 0.06
        start_xy = stroke.start[:2]

        def base_at(*, point: np.ndarray) -> np.ndarray:
            if mode == "fixed":
                return base
            return np.r_[base[:2] + point[:2] - start_xy, base[2]]

        hover = body.copy()
        hover[2, 3] += carry
        target = self.hand(body=hover)
        waypoints: list[tuple[str, np.ndarray, np.ndarray]] = []
        for height in np.linspace(carry, 0, 7)[1:]:
            lowered = body.copy()
            lowered[2, 3] += height
            waypoints.append(("lower", lowered, base))
        count = max(1, int(np.ceil(stroke.length / self.push_step)))
        for distance in np.linspace(0, stroke.length, count + 1)[1:]:
            pushed = body.copy()
            point = stroke.point(distance=distance)
            pushed[:3, 3] += point - stroke.start
            waypoints.append(("push", pushed, base_at(point=point)))
        end_base = base_at(point=stroke.end)
        end = waypoints[-1][1]
        # release_swept_food: 3 cm back against travel and 2 mm up, then 6 cm up, both
        # as 1 cm Cartesian lines (TableWiper.line's step).
        for fraction in np.linspace(0, 1, 4)[1:]:
            pose = end.copy()
            pose[:3, 3] += fraction * (-0.03 * stroke.direction + [0, 0, 0.002])
            waypoints.append(("release", pose, end_base))
        released = waypoints[-1][1]
        for fraction in np.linspace(0, 1, 7)[1:]:
            pose = released.copy()
            pose[2, 3] += 0.06 * fraction
            waypoints.append(("lift", pose, end_base))
        failures = []
        for seed in self.seeds():
            try:
                arm, _ = shelf.ik(
                    position=target[:3, 3],
                    rotation=target[:3, :3],
                    base=base,
                    holding=True,
                    attempts=12,
                    seed=seed,
                )
                achieved = shelf.site_at(base=base, arm=arm)
                if (
                    np.linalg.norm(achieved[:3, 3] - target[:3, 3]) > 0.005
                    or tool.rotation_error(actual=achieved[:3, :3], target=target[:3, :3]) > 2.0
                ):
                    raise SkillFailed("hover IK residual")
                tool.check_configuration(base=base, arm=arm, holding=True)
            except SkillFailed as exc:
                failures.append(f"hover: {exc}"[:200])
                continue
            worst_angle = worst_error = 0.0
            for phase, pose, here in waypoints:
                hand = self.hand(body=pose)
                shelf.base_target = here.copy()
                try:
                    arm = tool.ik(
                        position=hand[:3, 3], rotation=hand[:3, :3], holding=True, seed=arm
                    )
                except SkillFailed as exc:
                    failures.append(f"{phase}: {exc}"[:200])
                    break
                reached = shelf.site_at(base=here, arm=arm)
                worst_error = max(worst_error, float(np.linalg.norm(reached[:3, 3] - hand[:3, 3])))
                worst_angle = max(
                    worst_angle,
                    tool.rotation_error(actual=reached[:3, :3], target=hand[:3, :3]),
                )
            else:
                # Margin: the IK accepts 8 mm / 2 deg, and 2 deg of blade roll drops one
                # end of the blade 4.4 mm, into the rack wires at a 3 mm glide.
                return True, f"worst {worst_error * 1000:.2f} mm {worst_angle:.2f} deg"
        return False, failures[0] if len(set(failures)) == 1 else " | ".join(failures)[:400]

    def seeds(self) -> list[np.ndarray]:
        shelf = self.shelf
        rng = np.random.default_rng(7)
        current = shelf.data.qpos[shelf.qa].copy()
        return [current, shelf.home_arm.copy()] + [
            shelf.home_arm + rng.normal(0, 0.8, 7) for _ in range(self.restarts)
        ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("fixed", "slide"), default="fixed")
    parser.add_argument("--xs", type=float, nargs="+", default=[1.95, 2.05, 2.15])
    parser.add_argument("--y-step", type=float, default=0.1)
    parser.add_argument("--y-max", type=float, default=0.5)
    parser.add_argument("--y-min-abs", type=float, default=0.0, help="skip bases nearer y=0")
    parser.add_argument(
        "--heading-offsets", type=float, nargs="+", default=[-1.05, -0.7, -0.35, 0, 0.35, 0.7, 1.05]
    )
    parser.add_argument("--deltas", type=float, nargs="+", default=[-0.03, 0, 0.015, 0.03, 0.04])
    parser.add_argument("--strokes", nargs="+", default=["gather_plus", "gather_minus"])
    parser.add_argument("--tilt", type=float, default=45)
    parser.add_argument("--push-step", type=float, default=0.006)
    parser.add_argument("--restarts", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0, help="stop after N checks (timing)")
    parser.add_argument(
        "--bases",
        type=float,
        nargs="+",
        default=None,
        help="explicit x y heading triples instead of the grid",
    )
    parser.add_argument("--bases-file", type=Path, default=None, help="--bases, from a file")
    args = parser.parse_args()
    if args.bases_file is not None:
        args.bases = [float(v) for v in args.bases_file.read_text().split()]
    if args.output.exists():
        parser.error("use a new output file")
    env = RackServingEnvironment(config=RackServingConfig())
    rows = []
    try:
        env.reset_to_seed(seed=0)
        env.take_action(action=np.array([SKILL_IDS["PickScraper"], 0, 0, 0, 0], dtype=float))
        if env.last_skill_error():
            raise SystemExit(f"PickScraper failed: {env.last_skill_error()}")
        backend, controllers = env.backend(), env.rack_controllers()
        shelf, tool = controllers.shelf, controllers.tool
        shelf.in_hand = np.linalg.inv(shelf.site_pose()) @ tool.wiper_pose()
        geometry = controllers.stroke_geometry()
        # push_stroke sets these before every stroke; the fixture check's rack-top
        # support allowance reads them.
        tool.top, tool.front = geometry["rack_top"], geometry["rack_front"]
        config = backend.config
        names = list(backend.cube_names)
        corners = np.stack([backend.corners(name=name) for name in names])
        strokes: dict[tuple[str, float], Stroke] = {}
        squeeze = GatherAndSweepPlanner.squeeze(s=0.07, **geometry)
        sweep = GatherAndSweepPlanner.sweep(
            d=0.10, plate_clear_z=controllers.plate_clear_z(), **geometry
        )
        for delta in args.deltas:
            plus, minus = GatherAndSweepPlanner.gather(
                food_corners=corners, delta=delta, food_width=config.dumpling_width, **geometry
            )
            strokes["gather_plus", delta] = plus
            strokes["gather_minus", delta] = minus
            strokes["squeeze_plus", delta] = squeeze[0]
            strokes["squeeze_minus", delta] = squeeze[1]
            strokes["sweep", delta] = sweep
        reach = Reach(
            controllers=controllers,
            tilt=args.tilt,
            push_step=args.push_step,
            restarts=args.restarts,
        )
        ys = np.round(np.arange(-args.y_max, args.y_max + 1e-9, args.y_step), 4)
        rear_heading = np.pi - 0.30
        checks = 0
        hover_failed: dict[tuple, str] = {}
        began = time.monotonic()
        meta = {
            "args": {k: str(v) for k, v in vars(args).items()},
            "geometry": geometry,
            "strokes": {
                f"{k[0]}@{k[1]}": {
                    "start": v.start.round(4).tolist(),
                    "end": v.end.round(4).tolist(),
                }
                for k, v in strokes.items()
            },
        }
        grid = (
            [np.array(args.bases[i : i + 3]) for i in range(0, len(args.bases), 3)]
            if args.bases
            else [
                np.array([x, y, rear_heading + offset])
                for x in args.xs
                for y in ys
                if abs(y) >= args.y_min_abs - 1e-9
                for offset in args.heading_offsets
            ]
        )
        if True:
            if True:
                for base in grid:
                    for name in args.strokes:
                        deltas = args.deltas if name.startswith("gather") else args.deltas[:1]
                        for delta in deltas:
                            stroke = strokes[name, delta]
                            key = (name, *base.round(4).tolist())
                            if key in hover_failed:
                                ok, why = False, hover_failed[key]
                            else:
                                ok, why = reach.check(stroke=stroke, base=base, mode=args.mode)
                                if not ok and all(
                                    part.startswith("hover") for part in why.split(" | ")
                                ):
                                    # The hover pose is the stroke start, which delta
                                    # does not move.
                                    hover_failed[key] = why
                            rows.append({
                                "base": base.round(4).tolist(),
                                "stroke": name,
                                "delta": delta,
                                "ok": ok,
                                "why": why,
                            })
                            checks += 1
                            if args.limit and checks >= args.limit:
                                raise StopIteration
                    print(
                        json.dumps({
                            "base": base.round(3).tolist(),
                            "ok": [r["ok"] for r in rows[-len(args.strokes) * len(args.deltas) :]],
                            "elapsed": round(time.monotonic() - began, 1),
                        }),
                        flush=True,
                    )
                args.output.write_text(json.dumps({**meta, "rows": rows}) + "\n")
    except StopIteration:
        pass
    finally:
        args.output.write_text(json.dumps({**meta, "rows": rows}) + "\n")
        print(f"checks {len(rows)} in {time.monotonic() - began:.1f}s", flush=True)
        env.close()


if __name__ == "__main__":
    main()
