"""Film one two-route Serving skill and check its declared effects against what happened.

Runs `--setup` steps unfilmed from the spaced start, then films `--focus` in the lab2
scene. A step is a skill name with optional values after a colon: `PickScraper`,
`Sweep:0.08`, `GatherAndSweep:0.03,0.015,0.12`, `ReturnScraper`, `PickFood:2` (food
index), `ReturnFoodToRack:2`, `human_batch_return`, `human_tool_recovery`. Prints one
JSON object: the focus skill's declared preconditions, add effects and delete effects
(grounded, from the provider's own skill definitions), and for each whether it held in
the measured state before or after. Probe only; never launches learning.
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
from hitl_pmp.environments.racktoserving3d.skill_provider import (
    RackServingActions,
    RackServingSkillProvider,
)
from scripts.probe_two_route_layout import TwoRouteProbeVideo

HUMAN = ("human_batch_return", "human_tool_recovery")


class TwoRouteSkillFilm:
    @staticmethod
    def parse(*, step: str) -> tuple[str, list[float]]:
        name, _, values = step.partition(":")
        return name, [float(v) for v in values.split(",") if v]

    @staticmethod
    def ground(*, provider: RackServingSkillProvider, step: str) -> tuple[GroundSkill, np.ndarray]:
        name, values = TwoRouteSkillFilm.parse(step=step)
        if name in HUMAN:
            grounds = {g.skill.name: g for g in provider.human_cube_bin_reset_skills()}
            return grounds[name], np.zeros(0)
        skill = next(skill for skill in provider.skills() if skill.name == name)
        if name in ("PickScraper", "ReturnScraper"):
            objects: tuple[Any, ...] = (Objects.robot, Objects.scraper)
        elif name in ("Sweep", "GatherAndSweep"):
            objects = (Objects.robot, Objects.scraper, Objects.rack, Objects.tray, *Objects.food)
        elif name == "PickFood":
            objects = (Objects.robot, Objects.food[int(values[0])], Objects.rack, Objects.tray)
        else:
            objects = (Objects.robot, Objects.food[int(values[0])], Objects.rack)
        params = np.array(values if name in ("Sweep", "GatherAndSweep") else [])
        return GroundSkill(skill=skill, objects=objects), params

    @staticmethod
    def run(*, env: RackServingEnvironment, ground: GroundSkill, params: np.ndarray) -> str | None:
        if ground.skill.name in HUMAN:
            env.reset_movables(destination=ground.skill.name)
            return None
        env.take_action(action=RackServingActions.encode(ground_skill=ground, params=params))
        return env.last_skill_error()

    @staticmethod
    def named(*, atom: Any) -> tuple[str, tuple[str, ...]]:
        return atom.predicate.name, tuple(obj.name for obj in atom.objects)

    @staticmethod
    def text(*, atom: tuple[str, tuple[str, ...]]) -> str:
        return f"{atom[0]}({', '.join(atom[1])})"

    @staticmethod
    def main() -> None:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--setup", nargs="*", default=[])
        parser.add_argument("--focus", required=True)
        parser.add_argument("--seed", type=int, default=0)
        parser.add_argument("--video", type=Path, required=True)
        parser.add_argument("--clip-name", required=True)
        args = parser.parse_args()
        config = RackServingConfig(
            start_layout="spaced",
            surface_kind="wire_solid",
            scraper_attachment="grasp_weld",
            bulk_transfer_only=True,
            two_route_rewards=True,
        )
        env = RackServingEnvironment(config=config, scene_bg=True, skill_step_limit=12000)
        provider = RackServingSkillProvider(env=env)
        film = TwoRouteSkillFilm
        try:
            env.reset_to_seed(seed=args.seed)
            backend = env.backend()
            setup_errors = []
            for step in args.setup:
                ground, params = film.ground(provider=provider, step=step)
                setup_errors.append(film.run(env=env, ground=ground, params=params))
            ground, params = film.ground(provider=provider, step=args.focus)
            before = frozenset(env.get_current_state().abstract_atoms)
            args.video.mkdir(parents=True, exist_ok=True)
            video = TwoRouteProbeVideo(
                backend=backend,
                path=args.video / f"{args.clip_name}.mp4",
                title=f"SKILL: {args.focus} | after {' -> '.join(args.setup) or 'the spaced start'}"
                f" | env seed {args.seed}",
            )
            video.mark(label="before")
            video.frame(repeat=45, force=True)
            backend.set_tick_observer(observer=video.frame)
            video.mark(label=ground.skill.name)
            try:
                error = film.run(env=env, ground=ground, params=params)
            finally:
                backend.set_tick_observer(observer=None)
            video.mark(label="after")
            video.frame(repeat=60, force=True)
            video.close()
            after = frozenset(env.get_current_state().abstract_atoms)
            static = {"FoodOrder"}
            ground_before = {film.named(atom=a) for a in ground.preconditions}
            adds = {
                film.named(atom=a)
                for a in ground.add_effects
                | frozenset(
                    atom
                    for effect in ground.conditional_add_effects
                    if {film.named(atom=c) for c in effect.conditions} <= before
                    for atom in effect.add_effects
                )
            }
            deletes = {film.named(atom=a) for a in ground.delete_effects} - adds
            print(
                json.dumps({
                    "clip": f"{args.clip_name}.mp4",
                    "focus": args.focus,
                    "setup": args.setup,
                    "seed": args.seed,
                    "setup_errors": setup_errors,
                    "skill_error": error,
                    "plate_count_after": backend.plate_count(),
                    "clip_marks": video.marks,
                    "preconditions": [
                        {"atom": film.text(atom=a), "held_before": a in before}
                        for a in sorted(ground_before)
                        if a[0] not in static
                    ],
                    "add_effects": [
                        {"atom": film.text(atom=a), "holds_after": a in after} for a in sorted(adds)
                    ],
                    "delete_effects": [
                        {
                            "atom": film.text(atom=a),
                            "held_before": a in before,
                            "absent_after": a not in after,
                        }
                        for a in sorted(deletes)
                    ],
                    "atoms_before": sorted(film.text(atom=a) for a in before),
                    "atoms_after": sorted(film.text(atom=a) for a in after),
                }),
                flush=True,
            )
        finally:
            env.close()


if __name__ == "__main__":
    TwoRouteSkillFilm.main()
