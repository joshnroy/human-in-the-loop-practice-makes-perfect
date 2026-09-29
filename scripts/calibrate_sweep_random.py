"""Separate, fixed-context random-stock calibration; never a practice experiment."""

import argparse
import json
from pathlib import Path

import numpy as np

from hitl_pmp.core.method.types import GroundSkill
from hitl_pmp.environments.sweep_drawer3d.environment import SweepDrawerEnvironment
from hitl_pmp.environments.sweep_drawer3d.skill_provider import SweepDrawerSkillProvider
from hitl_pmp.environments.sweep_drawer3d.symbolic import SweepSymbols


class Calibration:
    @staticmethod
    def execute(*, env, provider, name, rng, fixed_params=None):
        skill = next(s for s in provider.skills() if s.name == name)
        ground = GroundSkill(skill=skill, objects=(SweepSymbols.SCENE,))
        if not all(
            a.predicate.holds(env.get_current_state(), a.objects) for a in ground.preconditions
        ):
            return None
        params = (
            provider.sample_params(ground_skill=ground, rng=rng)
            if fixed_params is None
            else np.asarray(fixed_params, dtype=float)
        )
        env.take_action(
            action=provider.compute_action(
                ground_skill=ground, params=params, state=env.get_current_state()
            )
        )
        return {
            "name": name,
            "params": params.tolist(),
            "success": all(
                a.predicate.holds(env.get_current_state(), a.objects) for a in ground.add_effects
            ),
        }

    @staticmethod
    def run():
        parser = argparse.ArgumentParser()
        parser.add_argument("--skill", choices=SweepSymbols.TRAINABLE, required=True)
        parser.add_argument("--output", type=Path, required=True)
        args = parser.parse_args()
        args.output.mkdir(parents=True, exist_ok=False)
        plan = {
            "purpose": "random competence calibration only; not experiment scores",
            "seeds": [0, 1, 2],
            "draws_per_seed": 8,
            "skill": args.skill,
            "parameter_rng_seed": 20260928,
            "estimator": "Beta(1,1) posterior mean: (successes+1)/(attempts+2)",
            "preparation": (
                "Sweep only: fixed OpenResetDrawer then nominal PickWiper(.7,-pi); "
                "failed preconditions logged, no retry or seed replacement"
            ),
            "prior_use": "frozen random competence in the existing ModelB domain descriptor",
            "approved_parameter_bounds": {
                "OpenDrawer": [[0.65, 0.95], [-13 * np.pi / 12, -11 * np.pi / 12]],
                "PickWiper": [[0.55, 0.85], [-13 * np.pi / 12, -11 * np.pi / 12]],
                "Sweep": [[0.40, 0.70], [-13 * np.pi / 12, -11 * np.pi / 12]],
            },
        }
        (args.output / "plan.json").write_text(json.dumps(plan, indent=2))
        rng = np.random.default_rng(plan["parameter_rng_seed"])
        records = []
        for seed in plan["seeds"]:
            for draw in range(plan["draws_per_seed"]):
                env = SweepDrawerEnvironment(
                    canonical_seed=seed,
                    evaluation=True,
                    output_dir=args.output / f"seed{seed}-draw{draw}",
                )
                record = {"seed": seed, "draw": draw, "skill": args.skill, "preparation": []}
                try:
                    env.hard_reset()
                    provider = SweepDrawerSkillProvider(env=env, human_reset_enabled=False)
                    provider.validate_trainable_support()
                    if args.skill in ("PickWiper", "Sweep"):
                        for name in (
                            ("OpenResetDrawer", "PickWiper")
                            if args.skill == "Sweep"
                            else ("OpenResetDrawer",)
                        ):
                            result = Calibration.execute(
                                env=env,
                                provider=provider,
                                name=name,
                                rng=np.random.default_rng(0),
                                fixed_params=(0.7, -np.pi) if name == "PickWiper" else (),
                            )
                            record["preparation"].append(result)
                    record["attempt"] = Calibration.execute(
                        env=env, provider=provider, name=args.skill, rng=rng
                    )
                    record["cubes_in_goal"] = int(
                        sum(
                            env.get_current_state().get(
                                obj=SweepSymbols.SCENE, feature_name=f"InDrawer{i}"
                            )
                            > 0.5
                            for i in range(5)
                        )
                    )
                finally:
                    env.close()
                records.append(record)
                with (args.output / "attempts.jsonl").open("a") as stream:
                    stream.write(json.dumps(record) + "\n")
                print(f"{len(records)}/24 {json.dumps(record)}", flush=True)
        attempts = [r["attempt"] for r in records if r["attempt"] is not None]
        successes = sum(r["success"] for r in attempts)
        summary = {
            "skill": args.skill,
            "contexts": len(records),
            "attempts": len(attempts),
            "precondition_unavailable": len(records) - len(attempts),
            "successes": successes,
            "random_competence": None if not attempts else (successes + 1) / (len(attempts) + 2),
            "estimator": plan["estimator"],
        }
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    Calibration.run()
