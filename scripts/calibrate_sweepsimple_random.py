"""Approved fixed-context floor-skill calibration, separate from practice scores."""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from hitl_pmp.core.method.types import GroundSkill
from hitl_pmp.environments.sweep_simple3d.environment import SweepSimpleEnvironment
from hitl_pmp.environments.sweep_simple3d.skill_provider import SweepSimpleSkillProvider
from hitl_pmp.environments.sweep_simple3d.symbolic import SimpleSymbols


class SimpleCalibration:
    @staticmethod
    def target_index(*, seed_index: int, draw_index: int) -> int:
        return (8 * seed_index + draw_index) % 5

    @staticmethod
    def execute(*, env, provider, name, cube_index, rng, fixed_params=None):
        skill = next(s for s in provider.skills() if s.name == name)
        objects = (SimpleSymbols.SCENE,)
        if len(skill.parameters) == 2:
            objects += (SimpleSymbols.CUBES[cube_index],)
        ground = GroundSkill(skill=skill, objects=objects)
        if not all(
            a.predicate.holds(env.get_current_state(), a.objects) for a in ground.preconditions
        ):
            return None
        params = (
            provider.sample_params(ground_skill=ground, rng=rng)
            if fixed_params is None
            else np.asarray(fixed_params)
        )
        env.take_action(
            action=provider.compute_action(
                ground_skill=ground, params=params, state=env.get_current_state()
            )
        )
        return dict(
            name=name,
            cube_index=cube_index if len(objects) == 2 else None,
            params=params.tolist(),
            success=all(
                a.predicate.holds(env.get_current_state(), a.objects) for a in ground.add_effects
            ),
        )

    @classmethod
    def run(cls) -> None:
        from check_slurm_memory import main as check_memory

        check_memory()
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--skill", choices=SimpleSymbols.TRAINABLE, required=True)
        parser.add_argument("--output", type=Path, required=True)
        args = parser.parse_args()
        source = Path(__file__).resolve().parents[1]
        repositories = [
            source,
            source / "reference/kindergarden",
            source / "reference/kinder-baselines",
        ]
        provenance = []
        for repository in repositories:
            if subprocess.check_output(["git", "status", "--porcelain"], cwd=repository, text=True):
                raise RuntimeError(f"Calibration requires clean frozen source: {repository}")
            provenance.append(
                dict(
                    path=str(repository),
                    revision=subprocess.check_output(
                        ["git", "rev-parse", "HEAD"], cwd=repository, text=True
                    ).strip(),
                )
            )
        args.output.mkdir(parents=True, exist_ok=False)
        plan = dict(
            purpose="calibration only, excluded from experiment scores",
            seeds=[0, 1, 2],
            draws_per_seed=8,
            skill=args.skill,
            parameter_rng_seed=20260928,
            target_rotation="(8 * seed_index + draw_index) % 5",
            approval="Josh explicitly approved rotation across5targets;24contexts unchanged",
            estimator="Beta(1,1): (successes+1)/(attempts+2)",
            preparation=(
                "Sweep only: nominal PickFloorWiper(.7,0); unavailable preconditions logged "
                "without parameter draw or replacement"
            ),
            approved_parameter_bounds=SweepSimpleSkillProvider.APPROVED_PARAMETER_BOUNDS,
            source_repositories=provenance,
        )
        (args.output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
        rng = np.random.default_rng(plan["parameter_rng_seed"])
        records = []
        for seed_index, seed in enumerate(plan["seeds"]):
            for draw in range(8):
                cube_index = cls.target_index(seed_index=seed_index, draw_index=draw)
                env = SweepSimpleEnvironment(
                    canonical_seed=seed,
                    evaluation=True,
                    output_dir=args.output / f"seed{seed}-draw{draw}",
                )
                record = dict(
                    seed=seed,
                    draw=draw,
                    cube_index=cube_index,
                    skill=args.skill,
                    preparation=[],
                    attempt=None,
                    error=None,
                    completed=False,
                )
                try:
                    env.hard_reset()
                    provider = SweepSimpleSkillProvider(env=env, human_reset_enabled=False)
                    provider.validate_trainable_support()
                    if args.skill == "SweepCubeToGoal":
                        record["preparation"].append(
                            cls.execute(
                                env=env,
                                provider=provider,
                                name="PickFloorWiper",
                                cube_index=cube_index,
                                rng=np.random.default_rng(0),
                                fixed_params=(0.7, 0.0),
                            )
                        )
                    record["attempt"] = cls.execute(
                        env=env, provider=provider, name=args.skill, cube_index=cube_index, rng=rng
                    )
                    record["completed"] = True
                except Exception as error:
                    record["error"] = repr(error)
                    raise
                finally:
                    env.close()
                    with (args.output / "attempts.jsonl").open("a") as stream:
                        stream.write(json.dumps(record) + "\n")
                    print(json.dumps(record), flush=True)
                records.append(record)
        attempts = [r["attempt"] for r in records if r["attempt"] is not None]
        successes = sum(a["success"] for a in attempts)
        summary = dict(
            skill=args.skill,
            contexts=len(records),
            attempts=len(attempts),
            precondition_unavailable=len(records) - len(attempts),
            successes=successes,
            random_competence=None if not attempts else (successes + 1) / (len(attempts) + 2),
            estimator=plan["estimator"],
            target_rotation=plan["target_rotation"],
        )
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    SimpleCalibration.run()
