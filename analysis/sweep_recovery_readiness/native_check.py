"""Native end-to-end contexts; report all failures without changing controllers."""

import json
from pathlib import Path

import numpy as np

from hitl_pmp.core.method.types import GroundSkill
from hitl_pmp.environments.sweep_drawer3d.environment import SweepDrawerEnvironment
from hitl_pmp.environments.sweep_drawer3d.skill_provider import SweepDrawerSkillProvider
from hitl_pmp.environments.sweep_drawer3d.symbolic import SweepSymbols


class NativeCheck:
    @staticmethod
    def run():
        out = Path("scratchpad/native-readiness")
        out.mkdir(parents=True, exist_ok=True)
        records = []
        for seed in (0, 1, 2, *range(10000, 10010)):
            env = SweepDrawerEnvironment(
                canonical_seed=seed, evaluation=True, output_dir=out / str(seed)
            )
            try:
                env.hard_reset()
                record = {"seed": seed, "steps": []}
                for name, distance in (("OpenDrawer", 0.8), ("PickWiper", 0.7), ("Sweep", 0.55)):
                    skill = next(s for s in SweepSymbols.skills() if s.name == name)
                    ground = GroundSkill(skill=skill, objects=(SweepSymbols.SCENE,))
                    applicable = all(
                        a.predicate.holds(env.current_state, a.objects)
                        for a in ground.preconditions
                    )
                    if not applicable:
                        record["steps"].append({"name": name, "applicable": False})
                        break
                    before = env.session().ticks
                    env.take_action(
                        action=np.array([env.ACTION_NAMES.index(name), distance, -np.pi])
                    )
                    success = all(
                        a.predicate.holds(env.current_state, a.objects) for a in ground.add_effects
                    )
                    record["steps"].append({
                        "name": name,
                        "applicable": True,
                        "success": success,
                        "ticks": env.session().ticks - before,
                    })
                record["goal"] = all(
                    a.predicate.holds(env.current_state, a.objects)
                    for a in SweepDrawerSkillProvider.deployment_goal_atoms()
                )
                records.append(record)
                (out / "summary.json").write_text(json.dumps(records, indent=2))
                print(json.dumps(record), flush=True)
            finally:
                env.close()


if __name__ == "__main__":
    NativeCheck.run()
