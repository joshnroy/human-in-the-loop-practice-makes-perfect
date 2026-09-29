"""Native end-to-end contexts; report all failures without changing controllers."""

import json
from pathlib import Path

import numpy as np

from hitl_pmp.core.method.types import GroundSkill
from hitl_pmp.environments.sweep_drawer3d.environment import SweepDrawerEnvironment
from hitl_pmp.environments.sweep_drawer3d.primitives import WiperHold
from hitl_pmp.environments.sweep_drawer3d.skill_provider import SweepDrawerSkillProvider
from hitl_pmp.environments.sweep_drawer3d.symbolic import SweepSymbols
from hitl_pmp.environments.sweep_drawer3d.types import SweepDrawerScene as S


class NativeCheck:
    @staticmethod
    def run(*, label="nominal", distances=(0.8, 0.7, 0.55), rotation=-np.pi):
        out = Path("scratchpad/native-varied-exact-tool") / label
        out.mkdir(parents=True, exist_ok=True)
        records = []
        for seed in (0, 1, 2, *range(10000, 10010)):
            env = SweepDrawerEnvironment(
                canonical_seed=seed, evaluation=True, output_dir=out / str(seed)
            )
            try:
                env.hard_reset()
                record = {"seed": seed, "steps": []}
                for name, distance in zip(
                    ("OpenDrawer", "PickWiper", "Sweep"), distances, strict=True
                ):
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
                        action=np.array([env.ACTION_NAMES.index(name), distance, rotation])
                    )
                    success = all(
                        a.predicate.holds(env.current_state, a.objects) for a in ground.add_effects
                    )
                    record["steps"].append({
                        "name": name,
                        "applicable": True,
                        "success": success,
                        "ticks": env.session().ticks - before,
                        "wiper_z": float(env.session().position(name=S.WIPER)[2]),
                        "physical_wiper_hold": bool(
                            env.session().gripper() > 0.2
                            and WiperHold.in_hand(
                                gripper=np.array(env.recovery().primitives.scene.ee_now().position),
                                wiper=env.session().position(name=S.WIPER),
                            )
                        ),
                        "event": json.loads(
                            (out / str(seed) / "sweep_events.jsonl").read_text().splitlines()[-1]
                        ),
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
    import sys

    offset = int(sys.argv[1])
    distances = (0.65, 0.55, 0.40) if offset < 0 else (0.95, 0.85, 0.70)
    NativeCheck.run(
        label=f"boundary{offset}", distances=distances, rotation=-np.pi + offset * np.pi / 12
    )
