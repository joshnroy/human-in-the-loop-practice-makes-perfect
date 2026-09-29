"""Report existing robot recovery admission in a valid blocked Sweep state."""

import json
from pathlib import Path

import numpy as np

from hitl_pmp.core.method.types import GroundSkill
from hitl_pmp.environments.sweep_drawer3d.environment import SweepDrawerEnvironment
from hitl_pmp.environments.sweep_drawer3d.symbolic import SweepSymbols

env = SweepDrawerEnvironment(
    canonical_seed=10001, evaluation=True, output_dir=Path("scratchpad/blocked-recovery")
)
try:
    env.hard_reset()
    for name, distance in (
        ("OpenDrawer", 0.8),
        ("PickWiper", 0.7),
        ("Sweep", 0.55),
        ("ParkWiper", 0),
    ):
        env.take_action(action=np.array([env.ACTION_NAMES.index(name), distance, -np.pi]))
        print(
            json.dumps({
                "action": name,
                "event": json.loads(
                    (env.output_dir / "sweep_events.jsonl").read_text().splitlines()[-1]
                ),
            }),
            flush=True,
        )
    applicable = []
    for skill in SweepSymbols.skills():
        ground = GroundSkill(skill=skill, objects=(SweepSymbols.SCENE,))
        if all(a.predicate.holds(env.current_state, a.objects) for a in ground.preconditions):
            applicable.append(skill.name)
    print(json.dumps({"applicable": applicable}), flush=True)
finally:
    env.close()
