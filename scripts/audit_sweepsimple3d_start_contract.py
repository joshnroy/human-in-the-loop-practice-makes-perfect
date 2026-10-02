"""Audit the shared reset/start contract without running learner actions."""

import argparse
import json
from pathlib import Path

from hitl_pmp.environments.sweep_simple3d.regions import SimpleRegions
from hitl_pmp.environments.sweep_simple3d.session import SweepSimpleSession


class NativeStartAudit:
    @staticmethod
    def run() -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--output", type=Path, required=True)
        args = parser.parse_args()
        if args.output.exists():
            raise FileExistsError(args.output)
        records = []
        for seed in (0, 1, 2, *range(10000, 10010)):
            session = SweepSimpleSession(seed=seed)
            try:
                validation = SimpleRegions.validate(session=session)
                records.append({
                    "seed": seed,
                    "validation": validation.model_dump(),
                    "wiper_quaternion_xyzw": list(session.quaternion(name="wiper_0")),
                    "native_task_goal_initially_true": bool(
                        session.env.unwrapped._object_centric_env._check_goals()
                    ),
                })
            finally:
                session.close()
        result = {
            "upright_axis_atol": 1e-3,
            "upright_axis_rtol": 0.0,
            "wiper_support_tolerance_m": 0.01,
            "valid": sum(r["validation"]["valid"] for r in records),
            "total": len(records),
            "records": records,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(f"Valid native starts: {result['valid']}/{result['total']}")


if __name__ == "__main__":
    NativeStartAudit.run()
