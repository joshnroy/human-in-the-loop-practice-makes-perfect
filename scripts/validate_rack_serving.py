"""Bounded native contract checks using explicitly privileged test setup, not robot resets."""

import argparse
import json
from pathlib import Path

import numpy as np

from hitl_pmp.environments.racktoserving3d.config import RackServingConfig
from hitl_pmp.environments.racktoserving3d.environment import RackServingEnvironment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("preserve existing validation reports; choose a new path")
    report = {"checks": [], "complete": False, "uses_privileged_test_setup": True}

    def check(*, name, passed, evidence=None):
        report["checks"].append({"name": name, "passed": bool(passed), "evidence": evidence})
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    for surface in ("wire", "concept"):
        env = RackServingEnvironment(
            config=RackServingConfig(
                surface_kind=surface,
                mode="deployment",
                cooling_seconds=0.2,
            )
        )
        try:
            env.reset_to_seed(seed=0)
            backend = env.backend()
            model, data = backend.model(), backend.data()
            rack, tray = backend.support(name="rack_0"), backend.support(name="box_0")
            check(
                name=f"{surface}: measured downward 10 mm transfer gap",
                passed=rack.top_z >= tray.top_z and abs(rack.x_min - tray.x_max - 0.01) < 0.001,
                evidence={"gap_m": rack.x_min - tray.x_max, "drop_m": rack.top_z - tray.top_z},
            )
            check(
                name=f"{surface}: five food proxies supported by rack",
                passed=all(backend.location(name=name) == "OnRack" for name in backend.cube_names),
            )
            for human in ("human_batch_return", "human_tool_recovery"):
                before = data.qpos.copy()
                ledger = backend.ledger
                rejected = False
                try:
                    env.reset_movables(destination=human)
                except PermissionError:
                    rejected = True
                check(
                    name=f"{surface}: deployment rejects {human} without writes or cost",
                    passed=rejected
                    and np.array_equal(before, data.qpos)
                    and ledger == backend.ledger,
                )
            env.take_action(action=np.array([6.0, 0, 0, 0]))
            check(
                name=f"{surface}: physical wait reaches configured elapsed-time threshold",
                passed=backend.ledger.ready and env.last_skill_error() is None,
                evidence={"elapsed_seconds": backend.ledger.elapsed_seconds},
            )
            for label, position in (
                ("OnTray", [backend.config.tray_center_x, 0, tray.top_z + 0.014]),
                ("SpilledCounter", [1.2, 0.35, backend.config.counter_height + 0.014]),
                ("SpilledFloor", [0.8, 0.55, 0.014]),
            ):
                backend._commit(
                    placements={"cube_0": (np.asarray(position), np.array([1.0, 0, 0, 0]))}
                )
                for _ in range(500):
                    backend.sim().step()
                actual = backend.location(name="cube_0")
                check(
                    name=f"{surface}: actual {label} classification",
                    passed=actual == label,
                    evidence={"observed": actual},
                )
            rack_body = model.body("rack_0").id
            rack_start = data.xpos[rack_body].copy()
            data.xfrc_applied[rack_body, 0] = 10
            for _ in range(100):
                backend.sim().step()
            data.xfrc_applied[rack_body] = 0
            displacement = float(np.linalg.norm(data.xpos[rack_body] - rack_start))
            check(
                name=f"{surface}: rack physically movable, not secretly fixed",
                passed=model.body_jntnum[rack_body] > 0 and displacement > 0.001,
                evidence={"displacement_m_under_test_force": displacement},
            )
            tray_body = model.body("box_0").id
            check(
                name=f"{surface}: tray graspability not encoded as immovability",
                passed=model.body_jntnum[tray_body] > 0,
            )
        finally:
            env.close()
    env = RackServingEnvironment(config=RackServingConfig(initial_elapsed_seconds=400))
    try:
        env.reset_to_seed(seed=1)
        backend = env.backend()
        before = backend.ledger.elapsed_seconds
        for human in ("human_batch_return", "human_tool_recovery"):
            env.reset_movables(destination=human)
        check(
            name="training: two explicit interventions charged, cooling not restarted",
            passed=backend.ledger.human_cost == 10 and backend.ledger.elapsed_seconds >= before,
            evidence=backend.ledger.model_dump(),
        )
    finally:
        env.close()
    report["complete"] = True
    report["passed"] = all(row["passed"] for row in report["checks"])
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
