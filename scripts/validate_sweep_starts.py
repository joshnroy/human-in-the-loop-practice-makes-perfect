"""Validate a fixed candidate seed pool before any action; never resample a seed."""

import argparse
import hashlib
import json
from pathlib import Path

from hitl_pmp.environments.sweep_drawer3d.session import SweepDrawerSession
from hitl_pmp.environments.sweep_drawer3d.start_regions import SweepRegions


class SweepStartManifest:
    @staticmethod
    def run(*, candidates: Path, output: Path) -> None:
        raw = candidates.read_bytes()
        plan = json.loads(raw)
        rows = []
        for split in ("practice", "evaluation"):
            seeds = plan[f"{split}_candidate_seeds"]
            if len(seeds) != len(set(seeds)):
                raise ValueError(f"Duplicate {split} candidate seed")
            for seed in seeds:
                session = SweepDrawerSession(seed=seed)
                try:
                    result = SweepRegions.validate(session=session)
                    row = {
                        "split": split,
                        **result.model_dump(),
                        "primitive_actions_before_validation": len(session.steps),
                        "controller_ticks_before_validation": session.ticks,
                    }
                    assert row["primitive_actions_before_validation"] == 0
                    rows.append(row)
                    print(json.dumps(row), flush=True)
                finally:
                    session.close()
        accepted = {
            split: [r["seed"] for r in rows if r["split"] == split and r["valid"]]
            for split in ("practice", "evaluation")
        }
        counts = {
            split: {"valid": len(accepted[split]), "total": len(plan[f"{split}_candidate_seeds"])}
            for split in ("practice", "evaluation")
        }
        payload = {
            "candidate_sha256": hashlib.sha256(raw).hexdigest(),
            "candidate_plan": plan,
            "rows": rows,
            "counts": counts,
            "valid_practice_seeds": accepted["practice"],
            "valid_evaluation_seeds": accepted["evaluation"],
            "excluded": [
                {"split": r["split"], "seed": r["seed"], "reasons": r["reasons"]}
                for r in rows
                if not r["valid"]
            ],
            "selection_rule": (
                "Ascending candidate order; first3 valid practice and "
                "first10 valid evaluation; no resampling or efficacy selection"
            ),
            "executed_learning_runs": 0,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    SweepStartManifest.run(candidates=args.candidates, output=args.output)
