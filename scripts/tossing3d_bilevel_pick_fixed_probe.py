"""Run one seed with pick_cube pinned to a *different* fixed standoff.

The sampled arm beating the zero-parameter arm does not by itself say that sampling
is what helped -- it could be that (0.55, 0.0) specifically is a bad place to stand
and any other fixed point would do. This pins an arbitrary fixed point instead and
asks the same question, so the two explanations can be told apart.

Reuses tossing3d_bilevel_pick_probe's ``zero-param`` mode, which returns
``PickCubeController.STANDOFF`` verbatim, by overwriting that class attribute.
"""

import argparse
import json
import time
import traceback
from typing import Any

from scripts import tossing3d_bilevel_pick_probe as probe


def main() -> None:
    """Run one seed at one fixed standoff and write its record to --output."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--samples-per-step", type=int, default=5)
    parser.add_argument("--max-abstract-plans", type=int, default=1)
    parser.add_argument("--planning-timeout", type=float, default=1800.0)
    parser.add_argument("--max-skill-horizon", type=int, default=400)
    parser.add_argument("--distance", type=float, required=True)
    parser.add_argument("--rot", type=float, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    args.pick_mode = "zero-param"

    record: dict[str, Any] = {
        "seed": args.seed,
        "samples_per_step": args.samples_per_step,
        "max_abstract_plans": args.max_abstract_plans,
        "planning_timeout": args.planning_timeout,
        "pick_mode": f"fixed({args.distance}, {args.rot})",
    }
    start = time.perf_counter()
    try:
        from kinder_models.dynamic3d.tossing import (  # noqa: PLC0415
            parameterized_skills,
        )

        parameterized_skills.PickCubeController.STANDOFF = (args.distance, args.rot)
        record.update(probe._run(args))  # noqa: SLF001
        record["outcome_error"] = None
    except BaseException:  # noqa: BLE001 - a crash is a result, not a reason to stop
        record["outcome"] = "crashed"
        record["outcome_error"] = traceback.format_exc()
    record["total_seconds"] = time.perf_counter() - start

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)


if __name__ == "__main__":
    main()
