"""Small LOCAL simulation calibration grid, separately for every object and context.

These trials choose preview proposals, not hardware parameters or success-rate estimates.
Each trial starts a fresh scene; every in-episode human relocation is costed.
"""

import argparse
import concurrent.futures
import itertools
import json
import multiprocessing
from pathlib import Path
from typing import Any

from .domain import Calibration, KitchenConfig


def trial(*args: Any) -> dict[str, Any]:
    from .simulator import RecyclingSimulator

    (job,) = args
    name, context, proposal, seed, settings = job
    config = KitchenConfig.model_validate(settings)
    key = f"{name}/{context}"
    config = config.model_copy(update={"calibrations": {key: Calibration(**proposal)}})
    simulator = RecyclingSimulator(config=config, seed=seed, pretty=False)
    try:
        if context == "near":
            simulator.human(action="relocate_bin", destination=(0.65, 1.4, 0))
        assert simulator.episode.context() == context
        picked = simulator.pick(name=name)
        tossed = simulator.toss(name=name) if picked["success"] else None
        return {
            "object": name,
            "context": context,
            "seed": seed,
            "proposal": proposal,
            "pick": picked,
            "toss": tossed,
            "observation": simulator.episode.observations[name].model_dump(),
            "events": simulator.episode.events,
            "human_cost": simulator.episode.human_cost,
        }
    finally:
        simulator.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--workers", type=int, default=3, choices=range(1, 5))
    parser.add_argument("--object", action="append", dest="objects")
    parser.add_argument("--context", action="append", choices=("near", "far"), dest="contexts")
    arguments = parser.parse_args()
    config = (
        KitchenConfig.model_validate_json(arguments.config.read_text())
        if arguments.config
        else KitchenConfig()
    )
    arguments.output_dir.mkdir(parents=True, exist_ok=False)
    jobs = []
    for name in arguments.objects or [spec.name for spec in config.objects]:
        kind = config.object(name=name).kind
        offsets = (0.02, 0.03) if kind == "bottle" else (0.045, 0.05)
        speeds = (150, 170, 190) if kind == "bottle" else (130, 150, 170)
        for context in arguments.contexts or ("near", "far"):
            azimuth = {"cube": 0.4, "bottle": 0.0, "can": -0.4, "packaging": 0.4}[kind]
            for offset, speed, release in itertools.product(offsets, speeds, (640, 720)):
                proposal = Calibration(
                    grasp_offset=offset,
                    toss_speed_deg_s=speed,
                    release_ms=release,
                    toss_azimuth=azimuth if context == "far" else 0,
                )
                jobs.append((name, context, proposal.model_dump(), 0, config.model_dump()))
    results = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=arguments.workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        for result in pool.map(trial, jobs):
            results.append(result)
            print(json.dumps(result), flush=True)
            with (arguments.output_dir / "trials.jsonl").open("a") as stream:
                stream.write(json.dumps(result) + "\n")
    selected = {}
    unresolved = []
    successful_toss = []
    for name, context, *_ in jobs:
        key = f"{name}/{context}"
        candidates = [
            result
            for result in results
            if result["object"] == name and result["context"] == context
        ]
        successful = [
            result for result in candidates if result["toss"] and result["toss"]["success"]
        ]
        if successful:
            selected[key] = Calibration(**successful[0]["proposal"])
            if key not in successful_toss:
                successful_toss.append(key)
        elif key not in unresolved:
            unresolved.append(key)
            picked = [result for result in candidates if result["pick"]["success"]]
            if picked:
                selected[key] = Calibration(**picked[0]["proposal"])
    output = config.model_copy(update={"calibrations": selected})
    (arguments.output_dir / "simulation-proposals.json").write_text(
        output.model_dump_json(indent=2)
    )
    (arguments.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "trials": len(results),
                "selected": list(selected),
                "unresolved_toss": unresolved,
                "successful_toss_contexts": successful_toss,
                "scope": "One seed per proposal; independent object/context simulation checks only",
                "hardware_validated": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
