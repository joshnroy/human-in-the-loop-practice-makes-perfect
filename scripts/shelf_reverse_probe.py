"""Explicit saved-state reverse-skill diagnostic; not a continuous full proof."""

import argparse
import json
import pickle
import traceback
from pathlib import Path
import numpy as np
import imageio.v2 as imageio
from shelf_physical_demo import Demo
from shelf_reverse_controllers import RetrieveShelfController

parser = argparse.ArgumentParser()
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
saved = pickle.loads((args.source / "final.pkl").read_bytes())
demo = Demo(output=args.output, seed=1, max_actions=1)
demo.env.set_state(saved["state"])
demo.state = saved["state"]
demo.initial = saved["initial"]
demo.records = saved["actions"]
demo.ticks = saved["ticks"]
demo.save()
imageio.imwrite(args.output / "final-forward.png", demo.env.render())
trials = []
try:
    for i in range(8, 0, -1):
        cube = demo.state.get_object_from_name(f"cube{i}")
        for distance in (1.025, 1.0, 1.05):
            base = [demo.state.get(demo.cupboard, "x") - distance, demo.state.get(cube, "y"), 0.0]
            controller = RetrieveShelfController(objects=[demo.robot, cube], pybullet_sim=demo.sim)
            row = {"cube": cube.name, "base": base}
            try:
                parameters = np.array([*base, 0.5, 0.5, 0.5, 0.5])
                controller.reset(demo.state, parameters)
                row["planned"] = True
                trials.append(row)
                demo.execute("retrieve_shelf", cube, controller, parameters)
                raise SystemExit(0)
            except Exception as error:
                row["planned"] = False
                row["error"] = type(error).__name__ + ": " + str(error)
                row["traceback"] = traceback.format_exc()
                trials.append(row)
finally:
    (args.output / "reverse-planning.json").write_text(json.dumps(trials, indent=2))
    demo.snapshot("reverse-final")
    demo.writer.close()
    demo.env.close()
