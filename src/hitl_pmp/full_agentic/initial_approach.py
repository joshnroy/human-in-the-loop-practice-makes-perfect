"""Bootstrap composition of the shared, untrained generated skill library."""

import json
from pathlib import Path
from typing import Any

from policies import open_gripper, pick, toss


class GeneratedApproach:
    def __init__(self, action_space: Any, observation_space: Any, primitives: Any) -> None:  # noqa: PLR0917 -- supplied RoboCode client/interface contract
        library = json.loads(Path(__file__).with_name("initial_library.json").read_text())
        self.options = {o["option_id"]: o for o in library["manifest"]["options"]}
        self.modules = {"pick": pick, "toss": toss, "open_gripper": open_gripper}

    def reset(self, state: Any, info: Any) -> Any:  # noqa: PLR0917 -- supplied RoboCode client/interface contract
        self.stage = 0
        self.steps = 0
        self.memory: dict[str, Any] = {}

    def get_action(self, state: Any) -> Any:  # noqa: PLR0917 -- supplied RoboCode client/interface contract
        sequence = ["pick", "toss", "open_gripper"]
        if self.stage >= len(sequence):
            return None
        name = sequence[self.stage]
        option = self.options[name]
        result = self.modules[name].policy(state, self.memory, option["parameters"])
        self.memory = result["memory"]
        self.steps += 1
        if result["done"] or self.steps >= option["max_steps"]:
            self.stage += 1
            self.steps = 0
            self.memory = {}
        return result["action"]
