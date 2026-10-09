"""Copied into the disconnected policy container; never imports generated code on host."""

import json
import random
import sys
from pathlib import Path
from typing import Any


def run_controller(*, controller_type: Any, request: Any, skill: str, limit: int) -> bool:
    import numpy as np

    observation = request(payload={"operation": "observe"})
    controller = controller_type(observation["action_spec"], {"mode": "object_state"}, {})
    controller.reset(observation, skill)
    for _ in range(limit):
        action = controller.get_action(observation)
        if action is None:
            return True
        observation = request(payload={"operation": "step", "action": np.asarray(action).tolist()})
    return False


def main() -> None:
    from policy_worker import ContainerPolicyWorker as Relay

    # Authenticate even if the generated module fails to import.
    Relay.request(payload={"operation": "observe"})
    try:
        import numpy as np

        config = json.loads(Path("/sandbox/skill_config.json").read_text())
        random.seed(config["seed"])
        np.random.seed(config["seed"])
        sys.path.insert(0, "/sandbox/submission")
        from skills import GeneratedSkills

        done = run_controller(
            controller_type=GeneratedSkills,
            request=Relay.request,
            skill=config["skill"],
            limit=config["limit"],
        )
        Relay.request(payload={"operation": "finish", "done": done})
    except BaseException as exc:
        Relay.request(
            payload={
                "operation": "finish",
                "done": False,
                "error": f"{type(exc).__name__}: {exc}"[:2000],
            }
        )


if __name__ == "__main__":
    main()
