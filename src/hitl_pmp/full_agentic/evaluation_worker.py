"""Container-only frozen GeneratedApproach evaluation, with no practice client."""

import json
import random
import sys
from pathlib import Path

from policy_worker import ContainerPolicyWorker as Relay


def main() -> None:
    try:
        sys.path.insert(0, "/sandbox/submission")
        from approach import GeneratedApproach

        random.seed(0)
        import numpy as np

        np.random.seed(0)
        observation = Relay.request(payload={"operation": "observe"})
        approach = GeneratedApproach(observation["action_spec"], {"mode": "object_state"}, {})
        approach.reset(observation, {})
        for _ in range(
            json.loads(Path("/sandbox/evaluation_config.json").read_text())["max_steps"]
        ):
            action = approach.get_action(observation)
            if action is None:
                break
            # The wire protocol uses JSON; controllers may return NumPy vectors
            # or schedules. Keep values/shape unchanged and validation in the relay.
            observation = Relay.request(
                payload={"operation": "step", "action": np.asarray(action).tolist()}
            )
        Relay.request(payload={"operation": "finish", "done": True})
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
