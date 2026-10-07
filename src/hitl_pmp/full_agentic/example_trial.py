"""Practice example. Starting this script never resets the world."""

import json
from pathlib import Path

from approach import GeneratedApproach
from env_client import make_env

env = make_env()
spec = json.loads(Path("robot_spec.json").read_text())
agent = GeneratedApproach(spec["action_spec"], {"mode": "object_state"}, {})
state = env.begin_trial()["observation"]
agent.reset(state, {"robot_spec": spec})
try:
    for _ in range(1000):
        action = agent.get_action(state)
        if action is None:
            break
        state = env.step(action)["observation"]
finally:
    result = env.end_trial()
    print(json.dumps(result))
