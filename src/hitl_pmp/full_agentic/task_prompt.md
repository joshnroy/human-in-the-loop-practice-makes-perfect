You are writing an approach for an environment that you can only access as a black box.

The environment is described below.

A mobile TidyBot with a seven-joint arm and gripper must place a cube
in a bin across a low barrier. Practice may leave the cube beyond autonomous reach.
The initial Python skill code is the same untrained generated library used by the
existing agentic experiment. approach.py contains a simple bootstrap composition;
you may rewrite it and its helpers freely. No fixed skill slots, state clusters,
belief model or planner are required. robot_spec.json contains known kinematics
and the permitted local PyBullet planning scene. Planning previews do not simulate
grasp/toss dynamics and are not practice evidence. Native environment source,
controllers, and simulator reset operations are unavailable.


The environment is a black box. Environment source and native controllers
are unavailable. The existing robot API and observation fields are documented in
robot_api.md and robot_spec.json. Observations are JSON dictionaries, exactly as
provided to the initial robot policies, not Gym vectors or ObjectCentricState objects.

env_client.py is the practice-only interface. Every make_env() connects to the SAME
persistent world. There is no free reset, set_state, or independent test world.
Observe before making decisions. Parallel analysis and coding are available;
serialize physical experiments in this shared world. A stale-world error requires
observing again and deciding from the actual current state. After a connection
failure, retry_last_request() retrieves the durable receipt without repeating
the physical action. Do not replay uncertain actions with new request IDs.

from env_client import make_env
env = make_env()
record = env.observe()
obs = record["observation"]
env.begin_trial()                 # snapshot current code; robot invocation cost 1
record = env.step(action)         # one documented robot control period
env.end_trial()                   # no reset; no simulator change
record = env.request_help("reset_cube_far")  # human request cost 5

A trial is one bounded controller execution, with RoboCode's 1000-step limit.
You can choose any code, parameters, controller structure and stopping point.
The host supplies the total counted-step budget below. No additional trial-count or human-request-count cap is imposed.
The server also records all control steps so different controller granularities
remain visible. Trial boundaries do not create new scenes or enforce learning.

Human assistance is available during practice. The two supplied interventions are:
- reset_cube_far: place the cube in the robot-side pickup area and the bin across
  the barrier.
- reset_cube_and_bin_near: place both cube and bin on the robot side.
Pause execution with end_trial() before requesting assistance. Both operations
sample fresh placements for BOTH objects. They preserve the arm and normally the
base; the supplied handler may move the base to its initial pose if it blocks all
valid bin placements. They do not open the gripper or retract the arm.
Observe the returned state. These are the existing deterministic host mechanisms;
this pilot does not introduce unknown helper reliability or invented human outcomes.
The request count and configured cost are charged by the host. A host failure is
an operational failure, not successful help. You may request help whenever its
expected benefit justifies its cost, including before becoming stuck.

Use the existing linear-cost objective: autonomous task success probability minus
3e-6 times accumulated robot-invocation and human-request cost. The current cost,
step count and resource status are returned with every tool result. Model spending
is independently capped by the coding CLI. Evaluation results are hidden.

The host records observations and actions in evidence/practice.jsonl. You may keep
your own files and models. All four responsibilities--policy learning, state
representation, uncertainty tracking, and planning--are yours; no fixed graph,
belief estimator, or practice schedule is imposed.

Your approach.py is the frozen deployment program. It must NOT import env_client.
The constructor receives the action-spec dictionary, {"mode":"object_state"},
and an empty primitives dictionary. reset(state, info) receives the current
observation and info["robot_spec"]. reset initializes YOUR program; it does not
reset the physical practice world. get_action(state) returns a valid low-level
action; returning None ends the task. Supplied supporting Python and static data
files are included with the frozen program. No practice evidence is included.
No model calls or human help are available during held-out evaluation.
The evaluator uses the native task criterion, which can be stricter than merely
being inside the bin. There is no success oracle in practice observations.

When you decide adaptation is finished, end any active trial, commit the current
program, call env.finish_adaptation(), and end your coding session. This snapshots
the final artifact. Keep approach.py usable throughout, as in ordinary RoboCode.


Write `approach.py` containing a class `GeneratedApproach` with the following interface:

```python
class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        """Initialize with the environment's gym spaces."""
        ...

    def reset(self, state, info):
        """Called at the start of each episode with the initial state."""
        ...

    def get_action(self, state):
        """Return a valid action for the given state."""
        ...
```

The class can maintain internal state between calls (e.g., a computed plan). The `reset` method is called at the start of each episode. The `get_action` method is called each step and must return a valid action.

Read robot_api.md and robot_spec.json for allowed robot observations, actions and planning tools. No native skill primitives are supplied.

Write the best approach you can — ideally one that solves the environment optimally. Your `approach.py` should only use packages available in the current environment. Write test scripts that use the real environment to verify your approach works. Once you have a strong candidate, search for instances it does not yet solve, then use those failures to improve it.

IMPORTANT: Use `/opt/robocode-strict/bin/python` to run your test scripts, since that interpreter has all required packages installed. For example:
```bash
/opt/robocode-strict/bin/python test_approach.py
```
