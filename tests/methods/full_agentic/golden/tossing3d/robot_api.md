### Practice API
`make_env()` connects to the persistent world. `observe()` returns its current observation. `step(action)` executes one robot control period and returns the updated observation and accounting. `retry_last_request()` retrieves a receipt after an uncertain connection failure without executing again. `finish_adaptation()` ends practice and freezes the current controller.
`request_help(intervention_id)` invokes the named human skill and returns the resulting observation, outcome, and accounting.
API signatures:
```python
env = make_env()
env.observe()
env.step(action)
env.request_help(intervention_id)
env.retry_last_request()
env.finish_adaptation()
```
### Robot API
Observations contain named, typed objects with numeric features, robot proprioception, timestamps, and the current `action_spec`. Object positions use world-frame metres; orientations use wxyz quaternions; linear velocities use metres/second and angular velocities use radians/second. Robot base position is in metres, and joint angles and base yaw are in radians. Use the provided field names, units, action dimensions, and bounds.
A normal command is an 18-element finite numeric vector: indices 0–1 are base x/y position deltas; index 2 is base yaw delta; indices 3–9 are arm joint-position deltas; index 10 is the gripper command (0 open, 1 close); indices 11–17 are arm joint-velocity targets. Position deltas are relative to the current measured configuration. A schedule contains exactly `action_spec.schedule_rows` such vectors, spanning one control period at `action_spec.schedule_timestep_s`. Either command form counts as one robot step.
