A mobile robot must place a cube in a bin across a barrier. Practice uses one persistent world and object-centric state observations. Use `env_client.py` for serialized robot commands and human requests. The deployment interface is `GeneratedApproach`; write its implementation.
Available human skills during practice: `reset_cube_far` samples a cube placement in the robot-side pickup area and a bin placement across the barrier; `reset_cube_and_bin_near` samples both on the robot side. Both reposition both objects, preserve the arm and gripper state, and normally preserve the base; the host may return the base to its initial pose if it blocks valid bin placements. The supplied interventions are deterministic host operations. Each has a separate cost function cₕ(e) and duration function dₕ(e), specified in the configuration below.
Maximize expected autonomous deployment success minus the weighted accumulated practice cost. R denotes executed robot control periods. Hₕ denotes dispatched invocations of human skill h; cₕ is its cost function and dₕ its counted-duration function. Each function is evaluated for execution event e.
$$
C_R = \sum_{e\in R} r(e)
$$
$$
C_H = \sum_h \sum_{e\in H_h} c_h(e)
$$
$$
C = C_R + w_H C_H
$$
$$
J = \mathbb{E}[\text{autonomous deployment success}] - \lambda C
$$
$$
B = |R| + \sum_h \sum_{e\in H_h} d_h(e)
$$
Configured cost and duration functions (constant value unless a host function is named):
```json
{
  "robot_step": {
    "value": 1.0,
    "function": null
  },
  "human_skills": {
    "reset_cube_far": {
      "cost": {
        "value": 40.0,
        "function": null
      },
      "duration": {
        "value": 30.0,
        "function": null
      }
    },
    "reset_cube_and_bin_near": {
      "cost": {
        "value": 25.0,
        "function": null
      },
      "duration": {
        "value": 20.0,
        "function": null
      }
    }
  },
  "human_weight": 2.0,
  "objective_lambda": 3e-6
}
```
The host returns current robot-step prices and per-human-skill price/duration quotes.
w_H weights human cost relative to robot cost. λ weights total cost relative to deployment success. Every executed robot control period counts, including unsuccessful attempts, preparation and recovery. Each dispatched human invocation counts once, including failure. Receipt replay, code edits and held-out evaluation add no practice cost.
Practice experience is B = robot control periods + Σdₕ(e), using the configured duration of each human skill. The host returns costs, counted experience, and remaining budget. The ceiling is 85,000 counted steps. Frozen copies are evaluated separately every 1,700 counted steps; results are hidden from you. When finished, commit your controller and call `env.finish_adaptation()`.
Before each practice batch, and whenever you switch strategy or choose to finish, emit a brief "Decision summary:" in your ordinary response (at most two sentences): chosen practice side and next action, expected benefit and robot/human cost tradeoff, supporting observations, and reason for any switch or stop. State uncertainty when estimates are unavailable.
Deployment calls the interface below on current observations. The constructor receives action metadata, the object-state observation-mode descriptor, and an empty primitives dictionary. `reset` initializes controller memory; `get_action` returns one valid low-level command or `None` to end execution. Each held-out task permits 500 robot control steps. Human assistance and model calls are unavailable during deployment.
Write `approach.py` containing a class `GeneratedApproach` with the following interface:
```python
class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives): ...

    def reset(self, state, info): ...

    def get_action(self, state): ...
```
Use the supplied interpreter to execute practice scripts:
```bash
/opt/robocode-strict/bin/python your_script.py
```

Practice has a total budget of 85000 counted steps: one per robot control period plus each human skill's configured duration. The host takes frozen deployment snapshots every 1700 steps without resetting the world or requesting a learning update. Evaluation results are never returned. Stop gracefully when the remaining budget is zero.
