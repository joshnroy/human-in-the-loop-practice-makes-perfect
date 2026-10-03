# Controller interface and native action space

A controller file exports this function:

```python
def policy(observation: dict, memory: dict, parameters: dict) -> dict:
    # Compute a complete low-level command from this observation and local memory.
    return {"action": action, "memory": next_memory, "done": False}
```

The runtime invokes the function in its isolated worker. `memory` starts empty
for each option invocation and is returned to the next call of that same option.
All inputs and outputs are JSON-compatible. There are no environment objects,
simulator handles, oracle skills, network tools, or host filesystem access in this
interface. Python's standard library, NumPy 1.26.4, and SciPy 1.14.0 are available
in the configured strict image. Do not assume a robotics library, image package,
simulator package, or other optional dependency is installed. Keep bounded work
per call. Return ordinary JSON numbers and lists, not NumPy arrays or scalar types.

`parameters` contains optional numeric settings from the accepted manifest. The
coding learner may revise the complete policy source, including feedback logic,
motion phases, and release logic. It is not restricted to adjusting those settings
or producing a four-number toss proposal. For the initial experiment, keep the
manifest fixed and put policy revisions in the controller files.

## Actual observations

The observation envelope contains these common fields:

```text
observation_mode object_state or rgb
proprioception  dictionary of robot measurements, each a finite number
control_step    dispatched control-period count local to this bridge instance
simulation_time_s actual simulator clock in seconds, shared within the current world
action_spec     the live action shape, bounds, and control metadata
execution_seed integer seed assigned to this option invocation by the host
```

The generator receives `initial_observation.json`, captured after the practice
world is initialized. It is an example from that actual world, not a forecast or
a reusable constant for later control. Each policy invocation receives current
observations through this interface. The run fixes the observation mode; do not
switch to another source when perception or a policy fails.

`simulation_time_s` is finite and nonnegative. Use it to compare observation times
and durations across the observing and executing bridges. It resets when the world
is initialized or an evaluation task is restored, so do not compare timestamps
across those boundaries. `control_step` is local to a bridge: an observing bridge
may report zero after another bridge has executed many commands. It is not a
global clock or evidence that the world was reset.

### Object-state condition

`object_state` additionally supplies:

```text
objects         [{"name": "...", "type": "...", "features": {"label": number}}]
state_spec      feature units and coordinate-frame conventions
```

This is an intentionally privileged semantically labeled vector, not a claim
that a real camera directly measures object state. Read object names, types, and
features from the actual observation rather than assuming index order or inventing
a missing object. Physical-object features include `x`, `y`, `z` (world position
in metres), `qw`, `qx`, `qy`, `qz` (body orientation as a wxyz quaternion), `vx`,
`vy`, `vz` (world linear velocity in metres/second), and `wx`, `wy`, `wz`
(object-local angular velocity in radians/second). `bb_x`, `bb_y`, `bb_z` are
full bounding-box dimensions along the object frame's axes, not half extents.
Interpret them with the orientation; they are not an oracle for free space,
contact, grasping, or motion-plan feasibility. Only use fields actually supplied;
`state_spec` is authoritative. Robot entries use the measured robot fields
described below. No images are required for generation, classification, or
option checks in this condition.

### Legacy RGB condition

`rgb` supplies `images`, a list of actual rendered PNG data URLs, and `image_views`
identifying those views. It does not supply `objects` or dynamic cube/bin poses.
The visual observer receives these images as image inputs alongside robot
measurements. This legacy camera configuration is not by itself a claim of
real-TidyBot++ sensor parity; later observation conditions must specify their
camera sets explicitly. A recent frame may be supplied with a timestamp; do not
interpret a prior frame as the current state.

If a policy uses randomness, derive it from `execution_seed`; never seed from
wall-clock time or an unseeded generator. For NumPy, use
`numpy.random.default_rng(observation["execution_seed"])` and retain its JSON-safe
bit-generator state in `memory` between calls, restoring that state before drawing
again. Do not recreate the same initial generator on every control step. Random
choices must reproduce when the invocation seed and observation sequence match.

Native robot measurements include `pos_base_x`, `pos_base_y`, `pos_base_rot`,
`pos_arm_joint1` through `pos_arm_joint7`, `pos_gripper`, and their `vel_` counterparts
(`vel_base_x`, `vel_base_y`, `vel_base_rot`, `vel_arm_joint1` through
`vel_arm_joint7`, `vel_gripper`). These are robot proprioception, not object poses.
Base x/y are metres, base yaw and arm joints are radians, and velocity fields use
the corresponding units per second. The bridge supplies measured finger-driver
joint position in radians as `pos_gripper` and the normalized
actuator command separately as `gripper_command`. This differs deliberately from
the native object-centric observation's `pos_gripper` command field. Neither a
closed command nor finger position alone proves a cube is held. Establish grasp
success from the supplied observations and the fixed option contract; proximity
alone does not establish a secure grasp.

In `object_state` mode, robot proprioception also includes `pos_tool_x`,
`pos_tool_y`, `pos_tool_z` (world position in metres) and `quat_tool_w`,
`quat_tool_x`, `quat_tool_y`, `quat_tool_z` (world orientation, wxyz quaternion).
These describe the robot's pinch-site frame from measured configuration and known
robot kinematics. They are robot geometry, not contact, holding, or success labels.
Relative cube/tool geometry can support a grasp judgment but is not sufficient
by itself to prove that the cube follows the hand securely.

### Physical history for observation judgments

The host may append `recent_observations` to a numeric observation sent to the
state/option judge. This contains at most eight chronological physical numeric
frames sampled from the host-recorded execution trajectory, spanning its first
frame, interior frames, and final three frames when enough are available. Use
`simulation_time_s` for their ordering and elapsed durations; sampling may skip
steps, so adjacent entries are not necessarily one control period apart. Equal
timestamps may repeat the same physical instant and do not provide independent
motion evidence. Do not order different bridge observations by `control_step`.
This history contains measurements,
not native success labels, expected graph destinations, or controller claims.

Use supplied history when a fixed description requires temporal evidence, such
as the cube moving with the hand rather than merely being near it. Judge current
membership from the latest observation, using earlier frames only to establish
the required physical relationship over time. Do not treat a previous pose as
the current pose, invent motion between samples, or assume a grasp continued
after the last supporting measurement. Missing or inconclusive temporal evidence
requires an unresolved judgment when the contract depends on it.

This bounded history is for judges only. The policy still receives the current
observation and its own persistent `memory`; it must retain any measurements it
needs between calls. The session-boundary coding learner receives the full
recorded trajectory. Neither mechanism adds histories to hypothetical search.

Neither mode supplies predicate values, goal flags, contact-oracle labels, future
states, or native planner feasibility. Object coordinates are available only in
the explicitly configured `object_state` condition. A numerical controller
cannot call the observer itself through this API. Any additional perception
features must be explicitly supplied by the runtime, not assumed to exist. If
observations are inadequate for a proposed controller, report the gap instead of
fabricating fields or an oracle function.

## Known robot specifications

`robot_spec.json` supplies the filtered robot kinematic chain from the initialized
robot model, its tool frame, joint limits, gripper convention, and `action_spec`.
These are known static robot specifications. They contain no dynamic object
coordinates, object meshes, pretrained controllers, or simulator interfaces.

The chain is rooted in `world`. Each link supplies a fixed `position_m` and
`quaternion_wxyz` relative to its parent, followed by its local joints. A joint
describes its type, local axis and pivot, reference position, limits, and the
measurement keys for position and velocity. Compose the fixed link transform
and then each joint's transform using measured position minus reference position;
rotations act about the joint's local pivot. The tool's fixed transform is relative
to its stated parent. This permits generated forward-kinematics and numerical
inverse-kinematics calculations using NumPy/SciPy without a native solver helper.
Do not interpret position deltas as tool-frame Cartesian commands, assume an
unlisted robot geometry, or import a simulator to compute these transforms.
The runtime supplies observations, not the generator's working directory. Embed
needed static kinematic constants in the generated controller source; do not open
`robot_spec.json`, `human_input.json`, or `initial_observation.json` from a running
policy. Dynamic cube/bin positions must always come from the current observation
in object-state mode, never constants copied from the initial scene.

## Low-level commands

Tossing3D enables the native 18-dimensional TidyBot action space:

| Indices | Meaning | Units and interpretation |
| --- | --- | --- |
| `0:2` | Base x/y commands | Position deltas in world coordinates, metres |
| `2` | Base yaw command | Yaw delta, radians |
| `3:10` | Seven arm position commands | Joint-angle deltas, radians |
| `10` | Gripper command | Absolute normalized force command: `0` opens, `1` closes |
| `11:18` | Seven arm velocity targets | Radians per second; these are targets, not deltas |

Use a JSON list with exactly the live action dimension and finite numeric entries.
The first ten coordinates have native bounds `[-0.1, 0.1]`; the gripper has bounds
`[0, 1]`. Native velocity bounds are infinite, but the host may impose tighter
finite limits. Obey the supplied live bounds and configured action limits. Do not
output infinities, NaNs, or a whole-skill action name.

For substep release timing the bridge also accepts
`{"schedule": [[18 values], ...]}` with exactly `action_spec.schedule_rows` rows.
Each row uses the same coordinate order and bounds. Rows span one control period
at `action_spec.schedule_timestep_s` (currently one millisecond). A schedule is
still one host control step, not an unlimited sequence; a plain 18-vector remains
valid for ordinary commands. Use only the published schedule length and timing.

The default native configuration uses delta position commands at 10 control steps
per second. The supplied `action_spec` is authoritative if the runner changes
these settings. Position deltas are relative to the robot's current measured
configuration, not accumulated target offsets. The simulator converts arm
position and velocity targets to torques with its existing controller. This is
joint-space control, not end-effector Cartesian displacement or direct torque.

Maintain the intended gripper command at each step. An all-zero action opens the
gripper; it is not a neutral action for a held cube. A zero arm position delta with
a nonzero velocity target can still move the arm. Think through both position
and velocity terms when defining holds, accelerations, and release.

`done` is a request to finish execution, not a success label. The host separately
checks the option's termination and success descriptions. Timeouts, worker
exceptions, and invalid action outputs are recorded explicitly; they do not
authorize a reset or constitute evidence that the cube reached the bin.
