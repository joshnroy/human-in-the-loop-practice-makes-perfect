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
interface. Python's standard library, NumPy 1.26.4, and SciPy 1.14.0 are available.
The planning-enabled image additionally provides only PyBullet and the isolated
planning scene described below. MuJoCo, KINDER, native controllers, and
`pybullet_helpers` are not available. Do not assume other optional dependencies.
Keep bounded work
per call. Return ordinary JSON numbers and lists, not NumPy arrays or scalar types.

The function above describes the initial helper library only. You may replace its
code, parameters, skill identities, state representation, belief model, and planner.
The host imposes no learning boundary. Deployment uses `GeneratedApproach` from the
task prompt and a frozen copy of its supporting files; practice continues while
that copy is evaluated independently.

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




`robot_spec.json` supplies the filtered robot kinematic chain from the initialized
robot model, its tool frame, joint limits, gripper convention, and `action_spec`.
These are known static robot specifications. In the object-state condition they
also contain `planning_scene`, the approved arm URDF metadata and primitive bin
dimensions. Neither specification contains dynamic object coordinates, pretrained
controllers, task-success labels, or a live simulator interface.

The chain is rooted in `world`. Each link supplies a fixed `position_m` and
`quaternion_wxyz` relative to its parent, followed by its local joints. A joint
describes its type, local axis and pivot, reference position, limits, and the
measurement keys for position and velocity. Compose the fixed link transform
and then each joint's transform using measured position minus reference position;
rotations act about the joint's local pivot. The tool's fixed transform is relative
to its stated parent. This permits generated forward-kinematics and numerical
inverse-kinematics calculations using NumPy/SciPy without a native solver helper.
Do not interpret position deltas as tool-frame Cartesian commands or assume an
unlisted robot geometry. The planning model below supplies a separate URDF tool
frame; do not silently equate it with the measured pinch-site frame.
The runtime supplies observations, not the generator's working directory. Embed
needed static kinematic constants in the generated controller source; do not open
`robot_spec.json`, `human_input.json`, or `initial_observation.json` from a running
policy. Dynamic cube/bin positions must always come from the current observation
in object-state mode, never constants copied from the initial scene.

## Isolated PyBullet planning copy

The coding sandbox includes `/opt/hitl-planning/planning_scene.py` and the approved
Kinova Gen3 URDF with its referenced meshes. It does not include native skill
implementations or their grasp transforms. Add `/opt/hitl-planning` to `sys.path`
to import `PlanningScene`. Construct it from an explicitly supplied object-state
snapshot and `robot_spec.json`'s `planning_scene` entry:

```python
with PlanningScene(observation=snapshot, specification=robot_spec["planning_scene"]) as scene:
    pose = scene.tool_pose()
    proposed_joints = scene.inverse_kinematics(position_m=target_position)
    scene.set_configuration(base=base_pose, arm=proposed_joints, gripper=finger_angle)
    distances = scene.collision_distances(distance_m=0.01)
    scene.render_state(path="planning-state.png")
```

This is an arm-only kinematic collision model matching the geometry construction
used for native motion planning: observed cube/static boxes and a hollow bin with
known wall thickness. It omits the mobile-base collision body and task dynamics.
IK returns a numerical candidate, not a feasibility verdict; inspect residuals,
limits, and collisions yourself. Signed contact distances do not label intended
gripper contact as failure. `client_id`, `robot_id`, and `bodies` refer only to this
private PyBullet copy. They never identify or control the live world.

The local `render_state` and `render_policy` tools operate only on that copy.
`scene.render_policy(policy=policy, parameters={}, max_steps=caller_budget,
output_dir="preview")` previews plain 18D commands by applying position deltas
kinematically. It keeps objects fixed, ignores velocity targets, and rejects
millisecond schedules. The caller supplies `max_steps`; no extra fixed preview
step limit is imposed. It writes PNG frames and `preview.json`, marked
`planning_only`, and restores the copy's initial robot configuration afterward.
These previews can expose approach/collision mistakes but cannot establish a
grasp, ballistic toss, state-cluster membership in the real scene, skill success,
or competence. Actual recorded observations remain the evidence for those claims.

Use the initial observation for generation-time geometry. During improvement,
select an actual timestamped snapshot from the supplied new trajectory and pass
it explicitly; `initial_observation.json` is not the current practice state.
Always close planning clients with the context manager above. No network,
MuJoCo rollout, live reset, hidden state, or existing complete Pick controller
is exposed by these planning tools.

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

The supplied policy(observation, memory, parameters) signature describes initial helper skills. The required final interface is GeneratedApproach, as documented in the task prompt.
