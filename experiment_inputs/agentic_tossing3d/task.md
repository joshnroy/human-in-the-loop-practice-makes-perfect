# Tossing3D continual practice task

The run first initializes the world, asks `HUMAN_TASK_QUESTION`, and uses the
editable constant `HUMAN_TASK_RESPONSE` in `human_input.json` as its reproducible
answer. That response supplies desired tasks, suggested subtasks, and important
state features. It does not supply abstract-state IDs, policies, outcome labels,
or a transition graph. Generate and print those artifacts before practice begins.
This initial experiment uses the fixed skill slots below; generating new skill
types and allocating their statistical models is outside its scope.

A mobile TidyBot has a seven-joint arm and a gripper. One cube and an open bin are
in a room with a low barrier. The deployment task is to put the cube into the bin.
Picking up the cube and tossing it are distinct skills. A toss can leave the cube
on the opposite side of the barrier, where the robot cannot recover it using the
available skills. Moving the bin to the robot's side can permit repeated practice.

Generate full Python robot policies, not only a sampler over an existing toss
controller's parameters. A policy can control base motion, arm motion, the gripper,
and release timing using the documented low-level action interface. Do not call
an oracle pick/toss policy, a symbolic planner, or hidden simulator operations.

Use these robot option IDs and existing statistical-model skill names:

| Option ID | Belief skill | Intended role |
| --- | --- | --- |
| `pick` | `PickCube` | Acquire a reachable cube securely in the gripper |
| `toss` | `MoveToTossLocationAndToss` | Transport/release the held cube into the bin |
| `open_gripper` | `OpenGripper` | Release an unsuccessful grasp and leave the gripper open |

The harness offers two human options, both accounted for with belief skill
`ask_for_reset_cube_bin_only`:

| Option ID | Explicit harness mechanism | Intended result |
| --- | --- | --- |
| `reset_cube_far` | `opposite_side` | Cube recoverable on the robot side; bin on the opposite side |
| `reset_cube_and_bin_near` | `robot_side` | Cube and bin both on the robot side |

The human mechanism performs a real intervention through the host. Never generate
a controller that teleports objects, resets the environment, or claims a human
intervention has happened. Robot code cannot invoke human help itself. Choosing
human help is a planning decision, and its configured cost is charged separately.

Describe whole scenes recognizable from the configured observation mode in
natural language. The default `object_state` condition supplies semantically
labeled numeric object features and robot measurements; it needs no images.
The separate legacy `rgb` condition supplies images and robot measurements
without dynamic object ground-truth poses. Keep these information boundaries
explicit. Distinguish at
least the following situations whenever they lead to different continuations:

- Cube held securely with the bin on the robot side, versus held with the bin on
  the opposite side.
- Cube free and reachable on the robot side, retaining the bin-side distinction.
- Cube released into the bin, retaining whether the robot can retrieve it there.
- Cube resting beyond the barrier or otherwise not recoverable by these skills.
- An unsuccessful or partial grasp that can be addressed by opening the gripper.

These are design requirements for a generated registry, not predetermined labels
or a predicate vector. A cluster is one complete natural-language scene description.
Every abstract state has this generated membership definition and a generated set
of possible next abstract states. Store the latter as outgoing skill/outcome edges:
the source definition describes where a skill can start, and the destination
definitions describe where its success or failure may leave the robot. The observer
checks observed membership; it does not infer it from an expected transition.
Different clusters can allow the same option and still need distinct outgoing
destinations. Do not merge scenes solely because they enable the same next skill.
Do not create a Cartesian product of independent Boolean feature flags.

The registry must cover the states you intend to plan through without claiming
that the supplied observation reveals hidden geometry or future motion-plan
feasibility. If needed measurements are absent, the cube is occluded in an image,
or descriptions overlap, the observer
must be allowed to return unresolved. It must not guess from the last planned edge.

The observer judges the fixed natural-language success contract using only the
chosen observation mode. It does not read a native reward or scoring-box flag.
Native task scoring can be recorded separately by the harness as an audit: a cube
inside the physical bin need not satisfy the simulator's smaller scoring region.
Do not claim these two measurements are equivalent or use the native score to
replace the generated contract.

Edges connect source cluster, option, success/failure, and destination cluster.
Each outcome's destination weights sum to one. These weights are conditional
transition assumptions until validated with execution; they are not the model's
confidence in its prose and not skill-success probabilities. Failure may have
multiple destinations. Success probabilities come from the existing competence
belief and are never invented in the generated library.

The world persists between skills and practice sessions. Session completion is
not a free reset. Policy learning occurs once at the existing end-of-session
boundary using recorded trajectories; it is never called inside search branches.

Evaluation uses frozen policies in a separate evaluation world. The external
bundle's `evaluation_skill_order` prioritizes the existing toss, pick, and
open-gripper belief slots among options whose initiation checks pass. This is a
Tossing3D evaluation convention, not a rule in the reusable practice search or
a guarantee that an option is applicable. Never reset the practice world to
prepare an evaluation.
