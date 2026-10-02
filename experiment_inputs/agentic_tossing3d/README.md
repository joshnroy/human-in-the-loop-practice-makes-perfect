# Tossing3D inputs for language-defined options

This directory is an **experiment input**, separate from the reusable runtime and
the planning method. The prompts describe the experiment we want the model to
construct; they do not implement the search algorithm or supply a hidden predicate
classifier. `bundle.json` records the files and experiment-specific accounting and
human-intervention choices.

The first condition is `observation_mode: "object_state"`: semantically labeled
numeric object and robot observations. It intentionally exposes simulator object
measurements and requires no camera images. The legacy `rgb` condition remains
separate and supplies images plus robot measurements without object poses.

## Files

| File | Consumer | Purpose |
| --- | --- | --- |
| `human_input.json` | Intake, library generator, and policy learner | Exact human question and editable constant task response for reproducible runs |
| `task.md` | Library generator and policy learner | Task, permitted human help, and the distinctions the abstraction must retain |
| `robot_api.md` | Library generator and policy learner | Observation and low-level robot-control contract |
| `generate.md` | Library generator | Propose natural-language clusters, option contracts, a skill-chain, and complete Python policies |
| `classify.md` | Observation judge | Assign the actual numeric or visual observation to one whole-state cluster, or abstain |
| `verify_option.md` | Observation judge | Check initiation, termination, or success independently |
| `improve.md` | Coding learner | Revise policy source at a practice-session boundary while contracts remain fixed |

The runner supplies `task.md`, `robot_api.md`, `human_input.json`, and the resolved
experiment settings to generation and improvement. After initialization, it also
stages `initial_observation.json` and `robot_spec.json` with actual initial
observations and filtered robot kinematics. These latter files are run artifacts,
not manually fabricated state examples committed in this input directory. It
supplies the relevant registry or option and the actual observation with each
judgment request. The prompts are plain text; no template
substitution language or code import is needed. Preserve their exact bytes with
the run configuration so a result identifies the inputs it used.

The isolated coding agent writes its generation result to `generated_library.json`
and its code-only improvement result to `revision.json`. At an improvement
boundary, `library.json`, `evidence.json`, and the current policy files provide
the inputs; no live simulator relay or free-reset endpoint is exposed.

Edit `HUMAN_TASK_RESPONSE` in `human_input.json` to change the reproducible human
answer. The default is AI-authored task prose, not a human's recorded response.
`HUMAN_TASK_QUESTION` preserves the user's requested intake question. The answer
contains no abstract-state IDs, numeric motion policy, generated outcome labels,
or fabricated geometry. Snapshot the input with each run before generation.

`robot_options` fixes the exact option-ID to existing belief-skill mapping. The
generator may write each option's policy, but cannot exchange the pick and toss
statistical slots, create an extra option under a known slot, or invent a new slot.
The resolved bundle and cost settings accompany both generation and improvement.

`evaluation_skill_order` stores the Tossing3D evaluation preference using existing
belief-skill names: toss, then pick, then open gripper. Applicability still needs
an observation-based initiation check. This configuration belongs to the external
experiment; it does not alter practice search or the competence forecast.

## Explicit smoke-run configuration

The experiment launcher accepts `--observation-probability-weight 0` for the
initial execution/revision smoke run. It forwards this to the method's
`--agentic-observation-probability-weight`; the normal default remains `0.1`.
The initial diagnostic selected STOP at the default: the weighted observation
entropy penalty was about `0.101`, exceeding the projected improvement of about
`0.0162`. Setting the weight to zero is an explicit experimental configuration
change to exercise the full lifecycle, not evidence that the default search
chooses to practice. The learner, competence forecast, and search implementation
remain unchanged. Keep this override in the run snapshot and distinguish its
outcomes from the default condition; see the
[experiment log](../../docs/experiment-logs/2026-10-02-tossing3d-agentic-options.md)
for the recorded diagnostic and follow-up results.

## What is generated

Each abstract state consists of a generated natural-language membership definition
and its possible successors. The manifest stores definitions in `clusters` and
successors in `edges`, linked by state ID. For a state `k`, its successor set is all
destinations of edges whose source is `k`; each edge retains the skill and its
success/failure outcome. This avoids maintaining a second, potentially inconsistent
adjacency list. Outgoing skills are candidates for initiation from that region;
the observer still verifies the option's initiation contract before real execution.

The proposed cluster descriptions, option descriptions, controller source, and
skill-chain are model outputs. This directory intentionally contains no purported
generated library. Running a prompt is not evidence that a library was validated:
save the actual response, validated manifest, policies, model identifier, and
execution outcomes before reporting generation or task success.

The initial prompts have not themselves established native task success or a
learning benefit. Test fixtures and hand-written smoke controllers must be reported
as such; they are not model-generated skills or successful learning runs.

Competence, learning-rate, and cost beliefs, their updates and refit forecast,
search horizon and objective, sandbox restrictions, and the session boundary are
implemented outside these files. The LLM does not invent these values. The bundle
uses the current Tossing3D configured costs of one per robot skill and five per
human reset. If an experiment overrides them, its resolved input must say so and
the host replaces generated accounting fields with those configured values before
the manifest is used. Generated policy text cannot redefine the charged cost.

## Human actions and frozen contracts

The external mapping in `bundle.json` binds `reset_cube_far` to `opposite_side`
and `reset_cube_and_bin_near` to `robot_side`. These are explicit harness actions,
not generated Python controllers or inferred meanings of arbitrary option names.
The existing human intervention returns the cube to the robot's side and places
the bin on the selected side. Both use the existing human-reset belief skill.
The runner must validate each generated destination against the observed result.

Once accepted, hold cluster descriptions, option initiation/termination/success
descriptions, human mechanisms, and the proposed chain fixed during the initial
policy learning comparison. Code revisions happen only between practice sessions.
The runtime can retain empirical destination counts over the existing registry;
these are execution evidence, not new generated descriptions or model confidence.
An observed destination absent from the initial graph challenges that proposal and
must remain visible in the audit. Unknown cluster IDs are rejected. A code revision
invalidates old conditional-destination counts so they do not silently describe a
different policy. Do not rewrite the abstraction to make a rollout appear consistent.

## Control-source checks

The control description was checked against the native KINDER implementation:

- `reference/kindergarden/src/kinder/envs/dynamic3d/task_families.py`,
  `Tossing3DEnv.__init__`: enables arm-velocity targets.
- `reference/kindergarden/src/kinder/envs/dynamic3d/robots/tidybot_robot_env.py`,
  `TidyBot3DRobotActionSpace` and `TidyBotRobotEnv._action_to_ctrl`: action order,
  delta semantics, and gripper direction.
- `reference/kindergarden/src/kinder/envs/dynamic3d/envs.py`, `TidyBot3DConfig`
  and `ObjectCentricTidyBot3DEnv._get_object_centric_robot_data`: control frequency
  and robot proprioception.

The live `action_spec` remains authoritative. These are implementation references,
not files exposed to the isolated coding agent.

The strict Robocode image permits the Python standard library, NumPy, and SciPy.
An offline import check of image
`sha256:712a0a55dc7fc61cb30ab043a79f176f524c937885437bf084f7abb95ed9f531`
on 2026-10-03 returned NumPy 1.26.4 and SciPy 1.14.0. The source Dockerfile and
strict-runtime package check agree on those two optional packages. This check
does not establish that any generated controller works.
