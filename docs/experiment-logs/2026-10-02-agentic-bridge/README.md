# Native Tossing3D bridge verification

This is an interface check with trusted numeric actuator commands. It is **not**
a generated-policy, skill-success, learning, or human-intervention experiment.

The seed-125 world was initialized once. One registered probe option sent one
18-dimensional joint-space command and one 100-by-18 control schedule through the
Unix-socket robot relay. Both control periods completed: **2/2** requested periods,
charged to **1/1** option slot. The base x position changed from
`-0.027972321542299262` to `-0.023620206204228205` metres. These are deterministic
probe observations; they support no statistical performance inference.

The relay returned rendered pixels, robot proprioception and a control-step count.
It did not return native rewards, goal verdicts, object poses, or predicate atoms.
The action specification reported 10 Hz control, 1 ms schedule rows, and finite
actuator bounds. Both KINDER imports resolved inside the probe worktree's pinned
submodules. The process ran in a 6 GB memory-capped systemd scope.

| Before the probe | After two control periods |
| --- | --- |
| ![Actual initial camera observation](initial.png) | ![Actual camera observation after the probe](after.png) |

[Machine-readable result](result.json) records the native action specification,
observation schema, dependency paths and action accounting.

The first probe exposed an observation-quality issue: the cube and gripper are
small in the room camera. The bridge now supplies **two actual views**: the room
overview and robot wrist camera, with their names in `image_views`. The accepted
actuator specification is also included in every observation. A repeat of the
same probe passed with this complete envelope; [the separate result](multiview-result.json)
preserves that revision without overwriting the first record.

| Wrist view before the probe | Wrist view after two control periods |
| --- | --- |
| ![Actual robot wrist view before the probe](wrist-initial.png) | ![Actual robot wrist view after the probe](wrist-after.png) |

Reproduce from a worktree with its pinned submodules and scene assets populated:

```bash
scripts/with_env.sh systemd-run --user --scope -p MemoryMax=6G \
  -p OOMPolicy=continue python scripts/probe_agentic_tossing3d.py \
  --through-relay --output-dir artifacts/agentic-bridge-relay-smoke
```

The script selects the current worktree's simulator sources without changing
shared editable installs. Scene meshes/textures may be reused as read-only assets;
the simulator source and task definition remain pinned to the worktree.

The isolated generated-code launcher and live model calls are separate checks.
Passing this native probe does not establish that those integrations run or that
the learned skills solve Tossing3D.

## Agentic experiment protocol and current blocker

[The example runtime configuration](runtime.example.json) contains placeholders,
not selected models or credentials. Set both absolute paths, choose the coding
and multimodal models, and set the vision endpoint. The vision schema follows
Robocode's `OpenAICompatibleClient`: `provider`, `model`, `base_url`, and
`api_key_env`. An empty `api_key_env` is for a keyless local server; for a hosted
endpoint, supply the **name** of an existing credential environment variable.
Do not put secret values in the JSON. The coding backend can be `codex` or
`claude`; its model, budget and turn limit are explicit experiment settings.

The protocol first hashes and snapshots the runtime configuration and external
input bundle. It then invokes the exact disconnected policy launcher with a
trusted marker program. Robocode's namespace and strict-image checks execute
before that marker. This preflight makes **zero model calls** and does not
initialize the simulation.

```bash
scripts/with_env.sh python scripts/agentic_tossing3d_experiment.py \
  --runtime-config /absolute/path/to/runtime.json \
  --inputs experiment_inputs/agentic_tossing3d \
  --results-root artifacts/agentic-tossing3d-first-run \
  --max-workers 1 --num-seeds 1 --num-cycles 2 \
  --max-steps-per-interaction 10 --num-test-tasks 2 \
  --human-reset --human-reset-practice-cost 5 --memory-max 8G
```

Add `--preflight-only` to check infrastructure without generation or experiment
execution. Use a fresh results directory for each attempt; the protocol refuses
to overwrite a previous provenance snapshot. A saved generated library can be
provided using `--library /absolute/path/to/library.json`.

After a successful preflight, the script launches `scripts/run_sweep.py` inside
a systemd scope with the requested memory cap and `OOMPolicy=continue`. It uses
`--method agentic-options`, fixed simulator seeds `0..num_seeds-1`, an explicit
worker limit, and `--practice-reset-policy never`. Evaluation uses its own
environment. A paired comparison can run a second fresh protocol directory with
`--no-human-reset` and the same seed count and initial library. Initial generated
code, judgments, revisions, `stats.json`, and `timing.json` remain separate
artifacts; fixed simulator seeds do not imply deterministic model completions.

The checked workstation preflight failed before the trusted marker with
`Failed to create container process: Operation not permitted`, including when
run outside the agent filesystem sandbox. [Recorded failure](preflight.json)
and [input/image hashes](digests.json) preserve the exact invocation. The protocol
returned exit status 2, made zero model calls, and did not start a sweep. There
is no host-code or network-enabled fallback.

The next dependent action is to run the same preflight on a host that permits
the required user, network and process namespaces. After it passes, validate
generated controller execution and VLM judgments before drawing conclusions
from the practice experiment. Learning and human-benefit results remain unmeasured.
