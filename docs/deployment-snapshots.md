# Tossing simulation deployment snapshots

New step-protocol runs on `tossing3d` and `kitchen_recycling3d` default to
`--evaluation-policy method`. The snapshot explicitly records the executed policy:

| Practice method | Deployment policy |
| --- | --- |
| EES | Its existing competence-weighted goal planner and greedy learned samplers |
| DP / DP-period | The same maximum-product task MDP used by its allocation objective; observed action failure absorbs and ends the task |
| Curtis / expectimax | The finite-horizon canonical policy valued by their Tossing objective: pick, recover a failed pick by opening, and toss once; both toss outcomes terminate |

Curtis's determinized A* and expectimax choose **practice** actions. Their current
Tossing stopping value evaluates a fixed deployment policy with a four-skill
horizon. Evaluation executes that policy; it does not repurpose the practice
search or change the objective. The external control-step cap still applies.
Unexpected states without an applicable canonical action end unsolved.

Evaluation never trains the skill samplers, competence models or cost models,
and does not invoke human resets. DP snapshots retain both observed competence
models and unobserved per-skill priors. `results.json` records `deployment_policy`
beside the immutable snapshot hash, including when tasks run in parallel.

Version 2 snapshots store this policy specification, frozen learned samplers,
competence state and deployment RNG. They deliberately omit practice search
state. Full continuation requires a separate `checkpoint.pickle`; a deployment
snapshot cannot resume practice.

Unversioned historical snapshots still replay the shared EES policy they saved.
This change does not reinterpret the archived v13 results. Explicit
`--evaluation-policy shared-ees` retains that measurement for a new run. Other
environments retain their historical default; method-specific belief-space
deployment outside Tossing raises an error instead of silently selecting EES.

## Resuming simulated practice

On Tossing simulations, `--checkpoint-cycles` saves a separate full practice
checkpoint after eval0 and after each completed learning cycle. It waits for
outstanding evaluations before publishing the checkpoint. The method graph,
beliefs, samplers, accounting, counters and all relevant RNGs travel with the
MuJoCo integration state and domain runtime state. Continuation requires the
same MuJoCo version and compiled model fingerprint.

Add `--stop-after-cycles 1 --measure-after-learning-updates 1 2` to pause cleanly
after eval0, practice1 and eval1. Start a new process with the original flags and
budget plus `--resume OUTPUT_DIR` to continue. Keeping `--stop-after-cycles 1`
pauses after one additional cycle, yielding practice2 and eval2. Removing that
operational stop flag continues normally. Completed evaluations and learning
updates are not repeated; an interruption inside a later cycle restarts from
the last complete checkpoint. Unsupported simulation backends reject resume.
