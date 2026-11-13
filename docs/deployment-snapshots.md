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

Serving (`--env racktoserving3d`) resumes the same way, with the same three flags.
Its checkpoint carries more than the engine state, because Serving keeps state
where Tossing does not:

- the compiled model's parameter tables. A grasp or home weld writes its relative
  pose into `model.eq_data`, which no MuJoCo state vector holds, so a scene rebuilt
  from its seed would otherwise carry the weld's compiled pose. The tables go back
  before the engine state, and the fingerprint check then covers everything else;
- what the last step left in `MjData`. MuJoCo computes positions, bias forces and
  contacts on the way into a step, so after it they are one step behind the state,
  and Serving's controllers read them before their next step. Recomputing them is
  not the same thing while the scene is moving, as it is right after a scene reset
  (Tossing's skills end with a recompute, so Tossing is unaffected). The arrays and
  the contact list are written back; when the contact list's size differs, the
  last step is run again from the state it started from;
- what the scripted controllers keep between skills: arm and base targets, the
  scraper's pose in the hand, the grip, contact counters, and the IK scratch data;
- the backend's ledger (elapsed seconds, human cost and interventions), control-step
  count, layout and human RNGs, and the environment's memory of its last action
  (pick refusals, the two-route verdict, the atoms before it).

A resumed run's practice recording continues in `resume-NN-recording.jsonl`; the
scene rebuild's own settling steps are recorded nowhere and charged to nobody.
A paused process exits 0 and writes `stats.json`: read `status.json`'s `phase`
(`paused` or `complete`) to tell a finished run from a paused one. A killed run
resumes from its last checkpoint the same way; `dp_allocations.jsonl` and the
`pomdp` decision log then hold the killed session's rows twice.
`--method ees` needs `--ees-score-human-skills` typed on the resume command: the
first process records the resolved value, and the resume check compares against it.
