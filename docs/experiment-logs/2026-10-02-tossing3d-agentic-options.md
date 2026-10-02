# Tossing3D: generated extended options and skill-chain practice planning

## Question / goal

Can the existing belief-space practice search operate over natural-language state
clusters and generated Python skills, with a VLM checking observed state and option
contracts and a coding learner improving skills between practice sessions?

## Background

The poster is complete and printed. This is new algorithm work. The current
Tossing3D method uses predicate transitions and a learned numerical toss sampler.
This implementation replaces physical search transitions with a skill-chain and
executes full generated low-level controllers. It retains the existing competence,
learning-rate and cost inference, training clock, refit forecast and deployment
objective. Raw observations, trajectories and source code stay outside search.

## Hypothesis

The same search can select useful human interventions when skill-chain clusters
preserve recoverability and bin-side distinctions. Whether the current learning
forecast is adequate for code revisions remains an empirical question.

## Guidance given

Josh requested four stacked draft PRs, in order:

1. Reusable infrastructure: validated option/library contracts, VLM interface,
   disconnected coding/execution runtime and native Tossing3D control bridge.
2. Experiment inputs: generation/judging/improvement prompts and domain API guidance,
   outside the method implementation.
3. Method: skill-chain adapter to existing search, shared learner/belief interface,
   persistent execution and session-boundary learning.
4. Experiments: reproducible probes, evidence and this log.

No sweeping work is resumed. Packed-4 is a separate investigation.

## Methods

Descriptions and success contracts are frozen while policy code is improved.
Search never calls a VLM or coding agent on hypothetical branches. VLM ambiguity
is an unresolved observation, not a fabricated failure. Physical transitions are
conditioned on both the source cluster and skill success/failure. Human reset
mechanisms and accounting remain host controlled; generated policies have no reset
RPC. Practice uses a persistent world and evaluation uses a separate world.

The native control space is 18-dimensional: three base position deltas, seven arm
joint-position deltas, one gripper command and seven joint-velocity targets. The
host validates controls and execution limits. Policies are not restricted to
choosing the legacy toss parameters.

## Results

The four implementation layers are complete as drafts. **53/53 focused tests**
pass: 18 runtime, 11 native-interface boundary and 24 method/configuration tests.
Ruff, formatting, mypy, import boundaries and document-link checks pass.
The repository regression run completed with **2812/2815 cases passing**, two
skipped and one expected failure (exit status 0). Review fixes made while that
run was in progress were subsequently checked by the final 53-test focused suite.

Independent reviews identified and fixed session-cost carryover, practice
initiation rejections leaking into evaluation, missing actuator metadata in
observations, sandbox seed propagation, generated skill-role reassignment,
controller-error/trajectory propagation, and trusted runtime-file collisions.
Provider failures after execution retain the paid attempt and raw evidence;
they do not fabricate a skill failure. VLM termination/success are checked at
controller completion or its execution limit; `done` alone is not success.

The native MuJoCo probe executed **2/2 control periods** through the Unix relay,
charged as **1/1 option action**, without a reset after initialization. It used
trusted numeric commands and proves the control/observation integration only.
Both global and wrist camera images are real simulator observations. See the
[probe evidence and reproduction instructions](2026-10-02-agentic-bridge/README.md).

No generated-policy learning or task-success result is established. The exact
disconnected Apptainer preflight fails with `Operation not permitted` before
entering the container, including an escalated probe. The recorded preflight
made zero model calls and launched no sweep. The configured modern Robocode
checkout also needs installation in the runtime environment. There is no
host-execution fallback. Native bridge and offline tests do not establish live
VLM/coding-agent feasibility.

## Recommendation

The critical next step is to pass the same disconnected preflight on a compatible
host. Then validate genuine generated-library and VLM execution, persistent
sessions with actual code revisions, and finally paired fixed-seed comparisons.
The [experiment protocol](../../scripts/agentic_tossing3d_experiment.py) snapshots
inputs and hashes, checks isolation before model calls, and launches the existing
sweep runner with explicit worker and memory limits. No optional sweeping or
Packed-4 work is included.
