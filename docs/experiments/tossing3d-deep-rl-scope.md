# Deep RL skill learning and neural state recognition for Tossing3D

> **Note added 2026-10-21, when this proposal was committed.** It describes the agentic
> implementation as of 2026-10-05, before kindergarden `0dac89e` (2026-10-19) fixed the
> Tossing3D bin to the floor and before the later pick-controller changes. Its settings
> table is a snapshot, not the current configuration.
>
> **Committed 2026-10-21 from the main checkout's uncommitted files.** Links to
> external sites and private Notion pages were removed (their text is kept), and links
> to paths outside this repository became plain paths. Most `artifacts/` evidence named
> here is not committed; it lives in the main checkout's untracked `artifacts/`
> directory, and the `/tmp` paths no longer exist.

Research and design proposal for Josh, October 5, 2026. No implementation, experiments, package installation, or changes to existing runs were performed for this proposal. Numerical hyperparameters and acceptance thresholds below are proposed starting points, not measured results.

## Recommended experiment

Keep planning to practice, the persistent practice world, supplied human interventions, and separate evaluation worlds. Replace the robot learner with **separate SAC policies for Pick, Toss, and OpenGripper**, and replace language-model recognition with **one small neural network with a state-membership head for each shared abstract state**. Return a single accepted state ID or an explicit unresolved result to the existing planner. Freeze state meanings and graph structure in a new, matched comparison arm.

Use independently specified simulator supervision for classifier training and skill rewards in the first interpretable experiment. Keep those labels out of policy/classifier inputs and out of runtime action selection. This is a **simulator-supervised learning experiment with learned recognition**, not label-free learning. Preserve native task scoring as the primary evaluation measure; separately report the generated skill contract's success, because they currently differ.

The shortest credible proof is smaller than the full replacement: **learn a Toss residual around a frozen initial controller, with common fixed Pick/OpenGripper policies, then substitute a validated frozen NN recognizer, then enable practice search**. Oracle recognition is a component-debugging condition only. A residual result establishes learned controller improvement, not discovery of Pick and Toss from scratch. The complete neural-policy condition follows only after those gates pass. Sparse-reward learning from random 18-dimensional controls in the current small interaction budget is a substantial risk; it should not be the first integration test.

There are five decisions to settle before future implementation: permission to use simulator labels/rewards, whether to provide a common bootstrap corpus, whether the first policy result may use a frozen controller prior, whether to freeze abstraction in the coding comparison, and how to cap interactions and compute. This proposal recommends **no additional training simulator**. The decisions are enumerated below; this document does not authorize a run.

## What the current implementation actually does

Inspected both the main checkout and the dirty agentic worktree, including current source, the experiment log, handoff status, and the r8 saved launch/library artifacts. The main checkout is at `86a188fafb12fae78922e80c4581f21a8d706396`; the agentic worktree is at `6295d1b61fb79424edf60a8bee53c8edcf407737` **plus uncommitted changes**. A commit ID alone does not describe the implementation audited here.

The handoff JSON's top-level `protocol` still says three actions and its top-level status refers to r6. Its newer `comparison_runs` and `next_experiment_settings`, the r8 launch artifacts, and executable source agree on the following. The latest recorded status describes r8 paused until October 5 at 5 PM Eastern; that is a recorded status, not an independent live-service check.

| Setting | Verified current behavior |
| --- | --- |
| Practice session | **At most 20 skill invocations**, including selected human actions; search or unresolved recognition can end it sooner. This is not 20 low-level controls. |
| Practice cost objective | Deployment value minus `3e-6 × accumulated session cost`; configured robot cost 1, human cost 5. These are planning costs, not SAC rewards or hardware prices. |
| Beliefs | Local-trend model, grid inference, 25 competence bins × 16 learning-rate bins; rate decay 0.9 per training example. |
| Search | `DeterminizedAStarPlanner`, 100 iterations and remaining-invocation bound. Legacy `depth=3` is not its depth cap. The r8 launch explicitly sets observation-probability weight **0**, whereas `SearchConfig` defaults to 0.1. |
| Evaluation | Explicitly launched with **10 tasks**, four skill slots per task, native goal scoring, no human action selection; separate evaluation environment. Launcher default task count is 2, so 10 must remain explicit. |
| Evaluation policy | Ordered available skills: Toss, Pick, OpenGripper. Practice uses search; deployment is not the same search procedure. |
| Practice initialization | Canonical scene seed 125; under reset policy `never`, task sampling reads the current world without resetting. Test scene RNG uses run seed + 10000. |
| Pilot stopping | 50 cycles and three consecutive complete perfect post-practice evaluations were launch choices. Early-perfect stopping is optional, not a method requirement. |
| Library | Saved initial library has 10 abstract states, 3 robot skills and 2 supplied human interventions. Pick/Toss/OpenGripper caps are 300/400/120 control periods. |
| Recognition | One classification initially and after each invocation, including the final evaluation invocation; cached ID drives applicability/outcomes. Numeric object-state input goes to the class named `VLMJudge`; the current condition is text/numeric language-model judgment, not image interpretation. |
| History | Complete recorded trajectories are outside search and available to the judge on demand. The current transport removes inline `recent_observations` and action specifications. Temporal evidence remains necessary for some definitions. |
| Learning | `PolicyLearner.begin_session/record/end_session/abort_session`; code and manifest frozen during execution, updated at session boundaries. Evaluation records do not enter learner evidence. |

Sources: r8 launch protocol (`/tmp/hitl-agentic-tossing3d/artifacts/agentic-claude50-20261005-ab-optional-r8/protocol/protocol.json`, a /tmp path that no longer exists), saved initial library (`/tmp/hitl-agentic-tossing3d/artifacts/agentic-claude50-20261005-ab-optional-r8/protocol/library.json`, a /tmp path that no longer exists), method (`/tmp/hitl-agentic-tossing3d/src/hitl_pmp/methods/agentic_options/method.py:75`, a /tmp path that no longer exists), belief initialization (`/tmp/hitl-agentic-tossing3d/src/hitl_pmp/methods/agentic_options/cli.py:274`, a /tmp path that no longer exists), grid configuration (`/tmp/hitl-agentic-tossing3d/src/hitl_pmp/methods/belief_space/competence_inference/config.py:13`, a /tmp path that no longer exists), task sampling (`/tmp/hitl-agentic-tossing3d/src/hitl_pmp/environments/tossing3d/tasks.py:94`, a /tmp path that no longer exists), experiment log (`/home/josh/Documents/repos/research/human-in-the-loop-practice-makes-perfect/docs/experiment-logs/2026-10-03-tossing3d-agentic-claude50.md`, local path), handoff status (`/home/josh/Documents/repos/research/human-in-the-loop-practice-makes-perfect/artifacts/handoff-2026-10-03-agentic-claude50/status.json`, local path).

Three qualifications affect scientific interpretation:

1. **Abstraction is currently revisable.** `RevisionProposal` preserves option IDs, success-description text, controller slots, costs and caps, but permits revised state descriptions and graph edges. Preserving prose does not mechanically prove unchanged graph-derived success semantics. A frozen-abstraction comparison is a new experimental condition; existing runs are not already that condition.
2. **Known nonsuccess currently means failure.** `SkillChainModel.outcome_audit` returns unknown if source/destination is null; otherwise it tests membership in success destinations and complements that result for failure. It does not require membership in listed failure destinations. The validator checks disjoint listed endpoints, not exhaustive endpoint coverage. For the proposed explicit contract, require success membership, failure membership, or unresolved; a known off-graph endpoint must be logged separately. Apply any such change to both new comparison arms, not silently to RL alone. No running code was changed.
3. **Containment is not native scoring.** Saved `cube_in_bin_*` definitions explicitly describe the bin interior rather than the simulator's smaller goal region. Native `Holding` also uses a contact-based condition, whereas the saved held states require temporal carrying or unsupported suspension. Native predicates cannot simply be renamed into oracle labels for the generated abstraction.

Sources: revision validation (`/tmp/hitl-agentic-tossing3d/src/hitl_pmp/agentic_runtime/types.py:164`, a /tmp path that no longer exists), outcome derivation (`/tmp/hitl-agentic-tossing3d/src/hitl_pmp/methods/agentic_options/chain_model.py:32`, a /tmp path that no longer exists), native state abstraction (`/tmp/hitl-agentic-tossing3d/reference/kinder-baselines/kinder-models/src/kinder_models/dynamic3d/tossing/state_abstractions.py`, a /tmp path that no longer exists).

## Component mapping

| Component | Current agentic implementation | Proposed RL/NN version | Required change |
| --- | --- | --- | --- |
| Abstract states and graph | Language descriptions and conditional edges; revisable between sessions | Same audited definitions and topology, frozen for first comparison | Manifest lock/hash in both new arms; independently operationalize labels |
| State recognition | Numeric/text model call returning `cluster_id` or null | Shared MLP with 10 sigmoid membership heads and abstention resolver | General `StateRecognizer` protocol; confidence, evidence and model version |
| Applicability/outcome | Derived locally from graph and cached state ID; complement convention for known nonsuccess | Local explicit initiation/success/failure sets | Resolve complement-versus-explicit-failure convention jointly |
| Robot policy | Generated Python, internal memory, bounded joint commands/schedules | Per-skill SAC actor; frozen residual controller first, standalone NN later | NN executor/checkpoint binding; preserve native bridge and skill accounting |
| Human skill | Host executes cube/bin relocation to either side; graph predicts success | Same executor, price, applicability and destinations | Exclude from actor/critic training; retain intervention/history boundaries |
| Trajectories | JSON evidence and physical trajectory files | Append-only trajectory archive plus per-skill replay | Typed transitions with provenance, reward validity and termination cause |
| Improvement | Coding invocation at `end_session` | Bounded replay-only gradient block at `end_session` | `UpdateReport`, optimizer/replay checkpoints; zero environment access during update |
| Competence | Bernoulli outcomes; each resolved call adds pending example | Deployment-policy outcomes; training exposure counted separately | Behavior-mode metadata; split observation and learning clocks |
| Learning forecast | Grid posterior over competence/rate; refit from pending examples | Same model as an initially unvalidated forecast of standardized SAC blocks | Gate on actual updates; report forecast calibration and replay readiness |
| Search | Determinized search over compact cluster/belief state | Same algorithm, action cap, objective and cost convention | Versioned compact summaries only; no replay or tensors in branches |
| Evaluation | Separate worlds; native task goal; 10 tasks × four slots | Same, with frozen policies/NN/normalization | Independent score channel and leakage checks; unresolved rates reported |
| Portability | Generic-looking protocols still contain Tossing3D types and Python-source artifacts | Environment adapter plus artifact-neutral learner and belief adapter | Generalize boundaries incrementally; do not rewrite planner |

## RL formulation and actual controls

### Algorithm and policy family

Recommend **SAC**, an off-policy actor-critic method for continuous actions, because replay reuses scarce practice data and learning can be batched between sessions. Start with two 256-unit ReLU layers for each actor and twin critics, target critics, tanh-squashed Gaussian actions, and automatic entropy tuning. Independent policies avoid cross-skill interference and make the planner's independent competence assumptions easier to audit. This recommendation is an engineering inference from the interaction constraints, not published evidence that SAC will solve this Tossing3D task. SAC paper, SAC algorithms and applications.

Use a small configuration rather than an algorithm sweep: proposed learning rate `3e-4`, batch 128, target update coefficient 0.005, discount 0.99, replay capacity 200,000 numeric transitions **per learned skill**. These values must be recorded and can be revised on development data before the comparison is locked. There is no evidence yet for an interaction count sufficient to learn Pick from scratch.

The implementation should reuse a reviewed SAC update kernel. The local KINDER SAC implementation (`/home/josh/Documents/repos/research/human-in-the-loop-practice-makes-perfect/reference/kinder-baselines/kinder-rl/src/kinder_rl/sac_agent.py`, local path) already has MLP actors, twin critics, replay, entropy tuning and target updates. However, its `train()` constructs vector training environments and resets them; its evaluation helper resets training environments. Its action scaling uses the supplied Box bounds directly. It cannot be invoked unchanged here. Stable-Baselines3 exposes a replay-only `train(gradient_steps, batch_size)` update path, but its usual collection/`learn()` lifecycle also needs an adapter. Prefer a pinned SAC update backend behind our collector, not either library's autonomous experiment loop. SB3 SAC source, SB3 collection source.

### Native action space

The saved live `robot_spec` and bridge agree: 10 Hz outer control, 18 numeric coordinates, and optional **100 × 18** schedules at 1 ms spacing for one outer period.

| Coordinates | Actual interpretation | Current bridge limits |
| --- | --- | --- |
| 0–2 | World-frame base x/y/yaw position deltas | Approximately ±0.1 per period, metres/metres/radians |
| 3–9 | Seven arm joint position deltas | Approximately ±0.1 radians |
| 10 | Absolute normalized gripper force command | 0 opens, 1 closes |
| 11–17 | Seven arm joint velocity targets | ±12 radians/s imposed by bridge; native bounds are infinite |

These are neither Cartesian end-effector velocities nor direct torques. Existing robot control converts arm targets into torques. An all-zero action opens the gripper; it is not a neutral holding action. Normalize against the **finite bridge bounds**, never the unbounded native velocity Box. Read bounds/frequency from the adapter rather than hardcoding them for every environment. Bridge (`/tmp/hitl-agentic-tossing3d/src/hitl_pmp/environments/tossing3d/agentic_bridge.py:355`, a /tmp path that no longer exists), saved robot specification (`/tmp/hitl-agentic-tossing3d/artifacts/agentic-claude50-20261005-ab-optional-r8/agentic-options/0/robot_spec.json`, a /tmp path that no longer exists), native conversion (`/tmp/hitl-agentic-tossing3d/reference/kindergarden/src/kinder/envs/dynamic3d/robots/tidybot_robot_env.py:372`, a /tmp path that no longer exists).

For standalone NN control, Pick/OpenGripper can start with an 18-dimensional action. For Toss, recommend those 18 values **plus one within-period gripper-switch time**: a fixed decoder holds the position/velocity targets over the period, retains the preceding gripper command until the chosen row, then applies the new gripper command. This 19-dimensional policy emits a legal native schedule without predicting 1,800 unconstrained numbers. Quantization to milliseconds belongs in the recorded decoder. A constant-vector action is the zero-delay special case.

This is a restricted schedule family: the coding policy can vary other coordinates within a period. The saved initial Toss already emits schedules, so a plain 10 Hz gripper policy would remove an existing control capability. Either give the matched coding arm the same decoder for a controlled representation comparison or retain full schedules and disclose the policy-family difference. Do not change physics frequency or give RL more simulator steps to hide it.

For the **first Toss-only component proof**, a simpler residual policy is credible: keep the initial controller's approach, phase logic, termination and legal schedules fixed; learn bounded joint-command corrections and a bounded release-time shift. Initialize the deterministic residual to zero, include base-controller phase/numeric memory in the policy/critic observation, and record the residual action, base action, executed schedule and decoder version. Fix correction scales before measurement. This avoids pretending that the network discovered the approach routine. If the initial Pick cannot supply held starts, the component is blocked on exploration/support; extra oracle grasps or restored held states would be a separately budgeted diagnostic condition. Residual RL supports this design pattern; TossingBot supports the relevance of structured throwing priors, but neither establishes results for this simulator.

### Observations

Use only the current bridge's numeric physical observations: measured base/joint positions and velocities, measured finger position, separate gripper command, calibrated tool pose, and typed cube/bin/barrier poses, dimensions and velocities. Include cube-to-tool, cube-to-bin and robot-to-barrier relative geometry; encode angles without wrap discontinuities and normalize quaternion signs or use a continuous orientation encoding. Include time remaining within the skill and a short ordered history with elapsed times and validity masks. Four recent control frames are a starting point for policies; preserve a longer evidence summary for recognition when needed.

Use a fixed typed-slot vector plus presence masks for Tossing3D's one cube/bin/barrier. Keep that behind an `ObservationEncoder`; a future variable-object environment can substitute a permutation-invariant encoder. A transformer, image encoder or learned world model is not required now. Do not feed absolute episode IDs, future states, contacts, native goal flags, native predicate values, oracle feasibility or human acknowledgment as physical-state evidence.

These object poses/types are themselves privileged simulator measurements. Sharing them with the coding arm controls that privilege within this comparison; it does not establish camera-based or real-robot performance. A later image condition must separately measure representation and perception learning.

### Rewards and termination

Define the skill MDP as a **bounded invocation**. For the initial residual condition, retain the fixed base controller's completion rule and existing cap. For a standalone NN's first component test, use the fixed cap as its horizon, with time remaining in the observation. Early learned termination is an extension, not another classifier call at every step. Classification still occurs once after the invocation; policy completion is not a success label.

The primary sparse skill reward is +1 for independently certified contract success at invocation end and 0 for independently certified nonsuccess. An uncertain label is **missing**, not a zero reward or failure. An incomplete invocation with an unresolved terminal reward stays quarantined from return learning until adjudicated. A NN-unresolved state can still yield usable replay when the independent training label is known. Native goal reward is logged separately and must not silently replace the broader Toss contract.

For the smallest learnable component, allow a published, fixed geometric shaping potential: approach/lift progress for Pick, distance to the target region after release for Toss, and finger opening/hand clearance for OpenGripper. Prefer `r_shaped = r_contract + beta * (gamma * Phi(next) - Phi(current))`, with bounded potential, the same discount, and zero terminal potential. Shaping must use declared observations or explicitly privileged training labels, and must not reward repeatedly opening the gripper or merely touching the cube. Sparse contract success and native task score remain the reported outcomes. This potential form has policy-invariance conditions; it does not guarantee finite-sample SAC improvement. Reward-shaping theorem.

A finite skill horizon ends that skill's return: zero bootstrap, even though the physical world continues. A logging interruption is not automatically a terminal failure. Never connect an RL transition across a human relocation or from the final state of one option into a different option's reset observation. Record the reason separately: controller completion, finite horizon, physical fault, operator abort, and perception unresolved. Settling/waiting commands that advance physics consume the same control budget; there is no free post-toss wait.

### Replay and updates

Freeze actors, recognizer and normalization during each practice session and evaluation sweep. Collect numeric transitions throughout actual robot execution. Human actions provide classifier boundary examples and intervention records, not robot action/reward samples. Keep full trajectories on disk and compact float arrays in learner-owned replay.

Proposed initial update contract: at session end, once a skill has at least 128 valid transitions from four distinct invocations, perform **64 SAC gradient iterations per newly credited usable invocation**, with at most 20 such credits processed per session. One iteration updates both critics, actor, entropy coefficient and target parameters; report optimizer counts separately. Retain startup credits as a small backlog and consume each exactly once. Use uniform replay first; no HER, prioritized replay, synthetic trajectories or cross-skill relabeling. Report both unique transitions and total replay draws. A replay sample reused 100 times is still one physical transition.

The 64-iteration unit is a proposed experimental control, not a claim that one attempt has a universal learning effect. Permit extra replay-only fitting as an explicit compute-budget variant; it must not masquerade as new practice or silently advance the attempt clock. Keep policy and classifier update budgets separate. The first integrated NN condition should use a pretrained **frozen** recognizer so changing recognition does not confound changing competence.

### Alternatives

| Alternative | When it is useful | Why it is not the first full design |
| --- | --- | --- |
| PPO | Simple on-policy implementation or abundant simulator interaction | Repeated fresh rollouts are expensive here; less reuse of limited persistent-world data |
| TD3 | Deterministic continuous controller with explicit exploration noise | Plausible second algorithm; adds a comparison before the data/label/control contracts work |
| Shared skill-conditioned SAC | Many related skills with transferable features | Cross-skill gains/regressions violate the present independent improvement forecast; needs per-skill critics/diagnostics and eventually transfer-aware beliefs |
| Neural distribution over existing toss parameters | Very short test of neural sampler learning | Compares parameter learning around a supplied motion controller, not replacement of generated feedback policies |
| Residual SAC | First Toss component result and shared initial controller improvement | Restricted policy family; inherits approach/release structure and needs base-memory-aware replay |
| Offline learning or behavior cloning first | A common, approved, action-complete bootstrap corpus exists | Adds prior data; pure offline SAC may extrapolate beyond dataset support. BC initialization is more transparent, but must be counted |
| Learned dynamics, abstraction discovery, image policies | Later generalization questions | Adds separate failure modes and experimental factors before basic integration is understood |

## Neural abstract-state recognition

The initial manifest has five families, each split by near/far bin placement: free pickup-ready cube, securely held cube, failed grasp, cube in bin, and cube beyond/on the barrier. The definitions deliberately leave gaps: a moving cube, a tipped bin, some intermediate bin locations, inaccessible near-wall placements, or insufficient carrying evidence may match none. Their prose does not establish a formally exhaustive partition.

| Representation | Benefits | Limitations and recommendation |
| --- | --- | --- |
| One multiclass softmax, 10 states plus OTHER | Cheapest training and direct compatibility with one planner ID | Forces exclusivity; an OTHER class does not represent epistemic uncertainty or every unseen situation. Use only after a partition audit, with abstention in addition to OTHER |
| Separate state-membership outputs | Labels can be true/false/unknown per state; overlaps are visible | Outputs need not sum to one. Recommend one shared two-layer, 128-unit MLP with 10 sigmoid heads; masked binary cross-entropy for unknown labels |
| Fully separate network per state | Easy per-state maintenance | More duplicated fitting and inconsistent calibration; unnecessary for ten states |
| Factored predicates/relational state or belief over states | Natural overlaps, variable objects, ambiguity-aware planning | Changes the abstraction and planner interface. Defer unless the fixed-state audit shows single-ID planning cannot represent the task |

Run all heads in **one inference call per boundary**. A conservative resolver accepts a state only when exactly one membership is confidently true, competing memberships are confidently false, required features/history are present, and the input is within supported coverage. Calibrate head thresholds on held-out development trajectories; values such as 0.95/0.05 are candidate gates, not known probabilities of correctness. If no state qualifies, multiple states qualify, or decisive history is missing, return `unresolved` with a reason. Preserve the whole membership vector in the audit record; keep it out of hypothetical branches initially.

Distinguish three cases in supervision: **known outside all definitions**, **overlap between definitions**, and **insufficient evidence**. Do not label every difficult sample as OTHER. If two true memberships imply different applicability or outcomes, do not choose the larger logit. Require an explicitly shared partition rule or abstain. Mapping arbitrary membership sets to graph nodes is abstraction engineering, not merely changing a classifier.

Use held-out trajectory/session splits, and scene splits where the approved data includes multiple scenes. Adjacent frames cannot be randomly split between training and validation. Report per-state precision/recall, a confusion matrix, calibration/Brier score, risk versus coverage, unknown rate, and false applicability/success rates. Report both the label class balance and distribution of *accepted* decisions; a classifier that abstains on every difficult state is not operationally reliable. Temperature/logit calibration can help, but does not establish out-of-distribution reliability. A three-model ensemble is an optional escalation if a single calibrated network fails the gate. Calibration study.

Evidence needed includes:

- **Holding:** relative tool/cube motion over time and support exclusion. The current description asks for at least 0.3 seconds with meaningful tool movement, or clear unsupported suspension. A closed command or nearby cube alone is insufficient.
- **Resting in bin:** oriented interior geometry, below-rim location, low motion, hand separation and bin uprightness. This is distinct from the native scoring window.
- **Failed grasp versus free:** actual finger position, command, cube support/location and hand clearance, not command alone.
- **Recoverability and side:** live barrier/bin geometry and the contract's reachability assumptions. Near-side position alone does not prove collision-free reachability.

Construct the history from already collected frames with timestamps and segment IDs; do not move the robot just to obtain certainty without charging a practice action. Clear physical continuity on human relocation, even if simulator time is unchanged. The first shared bootstrap data must include the initial free state; otherwise abstention before the first action can deadlock the system. An unresolved boundary ends the session and permits replay learning, as today. Any new active sensing or human classification-query action is an explicit extension with a cost, not an automatic escape hatch.

## Labels, reward authority and independent evaluation

| Source | Proposed use | Privilege, costs and limitations |
| --- | --- | --- |
| Simulator measurements and independent contract labeler | First classifier targets and skill terminal rewards; optional training shaping | Privileged labeled data. Define predicates from frozen contracts, not NN predictions. Native contact/goal checks are evidence, not interchangeable definitions |
| Human labels of saved trajectories | Adjudicate ambiguous language, boundary cases and temporal holding; alternative training condition | Count labels, minutes and disagreements separately from physical interventions. Do not assume a human can infer hidden contacts from a single image |
| Existing LLM judgments | Candidate annotations for review or a distillation condition | Learned/noisy judgments, not ground truth. Agreement measures imitation of the teacher and can preserve its errors |
| Learned NN judgments | Planning applicability and observed skill outcomes in the actual NN-recognition condition | Calibration/abstention required; errors can bias competence and available practice |
| Native simulator task goal | Primary independent evaluation score on separate worlds | Privileged evaluator; never fed into classifier training, replay, normalization, checkpoint selection or learning beliefs from evaluation |

First write an **independent labeling specification** for each state and skill. Replace “about” thresholds with agreed numeric thresholds and explicit ambiguity bands, document support/contact handling, and version it. That is operationalizing a fixed abstraction, but its ambiguities must be settled identically for both arms. A simulator does not contain a ready-made oracle for “securely carried over time” or “the robot can reach it.” Where supplied observations cannot determine a privileged label, the recognizer has irreducible uncertainty; abstention or a shared observation extension is appropriate.

For new authorized collections, capture permitted privileged evidence in a host-side training record at the time of execution, separate from the observation envelope. Existing numeric trajectories may lack contacts or native scoring metadata; do not claim they contain ground-truth labels that were never recorded. Label what their geometry/history supports, seek human adjudication, and leave the rest missing. Reconstructing missing labels by simulator rollout is additional simulation access. Also reuse initialized scoring infrastructure: the inspected `Tossing3DStateAbstractor` constructor calls `sim.reset()`, so constructing a new labeler that way during practice would violate the persistent-world contract.

Use three distinct label channels: training authority, online recognition, and evaluation scorer. In the main proposed NN condition, **SAC rewards come from the independent training authority; planning and competence evidence come from NN recognition**. Shadow oracle outcomes diagnose recognition errors but do not secretly choose skills or correct competence. During component debugging, oracle recognition/competence is allowed and clearly labeled as a different condition.

Using the NN simultaneously as planner recognizer, reward source and success reporter would create a feedback loop: the actor may exploit false positive regions; classifier updates change replay rewards; reported learning can reflect a changing judge. A later learned-reward condition must freeze/version the reward model, log relabeling, evaluate on untouched native scoring and independent contract labels, and compare against the independent-reward condition. Cross-fitting and held-out labels reduce data reuse but do not by themselves prevent reward exploitation.

Keep the existing 10-task evaluation sweep for pilot continuity, with frozen policy/recognizer/normalization, fixed task seeds and no human help. Report native task success, contract success, disagreements, abstentions and physical faults. Evaluation may score a state even when the recognizer abstains; that does not make the skill's unknown outcome a failure label. Ten tasks and a single training seed are a smoke test, not a precise estimate or a method-ranking result. For a confirmatory study, predeclare multiple training seeds and a final untouched task set; repeated checkpoint sweeps are development monitoring, not an inexhaustible test set. Disable outcome-driven early stopping for the main matched-budget comparison, or report stopping/censoring consistently and retain all stopped runs.

## Fixed abstraction and the comparison claim

**Freeze definitions, state IDs, initiation/terminal references and graph topology for the first comparison.** Start from the saved initial library, audit it, and share the resulting version across the new arms. Keep the same graph prior and the same policy-version rule for updating conditional destination counts. Learning transition probabilities from observed destinations is different from adding/removing edges or redefining states; document this distinction.

The smallest attribution sequence is:

1. Fixed policies + oracle versus NN recognition: classifier reliability and its effect on action selection.
2. Coding versus RL improvement with the **same frozen recognizer**, contracts and supervision access: policy-learning comparison.
3. Coding + language recognition versus RL + NN recognition: complete-system comparison, which changes two components.

A full 2 × 2 policy-learner/recognizer design is useful later, but not necessary before the component gates. Retain the motion-planning learner as a separately identified controller-prior baseline. If it keeps native predicates, its comparison changes perception and abstraction too. To isolate its policy learner, route its executions through the shared recognizer/graph; that requires an adapter.

Allowing the coding arm to revise its abstraction while freezing RL would measure system-level adaptation under different capabilities. Learning/revising abstraction for both would require state alignment, relabeling old replay, invalidating transition counts, and separating recognition change from policy change in competence. That is a subsequent research question, not a prerequisite for this design.

## Learning and search interface

The useful seam already exists: PolicyLearner (`/tmp/hitl-agentic-tossing3d/src/hitl_pmp/agentic_runtime/learner.py:15`, a /tmp path that no longer exists) owns real evidence; LearnerBelief (`/tmp/hitl-agentic-tossing3d/src/hitl_pmp/methods/agentic_options/belief.py:20`, a /tmp path that no longer exists) supplies compact forecasts; SkillChainModel (`/tmp/hitl-agentic-tossing3d/src/hitl_pmp/methods/agentic_options/chain_model.py:16`, a /tmp path that no longer exists) exposes transitions to search. Preserve lifecycle and search methods where possible.

The current local-trend transition is approximately

```text
c_next   = clip(c + m * eta + competence_noise, 0, 1)
eta_next = max(0, 0.9**m * eta + rate_noise)
```

The grid discretizes competence and a nonnegative learning rate (current rate axis ends at 0.15). `m` is the number of training examples/attempts, **not the number of optimizer steps or session boundaries**. Current agentic `observe` both conditions on a resolved outcome and increments pending examples; `end_cycle` advances the belief even if a coding revision changed nothing. Even zero-example boundaries apply process noise. The stopping value hypothetically refits the pending data before computing a canonical four-skill-slot deployment value. Observation/refit logic (`/tmp/hitl-agentic-tossing3d/src/hitl_pmp/methods/belief_space/tossing3d_observation_model.py:215`, a /tmp path that no longer exists), transition kernels (`/tmp/hitl-agentic-tossing3d/src/hitl_pmp/methods/belief_space/competence_inference/models.py:201`, a /tmp path that no longer exists), deployment value (`/tmp/hitl-agentic-tossing3d/src/hitl_pmp/methods/belief_space/tossing3d_deployment_model.py`, a /tmp path that no longer exists).

**Can it forecast RL improvement?** It is a reasonable first short-horizon heuristic under a fixed update recipe, but not yet a validated model. SAC can regress, jump after first successes, overfit reused data, or learn from unresolved-for-planning trajectories. The model's nonnegative trend cannot represent sustained negative learning except through noise and later observations. Global competence also mixes near/far bin contexts and changing practice-state distributions. Its canonical deployment value is a structural approximation when the generated graph or contract success differs from native task success. Shared use controls this approximation across arms; it does not make it accurate.

Define three separate clocks:

- **Outcome observation:** one completed real invocation; update competence once if its online outcome is resolved and it is evidence about the modeled execution policy. Never update again from replay or evaluation.
- **Learning exposure:** one new usable invocation credited to the fixed 64-iteration recipe. All its low-level steps remain separate physical-interaction counts. Unknown recognition need not erase independent reward-labeled data.
- **Optimizer work:** actual actor, critic, entropy, classifier and optional BC steps. Replaying old data adds optimizer work, not independent Bernoulli evidence or new exposure credits.

Do not pool exploratory SAC outcomes with deterministic deployment competence. Proposed default: choose exploratory versus deterministic execution by a recorded, state-independent coin at invocation start (initially 50/50); only deterministic attempts condition deployment competence, while both can train. Search uses a separate compact behavior-success estimate for exploratory transitions, or a declared mixture with deployment competence. The current non-agentic sampler already distinguishes exploratory evidence, but `TossingLearnerBelief.observe` does not. This is an essential RL change. A simpler alternative is to deploy the exact same frozen stochastic policy used during practice and model its competence; then that stochastic evaluation convention must be shared or disclosed. Never use held-out evaluation to fill missing competence observations.

For hypothetical search, clone only compact belief masses, eligible/backlog counts, replay-ready flags, behavior-mode summaries and artifact versions. Each imagined robot attempt adds a possible exposure credit under the locked update rule; stopping projects the corresponding learning transition **once**. No actor training, replay sampling, environment calls or imagined success labels enter a branch. Human actions affect world state/cost but add no robot-learning credit. At a real boundary, reconcile predictions with the `UpdateReport`; if no gradient block ran, apply no positive training drift. Calendar-boundary process noise may remain as the existing model convention, but cannot be reported as learned improvement.

Essential interface changes are small in scope, though not merely a config switch:

1. Generalize `judge: VLMJudge` to `StateRecognizer`, returning membership/accepted ID, status/reason and version. Invalidate cached classifications whenever a recognizer version changes, without stepping physics. Start with a frozen NN to avoid this additional confound.
2. Generalize `GeneratedLibrary.controllers: dict[path, source]` and `changed_files` into a policy artifact binding/revision report that can reference NN weights, decoder, encoder and normalizer. Keep the existing session lifecycle and environment action IDs. A first adapter can wrap this behind `PolicyLearner`; do not encode weights as fake Python source.
3. Add typed transition/outcome evidence and an `UpdateReport` with affected skills, consumed exposure credits, actual gradient counts, policy/classifier versions and skipped-update reasons. Only install a complete finite checkpoint atomically between sessions.
4. Split `observe_outcome`, `record_training_exposure` and `advance_after_update`; account for exploration, known costs even when outcomes are unknown, and unlearned human skills. Decouple observed next state from a possibly unresolved outcome so valid state evidence is not discarded.
5. Define an environment adapter for observation encoding, finite controls/schedules, invocation execution, human skills and independent scoring. Keep the reward/label authority separate. `Tossing3DBeliefState`, `Tossing3DTheta`, hardcoded Pick/Toss deployment value and native skill names still require a domain-specific belief/value adapter for another environment.

Assess forecasts on prequential **future deterministic practice outcomes**, stratified by source region and policy version. Compare one-update and several-update predictions against a no-improvement baseline using log/Brier loss and interval coverage. A falling critic loss is not a competence observation. If the forecast systematically fails, first stop claiming improvement prediction is validated; compare a no-learning forecast or a recalibrated time scale. Signed trends, context-conditioned competence and transfer-aware models are explicit later model changes.

## Continual execution without free resets

The learner update process receives files/tensors, not a live environment handle. Gradient updates leave the practice world's positions, velocities, simulation time, action counters and RNG state unchanged. There is one persistent practice world; a new session does not sample a new training scene or reset the last one. After a miss, search must choose an available recovery/human skill or stop. If the current graph cannot recover, that is a measured limitation.

New rollouts occur only through the planner/collector and consume invocation and low-level interaction budgets. Human cube/bin relocation uses the existing supplied mechanism and its price; it is not a robot policy to learn. Labeling a saved trajectory advances no physics, but consumes annotation/compute budgets. The coding arm's isolated PyBullet kinematic model is not a dynamics training world: it keeps objects fixed and cannot validate a throw or grasp. Count its model/IK/collision queries separately if comparing compute access.

A separate MuJoCo training simulator, state restoration to held-cube starts, vectorized resettable worlds, scripted oracle demonstrations, or replay resimulation would all add experience or privileges. None is included in the recommended budget. If Josh chooses such a condition, specify its interaction/reset budget, who gets access, what crosses back into the learner, and how evaluation seeds stay sealed. An oracle simulator used only for future component debugging still generates development experience; count it separately and do not transfer its weights/data into a no-extra-simulation result without declaring pretraining. Reset-free RL work motivates explicit recovery accounting, but its learned reset policies are not interchangeable with the supplied human skills here. Leave No Trace.

## Pseudocode for the proposed lifecycle

This is design pseudocode, not an implementation. `label_authority` is accessible to the training recorder, never to policy/NN input construction. `recognizer` is oracle only in labeled component-debug conditions; it is the frozen NN in the actual NN experiment.

```text
lock shared manifest, label spec, controls, budgets, evaluation tasks
initialize persistent practice world ONCE using the shared protocol
load permitted bootstrap data; record its origin, interactions and labels
initialize learner, replay, compact beliefs, and frozen recognition checkpoint
evaluate_checkpoint_in_separate_worlds()       # no training side effects

z = classify_once(current_numeric_observation, recorded_history)

for session in permitted_schedule:
    learner.begin_session()                  # freezes executable artifacts
    remaining = 20
    session_cost = 0

    while remaining > 0:
        if z.unresolved:
            record_session_stop("recognition unresolved")
            break                            # no failure label and no free reset

        action = determinized_search(
            state_id=z.id,
            belief=compact_beliefs,
            pending_exposures=compact_exposure_summary,
            remaining_actions=remaining,
            cost=session_cost,
            graph=shared_graph,
            max_iterations=100,
            observation_probability_weight=0,
            objective=deployment_value - 3e-6 * cost)
        # Search only conditions hypothetical beliefs and predicts one update.
        # It cannot sample replay, train a network, query a judge or step physics.
        if action is STOP:
            break

        source = z
        behavior_mode = NOT_APPLICABLE
        if action.is_human:
            invoke_existing_human_executor(action)
            end_physical_history_segment()
            trace = record_intervention_boundary()
        else:
            behavior_mode = choose_recorded_exploration_mode(action)
            trace = execute_frozen_skill(
                action, numeric_observations_only, behavior_mode,
                native_control_decoder, existing_skill_cap)
            # Every dispatched period is logged, including settling commands.

        remaining -= 1
        session_cost += configured_action_cost(action)
        z = classify_once(current_numeric_observation, recorded_history)
        outcome = derive_explicit_terminal_membership(source, action, z)
        record_real_cost_even_if_outcome_unknown(action)
        record_destination_if_known(source, action, z, outcome)

        labels = label_authority.annotate_recorded_training_trace(trace)
        learner.record(trace, source, z, outcome, labels,
                       behavior_mode, policy_version, recognizer_version)
        if action.is_robot:
            append_valid_reward_labeled_transitions_to_skill_replay(trace, labels)
            quarantine_missing_reward_or_incomplete_transitions(trace, labels)
            credit_new_usable_invocation_once_if_eligible(trace, labels)
            if outcome.resolved and behavior_mode == DETERMINISTIC:
                condition_deployment_competence_once(action, outcome)
            elif outcome.resolved:
                condition_behavior_competence_once(action, outcome)
        # Human outcome/cost diagnostics remain distinct from robot training.
        # Unknown outcome never calls condition_outcome(False).

    # Final skill already settled above, including an action at the cap.
    report = learner.end_session(
        replay_only=True, gradient_recipe=locked_recipe)
    install_complete_checkpoints_atomically(report)
    advance_learning_forecast_once(report.actual_consumed_exposures)
    invalidate_policy_dependent_transition_caches(report.changed_skills)
    if report.recognizer_changed:             # deferred in first integrated study
        z = classify_once(current_numeric_observation, recorded_history)
    # Policy-only updates leave the physical state and cached state meaning intact.
    evaluate_checkpoint_in_separate_worlds()

function evaluate_checkpoint_in_separate_worlds():
    freeze policy, recognizer, normalizer, replay, optimizer and learning beliefs
    for task in the 10 locked evaluation tasks:
        reset ONLY evaluation world to task
        clear evaluation history
        z = classify_once(current_numeric_observation, evaluation_history)
        for slot in 1..4:
            if independent_native_goal_satisfied(): break
            if z.unresolved: record_abstention(); break
            skill = ordered_available_robot_skill(z, [Toss, Pick, OpenGripper])
            if no skill: break
            execute_frozen_deployment_skill(skill)
            z = classify_once(current_numeric_observation, evaluation_history)
            # Includes the fourth/final invocation; no human help.
        record_independent_native_score_and_shadow_contract_audit()
    assert no evaluation transition/label/statistic entered learning state
    assert persistent practice world is unchanged by evaluation
```

## Fair comparisons and budget ledger

The scene/assets/physics revisions, native control bridge, initial scene, task distribution, human mechanisms/prices, abstract contracts, session cap, search implementation/configuration and independent evaluation can be shared. Policies, initialization, representation, exploration, labels, internal planning access and compute are not automatically equal. Historical motion-planning results and existing abstraction-revising coding runs are context, not a matched baseline for this new condition.

| Budget | What to record and constrain |
| --- | --- |
| Physical interactions | Robot invocations by skill, control periods, underlying simulation time/substeps, aborted attempts, sensing/waiting, and bootstrap/demo interactions. Evaluation interactions in a separate ledger |
| Human interventions | Count and modeled cost by intervention type, actual time if available; annotation/query counts and minutes separately |
| Gradient updates | Actor, critic, target, entropy, classifier and BC steps; batch size, unique data, replay draws, per-skill ratios |
| Wall time and compute | Collection, training, search, perception, rendering, provider waits and total elapsed; CPU/GPU model and active hours, memory, planner/model-query counts |
| Model costs | Coding/generation and judgment calls, tokens/cache use, retries/failed calls, dollars; report NN local compute even if API cost is zero |
| Prior information | Initial controller origin, code/model/kinematic access, demonstrations, labeled corpus, pretrained weights and their generation budget |

Use shared upper bounds on robot control periods and human interventions for the primary interaction-limited comparison, retaining the 20-invocation session boundary. Report learning curves against both control periods and interventions, plus compute/cost axes. Predeclare how a final partially affordable invocation is handled; either reserve its maximum allowance before dispatch or censor a cutoff explicitly. A stopped/deadlocked learner keeps its score at the reached budget and its failure/coverage statistics; it does not receive compensating free resets.

Fifty sessions × 20 invocations gives at most 1,000 invocation slots, not 1,000 equal data items. With the current maximum 400-period Toss cap, a loose upper bound is 400,000 robot control periods, or 40,000 simulated seconds at 10 Hz, before accounting for shorter skills, early completion and human slots. This is an arithmetic ceiling, not a runtime estimate. Equal session counts do not equalize physical, annotation, neural optimization or language-model budgets.

For a policy-only comparison, give both learners access to the same independent outcome/shaping evidence and common bootstrap corpus. If RL receives dense simulator feedback and coding receives only terminal language judgments, call that a different-supervision comparison. If a common corpus contains historical agentic practice, its collection and source policy are prior experience available to all arms; exclude historical evaluation trajectories. A data-origin audit is required before using existing artifacts—this proposal did not establish that they cover all ten states.

## Staged implementation plan and acceptance gates

Effort estimates are for one engineer familiar with this codebase, after design decisions and data access are settled. They exclude measured training time and do not promise successful learning. Work would be in an isolated future checkout with separate result paths.

| Stage | Scope and dependencies | Meaningful acceptance gate | Estimated effort |
| --- | --- | --- | --- |
| 0. Lock experiment contract | Audit ten definitions, graph, success/scoring mismatch, initial artifacts, data provenance, supervision and budgets | Every state/skill has adjudicated examples and counterexamples; known outside/overlap/unknown cases represented; frozen manifest and fair comparator selected | 1–2 days |
| 1. Execution and replay boundary | Stage 0; adapter, observation schema, finite controls, schedule decoder, reward recorder, replay/checkpoints | Recorded commands match dispatched schedules and terminal observations; no transitions bridge human relocations; update/evaluation leave practice state/RNG/time unchanged; aborted last action retained | 2–4 days |
| 2. Policy component with oracle recognition | Stage 1; learn Toss residual first, common fixed supporting policies, independent rewards | A predeclared interaction budget yields held-out contract/native improvement over frozen-controller and matched exploration-without-learning controls across at least three development seeds; actor updates change behavior, not just losses. Report no-learning if support is insufficient | 3–5 days engineering, learning duration unknown |
| 3. Classifier component | Stage 0 data audit; fit/calibrate NN on approved practice data, initially shadow-only | Group-held-out risk/coverage and critical confusions pass; unseen/tipped/moving/ambiguous/history-reset cases abstain; no task labels in features; initial state recognized reliably | 2–4 days plus annotation |
| 4. Planning-to-practice integration | Stages 2 and 3; NN deployed frozen, update report and competence/exploration split | Exactly one boundary inference plus documented version invalidations; unresolved never updates failure; no-update sessions produce no learning drift; hypothetical search performs zero live/replay/gradient calls; all costs match execution ledger | 3–5 days |
| 5. Full neural skills and matched pilot | Stage 4; standalone policy initialization and timing decoder, Pick/OpenGripper SAC, common frozen-abstraction coding arm | Same scene/human/evaluation contracts; budget-limited curves and forecast calibration; final independent scoring with all seeds/stops retained; no evaluation leakage | 4–8 days plus development/training |

Stages 0–4 are roughly **11–20 engineering days**; full standalone-policy work raises this to roughly **15–28 days**, plus annotation and uncertain learning time. Classifier and policy component work can proceed independently after the contract is fixed, but no extra agents or experiments were launched for this scoping task.

Proposed classifier gate: at least 90% coverage on represented invocation-boundary states, macro recall at least 90%, and a one-sided 95% upper bound below 1% for critical false Holding/Toss-success/applicability decisions. These are candidate criteria for Josh to approve. With zero observed errors, demonstrating a 1% upper bound takes roughly 300 independent decisions per critical category; correlated frames do not meet that requirement. Smaller datasets should report uncertainty and remain component pilots rather than claim that gate passed. Also test distribution shifts and report unknown rates there; a high in-distribution score is not sufficient.

Proposed policy gate is behavioral improvement under fixed interaction/supervision budgets, not a small TD loss or one lucky 10/10 sweep. Use development tasks separate from the final benchmark and account for all debugging rollouts. Stage 2 can fail usefully: if Toss never receives held starts, if the residual cannot correct the seed policy, or if replay lacks successful support, resolve that concrete issue before spending on full search integration.

Hardware estimate: CPU is sufficient for numeric MLP inference and initial training checks; one commodity NVIDIA GPU with roughly 8–12 GB VRAM is ample for these proposed MLP updates, but is optional. A workstation with 16–32 GB RAM and several CPU cores should support one practice plus one evaluation world and bounded numeric replay, subject to measuring the simulator's actual resident memory. Do not reserve resources from current runs. Numeric replay with roughly 200 features uses about 340 MB for 200,000 transitions storing two observations plus a ~20-dimensional action, before metadata/history overhead; three skills and stacked histories can require several GB. Cap raw video/trajectory retention separately. Existing simulator assets are already large, and no image training cluster is warranted for this design. Runtime throughput and GPU benefit remain unmeasured.

## Decisions for Josh

| Decision | Recommendation | What choosing differently would establish |
| --- | --- | --- |
| Training labels | Independent simulator contract labeler plus human adjudication of ambiguous definitions | Human-only labels test annotation efficiency and perceptual sufficiency; LLM labels test distillation, not simulator truth |
| RL reward | Fixed independent contract reward; published geometric potential for initial component learning; same evidence accessible to comparators | Learned NN reward introduces reward-model learning/exploitation as another research factor |
| Simulator access | Persistent practice world only; no extra rollout simulator, state restoration or uncounted oracle starts | Additional simulator pretraining measures a simulation-assisted method and needs a separate experience budget |
| Bootstrap data | Permit an audited common corpus of action-complete **practice** trajectories and labels, available to all arms and costed | No initial corpus tests cold-start discovery; immediate classifier coverage and skill exploration may be inadequate |
| First policy scope | Toss residual first, common frozen supporting skills; standalone three-skill SAC is a later gate | Full from-scratch control tests a stronger claim with substantially greater exploration risk |
| Abstraction | Freeze a shared audited manifest/graph in a new coding comparison arm; leave current runs untouched | Revisable coding abstraction versus fixed RL is an end-to-end capability comparison |
| Outcome semantics | Explicit success/failure membership, with unresolved off-graph endpoints, in both new arms | Retaining complement semantics requires explicit coverage/termination assumptions and reporting its consequences |
| Recognition during integration | Supervised NN, pretrained and frozen; shadow oracle only for audit | Online NN adaptation needs classifier versioning, calibration budget and competence-drift controls |
| Exploration and competence | Record exploratory/deterministic modes; separate behavior transition estimates from deployment competence | Stochastic deployment is simpler but changes the competence target/evaluation convention |
| Control representation | Preserve 1 ms release capability through an explicit decoder; document restricted schedule family | A full schedule actor increases complexity; plain 10 Hz actions confound release resolution |
| Comparison budgets | Interaction/intervention caps primary; report compute, labels, gradients and model costs independently | A wall-time- or dollar-limited study answers a different resource question |
| Pilot and final evidence | Keep 10-task pilot sweeps; multiple training seeds and untouched final evaluation for claims | The 50-cycle/three-perfect-sweep pilot stop alone supports engineering readiness, not general method superiority |

## Primary literature and implementations inspected

- Haarnoja et al., SAC, ICML 2018 and SAC algorithms and applications: off-policy continuous control and entropy tuning. They motivate the algorithm, not a Tossing3D sample-complexity estimate.
- Johannink et al., Residual Reinforcement Learning for Robot Control, authors' project: rationale for the narrowly scoped residual-policy proof.
- Zeng et al., TossingBot: structured throwing and residual physics; different task/controller setup, not a drop-in Tossing3D baseline.
- Guo et al., On Calibration of Modern Neural Networks: calibration motivation; no guarantee of unknown-state detection.
- Ng, Harada and Russell, Policy Invariance under Reward Transformations: conditions on potential-based shaping.
- Eysenbach et al., Leave No Trace: reset costs and learned recovery; does not authorize extra resets or replacing human skills here.
- Sutton, Precup and Singh, Between MDPs and semi-MDPs: temporally extended skills/options; no need for option discovery in the first experiment.
- Local KINDER SAC source (`/home/josh/Documents/repos/research/human-in-the-loop-practice-makes-perfect/reference/kinder-baselines/kinder-rl/src/kinder_rl/sac_agent.py`, local path), local replay/wrappers (`/home/josh/Documents/repos/research/human-in-the-loop-practice-makes-perfect/reference/kinder-baselines/kinder-rl/src/kinder_rl/gym_utils.py`, local path), CleanRL SAC source, SB3 SAC source, and SB3 collection source: inspected update and environment-lifecycle boundaries. Online documentation is unpinned; a future implementation must pin its dependency version.

Audit identifiers: agentic-worktree KINDER checkout `f0d554b7b14731f33d55111f99a37f0ae2853e48`; kinder-baselines `427ad6cdfcc79efb489ebabc8fd39d77e104bb4e`. SHA-256 of inspected r8 library bytes: `2bee39ec4cafea4671a2635b8ecfca4e72169f0a5b7c83e5ac7407fba353a02c`. SHA-256 of inspected dirty `method.py`: `88462c65a9202eeff4a5d6a1be406b794671fefb577945e063797b109869f833`; `chain_model.py`: `0ba9aeb8d2f21ffb721c8b926c1957128ff0ef549e92b8264498c3110ca54af9`; `agentic_bridge.py`: `fd55d7331fac9e7c4c1928df7a3147e6a23b00a5135b8fd2c8ae620e4cf56885`. These identify the inspected snapshot without modifying or copying live execution files.
