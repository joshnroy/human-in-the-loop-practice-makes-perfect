# Expectimax with an agentic robot-skill learner

## Question / goal

Does trajectory-driven code learning reduce the practice experience needed by the
structured expectimax system in fragile Tossing3D? This is one seed for tuning,
not an uncertainty estimate or a claim that the planner causes any observed gap.

## Background

The corrected structured experiment used the binary-success MLP sampler and the
composed long-range controller. The full-agentic methods could inspect detailed
trajectories and edit their controllers. The older October 5 hybrid also generated
abstract states and used determinized A*; it stopped on state-coverage gaps. This
experiment fixes the structured state representation and changes robot skill learning.

## Hypothesis

Learning from continuous trajectories and revising robot controller code can improve
experience efficiency while the structured planner still chooses practice and human help.

## Guidance given

The original full-agentic objective, fragile-object description, and existing expanded
same-side practice/transfer hint are retained. The extra accumulated-future-cost wording
is absent. The hybrid-specific contract defines three robot skill interfaces and makes
the planner responsible for physical execution. The coding agent receives the same
numeric observation schema, action API, Python packages and ordinary coding tools.
It receives full practice trajectories and costs, but no evaluation feedback, symbolic
ground truth, host source, robot-spec file, PyBullet scene, or geometry/planning service.
The initial source contains only an interface stub, not a previously learned controller.

## Methods

| Setting | Value |
| --- | --- |
| Environment | Fragile Tossing3D, barrier layout, 1 m square mat, 100 kg bin |
| Seed | 0 |
| Planner | Expectimax, depth 6, observation penalty 0 |
| Beliefs | Local trend; grid with 25 competence × 16 learning-rate bins |
| Human forecast | Posterior competence and learning-rate forecasts |
| Cost | Robot control step 1; each human invocation 100; bare-ground impact 10 |
| Human duration | 1 counted step for each invocation |
| Objective lambda | 0.000003 |
| Learning | Coding revision after planner STOP or 20 executed skill invocations |
| Measurement | Every 1,700 counted practice steps, separate frozen evaluation process |
| Ceiling | 85,000 counted steps |
| Evaluation | 10 held-out episodes, at most 500 control steps each |
| Early stop | Three consecutive complete 10/10 evaluations |
| Model | Claude Opus 5.5, high effort, persistent RoboCode conversation |
| Model budget | $20 total across initialization and revisions |
| Robot controller limits | Existing limits: pick 400, toss 1,000, open gripper 100 |
| Monitoring | Workstation background monitor every 15 minutes (updated after launch) |

Generated robot policies replace the controllers for PickCube,
MoveToTossLocationAndToss and OpenGripper. The robot skill identities and structured
initiation/effect definitions remain fixed. No MLP candidate sampling occurs; therefore
the hybrid's epsilon is zero and its forecast has no binary-classifier warmup gate.
The same outcome-based Bayesian updates apply; each completed toss trajectory advances
the toss learning clock. Skill cost forecasts use measured execution costs and durations.

The host freezes Python source at each accepted revision. Generated code runs only in
the disconnected RoboCode policy container. Evaluation receives immutable source and
deployment competence snapshots, not a live learner or practice environment. Baseline
evaluation follows initial code generation without physical training experience.

## Results

Implementation validation: an unpaid simulator smoke executed 12/12 permitted robot
steps, accounted for 12 cost units, completed four separate one-task evaluations, and
performed one learning update. This verifies plumbing, not task competence.

An additional interruption smoke stopped mid-controller at exactly 4/4 robot steps,
accounted for four cost units, and completed three separate one-task evaluations.
Neither smoke made a model call or supplies learned code to the experiment.

The targeted planner/measurement suite passed 386 tests; the final hybrid and measurement
checks passed 18 tests. Source type checking and dependency checks passed. The full
repository suite completed with 2,901 passed, 6 failed, 3 skipped and 1 expected failure
in 17 minutes. The final hybrid/measurement follow-up tests passed separately after the
last source changes. All six failures reproduce against the unchanged frozen source
from the preceding experiment: two original-bin stiffness assertions, a fixed-seed
bin-position assertion, an evaluation-task snapshot assertion, and two old controller
witness assertions. This is not a claim of a clean full repository suite. No contact
physics or task distribution was changed to satisfy tests. Baseline reproduction logs
are saved alongside the full-suite log in the runtime artifact directory.

### Launch record — October 9, 2026

The single seed-0 workstation run launched at 09:55 EDT. Its service is active and the
RoboCode stream confirms Claude Opus 5.5 is responding during initial code generation.
No practice or evaluation result is available at this launch checkpoint.

| Provenance | Pinned value |
| --- | --- |
| HITL source | `f4dfe3418fe89f241a7161abc436b9713b075cb3` |
| KINDER source | `ec4d4eadc877cf944aae49efb366300d7f72138c` |
| kinder-baselines source | `527fb91f6a92069e2c0aa3166f5f524bd8e4f067` |
| RoboCode source tree SHA-256 | `9f268a0e1da27faffbefc592883fec5a447c682df78dd50e0ca44dbdfa1502ff` |
| Policy/coding image | `sha256:712a0a55dc7fc61cb30ab043a79f176f524c937885437bf084f7abb95ed9f531` |
| Persistent run service | `hitl-hybrid-expectimax-agentic-seed0-20261009` |
| Persistent monitor service | `hitl-hybrid-expectimax-agentic-monitor-10m-20261009` |
| tmux journal view | `hybrid-expectimax-seed0-20261009` |

The run uses the repository's sweep harness with one worker, a 15 GiB host memory cap,
six-core CPU ceiling, and 4 GiB/two-core policy-container limits. The service survives
terminal disconnection. The read-only monitor checks every 600 seconds, writes history,
and reports completion or actionable failures; it does not restart experiments or feed
held-out results into the learner. No additional Della or full-agentic run was launched.

Runtime artifacts are stored under `artifacts/hybrid-expectimax-agentic-seed0-20261009/`.
The frozen `source-versions.json`, `resolved-command.json`, `workstation-launch-receipt.json`,
and `workstation-monitor-history.jsonl` record execution provenance. Results live under
`results/pomdp-agentic-skills/0/`: configuration, exact prompts, coding streams, accepted
revisions, practice trajectories, cost/step events, frozen evaluations, and timing.

Initial planning estimate: 20–40 minutes to the first post-learning evaluation and
6–12 hours for the run, conditional on planner/code-generation time and the $20 model
ceiling. These are estimates, not observed runtimes. Stop at the actual budget or
three-perfect-evaluation endpoint and report the reason.

## Recommendation

Compare performance and same-/opposite-side cumulative resets against counted practice
steps. Also inspect damage, controller revisions and cost forecasts. Preserve the actual
endpoint if model cost, practice steps or early stopping ends the run; do not extend curves.

### Monitoring update — October 9, 2026

At the user’s request, the monitoring interval changed from 10 to 15 minutes
(900 seconds). The replacement service is
`hitl-hybrid-expectimax-agentic-monitor-15m-20261009`. Only the monitor was restarted;
the experiment continues uninterrupted. Completion and actionable-failure notifications
remain enabled, with no routine notification when nothing needs attention.
