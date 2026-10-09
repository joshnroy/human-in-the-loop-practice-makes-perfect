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
| Monitoring | Workstation background monitor every 10 minutes |

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

The seed-0 experiment has not launched at the time of this entry. Runtime artifacts are
stored separately under `artifacts/hybrid-expectimax-agentic-seed0-20261009/`; its frozen
source manifest, launch receipt, measurements and monitor history are authoritative.

## Recommendation

Compare performance and same-/opposite-side cumulative resets against counted practice
steps. Also inspect damage, controller revisions and cost forecasts. Preserve the actual
endpoint if model cost, practice steps or early stopping ends the run; do not extend curves.
