# Fragile Tossing3D comparison

Use `--fragile-object --mat-size 4` and a shared `--practice-cost-config` containing
`damage_contact: {"value": 10}`. Each human reset costs 100 and lasts one counted
step. Robot control periods cost 1. The shared objective is

C = sum(robot step charges) + human_weight * sum(human charges) + sum(damage charges)

J = E[autonomous deployment success] - objective_lambda * C

Defaults for this comparison: human_weight=1, objective_lambda=3e-6. Damage does not
advance the experience clock. As with robot and human costs, damage_contact can be
a trusted host function; its context includes the impact and current observation.
Environment info reports the configured constant reference price; host accounting
is authoritative for function-valued charges.

Damage comes from actual MuJoCo contacts in the separate KINDER FragileTossing3D
variant. Never infer damage from 10 Hz recorded cube poses. The mat is visual only;
wall contact and physical dynamics are unchanged. Initial/human placement is exempt.
A ground-contact episode starts when a cube first contacts bare ground, including
sliding out of the mat; it is not charged repeatedly while resting. Separate bounces
can incur separate charges. Placement exemption ends after the oriented cube bottom
clears the floor by 5 mm. Success still requires reaching the bin.

Practice measurement snapshots include the current step's damage before evaluation
is queued. Agentic durable receipt replay does not execute or charge again. Structured
step_events.jsonl records impact positions; agentic responses report last_damage_events.
Both evaluation paths use the same fragile environment and report damage separately
from success without adding it to practice cost.

For the fragile experiment, structured methods learn the mean total cost of each
completed grounded robot skill (execution plus damage); interrupted executions incur
real charges but do not masquerade as completed-duration samples. Expectimax plans
at skill granularity. Before a skill has data, its existing duration prior applies
and no additional damage prior is supplied. This is a learned expected-cost estimate,
not an oracle trajectory forecast; means pool visits to the same grounded skill.
Human actions retain their configured prices and Bayesian competence forecasts.

The original prompt means the original *cost-sweep* prompt, including the previously
approved expanded same-side/retrieval/transfer hint and decision explanations. The
new-wording variant adds the versioned continuation_cost_prompt.md paragraph. Both
get identical fragile-object/mat/cost information. No initial learned policy, robot
geometry file, planning scene, simulator package, or additional controller hints
are supplied. Model: claude-opus-5-5, high effort, $20 per run.

Four seed-0 runs: EES and grid expectimax (depth 6, observation penalty 0) on Della;
original and new-wording agentic on the workstation. Cap 85,000 counted practice
steps; measurement every 1,700; ten held-out episodes of at most 500 robot steps.
Stop after three consecutive perfect completed evaluations. Monitor every 600 s.

Dependencies: new KINDER variant stacked on static-planning-colliders; existing
kinder-baselines tossing-pickup-robustness controllers. Both include latest main.
