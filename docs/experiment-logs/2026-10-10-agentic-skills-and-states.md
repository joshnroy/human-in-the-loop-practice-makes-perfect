# Fragile Tossing3D: POMDP + Agentic Skills & States (seed 0)

Authorized: one local seed, true LLCC, same fragile-object settings as the recent
comparison. This is a new run; the earlier POMDP + Agentic Skills result remains LCCC.

## Configuration

- Canonical name: **POMDP + Agentic Skills & States**; internal method `pomdp-agentic-skills-states`.
- Generated low-level skill code and natural-language state regions/transition graph.
- Runtime Claude classification of the same object-state observation available to the
  coding agent. No supplied predicates or baseline pickup feasibility gate in decisions.
- Classical expectimax depth 6, observation penalty 0; grid 25 × 16,
  local-trend Bayesian competence/learning-rate inference. Human success uses E[kappa].
- Existing classical pick/toss/open competence-based terminal utility is retained.
  Deployment follows a classical shortest successful robot path through the frozen
  generated graph, reclassifying after each skill. The simulator scores actual success.
- Seed 0; 1 m × 1 m visual cost-free mat; 100 kg movable bin; damage cost 10;
  human cost 100 for each reset; robot cost 1/step; human duration 1; lambda 3e-6.
- 85,000 counted steps; measurement every 1,700; 10 held-out tasks × 500 robot steps;
  early stop after three consecutive complete 10/10 evaluations.
- Skill caps: pick 400, toss 1,000, open 100. Revisions after planner STOP or 20 actions;
  unknown language states may request a representation repair without taking an action.
- RoboCode / Claude Code, Opus 5.5 high for coding, low for observation classification.
  One shared $20 budget includes coding and all practice/evaluation classification.
  Paid calls are serialized across practice/evaluation to prevent budget races.
- Fresh interface-only bootstrap, original hybrid prompt/hint; no robot geometry or
  planning tools supplied. State schema is supplied; state descriptions/graph are generated.

## Provenance and validation

Parent HITL source: 62fd75a8d9f529d1fd3a3f523f9e92d26701ca16 (LCCC implementation).
Simulator and model sources match the corrected LCCC seed 0: kindergarden
`ec4d4eadc877cf944aae49efb366300d7f72138c`, kinder-baselines
`527fb91f6a92069e2c0aa3166f5f524bd8e4f067`. Sources are frozen at launch.

TDD: added failing tests for generated-state routing, Bayesian reset probabilities,
normalized forecast costs, immutable code/state snapshots, both reset directions,
and one shared resumed/fresh-call model budget; then implemented the adapters.
402 focused and belief-space tests passed. Real isolated-container integration passed:
3 robot steps + 2 human resets = 5 counted steps; cost 203; measurements at steps 2/4;
both reset directions executed; no model calls in the smoke test.

Launch artifacts: `artifacts/llcc-fragile-seed0-20261010/`.
Launched 2026-10-10 16:21 EDT (20:21 UTC), persistent workstation service
`hitl-llcc-fragile-seed0-20261010`, with a separate 15-minute read-only monitor.
Frozen implementation: `c27912d45f1514469511e87ec3c32a1826e7e189`.
Startup protocol verified generated-language states and the intended planner, model,
and budgets. Initial coding is running; no evaluation result yet. Do not substitute LCCC results.

## Interpretation limits

The generator owns state descriptions and applicability/effects, while the competence
prior and terminal value approximation remain classical and supplied. LLCC does not
mean all four components are generated. Runtime classification adds model cost/latency;
held-out classifications never enter the coding evidence. Budget exhaustion or incomplete
evaluations are reported explicitly, never converted into a fabricated final score.

## Canonical reporting format (copy/paste)

Use these method names, ordered most classical to most agentic:
EES; POMDP Planner; POMDP + Agentic Skills; POMDP + Agentic Skills & States; Full-Agentic.
Put variants (original/new wording, costs, competence settings) in the method name.
Use results columns **Method | Final Eval | relevant metrics**. Final Eval is the actual
final score, Running, or Not run. An interrupted run without a final evaluation uses
N/A plus a short footnote; report earlier checkpoints separately. Omit C/L, separate
variant/status columns, and best-checkpoint columns from results tables. Keep the C/L
component decomposition on Algorithms comparison only. Never relabel LCCC as LLCC.
Page order: results TLDR/table; environment at a glance; practice/evaluation videos;
graphs; wording/configuration and other details. Preserve evidence, timestamp results,
and state which experiments actually ran.
