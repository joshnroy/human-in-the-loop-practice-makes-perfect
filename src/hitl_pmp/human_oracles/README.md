# human_oracles

Concrete implementations of `core.HumanOracle`, the versioned human-cost-model axis from
the design doc. A `HumanOracle` models what it costs to get a human (or an oracle
standing in for one) to do something the robot cannot — most centrally, to service a
`Problem.request_human_reset()` call.

These are deliberately **domain-agnostic**: a `HumanOracle` implementation should have
zero knowledge of any specific `Environment`'s dynamics, state layout, or action space.
It is swappable independent of which `Environment` it's paired with — that pairing
happens one level up, in a `Problem`.

## Planned versions (none implemented yet)

- **v0 — `oracle.py`**: unconditional. The human can always do anything the robot asks;
  no cost model, no feasibility check. The trivial baseline.
- **v1 — `cost_model.py`**: `(start_state, goal_state) -> cost`. Returns infinite cost if
  the transition is infeasible for the human, finite cost otherwise.
- **v2 — `uncertain_cost_model.py`**: extends v1 with a certainty/uncertainty estimate
  alongside the cost, so callers can reason about confidence in the cost estimate rather
  than treating it as ground truth.
- **v3 — `nl_cost_model.py`**: the human receives a natural-language and/or pictorial
  description of the goal rather than a raw `goal_state`. Real humans can't operate on
  raw state representations (arrays of numbers), so this version is capability- and
  communication-aware in a way v0–v2 are not.

## Status

**Nothing in this folder is implemented yet.** This README describes the intended
structure so future files land in the right place; the files listed above do not exist
yet.

## Relationship to `core`

`core/problem/human_oracle/human_oracle.py` defines the fixed abstract interface (`HumanOracle`)
that every version above implements. See [`../core/README.md`](../core/README.md) for
the Environment/HumanOracle/Problem split rationale.
