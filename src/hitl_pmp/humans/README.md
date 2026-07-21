# humans

Concrete implementations of `core.HumanOracle`, the versioned human-cost-model axis from
the design doc. A `HumanOracle` models what it costs to get a human (or an oracle
standing in for one) to do something the robot cannot — most centrally, to service a
`Problem.execute_human_command()` call.

These are deliberately **domain-agnostic**: a `HumanOracle` implementation should have
zero knowledge of any specific `Environment`'s dynamics, state layout, or action space.
It is swappable independent of which `Environment` it's paired with — that pairing
happens one level up, in a `Problem`.

## Planned versions (none implemented yet)

All versions implement two methods:

- `calculate_cost_for_human_command(*, command_start_state_description,
  command_goal_description) -> Cost` — a pure query, no side effect. Estimate what
  asking the human would cost, without actually asking; safe to call repeatedly for
  planning/ROI.
- `execute_human_command(*, command_start_state_description, command_goal_description,
  env: Environment) -> None` — actually ask. No return value (the cost was
  already known from `calculate_cost_for_human_command`); instead this is handed the
  one `Environment` *instance* directly and is responsible for updating it (e.g.
  `env.set_state(...)`) to reflect whatever actually happened, since only it knows
  what that was. This is deliberately hand-waved at the interface level — each
  version below implements its own policy for how the human actually goes about it,
  so different versions can model humans of different capability/efficiency without
  the interface changing. `HumanOracle` itself stays a static-method container (no
  constructor, no state of its own) even though `Environment`/`Problem`/`Tasks`/
  `Method` are real constructor-injected instances now — it never needed a global to
  begin with, since `execute_human_command` already receives the one `Environment`
  instance it needs as this explicit per-call argument.

`CommandStartStateDescription` currently just wraps a raw `State` (see the `TODO` in
`core/problem/human/types.py`); `CommandGoalDescription` already wraps the same
symbolic `Goal` that `Task.goal` uses, not a raw state — the human is being asked to
bring about a goal, not teleport to one exact numeric state. The versions differ in how
much of the above they actually implement:

- **v0 — `oracle.py`**: unconditional. The human can always do anything the robot asks;
  no cost model, no feasibility check — `execute_human_command` just calls
  `env.set_state(...)` directly to satisfy the goal. The trivial baseline.
- **v1 — `cost_model.py`**: `calculate_cost_for_human_command` reads the raw `state`
  off `command_start_state_description` and the `goal` atoms off
  `command_goal_description` and returns a cost; infinite if infeasible for the human,
  finite otherwise.
- **v2 — `uncertain_cost_model.py`**: extends v1 with a certainty/uncertainty estimate
  alongside the cost, so callers can reason about confidence in the cost estimate rather
  than treating it as ground truth.
- **v3 — `nl_cost_model.py`**: this is the version that needs
  `CommandStartStateDescription` to grow beyond wrapping a raw `State` — a
  natural-language and/or pictorial description instead, since real humans can't
  operate on raw state representations (arrays of numbers). Resolving the `TODO` on
  that type is a prerequisite for implementing this version.

## Status

**Nothing in this folder is implemented yet.** This README describes the intended
structure so future files land in the right place; the files listed above do not exist
yet.

## Relationship to `core`

`core/problem/human/human.py` defines the fixed abstract interface (`HumanOracle`)
that every version above implements. See [`../core/README.md`](../core/README.md) for
the Environment/HumanOracle/Problem split rationale.
