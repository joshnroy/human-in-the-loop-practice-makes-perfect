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
`core/problem/human/types.py`); `CommandGoalDescription` wraps the same symbolic `Goal`
that `Task.goal` uses **plus an optional `target_state`**. The `Goal` is the human being
asked to bring something about; the `target_state` is the human being asked for a
**reset** — "put the world back into exactly this configuration".

Those are two different commands, and the second is not a weakening of the first. A
physical human reset — picking the robot up and carrying it back — *is* a state teleport;
no goal is being achieved. And nothing domain-agnostic can turn a `Goal` into a `State`:
its truth is an opaque `holds` callable, so there is no way to synthesise a state that
satisfies it. That is why v0 below, as this README originally sketched it ("just calls
`env.set_state(...)` to satisfy the goal"), was not implementable until the field existed.

The versions differ in how much of the above they actually implement:

- **v0 — `oracle.py`** (`UnconditionalHumanOracle`, **implemented**): unconditional. The
  human always complies, immediately, at a flat `intervention_cost` of 1.0 — no
  feasibility check, no capability model, no dependence on how far the robot has drifted.
  `execute_human_command` deep-copies the command's `target_state` into `env.set_state`;
  `calculate_cost_for_human_command` returns `inf` for a command carrying no target
  state, since a v0 human genuinely cannot reason toward a symbolic goal. The trivial
  baseline.
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

**v0 (`oracle.py`) is implemented; v1-v3 are not.** This README describes the intended
structure so future files land in the right place; `cost_model.py`,
`uncertain_cost_model.py` and `nl_cost_model.py` do not exist yet.

Because v0 exists, human intervention is now *representable*, which is what
`Metrics.num_human_interventions()` needed in order to report anything but a hardcoded
zero. What drives it — when a human is asked, and what configuration they are asked to
restore — lives in the harness (`practice_loop.py`), not here: a `HumanOracle` models
what the human can do and what it costs, never when to call one.

## Relationship to `core`

`core/problem/human/human.py` defines the fixed abstract interface (`HumanOracle`)
that every version above implements. See [`../core/README.md`](../core/README.md) for
the Environment/HumanOracle/Problem split rationale.
