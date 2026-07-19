# adapters

This folder holds the bidirectional bridge between `core.Environment` and
Gym/Gymnasium's `Env` interface. It is glue, not a third fixed ABC — the five
core interfaces (`Environment`, `HumanOracle`, `Problem`, `Method`, `Metrics`)
are unaffected.

## Why `core.Environment` isn't just `gym.Env`

Gym/Gymnasium's `step()`/`reset()` loop assumes `reset()` is free and
automatic whenever an episode ends. This project's central thesis is the
opposite: some actions are irreversible, so ending an episode does not imply
a free reset — a human/oracle must sometimes intervene, at a cost, via
`Problem.request_human_reset()`. Baking Gym's assumption into the core
abstraction would silently contradict the research question, so
`core.Environment` stays a bespoke interface representing the one real-world
state (`take_action`/`get_valid_actions`/`get_current_state`/`set_state`/
`hard_reset`, backed by `gymnasium.spaces.Space`), and Gym-compatibility is
handled here instead — explicitly, as two separate, non-symmetric adapters.

## Two directions, not one

- **`from_gym.py`** (future) — `GymEnvAdapter(core.Environment)` wraps a
  **third-party** `gym.Env`/`gymnasium.Env` so it satisfies `core.Environment`,
  letting externally-published environments be imported into this project's
  `Problem`/`Method` framework. Usually you'd treat the wrapped env's raw
  observation as one untyped `Object` rather than inventing a full symbolic
  `Predicate` layer for it, unless a planning-based `Method` specifically
  needs symbolic reasoning over it.
- **`to_gym.py`** (future) — `GymAdapter(gymnasium.Env)` wraps **this
  project's** `core.Environment` to expose the Gym interface, for feeding
  into deep-RL libraries like Stable-Baselines3 or RLlib when a baseline
  needs one. This direction is purely mechanical: flatten the per-object
  `State` into one `Box` vector; it does not need to reconstruct any of the
  reset-cost semantics that `core.Environment` deliberately omits from Gym.

The two adapters are not mirror images of each other: `from_gym` is an
import path into the symbolic/reset-cost framework, while `to_gym` is an
export path that discards that structure for a flat vector.

Neither adapter is implemented yet — this README documents the intended
shape of the folder before any code lands.

## Precedent

The sibling `hitl-practice` repo (a fork of the Predicators TAMP codebase,
intentionally not extended here) has its own `predicators/envs/` conventions
worth citing but not copying — see the project design doc.
