# adapters

This folder holds bridges to **external frameworks** — code that translates
between this project's `core` vocabulary and somebody else's. It is glue, not a
third fixed ABC — the five core interfaces (`Environment`, `HumanOracle`,
`Problem`, `Method`, `Metrics`) are unaffected.

The import-linter contract puts this layer directly above `core` and below
`environments`, depending only on `core`. That is exactly the shape a bridge
wants: a domain under `environments/` can reach one, and `core` never learns
that the external framework exists.

Two bridges live here, to two different things:

- **`kinder/`** — KINDER's symbolic layer and its parameterized controllers, as
  `core` `Predicate`s and `Skill`s. Implemented; see below and its own README.
- The Gym/Gymnasium `Env` bridge (`from_gym.py` / `to_gym.py`). Not implemented;
  the rest of this file describes its intended shape.

## `kinder/` — KINDER environments, without reimplementing them

Every KINDER environment exposes the same two entry points, which is what makes
one generic bridge possible rather than one integration per domain:

- `create_lifted_controllers(action_space, ...) -> dict[str, LiftedParameterizedController]`
- `<Env>StateAbstractor(sim).state_abstractor(state) -> RelationalAbstractState`

`adapters/kinder/` turns those into `core.Skill`s and `core.Predicate`s **by
delegation**: a skill's parameters are drawn by the controller's own
`sample_parameters`, and a predicate's `holds` is a membership test in the atom
set the abstractor computed. Nothing here re-derives a bound or re-writes a
classifier, so there is no second copy to keep in agreement with upstream by
test. Nothing in it mentions any particular KINDER environment.

See `kinder/README.md` for the four modules, the abstractor's non-purity and the
cache invalidation it forces, and the two translation traps.

## Why `core.Environment` isn't just `gym.Env`

Gym/Gymnasium's `step()`/`reset()` loop assumes `reset()` is free and
automatic whenever an episode ends. This project's central thesis is the
opposite: some actions are irreversible, so ending an episode does not imply
a free reset — a human/oracle must sometimes intervene, at a cost, via
`Problem.execute_human_command()`. Baking Gym's assumption into the core
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

Neither Gym adapter is implemented yet — that section documents the intended
shape before any code lands. `kinder/` is implemented.

## Precedent

The sibling `hitl-practice` repo (a fork of the Predicators TAMP codebase,
intentionally not extended here) has its own `predicators/envs/` conventions
worth citing but not copying — see the project design doc.
