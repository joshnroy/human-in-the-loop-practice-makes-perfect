# environments

This is where **concrete** `Environment` implementations live — one subfolder per
domain, e.g. a future `environments/lightswitch/` or `environments/tossing_room/`.
Nothing concrete exists yet: this folder currently only documents the convention new
environments should follow. See [`../core/README.md`](../core/README.md) for why
`Environment` is the one real-world/ground-truth instance for a domain (not a
reusable dynamics function for hypothetical planning), with no notion of tasks,
humans, or reset cost.

## Convention for a domain subfolder

Each domain subfolder is expected to contain:

- `environment.py` — a concrete subclass of `core.Environment`: the domain's own
  dynamics (`take_action`, `get_valid_actions`, `get_current_state`/`set_state`/
  `hard_reset`), all operating on the one tracked `current_state`. No tasks, no
  humans, no reset cost — just the physics/logic of the domain.
- `tasks.py` — a concrete subclass of `core.Tasks`: `get_train_tasks`/`get_test_task`
  (sampling initial states, goals, train/test splits) specific to this domain.
- `problem.py` — a concrete subclass of `core.Problem` that sets `env`/`human`/`tasks`
  to this domain's `Environment`, a chosen `HumanOracle` from `../human_oracles/`, and
  this domain's `Tasks`, and implements `run_task_episode` (the one method `Problem`
  doesn't get for free as a passthrough).
- `predicates.py` — domain predicates, needed only if a planning-based `Method`
  requires symbolic `GroundAtom`s for this domain. Pure-RL-only domains can skip
  this file entirely.

## Precedent

The sibling repo `hitl-practice` (a fork of the "predicators" TAMP codebase, one
level up) organizes concrete environments similarly under `predicators/envs/`, and
vendors a PDDL translator under
`predicators/third_party/fast_downward_translator/` for planning-based domains.
This project intentionally does not extend that codebase (it's entangled and
difficult to extend), but the domain-subfolder convention above is worth citing as
precedent: `predicates.py` here plays the same symbolic-planning-support role that
predicate definitions play in `predicators/envs/`.

## Status

No concrete environments exist yet. This README documents the expected shape of a
domain subfolder so the first one (and every one after it) is consistent.
