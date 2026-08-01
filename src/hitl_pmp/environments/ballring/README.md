# ballring — simulated "Ball-Ring"

A faithful port of the Practice Makes Perfect paper's **simulated Ball-Ring**
environment, which in the reference `predicators` codebase (sibling repo
`../hitl-practice`) is `BallAndCupStickyTableEnv`
(`predicators/envs/ball_and_cup_sticky_table.py`). Dynamics are ported
symbol-for-symbol from that env's `simulate`; the geometry math
(`Circle.contains_point`/`contains_circle`) matches `predicators/utils.py`.

## The domain

Tables sit in a ring around the center of a unit room. One is **sticky**; the rest
are **normal**. The ball starts balanced on a normal table, a cup sits on the floor,
and the goal is `BallOnTable(ball, sticky-table-0)` — get the ball onto the sticky
target table.

The catch: a **bare ball placed on any table rolls off** (falls to the floor). Only
a ball *inside the cup* rides safely onto a table — and even the cup falls off the
smooth sub-region of the sticky table. So the one reliable plan is: pick the ball,
drop it into the cup, pick the cup (which lifts the contained ball too), navigate to
the sticky table, and place the cup on its safe (non-smooth) sub-region.

### State

Three singleton objects — `robot` (x, y), `ball` (x, y, radius, held), `cup` (x, y,
radius, held) — plus per-task `table` objects (x, y, radius, sticky,
sticky_region_x_offset, sticky_region_y_offset, sticky_region_radius). `held` is a
0/1 flag. The sticky table's "safe" circle is `(x + offset_x, y + offset_y)` with
radius `sticky_region_radius`.

### Action

5-D continuous `[move_or_pickplace, obj_type_id, ball_only, x, y]`, exactly
predicators' `action_space`. `move_or_pickplace` 0 = navigate to `(x, y)`, 1 =
pick/place; `obj_type_id` 1 = ball, 2 = cup, 3 = table; `ball_only` handles placing
just the ball while also holding the cup. It is continuous and unbounded in intent,
so `get_valid_actions()` returns `[]` (no finite menu) — a discrete skill layer is
`skills.py` (PR 2), not the raw env.

## Deterministic paper config (paper text vs. predicators code)

The reference env has **genuine placement stochasticity** (`settings.py` defaults),
but the paper's *simulated* Ball-Ring uses the deterministic overrides from
`scripts/configs/active_sampler_learning.yaml`'s `ball_and_cup_sticky_table` block.
Per the repo convention ("treat hitl-practice's code as ground truth over the paper
text"), those overrides are this env's field defaults:

| Quantity | Value | Effect on dynamics |
| --- | --- | --- |
| `pick_success_prob` | 1.0 | picks never randomly fail |
| `place_ball_fall_prob` | 1.0 | a **bare ball** on any table always falls |
| `place_smooth_fall_prob` | 1.0 | a **cup** on the smooth sticky region always falls |
| `place_sticky_fall_prob` | 0.0 | a cup on a normal table / safe region never falls |
| `num_tables` | 5 | ring size |
| `num_sticky_tables` | 1 | exactly one target (sticky) table |
| `horizon` (`max_episode_steps`) | 8 | the paper's `H_eval` for Ball-Ring Sim |

Because every fall probability is 0.0 or 1.0, `uniform() < prob` is decided without
the draw — the transition is deterministic. The single residual randomness is *where*
a fallen object lands (`_sample_floor_point_around_table`), seeded per-instance by
`noise_seed`; it never affects the happy path (a correctly placed cup never falls).
The 100-step free period, competence window/recency = 2, and epsilon = 0.5 are
**method-side** (EES) parameters, not part of this env, so they land with a Ball-Ring
method run config, not here.

One deviation from a literal port: predicators' `simulate` has defensive
post-condition `assert`s (a pick actually grasped, a place actually landed). They
hold whenever the skill layer supplies valid coordinates; here they are dropped so an
off-target raw action is a harmless no-op rather than a crash — no successful
transition changes.

## Files

- `environment.py` — `BallRingEnvironment(core.Environment)`: raw dynamics
  (`take_action`/`get_valid_actions`/`hard_reset`), geometry + state classifiers, and
  `sample_initial_state` (the port of predicators' `_get_tasks` inner loop).
- `predicates.py` — the twelve `Predicate` singletons (only `BALL_ON_TABLE` is a goal
  predicate); thin adapters over the env's classifiers.
- `tasks.py` — `BallRingTasks(core.Tasks)`: train/test task generation with the
  `seed + test_env_seed_offset` (10000) split, matching Light Switch.
- `problem.py` — `BallRingProblem(core.Problem)`: the facade, `horizon = 8`, and
  `run_task_episode` routing through `reset_to_task`.
- `cli.py` — `BallRingCli`: registers this domain's config flags and is its
  composition root (`run_method`), registered under `--env ballring` in
  `hitl_pmp/cli.py`.

## Scope / follow-ups

This is **PR 1**: the runnable environment + tasks + facade + CLI registration.

- **`skills.py`** (PR 2, stacked on this) — the lifted `Skill` operators
  (Pick/Place/Navigate/…) with `LiftedAtom` preconditions/effects over `predicates.py`,
  which is what lets `random-skills`/`skill-oracle` (and later EES) run on this domain.
- A Ball-Ring `renderer.py` and a Ball-Ring `Method` are later still. The
  `skill-oracle`/`random-skills`/`ees` method CLIs are currently hardcoded to Light
  Switch (a pre-existing `TODO(scale)`), so `--env ballring --method …` does not yet
  drive a method end to end; `BallRingCli.run_method` already works when handed a
  compatible `method_factory` (see `test_cli.py`).
