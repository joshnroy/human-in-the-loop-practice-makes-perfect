# Tossing3D: what the closed seven-PR stack measured

**Status: superseded.** Seven pull requests integrated KINDER's `Tossing3D-o1` into this
repo, measured EES on it, corrected two fidelity defects and shipped a task config of our
own. All seven were **closed without merging**, so that the environment could be validated
first. That validation happened (#68, `docs/kinder-environment-validation.md`), and the
domain then landed on `main` as a **fresh** integration — #69 (the coincident task config)
and #77 (`--env tossing3d`) — which reused none of the seven branches' code. Every line of
the closed stack is still only on a branch.

So this file is a record of what that stack measured. It is not a plan for resuming it, and
it does not describe the code on `main`: the live description of the domain as it exists
today is `src/hitl_pmp/environments/tossing3d/README.md`.

**Written 2026-08-04**, against `main` at `d6ae54c` and the top of the stack at `3565312`;
audited and corrected 2026-08-06 against `main` at `db2589f`.

### How to read the claims in here

Every number below is traceable to a PR body, a diff, or `docs/experiment-logs/2026-08-04-tossing3d-ees.md`
on branch `josh/experiment/tossing3d-ees` (and later branches) — that file is **not** on
`main`. Where a statement is an inference rather than something the record states, it is
marked **(inferred)** inline. Where the record is silent, this file says so rather than
filling the gap. Counts are `x/y` throughout — a percentage without its denominator is
never the primary record.

**Citations are symbol-first; the line number is a convenience that rots.** A reference
here names the symbol you can `grep` for and gives `file:line` after it, pinned to a stated
commit. Read the symbol as the claim and the line as a shortcut: line numbers in this file
went stale on two consecutive merges into `main` during the audit that produced it (#85,
then #90, both moving `ees_method.py`'s `PlanningFailure` sites). If a line does not point
where it says, `grep` the symbol — the claim is probably still true.

**Every measurement in §5 was taken on the closed branches**, whose adapter differs from
`main`'s in ways that matter — `main` has no `swing` dial, no `ORACLE_SWING`, no
`_release`, and defaults to the coincident scene. Do not read a number here as a number
about `main`.

---

## 1. Where the closed stack lives

**Closing a pull request does not delete its branch.** GitHub keeps the head ref; the
commits below are reachable on `origin` exactly as they were, and all seven PRs are still
in the `CLOSED` state with the branches and head SHAs listed here (re-verified 2026-08-06).
If a branch is ever deleted, the SHAs still identify the commits (`git fetch origin <sha>`).

The stack, bottom-first. Merge order is the table order.

| # | URL | branch | head SHA | base branch | what it is |
| --- | --- | --- | --- | --- | --- |
| 40 | https://github.com/joshnroy/human-in-the-loop-practice-makes-perfect/pull/40 | `josh/feature/tossing3d-port` | `f46dc1d13d32f769f203c3b799422921b6fd9387` | `main` | The KINDER integration itself: `environments/tossing3d/`, the oracle, the demo renderer, the pinned optional dependency. |
| 42 | https://github.com/joshnroy/human-in-the-loop-practice-makes-perfect/pull/42 | `josh/fix/tossing3d-pybullet-leak` | `a00b6327aec56511e5f2617b42f10ba0dd475ac2` | `josh/feature/tossing3d-port` | Releases the PyBullet client each skill execution opens (`KinderBackend._release`), plus a memory regression test. |
| 43 | https://github.com/joshnroy/human-in-the-loop-practice-makes-perfect/pull/43 | `josh/experiment/tossing3d-ees` | `22232380a57962911e5752700ac862449d56e1bb` | `josh/fix/tossing3d-pybullet-leak` | The EES-vs-random-skills experiment: analysis script, `docs/experiment-logs/2026-08-04-tossing3d-ees.md`, committed per-seed counts, figures. |
| 59 | https://github.com/joshnroy/human-in-the-loop-practice-makes-perfect/pull/59 | `josh/fix/tossing3d-goal-region` | `a77e6f444ea23c46569be25110531a936fe135e1` | `josh/experiment/tossing3d-ees` | Scores against the goal region KINDER actually tests (the inflated `Region.bbox`), and re-runs both arms. |
| 62 | https://github.com/joshnroy/human-in-the-loop-practice-makes-perfect/pull/62 | `josh/fix/tossing3d-demo-caption` | `3e2f70de07978c041a41efb3f67b3a1321c4a521` | `josh/fix/tossing3d-goal-region` | The "bin is scenery" diagnosis, the fine swing sweep, per-tick captions on the demo clip, and the GIF palette fix. |
| 64 | https://github.com/joshnroy/human-in-the-loop-practice-makes-perfect/pull/64 | `josh/feature/tossing3d-coincident-bin-goal` | `3565312f455066437cdc18150fe0280d1d508798` | `josh/fix/tossing3d-demo-caption` | Our own task config in which the bin and the goal region coincide, opt-in behind `--coincident-bin-goal`. |
| 61 | https://github.com/joshnroy/human-in-the-loop-practice-makes-perfect/pull/61 | `josh/fix/tossing3d-controller-reuse-explanation` | `fc50bfddbcdd568186832bb49c367ba552d2c4a7` | `josh/fix/tossing3d-pybullet-leak` | Corrects *why* reusing a ground controller breaks `Pick`. Comments and prose only. **Not in the linear stack — see below.** |

### #61 is a side branch, not a link in the chain

`fc50bfd` sits directly on top of `a00b632` (#42's head), and
`git merge-base --is-ancestor fc50bfd 3565312` is **false** — #61's commit is *not*
reachable from the top of the stack. #43 was branched from #42 before #61 existed, so
everything above #43 was built from a base that never contained it.

The consequence, on those branches only: the corrected explanation of why reusing a ground
controller breaks `Pick` is missing from `3565312`, and the *wrong* explanation is still
present there — in the environment README, in `kinder_backend.py`'s `_ground` docstring, in
`KinderBackend`'s class docstring, and a fourth time in
`docs/experiment-logs/2026-08-04-tossing3d-ees.md`, which arrived in #43 above #61's base.
**(inferred: #61 could not have fixed a file that was not on its base; the record does not
discuss this fourth site.)** None of those files is on `main`, and `main`'s integration was
written fresh rather than from `3565312`, so nothing needs cherry-picking. The corrected
mechanism is recorded in
[§5.6](#56-why-reusing-a-ground-controller-breaks-pick-corrected-twice) below.

### Other notes on the table

- **The stack numbering in the PR titles is stale and inconsistent.** #40/#42/#43 say
  "stack 1 of 3", "2 of 3", "3 of 3" — written before #59, #62 and #64 existed. #59 says
  "stack 4/4", #62 "stack 5/5", #64 "6/6". #61 carries no stack number. Read the table
  above, not the titles.
- Base branches above are the PR's declared base, not necessarily where `main` was. #40's
  merge-base with `main` is `d57e613`. `main` has moved a long way since, and now carries
  its own `Tossing3DEnvironment` and `tests/environments/tossing3d/`, so the stack is not
  merely behind `main` — it conflicts with a different implementation of the same domain.

---

## 2. Why the stack was closed

Josh chose to **validate the environment before any of it lands**. The seven PRs describe
a domain whose scoring semantics were misunderstood twice in a row (the goal region was
tested against the wrong box; the bin turned out to be scenery), on top of an upstream that
had a live memory bug in it. Merging that onto `main` first and validating second would put
unvalidated environment semantics underneath every future number.

That validation was then done independently (#68), and the domain was re-integrated from
scratch (#77) rather than by reopening the stack. The findings in §5 stand on their own
evidence, and #77 checked its own numbers against them.

---

## 3. What the environment is, and where the boundary runs

**This section describes the closed stack's adapter, not `main`'s.** The description of the
*task* is upstream's and still current; the "ours" half below was rewritten from scratch
in #77 and differs — see `src/hitl_pmp/environments/tossing3d/README.md`.

`Tossing3D-o1` is a KINDER benchmark task. A TidyBot++ mobile manipulator must get a cube
from the floor to a goal region on the far side of an immovable 5 m barrier. The base
cannot pass the barrier, so the cube can only get there through the air: the robot must
**toss** it. **A tossed cube cannot be retrieved** — success or miss, it ends up past the
barrier and no skill brings it back. That irreversibility is why the domain was chosen:
the project's V1 proposal names exactly this as EES's predicted failure mode.

### This is an integration, not a port

No dynamics are written in this repo. `kinder_backend.py` is the only module that touches
KINDER, and it imports it lazily so the rest of the package imports, typechecks and tests
on a machine with no MuJoCo.

**Upstream's, used verbatim:**

- The simulator — `kinder.register_all_environments()` then `kinder.make("kinder/Tossing3D-<variant>-v0")`.
- All physics. KINDER steps `SIMULATION_TIMESTEP = 0.0005 s` at `control_frequency = 10 Hz`,
  i.e. 200 substeps per env step.
- Every controller. `pick_shelf` from `kinder_models.dynamic3d.shelf`; `move_to_target`,
  `move_arm_to_conf` and `toss` from `kinder_models.dynamic3d.tossing`.
- **The success criterion.** `Tossing3D-o1.json`'s `"goal_state"` key (`:81`) declares one
  goal predicate, `["on", "cube_0", "blocks_goal_region"]`, evaluated by
  `ObjectCentricRobotEnv._check_goals` (`envs.py:1053-1167`) as containment in a ground
  region via `MujocoGround.check_in_region`.
- The rendering path and the `task_view` camera; `render_fps` metadata (20);
  `kinder.gif_utils.optimize_gif`.
- The task JSON scenes for `o1` and `o2`.
- The `Pick` sampler bounds — `sample_params` draws over KINDER's own
  `MOVE_TO_TARGET_DISTANCE_BOUNDS` / `MOVE_TO_TARGET_ROT_BOUNDS`.
- The windup and full-power arm configurations `KinderBackend.WINDUP_CONF` /
  `FULL_TOSS_CONF` — the values KINDER's own Tossing3D demo drives.

**Ours (adapter code, ~2000 lines at #40):**

| upstream gives | we map it to | why |
| --- | --- | --- |
| Gym `env.step(action_vector)` | `environment.py`'s `[skill_id, param0, param1]` encoding | one transition here is one *skill*, not one 20 ms control tick |
| an `ObjectCentricState` | a flat `State` over five `Object`s | predicates here are pure functions of `State`, with no simulator handle |
| `_check_goals()`, a method on the env | `predicates.py`'s `InGoalRegion` and friends | Fast Downward needs a symbolic layer evaluable off-simulator |
| imperative controllers with `sample_parameters` | `skills.py`'s lifted `Skill`s with pre/add/delete effects | EES task-plans over operators; a controller is not one |
| an episode seed | `tasks.py`'s `Task` (a seed plus a `Goal`) | the harness needs a train/test split and a restorable initial state |
| — | `problem.py` | the `Problem` facade and `H_eval = 3 + 2` |

Specifically **ours**, and therefore ours to defend:

- **The five predicates**, all pure arithmetic over feature vectors: `InGoalRegion`,
  `HandEmpty`, `Holding`, `Reachable`, `AtThrowPose`. Only `InGoalRegion` mirrors an
  upstream criterion; the other four exist so Fast Downward has a symbolic layer.
  `Reachable` (cube's x < barrier's x) is the domain's irreversibility expressed
  symbolically — making `Pick` require it is what stops the planner emitting a
  retrieve-and-retry plan the dynamics can never execute.
- **The three lifted skills** and their operator models: `Pick` (2 params),
  `MoveToThrowPose` (0), `Toss` (1). Two modelling choices are load-bearing and
  documented in `skills.py`: `Pick` requires `Reachable`, and `Pick` deletes
  `AtThrowPose` (because `pick_shelf` drives the base to the cube).
  `Toss` deletes `Reachable` — a toss makes the cube unreachable whether or not it lands
  in the region, which is what makes the planner's model of a *failed* toss honest.
- **The `swing` dial.** `execute_toss` interpolates between `WINDUP_CONF` and
  `FULL_TOSS_CONF`, so `swing = 1.0` reproduces KINDER's own demo toss exactly and smaller
  values release earlier in the arc. The interpolation is ours; the endpoints are theirs.
- **`SkillOraclePolicy`** and its constants: `ORACLE_SWING = 0.75`,
  `ORACLE_PICK_DISTANCE = 0.55`, `ORACLE_PICK_ROT = 0.0`.
- **`H_eval = 3 + 2`**, the step budget per evaluation episode.
- **`KinderBackend._release`**, the PyBullet workaround (§5.5). Deliberately **not** on
  `main`: the leak was fixed upstream, and a `close()` on top of that finalizer would
  double-disconnect.
- **`task_configs/tossing3d-o1-coincident-bin.json`**, the one scene here that is not
  upstream's (§5.8). `main` ships the same idea at a different path,
  `scripts/task_configs/Tossing3D-o1-coincident.json`, and defaults to it.
- The decision to **ignore KINDER's `terminated`**: an interaction period runs its full
  length regardless, because a solved state is absorbing and ending early would make a
  solved period cheaper in transitions than a failed one, biasing the x-axis of every
  learning curve.

---

## 4. Version pinning: the version *is* the experiment

On the closed stack, `pyproject.toml` declared an optional `tossing3d` extra pinned to
**exact commits, not branches**. **`main`'s `pyproject.toml` declares no KINDER dependency
at all** — the simulator is installed into a separate venv from the `reference/` checkouts
instead (see `CLAUDE.md`), so the pin below describes the closed branches only:

| package | repo | pin |
| --- | --- | --- |
| `kindergarden` | `Princeton-Robot-Planning-and-Learning/kindergarden` | `39eb7e084c1d54a69f71abfd7faebef62e4a059e` |
| `kinder_models` | `Princeton-Robot-Planning-and-Learning/kinder-baselines`, subdirectory `kinder-models` | `4c731dc81d68ee6888ef3a989034991cd0694630` |

Exact SHAs because this domain does not reimplement the benchmark — the dynamics, the
controllers and the success criterion are all upstream's, so a number measured here is
only meaningful against the version it was measured on. KINDER is a live upstream that
moves underneath the adapter, and the leaked PyBullet client `_release` worked around was
an *upstream* bug, so a version that fixed or reshaped it would silently change what the
numbers in the experiment log mean. **That version now exists**: `kinder-baselines` PR #87
(`9512b9e`, 2026-08-06) fixed the leak, so every number in §5 predates the fix.

### These SHAs are inferred, not recorded

**Nothing wrote down the KINDER commit the adapter was actually built against.** The pin
is a reconstruction, and the argument for it is partly timing and partly content:

- `39eb7e08` was upstream `main` continuously from 2026-07-28 until 2026-08-04 12:09 UTC.
- The adapter's first commit is 2026-08-04 03:08 UTC — **inside** that window.
- The EES sweep is 2026-08-04 15:37 UTC — **outside** it. By then `cdf1b8b` had landed.

So **timing alone does not exclude the sweep having run against `cdf1b8b`.** It is ruled
out on **content**: `cdf1b8b` changes cluttered-retrieval initial-state sampling, which
Tossing3D does not use.

Independently re-verified while writing this file, via the GitHub API:

- `39eb7e08` — committed 2026-07-28T16:55:05Z, "Update Table3D group GIF to the current demo (#122)".
- `cdf1b8b` (`cdf1b8ba0ed0d4fbf0390e336bea748e83d517d5`) — committed 2026-08-04T12:09:25Z,
  **"Fix cluttered retrieval initial-state sampling (#123)"**. The commit subject matches
  the content argument exactly.
- `39eb7e08...cdf1b8b` is `ahead_by: 1`, `total_commits: 1` — **exactly one commit apart.**
- `kinder-baselines` `main` was still `4c731dc8` (2026-06-29T15:13:44Z, "Add reward
  grounding prototype (#81)") when this was written, so that pin was unambiguous. It has
  since moved: `main` is `9512b9e` as of 2026-08-06, the leak fix.

### Upstream has moved one commit past the pin

Upstream `kindergarden` `main` is `cdf1b8b`, i.e. **one commit ahead of the pin**. Reading
source out of a checkout at `main` is therefore reading a slightly different tree than the
pinned one. For everything Tossing3D touches this is believed harmless (the single
intervening commit is the cluttered-retrieval fix above), but it is a real gap between what
the code cites and what a reader has locally.

The local `reference/` checkouts are not at `main` either, and since 2026-08-07 that is
recorded rather than incidental: both are **git submodules pinned to a fork branch** —
`joshnroy/kinder-baselines` @ `11eace5` and `joshnroy/kindergarden` @ `4113237`. Read the
exact pin with `scripts/update_reference_repos.sh --check`, which reports drift and never
resets a checkout. A checkout on some other commit is somebody's work, so the script says
so instead of moving it.

### The `kindergarden` / `kinder` naming trap

The **distribution** is named `kindergarden`; the **import package** is `kinder`.
`pip install kindergarden`, `import kinder`. Two consequences that have already cost time:

- The simulator-availability check in `tests/environments/tossing3d/` keys on the *import*
  name `kinder`, not on the distribution name. (On the closed stack that check lived in a
  `conftest.py`; on `main` it is an `importlib.util.find_spec("kinder")` guard in each
  simulator-backed test module.)
- The old install docs told you to clone
  `Princeton-Robot-Planning-and-Learning/kinder-models`, which **404s**. The controllers
  live in the `kinder-baselines` monorepo, subdirectory `kinder-models`. Corrected in #40.

Also: **`kindergarden` alone is not enough.** The parameterized controllers this domain
drives are in `kinder_models`, which pulls `bilevel_planning` in turn.

---

## 5. Empirical findings

### 5.1 The EES learning curve (10 seeds per arm)

Both arms share fixed seeds 0..9, run through `scripts/run_sweep.py`, so every comparison
is **paired** and is tested as such with an exact Wilcoxon signed-rank (all 2ⁿ sign
assignments enumerated). 10 cycles × 150 steps = 1500 online transitions, evaluated on 10
held-out test tasks at 11 checkpoints.

```bash
python -m scripts.run_sweep \
  --env tossing3d --methods ees random-skills --num-seeds 10 \
  --results-root <root> --max-workers 12 \
  --shared-args "--num-cycles 10 --max-steps-per-interaction 150"
```

| arm | seeds | pre-practice | end of training | sd of per-seed rates | worst seed | best seed |
| --- | --- | --- | --- | --- | --- | --- |
| **EES** | 10 | 33/100 | **67/100** | 22.1 | 4/10 | 10/10 |
| random skills | 10 | 14/100 | 21/100 | 7.4 | 1/10 | 3/10 |

Each arm evaluates 10 held-out tasks on each of 10 seeds, so the arm column is a genuine
pooled count of 100 evaluation episodes, not a mean of rates handed a denominator
afterwards. The `sd` is the spread of the ten per-seed rates in percentage points and is
**not** a binomial spread on the pooled count.

Per-seed end of training —
EES: `[7/10, 6/10, 9/10, 8/10, 4/10, 5/10, 5/10, 4/10, 10/10, 9/10]`.
Random skills: `[3/10, 2/10, 3/10, 1/10, 2/10, 2/10, 2/10, 3/10, 1/10, 2/10]`.

Evaluation episodes solved out of the 100 run at each checkpoint (post-correction values,
i.e. #59's re-run):

| transitions | 0 | 150 | 300 | 450 | 600 | 750 | 900 | 1050 | 1200 | 1350 | 1500 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EES | 33/100 | 22/100 | 20/100 | 25/100 | 49/100 | 41/100 | 64/100 | 58/100 | 59/100 | 61/100 | **67/100** |
| random skills | 14/100 | 23/100 | 23/100 | 25/100 | 22/100 | 24/100 | 18/100 | 26/100 | 22/100 | 23/100 | 21/100 |

| comparison | mean paired difference | test | verdict |
| --- | --- | --- | --- |
| EES end vs its own pre-practice | +34.0 pp (sd 23.2) | p = 0.0039, n = 9 non-tied | **established** |
| EES end vs random-skills end | +46.0 pp (sd 25.5) | p = 0.0020, n = 10 non-tied | **established** |
| random-skills end vs its own start | +7.0 pp (sd 12.5) | p = 0.1094, n = 7 non-tied | **not established** — 26 paired seeds needed for 80% power |
| EES worst post-practice checkpoint (300) vs pre-practice — *the dip* | −13.0 pp (sd 20.6) | p = 0.1328, n = 8 non-tied | **not established** — 20 paired seeds needed for 80% power |

**The dip is not a finding.** The mean curve falls from 33/100 to 20/100 at 300
transitions before climbing to 67/100, and an earlier draft asserted "EES gets worse
before it gets better" straight off that mean. It does not survive a test: per-seed
differences are `[-30, -40, 0, 0, -10, -10, +10, -10, -50, +10]` — seven seeds at or below
zero, two up, and the spread swamps the mean. The 300-transition checkpoint was also
chosen *post hoc* as the worst of ten, which inflates significance rather than deflating
it. `describe_trough` in `analysis/practice_makes_perfect/tossing3d_comparison.py` re-runs
this test on every invocation precisely so the claim cannot drift back into being
asserted — that script is on `josh/experiment/tossing3d-ees` and later, **not** on `main`.

**Why EES starts at 33/100 and the floor at 14/100:** EES *plans* the correct skill
sequence from the first checkpoint and only its sampler is untrained, so it already
inherits whatever fraction of the uniform swing prior happens to land in the goal region.
Random skills has to discover the sequence too. The two comparisons therefore answer
different questions, which is why both are reported. **The record gives two different
figures for that fraction and they disagree** — "~36% of the swing prior" and "roughly
half the sampler's `[0.25, 1.25]` prior", both in the same experiment log. Neither is
backed by a stated measurement, so neither is repeated as fact here; see §8.

**The raw counts are committed.** `docs/experiment-logs/2026-08-04-tossing3d-arms.json` on
`josh/experiment/tossing3d-ees` and later holds all 10 seeds of both arms at all 11
checkpoints as `(num_online_transitions, num_solved, num_total)` triples, copied out of the
runs' own `stats.json`. Every number above is re-derivable from that file. The results
directory itself no longer exists.

**This experiment cannot speak to the irreversibility hypothesis, in either direction.**
Two concrete reasons, both structural:

1. **The harness hands out a free reset.** `PracticeLoop.run` calls
   `problem.reset_to_task(task=task)` at the top of every practice cycle
   (`practice_loop.py:236` on `main` at `db2589f`; the line has moved since this was
   written) and `run_task_episode` resets per evaluation episode — a free
   reset every `--max-steps-per-interaction` steps. That is faithful to predicators and
   correct for a reproduction, and it supplies **precisely** the resets the hypothesis is
   about removing.
2. **Human intervention is not representable.** `Metrics.num_human_interventions()`
   returns a hardcoded `(0.0, 0)` because no `Method` here ever calls
   `Problem.execute_human_command`. The cost the hypothesis predicts has nowhere to be
   recorded.

The 450-transition demo checkpoint shows the irreversibility on screen — cube at x = 2.86,
past both the goal region and the bin, labelled literally `no-op (no plan)` — and the run
still recovers to 9/10 by 1500 transitions, because the harness resets. That is the
confound, in one frame.

### 5.2 The goal-region inflation bug — real defect, negligible effect

`InGoalRegion` tested the cube against the raw task-JSON range. **KINDER never compares
anything against that literal.** `MujocoGround._create_regions` (`objects/base.py:874-881`)
inflates the range by `ground_placement_threshold = 0.05` m (`base.py:840`) on every side,
z clamped at 0, and stores the result as `Region.bbox` — which is what
`Region.check_in_region` (`base.py:148-185`) tests — it takes a **point**, the object's
own position — and therefore what `ObjectCentricRobotEnv._check_goals`
(`envs.py:1053-1167`) decides success by.

| | x | y | z |
| --- | --- | --- | --- |
| what we tested | [1.90, 2.10] | [-0.10, 0.10] | [0.00, 0.10] |
| **what KINDER tests** | **[1.85, 2.15]** | **[-0.15, 0.15]** | **[0.00, 0.15]** |

Our box was 2/3 of the true width on x — the axis a toss controls — and the error was
one-directional: our box is strictly contained in KINDER's, so every disagreement was a
KINDER success scored here as a failure. The fix reads `Region.bbox` back rather than
re-deriving the inflation, so the two cannot drift apart again, and raises loudly if
`blocks_goal_region` ever stops being a single box.

Both arms were **re-run rather than rescored**, because the predicate is the goal atom
(the `Goal` built in `tasks.py:66-69`) and a `Toss` add-effect (`skills.py:73`) — both on
the closed branches, so those lines cannot be re-checked against `main` — and
`run_task_episode` returns
early on `is_satisfied`, so a wider box *could* have changed behaviour.

**Empirically it did not.** Of the 220 `(transitions, solved, total)` triples the two arms
record (11 checkpoints × 10 seeds × 2 arms, 2200 evaluation episodes), **214 are identical
and 6 moved, each by exactly +1** — the only direction the strictly-containing box allows:

| arm | seed | transitions | was | now |
| --- | --- | --- | --- | --- |
| EES | 7 | 750 | 2/10 | 3/10 |
| EES | 9 | 150 | 3/10 | 4/10 |
| random skills | 2 | 150 | 0/10 | 1/10 |
| random skills | 3 | 450 | 1/10 | 2/10 |
| random skills | 3 | 1350 | 2/10 | 3/10 |
| random skills | 6 | 1350 | 1/10 | 2/10 |

Six moved triples land in five pooled checkpoint cells: EES 21/100 → 22/100 at 150 and
40/100 → 41/100 at 750; floor 22/100 → 23/100 at 150, 24/100 → 25/100 at 450, and
21/100 → 23/100 at 1350 (two seeds).

**Pre-practice and end-of-training are identical for both arms** (33/100 and 67/100;
14/100 and 21/100), every per-seed final is identical, and **all four significance tests
return exactly what they did before**, including EES vs floor at +46.0 pp, p = 0.0020.

The evidence that these are scoring flips and not behavioural divergence is that they are
**isolated and non-cascading**: EES seed 7 differs at 750 then matches exactly at 900,
1050, 1200, 1350 and 1500; EES seed 9 differs at 150 and matches everywhere after. A
changed practice-phase abstract state would have altered the sampler's training data and
compounded, not healed at the next checkpoint. A plausible mechanism — after a toss the
cube is past the barrier and no skill applies, so the same actions get taken whether or
not `InGoalRegion` holds — is stated in the log **as a hypothesis and was not verified**.

**Why the old test suite could not catch it:**
`test_in_goal_region_agrees_with_kinders_own_goal_check` is a genuine differential test
against upstream's real `_check_goals()`, and it **passed** — a 12-state random walk of
whole skills essentially never lands the cube in a 5 cm boundary shell. Sound in design,
underpowered in coverage. #59 replaced the JSON-literal pin with an element-wise pin
against `Region.bbox` (the only check that can catch a wrong box) and added
deliberately-placed shell probes, with offline shell coverage in `test_predicates.py` for
machines without MuJoCo.

**A self-correction in #59 worth preserving:** its first commit message and first log draft
asserted "this changes behaviour, not just scoring". The re-run refuted that. The commit
was amended and the log rewritten rather than leaving a false claim in the fix's own
history. The defensible version is that a re-run was *required* because the predicate
**could** change behaviour, and that empirically it did not.

### 5.3 The bin is scenery

The single most misreadable thing about this domain. **A cube that lands in the bin is a
scored failure, every time.** Three independent checks against upstream:

1. **The goal never consults the bin.** One goal predicate,
   `["on", "cube_0", "blocks_goal_region"]` (`Tossing3D-o1.json`'s `"goal_state"`), evaluated as
   containment in a **ground** region. No bin body, no bin site, no second condition, in
   either shipped variant.
2. **The overlap is arithmetic only.** The 0.30 m bin sits at x = 2.2305, footprint
   x ∈ [2.0805, 2.3805], against the corrected region x ∈ [1.85, 2.15] — an overlap of
   x ∈ [2.0805, 2.1500], 69.5 mm. The bin's near wall occupies x ∈ [2.0805, 2.1005]
   (`Bin._create_xml_element`'s left wall, `primitive_objects.py:368-380`), so once the
   cube's 0.025 m half-extent is counted the
   part of that overlap a cube can *rest* in is x ∈ [2.1255, 2.1500] — 24.5 mm. Nothing
   lands there (see 5.4).
3. **Upstream's own prose is stale, not loose.** `docs/envs/Tossing3D.md`'s `## Description`
   paragraph (`:8`) says the robot "must toss the object into a bin". That was true at KINDER's initial commit, when the
   bin sat at x = 2.0005 with footprint x ∈ [1.8505, 2.1505] — the inflated goal region
   x ∈ [1.85, 2.15] to within half a millimetre, and the same on y. **The region *was* the
   bin.** Commit `1183de7` ("merge final changes from prpl-mono", 2026-03-20) moved
   `bin_init_region` to 2.23 and left `blocks_goal_region` behind.

**Retracted** across `predicates.py`, `test_predicates.py`, `test_kinder_fidelity.py` and
the EES log: the claim "landing in the bin is not itself a failure". As a practical matter
it always is. The geometric overlap is still stated, now as arithmetic that nothing
reaches.

### 5.4 The swing sweep: a staircase, not a curve

Landing x through this domain's own oracle (`throw_standoff` = 1.35,
`ORACLE_PICK_DISTANCE` = 0.55, so `swing` is the only thing varying), verdicts from
KINDER's own `_check_goals()`:

| swing | seed 0 | seed 2 | seed 1166418 (the demo's) |
| --- | --- | --- | --- |
| 0.50 | 1.657 ✗ short | 1.657 ✗ short | 1.656 ✗ short |
| 0.60 | 1.990 ✓ | 1.989 ✓ | 1.989 ✓ |
| **0.75 (oracle)** | **1.914 ✓** | **1.915 ✓** | **1.914 ✓** |
| 0.90 | 2.015 ✓ | 2.014 ✓ | 1.960 ✓ |
| 0.958 | 2.017 ✓ | 2.015 ✓ | 1.961 ✓ |
| 0.959 | 2.016 ✓ | 2.015 ✓ | **2.220 ✗ in the bin** |
| 0.960 | **2.216 ✗ in the bin** | **2.215 ✗ in the bin** | **2.220 ✗ in the bin** |
| 0.962 | 2.016 ✓ | 2.015 ✓ | 1.961 ✓ |
| 1.00 | 2.216 ✗ | 2.216 ✗ | 2.219 ✗ |
| 1.25 | 2.217 ✗ | 2.219 ✗ | 2.217 ✗ |

Two things to read off it. It is a **staircase** — 0.962 lands short again after 0.960
cleared the wall — and **nothing lands between x = 2.017 and x = 2.215** at a sweep
resolution of 0.001 across the step. Every in-bin resting position (identifiable by
z = 0.044, the bin's interior floor, against 0.025 on the ground) is 6.5–7 cm past the
region's far edge. `TossController` releases on the first control tick past a fixed 0.46
fraction of a trapezoidal profile *and* plans the swing through PyBullet, so the landing
point steps rather than sliding; **which of the two dominates is not measured** — only that
it steps.

A coarser earlier sweep (#40, before the goal-region correction) gives the same verdicts
under either box:

| swing | 0.25 | 0.50 | 0.60 | 0.75 | 0.90 | 1.00 |
| --- | --- | --- | --- | --- | --- | --- |
| seed 0 | 1.418 | 1.657 | **1.990** | **1.914** | **2.015** | 2.216 |
| seed 2 | 1.424 | 1.656 | **1.989** | **1.915** | **2.014** | 2.216 |

`ORACLE_SWING` stays at 0.75: it was measured to solve, and aiming at the region's overlap
with the bin would only produce a demo that fails its own goal check. **KINDER's own demo
toss is swing = 1.0, and it overshoots** — which is exactly what makes the dial worth
learning: the obvious value is the wrong one.

A previously quoted usable band of `[0.57, 0.93]` was **interpolation against the narrow
box, not a measurement**, and has been withdrawn. The sampled points bracket the solving
band to somewhere inside (0.50, 0.959).

### 5.5 The PyBullet client leak

**The integration leaked an entire live PyBullet physics server per skill execution.** A
40-step run reached **18.7 GB RSS** climbing at ~112 MB/s on a 60 GB box; a run of this
shape twice took a whole session down via the OOM killer. No sweep could run.

The mechanism is a seam between two individually reasonable pieces:

1. `LiftedParameterizedController.ground` is `return self.controller_cls(objects)`
   (`bilevel_planning/structs.py:145`) — grounding mints a **new** controller object every
   call.
2. Each new controller's `reset` runs `if self._pybullet_sim is None: self._pybullet_sim = PyBulletSim(x)`,
   and `PyBulletSim.__init__` does a `p.connect(p.DIRECT)` plus a kinova-gen3 URDF load.
   That `is None` guard is therefore `None` **every time**.
3. `PyBulletSim.close()` existed (`kinder_models/dynamic3d/utils.py:590` then; `:596` at
   upstream `main` today) and had **zero callers anywhere in the package**.

**This is fixed upstream and the section below is history.** `kinder-baselines` PR #87,
squash-merged as `9512b9e` on 2026-08-06, gives `PyBulletSim` a `weakref.finalize` that
disconnects its client when the sim is collected, so a sequential run releases as it goes;
`close()` now calls that finalizer rather than `p.disconnect` directly. What did *not* go
away is the cost of holding many sims alive at once. **`main` carries no `_release`**: a
`close()` on top of the finalizer would double-disconnect.

`KinderBackend` had already spotted the hazard and cached the **lifted** controllers to
avoid it. That cache cannot help: the `PyBulletSim` lives on the **ground** controller,
which `ground()` rebuilds every call. The mitigation was at the wrong level.

Three independent probes of per-skill ΔRSS agree at ~150 MB per client:

| skill | ΔRSS/call | connects | disconnects |
| --- | --- | --- | --- |
| `execute_toss` | +300 MB | 2 | 0 |
| `execute_pick` | +150 MB | 1 | 0 |
| `execute_move_to_throw_pose` | 0 MB | 0 | 0 |
| `reset()` / `render()` | 0 MB | 0 | 0 |

`tracemalloc` stayed flat at 6.6 MB — the *Python* object is collected; the C++ physics
server it opened is what survives. A **failed** skill leaks exactly as much as a successful
one, since the sim is built at the top of `reset()` before motion planning can raise, so
early training, when EES fails constantly, is the worst case.

**The fix:** call KINDER's own `PyBulletSim.close()` from `_run`'s `finally`, via
`KinderBackend._release`.

| | 60 skill executions | 600 skill executions |
| --- | --- | --- |
| before | 8.4 GB, still climbing | would not survive |
| after | ~0.66 GB | ~0.66 GB (+122 MB total, all warmup) |

**The physics are unchanged.** Releasing only frees a resource — the controller is never
touched again after `_run` returns — and seeds 0 and 2 reproduce the pre-fix landing
positions to three decimals, including the swing = 1.0 overshoot to x = 2.216.

`test_skill_executions_do_not_leak_memory` pins peak RSS across 40 skill executions via
stdlib `resource`, so it adds no dependency. Leaking costs >6 GB over that stretch and
reclaiming costs ~nothing, so the 500 MB threshold has margin on both sides.

**A second diagnosis was refuted along the way.** The environment README warned that a
marginal grasp could flip on "residual MuJoCo solver state". Wrong: **the flipping was
these leaked clients.** Before the fix, seed 1's grasp survived the base move or not
depending on run history *even with a fresh environment per trial*; after it, seed 1 drops
the cube on every swing, deterministically. A marginal seed here is reproducibly marginal,
not flaky.

### 5.6 Why reusing a ground controller breaks Pick (corrected twice)

The obvious tidy-up — memoize the grounding so no client is allocated to reclaim — is
**wrong**, and was tried and measured. A reused `PickShelfController` reports **success**
having done nothing: the cube sits at its start pose x = 0.659 on **6/6** swings.

| swing | 0.25 | 0.50 | 0.60 | 0.75 | 0.90 | 1.00 |
| --- | --- | --- | --- | --- | --- | --- |
| final cube x, with ground-controller cache | 0.659 | 0.659 | 0.659 | 0.659 | 0.659 | 0.659 |

**The mechanism as first published was wrong.** #42 originally blamed
`base_link_to_held_obj`: "`PyBulletSim` carries held-object state that `reset` does not
clear." `PickShelfController.reset` reassigns that field **unconditionally**, at the top
level of the method body, recomputed from the current state
(`kinder_models/dynamic3d/shelf/parameterized_skills.py:199`). It is stale-free by
construction and cannot be the cause.

**The real cause is the controller's own progress flags.** All bare `:NNN` in the table
below are in `PickShelfController`, `kinder_models/dynamic3d/shelf/parameterized_skills.py`,
at `kinder-baselines` `main` (`9512b9e`); the symbol in the first column is the durable
locator, the line is the shortcut.

| symbol | set `False` | set `True` | cleared by `reset()`? |
| --- | --- | --- | --- |
| `_navigated` | `__init__` `:81` | `step()` `:291` | **no** (only *read*, at `:170`) |
| `_pre_grasp` | `__init__` `:82` | `step()` `:305` | **no** |
| `_closed_gripper` | `__init__` `:83` | `step()` `:322` | **no** |
| `_lifted` | `__init__` `:84` | `step()` `:329` | **no** |
| `_approach_step_idx` | `__init__` | — | **yes** |
| `_retract_step_idx` | `__init__` | — | **yes** |
| `base_link_to_held_obj` | — | — | **reassigned unconditionally, `:199`** |

`terminated()` is `return self._lifted` (`:276`). On a reused controller all four flags are
stale-`True`, so `step()` skips its navigate, approach and close-gripper branches, falls
into the final `_pre_grasp and _closed_gripper` branch, emits one retract action, and
`terminated()` returns the stale `True`. `_run` checks `terminated()` *after* applying the
step, so it exits after **exactly one step, returning `True`** — a no-op that reports
success, which is strictly worse than a crash for anyone debugging it. (A second, upstream
corruption in the same reuse: `reset()` *reads* `_navigated` at `:170` and, finding it
stale-`True`, plans the grasp against the robot's current base pose rather than the planned
one. The `terminated()` short-circuit is the decisive one.)

`reset()` clearing the two step indices but not the four flags is what makes this look
deliberate on a skim, and is why it survived review.

**Corroboration, and it is a prediction the refuted mechanism does not make:** KINDER's own
`TossController` *does* clear its progress flag in `reset()`
(`tossing/parameterized_skills.py:371`) and derives `terminated()` from a reset-cleared
step index — and Toss was never observed to break under reuse. It is `PickShelfController`
specifically whose flags are `__init__`-only, which is exactly why the symptom was
Pick-specific.

**A correction to the correction:** the brief #61 worked from said a reused controller
"terminates instantly, **zero steps**". It is **one** step, not zero.

**General hazard this leaves:** KINDER controllers are **not** uniformly reset-safe. Toss
is, Pick is not. Anything that caches or reuses a ground controller in future has to check
per controller class rather than assume.

No test guards this, deliberately: the claim is about upstream state across a reuse this
repo does not perform, so there is no code path to assert against without reintroducing the
bug.

### 5.7 The oracle's weak link is the Pick, not the swing

At `ORACLE_PICK_DISTANCE = 0.55`, `rot = 0.0`, **2 of the first 4 seeds fail before the
throw ever happens**:

- **seed 1** loses the cube during `move_to_target` (ends at y = −0.495).
- **seed 3** never releases it (z = 0.3946 at every swing).

`test_the_oracle_swing_actually_reaches_the_goal_region` tolerates this by asserting
`solved >= 2` rather than 3 of 3. `test_a_full_power_toss_overshoots_the_goal_region`
originally asserted the overshoot on **seed 1 — the one seed of three where it does not
happen**; there the grasp is marginal and the cube slips out during `move_to_target`,
landing at x ≈ 1.58 having never been tossed, so the test was measuring the `Pick`, not the
swing, and failed deterministically. It now asserts on seeds 0 and 2.

This caps what the oracle can measure and was left unfixed. **(inferred: the record states
the fact and calls it out of scope, but does not propose a remedy.)**

### 5.8 The coincident-bin task config

Ours, not upstream's. On the closed stack it was
`src/hitl_pmp/environments/tossing3d/task_configs/tossing3d-o1-coincident-bin.json`,
selected only by `Tossing3DEnvironment(coincident_bin_goal=True)` / `--coincident-bin-goal`;
on `main` the same scene is `scripts/task_configs/Tossing3D-o1-coincident.json`, selected by
`--task-config coincident` and **the default**. It is upstream's `o1` with `bin_init_region` put back to x = 2.0 — the value **upstream's
own `o2` still ships** — which makes the bin footprint and the *inflated* goal box coincide
to under a millimetre. `blocks_goal_region` is byte-identical to upstream's.

Measured through KINDER's own `_check_goals()`, seeds 0 and 2:

| | stock `o1` | coincident |
| --- | --- | --- |
| goal box x (live `Region.bbox`) | [1.8500, 2.1500] | [1.8500, 2.1500] |
| bin footprint x (live MuJoCo geoms) | [2.0807, 2.3807] | **[1.8502, 2.1502]** |
| tosses landing in the bin | **6/6 scored failure** | **6/6 scored success** |
| tosses scored a success | 6/12, all resting on open floor | 6/12, **all inside the bin** |

Under the coincident config, "in the bin" and "in the goal region" are the same event in
**12/12** rollouts — the goal is satisfied by exactly the tosses that land in the bin and by
no others.

**Why the bin moved rather than the region.** Two reasons: x = 2.0 is a pairing upstream
already publishes (`o2`), so it is not a new scene; and it reverts the edit that caused the
mismatch rather than compensating for it, since the goal region is the task *specification*
and the bin is scenery that drifted.

**A third reason was given and is arithmetically wrong. Retracted.** It claimed that moving
`blocks_goal_region` onto the bin instead "would have implied a goal box at x ∈ [2.13, 2.33]
after inflation, extending ~18 cm past the bin's far wall onto open floor, so a cube that
overshot the bin entirely would score as a success". Re-derived from upstream `main`'s
`Tossing3D-o1.json`, `MujocoGround._create_regions` (`objects/base.py:874-881`) and
`Bin._create_xml_element` (`primitive_objects.py:368-380`):

- The raw `blocks_goal_region` range is `[1.90, -0.10, 0.0, 2.10, 0.10, 0.10]`, 0.20 m wide
  on x about 2.0. Recentring it on the bin (`bin_init_region` x ∈ [2.23, 2.231], midpoint
  2.2305) gives a **raw** range of x ∈ [2.1305, 2.3305]. So `[2.13, 2.33]` is the range
  *before* inflation, not after; after the 0.05 m per-side inflation it is
  x ∈ **[2.0805, 2.3805]**.
- The bin's outer length is 0.3 m, so its footprint is x ∈ [2.0805, 2.3805]; its walls are
  0.02 m, so the far wall's **outer face** is at **2.3805**. The quoted 2.33 is
  **5.05 cm short of** that face, not 18 cm past it. The 18 cm figure is `2.33 − 2.15` —
  the distance past the *current* goal box's far edge, a different reference point.
- The inflated box [2.0805, 2.3805] coincides with the bin footprint to **0.0 mm**, and
  does so **by identity, not by luck**: half the raw range (0.10) plus the inflation (0.05)
  is 0.15, which is exactly half the bin's 0.30 m length. Recentring this particular region
  on this particular bin can therefore never overshoot it, whatever x the bin is sampled
  at. That is the exact mirror of what moving the bin to 2.0 achieves. A cube that overshot
  the bin entirely comes to rest past 2.3805, i.e. **outside** that box, so it would
  **not** score. The stated consequence does not follow.

The two remaining reasons are unaffected, and moving the bin is still the right call — but
this ground was not one.

**It changes the physics, not just the scoring — and this was not predicted.** Moving the
bin 23 cm nearer puts it where the cube used to land, so it is now an **obstacle in the
flight path**: at swing 0.75 the cube comes to rest at x ≈ 1.71, bounced off the near wall,
rather than x ≈ 1.91. The swings that solve the scene move from 0.6–0.9 to ≥ 0.96.

**`ORACLE_SWING = 0.75` is not valid for the coincident scene and was deliberately not
retuned.** That constant is measured against stock `o1`, which is what every number on this
domain is measured against. **If the coincident scene is ever used for a learning run,
`ORACLE_SWING` must be re-measured against it first — its current value does not solve that
scene.**

On the closed stack, stock `o1` remained the default, guarded by
`test_stock_o1_is_untouched_by_shipping_the_variant`; **`main` inverts this** and defaults to
the coincident scene, with `--task-config stock` still selectable.
**Do not compare a number measured under the coincident config against one that was not.**
On the closed stack, passing the flag with `--variant o2` raised rather than silently
no-oping (o2's bin is already at 2.0). Provenance is pinned against the *installed*
upstream, so a KINDER bump that edits `o1` fails loudly rather than silently widening the
diff.

### 5.9 Concurrency did not contaminate the sweep — and the first check lied

12 concurrent runs on a 24-core box puts Fast Downward's 10 s wall-clock timeout at risk,
and a starved FD returns no plan, which would make a curve measure timeouts rather than
learning.

**This cannot be checked the obvious way.** `ees_method.py` swallows `PlanningFailure` at
all three call sites (lines 386, 778, 802 as written; **404, 829, 853** on `main` at
`db2589f`) without logging it, so a timeout and a genuinely
unreachable goal are indistinguishable and neither reaches `log.txt`. A re-run diff was
substituted, since a timeout would necessarily change the plan and hence the trajectory:

```text
sweep    (12-way concurrent, OMP=1): [6, 3, 3, 4, 7, 7, 7, 8, 8, 9, 7]
recheck3 (1 worker,          OMP=1): [6, 3, 3, 4, 7, 7, 7, 8, 8, 9, 7]
IDENTICAL
```

**The first attempt returned DIFFERS and would have become a false finding.** It drove
`hitl_pmp.cli` directly, missing the `OMP_NUM_THREADS=1` / `MKL_NUM_THREADS=1` that
`run_sweep` pins on every child (`SweepRunner._execute_one`'s `child_env`;
`run_sweep.py:153` as written, `:426` on `main` at `db2589f`), so it compared a many-thread run
against a one-thread run and was about to report "concurrency perturbs Tossing3D".

Two facts fall out, both worth more than the check itself:

- **Any re-run comparison in this repo must go through `run_sweep`, not the CLI.**
- **The sampler's numerics are thread-count dependent**, so a `--seed` determines a run only
  at a fixed thread count.

Corroborating cross-agent evidence on the same box: 13,105 live FD process observations,
maximum lifetime 0 s against the 10 s budget, and byte-identical `stats.json` at 20-way
concurrency.

### 5.10 The GPU cannot help this workload

The box has an idle RTX 5090 and a Tossing3D run costs ~1.1 s per transition, so it is a
fair question. Measured per subsystem; **no change is warranted**:

| subsystem | GPU path? | measured | verdict |
| --- | --- | --- | --- |
| MuJoCo physics | **none available** | 8.0 ms per env step = 200 substeps × 0.040 ms | dominant cost, unreachable |
| rendering | **already on GPU** | `GL_RENDERER = NVIDIA GeForce RTX 5090`, 1.20 ms/frame | nothing to do |
| PyBullet IK / motion planning | none | `p.connect(p.DIRECT)` is headless CPU by design | unreachable |
| EES sampler MLP (torch) | possible, unwise | 44 ms per 200 full-batch steps, 1 thread | slower on GPU, and would break byte-identical `stats.json` |

GPU MuJoCo means MJX, and `grep` finds no `mjx` and no `jax` anywhere in `kindergarden` or
`kinder-models`. MJX's advantage is thousands of environments in parallel, not
single-environment latency. Incidentally the KINDER venv's torch is `2.13.0+cpu`, so the
sweep never had a GPU option in the first place.

### 5.11 Rendering

- **The storyboard is 4 frames per episode by construction.** `core.Renderer` emits one
  frame per *transition*, and one transition here is a whole skill — several hundred MuJoCo
  ticks. That is a property of the core interface, not of this domain.
- **The smooth clip is a separate script**, `scripts/render_tossing3d_demo.py` on the closed
  branches (`main` has no such file; its demo script is `scripts/tossing3d_oracle_demo.py`,
  and `main` gets smooth clips from a `gymnasium.wrappers.RenderCollection` instead),
  following
  KINDER's own `generate_demo_video.py` step for step: per-tick `env.render()` via
  `KinderBackend.capture_frames_into`, `fps` from KINDER's own `render_fps` metadata (20),
  `imageio.mimsave` straight to GIF with no mp4 round trip, and `kinder.gif_utils.optimize_gif`.
  **276 frames instead of 4**, 171 after trimming.
- Cost: ~1 render per tick at ~4 ms on ~8 ms of physics, i.e. **≈ +50% wall-clock**. This
  is why smooth mode is opt-in and the storyboard stays the default for checkpoint
  recording.
- **`render()` returned a view, not a copy.** `np.asarray(...)` does not copy an already-uint8
  array, so it handed back a view into MuJoCo's reused render buffer — harmless one frame at
  a time, silently wrong when accumulating hundreds. Fixed in #40.
- **276 captured ticks contain only 150 distinct frames**; `SETTLE_STEPS` renders ~126
  identical stills after the cube stops. `trim_static_tail` drops only frames byte-identical
  to the last, keeping 1 s of hold.
- **`scene_bg=True` gets the textured MimicLabs room.** Our `_ensure_env` originally passed
  `scene_bg=False`; KINDER's own docs generator passes `True` for every Dynamic3D env. The
  old justification (that its render path "reaches for the OSMesa context") has **no support
  in KINDER's source** and, measured under `MUJOCO_GL=egl`, it just works. `--check-scene-bg`
  asserts it is purely cosmetic: the same rollout with and without produces a **bit-identical**
  cube trajectory.
- **Palette.** Adding the caption strip turned the green cube grey: the cube is ~400 of a
  frame's 327,680 pixels and median cut (imageio's GIF default) allocates palette entries by
  population, so its entry was marginal and the strip tipped it over. Measured over the cube's
  own pixels against the raw render:

  | pipeline | mean abs channel error | size |
  | --- | --- | --- |
  | no pre-quantisation, 64 colours, lossy 120 | 13.7 | 2.02 MB |
  | octree, 64 colours, lossy 120 | 8.9 | 0.79 MB |
  | **octree, 256 colours, lossy 0** | **2.9** | **1.18 MB** |

- **The caption** is `Tossing3DRenderer.caption`, shared between the storyboard and the demo
  script so they cannot drift, filled per control tick via `KinderBackend.capture_features_into`.
  It reports `z` rather than the `holding` flag because `holding` is the height proxy
  `z > 0.2` and, measured, 24 frames of the seed-0 rollout read `holding` while the cube was
  airborne, across 63 cm of flight.
- `gifsicle` is a requirement of the demo script only — not of CI, the test suite, or any
  sweep.
- **`core/renderer/renderer.py` writes GIFs through an mp4 round trip** —
  `VideoWriter.write_gif` takes an already-written `video_path` and re-reads it
  (`renderer.py:105-124` on `main` at `db2589f`; the quoted "lines 58-68" no longer point at
  it), pushing frames through lossy 4:2:0 H.264 before the palette is chosen. Cross-domain,
  still unfixed.

### 5.12 The demo GIFs are not the seed-0 run behind the curve

The four progression clips in the experiment log come from a **separate run**. Its
trajectory differs from the sweep's seed 0 — `[6,3,1,1,7,7,7,7,8,9,9]` vs
`[6,3,3,4,7,7,7,8,8,9,7]` solved-out-of-10 per checkpoint — so they illustrate the
behaviour rather than being the data.

The likely cause is thread count, not the render flag: that run drove the CLI directly and
so missed `OMP_NUM_THREADS=1`, which §5.9 independently shows moves a Tossing3D run
substantially. So `--num-render-checkpoints` is **not shown** to perturb anything. A matched
pair of `run_sweep` invocations differing only in the render flag would confirm it, and was
not run.

---

## 6. Known defects and open questions

### Unresolved

- **One upstream KINDER issue is still unfiled.** `PickShelfController.reset` does not clear
  its four progress flags, so a reused controller reports success having done nothing
  (§5.6). **The actionable ask is that `reset` clear those four flags, exactly as
  `TossController.reset` already clears `_has_released`.** Use §5.6's mechanism, not #42's
  original — a wrong mechanism sends maintainers after the wrong field.
  - **(inferred)** A second candidate, never framed as an upstream issue in the record:
    `docs/envs/Tossing3D.md`'s `## Description` paragraph describes the pre-`1183de7` scene
    (§5.3). That is the subject
    of kindergarden PR #126, still open.
- **The KINDER-backed claims are unverified on CI.** CI never installs KINDER, so at
  `3565312`: **11/11** tests in `test_kinder_fidelity.py` skip, and **7/8** of #64's own
  `test_task_configs.py` skip (1 of the 8 runs offline). CI therefore **cannot** catch a
  wrong goal box or a broken task-config provenance pin. The `79/79` tossing3d figure quoted
  in #64 was measured in a venv with the simulator installed, before that PR's final rebase,
  and was **not** re-run for the rebased head. The same gap holds on `main`: **15/15** of
  the suite's skips are these KINDER-gated tests — 13 in `test_kinder_fidelity.py` and 2 in
  `test_operator_fidelity.py`, re-counted at `db2589f`.
- **No `HumanOracle`.** This is the domain in the repo that most needs one.
  `Metrics.num_human_interventions()` reports `(0.0, 0)` — not because no intervention was
  needed, but because none is representable.
- **`o2` is not supported.** It requires two cubes in the goal region; the symbolic layer
  here is single-cube. The CLI accepts `--variant o2` because the backend does, but the goal
  would be under-specified.
- **The planning-failure rate cannot be read out of any run log.** `ees_method.py` catches
  `PlanningFailure` silently at three call sites and never logs it. Still true on `main`
  (lines 404, 829, 853 at `db2589f`). **Re-find them with
  `grep -n 'except PlanningFailure'` rather than trusting those numbers** — two of the
  three moved on each of the last two merges into `main` (#85, then #90), so a line pin
  here decays faster than anything else in this file.
- **The irreversibility hypothesis is untested.** Changing `PracticeLoop`'s reset behaviour
  is a separate design decision, deliberately not made anywhere in this stack. The EES log is
  the reproduction baseline such a change would be measured against.
- **`ORACLE_SWING` is invalid for the coincident scene** and was never re-measured against
  it (§5.8). Applies to the closed branches only — `main` has no swing dial and no
  `ORACLE_SWING`.
- **Which of two mechanisms makes the swing dial step** — release-tick quantisation or
  PyBullet path replanning — is not measured (§5.4).
- **Whether `--num-render-checkpoints` perturbs a run** is not shown either way (§5.12).
- **The dip's mechanism** is untested, and rests on a dip that is itself not established
  (§5.1).

### Resolved

- The goal-region inflation bug — fixed by reading `Region.bbox`, both arms re-run (§5.2).
- The PyBullet leak — worked around here as `_release` (§5.5), then **fixed upstream** by
  `kinder-baselines` PR #87 (`9512b9e`), which also settles who owned the bug. `main`
  carries no workaround.
- The wrong reuse mechanism — corrected, but see §1 for where the correction lives (§5.6).
- "Marginal grasps flip on residual MuJoCo solver state" — refuted; it was the leaked
  clients (§5.5).
- "Landing in the bin is not itself a failure" — retracted; as a practical matter it always
  is (§5.3).
- "Concurrency contaminated the sweep" — refuted, after the first check turned out to be
  invalid (§5.9).
- "EES gets worse before it gets better" — withdrawn as not established (§5.1).
- "The renderer's `task_view` docstring is inaccurate" — checked and **refuted**;
  `set_render_camera("task_view")` is called explicitly, exactly as KINDER's own
  `generate_demo_video.py:189` does.
- "Caching the ground controller is the obvious fix" — refuted by measurement (§5.6).
- A red test asserting the full-power overshoot on seed 1, the one seed where it does not
  happen — fixed to assert on seeds 0 and 2 (§5.7).

---

## 7. Environment gotchas that cost real time

**All six of these are now maintained in `CLAUDE.md` and the `hitl-env` skill**, which are
the authoritative copies; the versions that used to be spelled out here had already drifted
(the install command named a `[tossing3d]` extra that `main`'s `pyproject.toml` does not
declare). They are listed by name only, so the duplicate cannot drift again:

1. The editable install points at the main checkout, not at your worktree — set an absolute
   `PYTHONPATH`.
2. Fast Downward's sibling convention resolves to a nonexistent path from a worktree — set
   `FD_EXEC_PATH`. Measured on `main` at `d6ae54c` from a worktree with it unset:
   **29/29 failures**, all FD-dependent, and zero failures anywhere else. The count moves as
   `main` gains FD-backed tests.
3. `MUJOCO_GL=egl PYOPENGL_PLATFORM=egl` must be **exported**, not set inline after import;
   `register_all_environments()` forces `osmesa` whenever `DISPLAY` is unset.
4. KINDER goes in its own virtualenv, never in `hitl-pmp`.
5. The distribution is `kindergarden`; the import package is `kinder` (§4).
6. Any re-run comparison must go through `scripts/run_sweep.py`, not the CLI — the thread
   pinning is load-bearing and a `--seed` determines a run only at a fixed thread count
   (§5.9).

---

## 8. Loose ends found while writing this file

Recorded so the next reader does not have to re-derive them.

1. **#61 is not an ancestor of the stack top** (§1). The wrong mechanism survives at
   `3565312` in three places, plus a fourth in the experiment log that #61 never covered.
   Verified by `git merge-base --is-ancestor fc50bfd 3565312` and by grepping `3565312`.
   Moot for `main`, which was written fresh (§1).
2. **"the eight simulator tests" is stale in two places on the closed branches.** Both
   `src/hitl_pmp/environments/tossing3d/README.md` and `pyproject.toml`'s `tossing3d` comment
   say the conftest skips "the eight simulator tests". At `3565312` `test_kinder_fidelity.py`
   holds **11**. The count was accurate at #40 and was not updated as #42, #59 and #62 each
   added fidelity tests. Neither site exists on `main`.
3. **Stack numbering in PR titles is inconsistent** — "1 of 3" through "3 of 3", then "4/4",
   "5/5", "6/6", and #61 unnumbered (§1).
4. **The FD-failure count differs from the brief.** 29/29 measured here on `main` at
   `d6ae54c`, against 26 quoted in the brief for this file. Same mechanism, different `main`.
   **(inferred: the brief's 26 was presumably measured on an earlier `main`; the record does
   not say.)**
5. **#43's checkpoint table is the pre-correction one.** Its body shows EES 21/100 at 150 and
   40/100 at 750, and floor 22/100 at 150, 24/100 at 450, 21/100 at 1350. #59's re-run moves
   exactly those five cells. Both are correct as of their own PR; §5.1 above uses the
   post-correction values. Do not mix the two tables.
6. **The `79/79` tossing3d figure was not re-measured at #64's final head**, and no
   KINDER-backed number in the stack is verified by CI (§6).
7. **Upstream `kindergarden` `main` is one commit ahead of the pin** (§4). Source citations
   read out of a checkout at `main` are from `cdf1b8b`, not `39eb7e08` — and a local
   `reference/` checkout may not be at `main` at all.
8. **Suite counts across the stack are not comparable.** The PRs report 725, 738, 739, 742
   passed at different heads; each PR says explicitly that the number drifts upward on every
   rebase from tests `main` itself gained. Read them as "green at this SHA", never as a fixed
   figure.
9. **The experiment log gives two conflicting figures for how much of the swing prior
   solves.** `docs/experiment-logs/2026-08-04-tossing3d-ees.md` says both "~36% of the swing
   prior" and "roughly half the sampler's `[0.25, 1.25]` prior lands in the region". Neither
   is attached to a stated measurement, and both are used to explain the same thing (why the
   pre-practice checkpoint is 33/100 rather than near zero). One of them is wrong; the record
   does not say which. **(inferred: the `[0.57, 0.93]` interpolated band was withdrawn in
   §5.4 as not a measurement, and either figure would have been derived from something like
   it — so both are suspect.)** If this number matters to a future argument, measure it.
