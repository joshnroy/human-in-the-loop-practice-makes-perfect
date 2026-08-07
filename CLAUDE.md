# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Design doc / paper notes live in Notion (ask Josh for access):
https://app.notion.com/p/joshnroy/Human-in-the-Loop-Practice-Makes-Perfect-37133470fbc580aab736c283e49ee5db?source=copy_link

## Setup

Uses the `hitl-pmp` conda environment (Python 3.10). Activate it before running any
command below — a persistent shell can silently fall back to `base`, which has
mismatched dependency versions (e.g. numpy stubs) and produces confusing failures:

```bash
conda activate hitl-pmp
pip install -e ".[dev]"
```

**Fast Downward** is required by any planning-based `Method` (`--method ees`, and
`planning/`'s own tests). It is deliberately not vendored — build it once and it is
found automatically if it sits beside this repo, or point `FD_EXEC_PATH` at any other
checkout:

```bash
cd .. && git clone https://github.com/aibasel/downward.git && cd downward && ./build.py
```

A working `python` and that checkout are the whole dependency; the per-call budget is
enforced by `subprocess` itself. See `planning/fast_downward.py`'s deviations list.

### `reference/`: third-party checkouts, pinned as git submodules

`reference/` holds upstream repos this project reads for API/behavior reference —
`kindergarden`, `kinder-baselines`, `predicators`. All three are **git submodules**,
so `.gitmodules` plus one gitlink per path pins each to an exact commit through this
repo's own history. They were gitignored plain clones until that change; submodules
are the supported way to embed a git repo in another, which is what the old ignore was
working around.

| path | url | pinned at |
| --- | --- | --- |
| `reference/kinder-baselines` | `joshnroy/kinder-baselines` | `11eace5` |
| `reference/kindergarden` | `joshnroy/kindergarden` | `4113237` |
| `reference/predicators` | `Learning-and-Intelligent-Systems/predicators` | `5bd3f5b` |

**Two of the three point at forks on purpose.** The commits this repo depends on live
on unmerged branches, and a submodule *hard-fails* when its pinned SHA is force-pushed
away mid-review — unlike a recorded SHA, which merely reports drift. A fork Josh
controls cannot be force-pushed under us. `predicators` needs no fork: it is read-only
reference, pinned at the tip of its own default branch (`master`, not `main`).

**A fresh clone gets empty directories until you populate them:**

```bash
git submodule update --init          # or: git clone --recurse-submodules
```

Then sync or audit them with one idempotent command:

```bash
scripts/update_reference_repos.sh            # initialise anything missing
scripts/update_reference_repos.sh --check    # report only, clone nothing
```

It initialises submodules that are missing or uninitialised, and **reports** — never
resets — any that are dirty or sitting on a commit other than the pin. That refusal is
the point: `git submodule update` would silently detach someone's mid-investigation
checkout back onto the gitlink. It exits `0` when everything is current or was
initialised, `1` on a real failure, and `2` when nothing failed but something was
skipped or has drifted.

**`git worktree add` does not populate submodules**, which is deliberate leverage
rather than a wart: a worktree starts with empty `reference/` directories and stays
that way unless someone runs that script *inside it*, so kindergarden's 912 MB is
opt-in per worktree (measured 2026-08-07: 1.1 GB for all three — kindergarden 912 MB,
predicators 147 MB, kinder-baselines 5.0 MB). Nothing breaks without them: the
KINDER-backed tests gate on `importlib.util.find_spec`, so a worktree with an empty
`reference/` **skips** those tests rather than failing — exactly what CI does, since CI
never installs the optional extra either. Most worktrees should therefore never
populate `reference/` at all; `--check` answers "am I in sync?" without paying for the
clone to find out.

**KINDER** (the `Tossing3D` benchmark and its parameterized controllers) is two of
those repos, installed into a **separate venv — never `hitl-pmp`**, because it pulls
MuJoCo, PyBullet and OpenCV and `kindergarden` caps `requires-python` at `<3.13`. The
submodules must be populated first, or both `-e` paths are empty directories and pip
fails:

```bash
git submodule update --init reference/kindergarden reference/kinder-baselines
python3.10 -m venv ../kinder-venv
../kinder-venv/bin/pip install -e reference/kindergarden -e reference/kinder-baselines/kinder-models
```

Install from `reference/` specifically, so the tree that is *read* and the tree that
is *run* are the same one — a read-vs-run skew between two copies at different
commits has already caused a wrong SHA to be stated as fact. Verify it took:
`kinder.__file__` and `kinder_models.__file__` must both resolve under `reference/`.

**Only if you need IKFast**, install system BLAS/LAPACK once. `pybullet_helpers`
compiles IKFast from C++ the first time a controller asks for inverse kinematics —
which on Tossing3D means `pick_shelf`, not the toss sequence itself (`move_to_target`,
`move_arm_to_conf` and `toss` never call IK):

```bash
sudo apt install libblas-dev liblapack-dev libgfortran5   # IKFast / pick_shelf only
```

With those present the compile is stock: `compile.py`'s own default paths, no
`BLAS_DIR`/`LAPACK_DIR`/`LIBGFORTRAN_DIR` and no `CC`/`CXX`/`LDSHARED` overrides, even
in a venv seeded from conda's interpreter (which inherits conda's `compiler_compat`
build flags). Without them the build fails; do not substitute wheel-internal libraries.

Four traps, each of which costs an hour:

- **The distribution is `kindergarden`; the import package is `kinder`.**
  `import kindergarden` raises `ModuleNotFoundError`. `kinder_models` lives in the
  `kinder-baselines` monorepo, subdirectory `kinder-models`.
- **`export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl` is required**, and ordering matters.
  `register_all_environments()` forces `osmesa` when `DISPLAY` is unset
  (`src/kinder/__init__.py:67-74`); under `osmesa` `import mujoco` raises, and
  `_check_deps` swallows *every* exception, so all `Dynamic3D` envs are skipped in
  silence and `kinder.make("kinder/Tossing3D-o1-v0")` fails much later with
  `NameNotFound`. Import a dynamic3d **module** (e.g. `kinder.envs.dynamic3d.envs`,
  *not* the `kinder.envs.dynamic3d` package, which does not pull in `mujoco`) before
  that call, and set both variables back to `egl` after it.
- **The unbounded memory leak is fixed upstream**, but memory still needs care. Each
  skill execution used to strand one PyBullet client and ~136 MB forever; `PyBulletSim`
  now disconnects its client from a `weakref.finalize` when it is collected, so a
  sequential run releases as it goes. What did *not* go away is the cost of holding many
  sims alive **at once** — and a planner grounds a fresh controller, hence a fresh sim,
  per sampling attempt. So still never run an unbounded loop; wrap anything iterative in
  `systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0 -p OOMPolicy=continue -- <cmd>`
  (no sudo needed) and check the scope's cgroup `memory.max` before trusting it. This
  machine's `DefaultOOMPolicy=stop` means a kernel OOM takes down the whole session.
- **First `reset()` downloads ~1 GB** of MimicLabs scene assets from Google Drive into
  the checkout. Automatic and idempotent, but it needs network and a few minutes.

`docs/kinder-environment-validation.md` records what was actually measured at
upstream `main` — including that a cube landing **in** the bin scores a **failure**,
which is the single most misreadable thing about `Tossing3D`.

**Where the two KINDER pins come from, and what they deliberately leave out.**

`reference/kinder-baselines` is pinned at `11eace5`, the head of
`josh/feature/tossing-throw-controllers`. That branch is the **Tossing3D port stack**'s
second rung: `josh/feature/tossing-state-abstractions` → `-throw-controllers` →
`-oracle-policy` → `-bilevel-model`. The stack was opened on the lab repo as PRs #89–#92,
which are all **closed**; it now lives on the fork as `joshnroy/kinder-baselines`
**PRs #1–#4** (`#1` targets `main`, each later one targets its predecessor).

The pin is rung two rather than the top on purpose: **that is the only rung this repo
imports.** `kinder_backend.py` and `scripts/tossing3d_oracle_demo.py` pull
`kinder_models.dynamic3d.tossing.parameterized_skills` and
`kinder_models.dynamic3d.shelf.parameterized_skills`, and nothing else from the stack.
The oracle policy (#3) and the bilevel model (#4) are never imported — we carry our own
`environments/tossing3d/predicates.py` and `skill_oracle_policy.py`. **So the working
tree no longer carries either of them**; `docs/` prose that assumes they are on disk is
wrong. `9512b9e` — the PyBullet leak fix, upstream PR #87 — is an ancestor of the pin, so
that fix is present.

`reference/kindergarden` is pinned at `4113237`, the head of
`josh/bugfix/tossing3d-bin-outside-goal-region`: the bin fix every committed run recorded
in its `config_snapshot.json`. Upstream that is `kindergarden` PR #126, still **open**;
on the fork it is `joshnroy/kindergarden` **PR #1**.

Because both pins are gitlinks, a refresh moves nothing on its own — **the pin only
changes when someone commits a new gitlink here**. Bumping one is
`git -C reference/<name> fetch && git -C reference/<name> checkout <sha>` followed by
committing the changed gitlink in this repo, and it is a deliberate act, not a side
effect of running the sync script.

**Do not add a `_release`-style explicit `close()`** on top of the finalizer: that
double-disconnects. `close()` itself is safe and idempotent — it calls the finalizer — so
prefer it over `p.disconnect` by hand, which would strand a stale finalizer against a
reused client id.

`environments/tossing3d/` is the integration (`--env tossing3d`), and it imports KINDER
**lazily**, from one module — so the package still imports, typechecks and tests without
it, and CI (which never installs the extra) skips only the simulator-backed tests. See
that folder's own README for what is upstream's and what is ours.

### Contributing upstream to `kindergarden` / `kinder-baselines`

Neither repo has a `CONTRIBUTING.md` or a PR template, and **both `.gitignore` `CLAUDE.md`**
— `kinder-baselines` since its initial prpl-mono extraction — so that convention is
deliberate and their conventions live here instead. All of this is measured from their
trees, not assumed.

**Gates, and how they mislead.** Line length 88 (black, docformatter; pylint's
`max-line-length` is 89 but black is the authority); isort profile black with
`multi_line_output = 2` and `split_on_trailing_comma`; mypy `strict_equality` /
`disallow_untyped_calls` / `warn_unreachable`. CI is four jobs on `kinder-baselines`
(`autoformat`, `linting`, `static-type-checking`, `unit-tests`) and five on `kindergarden`
(plus `notebooks`). Three traps:

- **`run_autoformat.sh` reformats but never asserts a clean diff**, and CI runs that same
  script — so unformatted code does not fail CI. Running it repo-wide currently rewrites
  ~28 unrelated files from tool-version drift; format your own and revert the rest.
- **The pylint plugin path is per-package**: point `PYTHONPATH` at the package directory,
  since each package carries its own `pylint_plugins/` copy. With the wrong one pylint
  prints `E0013` and **silently runs without the plugin, still reporting 10.00/10**. That
  has already produced a meaningless pass here.
- **`np.random` is banned** by that plugin — `default_rng` and `Generator` are
  allow-listed.

**`gh pr edit` fails on both repos** with a Projects-classic GraphQL error. Use
`gh api -X PATCH .../pulls/<n> -F body=@file` and read `.body` back to confirm.

**Titles are `<package>: <lowercase imperative>`;** bodies are `## Summary` / `## Test plan`
(`- [x]` with the exact commands) / `## Followup`. Merged precedent is terse — their PR #66
was two new files, +180 src / +214 test.

**`kindergarden` specifics.** Dynamic3D env registration is **filesystem-derived**:
`_register_dynamic3d()` walks `tasks/`, treating each subdirectory as a class and each JSON
as a variant, so dropping a JSON in registers an environment with no code change — and a
malformed one raises out of `register_all_environments()` and breaks *every* environment.
`scripts/docs/generate_env_docs.py` **will not notice a task-JSON change**: it diffs against
`origin/main` but then matches on `inspect.getfile(env.unwrapped.__class__)`, the env class's
module file, so JSON, asset and scene edits go stale **silently** — regenerate with
`--env <Name>` or `--force`. The MimicLabs download (~2 GB unpacked) fires on `kinder.make()`
via `_ensure_assets_for_env`, not on first `reset()`, and is gated by
`DISABLE_AUTO_DYNAMIC3D_SCENES_DOWNLOAD`, which CI sets. Test files covering a TidyBot3D
task use a `test_tidybot3d_*` prefix; unit-test files are instead named for the module they
test. Tests call `env._check_goals()` with an inline `# pylint: disable=protected-access` at
the call site rather than wrapping it. Assert messages are **not** a convention there
(37/710).

**`kinder-baselines` specifics.** The env-model dispatcher string must equal the module
filename stem **including case** (`tidybot3d_sweep3D`). `invalid-name` and the `[DESIGN]`
limits are disabled, so CapWords locals are idiomatic, and `duplicate-code` is off or set to
`min-similarity-lines=100` — a near-copy env model is the accepted pattern. Selective CI
falls back to all ten packages whenever a change touches `.github/`, `scripts/`, a root
`run_*.sh`, or anything outside a package. **black and isort actively disagree** on
`from X import (Y as Z,)`, live on `main` today. Two runtime facts worth knowing before
debugging a planner: `TrajectorySamplingFailure` is **not an `Exception` subclass**, so
`except Exception` misses every sampling failure; and refinement requires
`final_abstract_state == ns`, **exact set equality**, not "the add effects hold".

## Commands

```bash
pytest                        # run all tests
pytest tests/core/problem/test_problem.py::test_take_action_delegates_to_env  # single test
ruff check .                  # lint
ruff check --fix .            # lint, autofix
ruff format .                 # format
mypy src                      # typecheck (src only; tests/ has relaxed untyped-def rules)
lint-imports                  # enforce the core/environments/methods dependency direction
coverage run --source=src/hitl_pmp -m pytest -q && coverage report -m  # coverage
pre-commit install            # optional: run lint/format/typecheck locally pre-commit
```

CI (`.github/workflows/ci.yml`) runs these on every push/PR to `main`, as **three jobs**: `lint`
(`ruff check .`, `ruff format --check .`, `lint-imports`), `typecheck` (`mypy src`), and `test`
(`pytest -q`). `main` only allows squash-merge (no merge commits, no rebase merge).

**A branch that is `BEHIND` gets rebased before it is surfaced** — it has never been tested against
what it would merge into, and `MERGEABLE` only means "no textual conflict". Prefer resuming the
agent that owns the branch over rebasing in the main checkout; a stale checkout of an agent's
branch has nearly clobbered work.

**One exception: an agent far into an experiment finishes first, then rebases.** Interrupting a
run to replay commits risks the run for no gain, so if the incoming changes cannot affect the
result, take the results and rebase afterwards. "Cannot affect the result" is a claim to check,
not assume — if `main` moved the dynamics, the sampler, the analysis module or anything else the
experiment reads, the numbers are stale and the rebase means a **re-run**, not a replay. Docs,
tooling and unrelated domains are the safe cases. State which it was in the PR.

Two things that go stale under any rebase and are easy to miss: `raw.githubusercontent.com`
figure URLs pinned to a **full SHA** (the old SHA no longer holds the figures — re-pin and `curl`
each for `200`), and `file:line` citations in prose. Cite a **symbol** with the line as a
convenience, so a shifted line does not make the text false.

**Run the gate locally; do not block on GitHub CI.** The local gate is the real check — it is the
same five commands, and it runs in ~1 minute against CI's ~10. Once it passes, open the PR, report
whatever state CI happens to be in, and finish. Polling GitHub until every check goes green wastes
minutes per PR and tells you nothing the local run did not. Two caveats worth knowing rather than
waiting for: CI does **not** install the optional `tossing3d` extra, so KINDER-backed tests must skip
cleanly there (gate on `importlib.util.find_spec`), and CI *does* have Fast Downward while a fresh
worktree may not — see `FD_EXEC_PATH` below. If CI later fails on something local passed, that
divergence is itself the bug worth reporting.

**Running the gate from a git worktree** needs two environment variables, or ~29 tests fail for
reasons unrelated to any change: `PYTHONPATH=<worktree>/src`, because the editable install resolves
`hitl_pmp` to the *main* checkout rather than the worktree; and `FD_EXEC_PATH=<path-to>/downward`,
because Fast Downward is found by a sibling-directory convention that resolves to a nonexistent path
from inside `.claude/worktrees/`.

## Ask before changing the state of anything public

**Never create or change the state of a publicly visible repository object without
explicit permission for that specific action.** Repos owned by the lab
(`Princeton-Robot-Planning-and-Learning/*`) count as public here, as does this one.

Covered: opening a PR **including a draft**, merging, closing, opening or commenting on
an issue, posting a review, requesting reviewers, adding labels, editing a PR or issue
body, and pushing a branch to a public remote.

Not covered, and never needs asking: reading, fetching, local branches and commits,
work inside a fork you were told to use, and CI that runs on its own.

**"It's only a draft" is not a reason to skip asking.** Maintainers are notified on
draft PRs, and anything posted under Josh's account is a statement he is accountable
for — technical claims in it will be read as his.

**And an agent's PR *stays* a draft — always, on every repo.** `gh pr create --draft`;
never `gh pr ready`, including flipping one back to ready as part of a rebase or a fix.
Josh promotes them himself, so this is not on the ask-first list above — it is a thing not
to do at all. It matters most on repos not owned by `joshnroy`
(`Princeton-Robot-Planning-and-Learning/*` and anything else lab- or third-party-owned).
Since a draft notifies maintainers anyway, draft-versus-ready is not about privacy: it is
about who is asserting "this is finished and worth your time". Assistant-authored
technical claims in these PRs have already needed public correction, so the person
accountable for a claim should be the one who releases it. This goes in every subagent
brief that can open a PR, the same way the permission rule does.

**Permission for one action is not permission for the next.** Approval to open a PR is
not approval to comment on it afterwards, to edit its body, or to open a follow-up
issue. Ask again.

To ask: name the exact object, the exact repo, and what it would say, then wait. This
applies to subagents too — put it in every brief that can reach a public repo.

## Task organization: tiers of tasks, each a stack of PRs

A **task** is the unit of planning; a **PR is a subtask**. Implementing a task means
shipping a stack of PRs in dependency order (see the section below for how to build
one). Keep work that belongs together in the same task — "implement the bilevel
planning model" and "consume it here" are one task with two subtasks, not two tasks.

Tasks are prioritized in tiers, and within a tier by dependency:

1. **Correctness** — the environment, its skills, or the measurement pipeline are
   *wrong*, so anything measured on them is suspect. These come first regardless of
   size, because every later result depends on them.
2. **Methods and experiments** — the long-running scientific work.
3. **Infra** — tooling, docs, ergonomics. **Exception:** infra that makes tier-2 work
   significantly faster gets promoted into tier 2, since that is what it is for.

Two scheduling rules on top of that ordering:

- **When several tasks are ready, start the longest-running agent work first** so the
  expensive jobs are running while the cheap ones are discussed and decided.
- **Surface open decisions before delegating**, not after — a wrong assumption costs a
  whole agent run, and agent wall-clock is roughly the number of tool calls times ~15s.

## Workflow: one independent feature per PR, stacked in dependency order

Multi-piece work (e.g. "port this paper baseline") gets decomposed into a list of
genuinely independent features *before* any branch is created, not discovered
mid-implementation. Two features are independent if neither imports/calls the
other; a feature that imports another is *dependent* on it, not independent, even
if they're conceptually part of the same effort.

- Write out the full dependency-ordered list up front (most-foundational first —
  a feature with zero dependencies on the others goes first; each later PR only
  depends on what's strictly below it in the list).
- Build and open **one PR at a time**, stacked on the previous one's branch,
  even when the whole set was scoped together. Don't bundle several independent
  features into one PR because they're related or were requested together — if a
  PR's diff spans more than one of the list's entries, split it before opening.
- If a later PR reveals that an earlier one's scope was wrong (e.g. a piece
  turns out to need infrastructure that didn't ship yet), fix the ordering going
  forward rather than quietly re-bundling — reopen/re-split as needed, and keep
  the running dependency list current so this doesn't recur.
- `scripts/` holds operational entrypoints that *drive* runs — notably
  `scripts/run_sweep.py`, which runs a (method × seed) grid in parallel into the
  `<results-root>/<method>/<seed>/` layout `analysis/` globs for, with fixed
  (never randomly drawn) seeds. Use it rather than hand-rolling a shell loop; a
  single `--seed` fully determines a run, pinned end-to-end by
  `tests/scripts/test_reproducibility.py`. Each run also gets a `timing.json`
  beside its `stats.json` (wall-clock, exit status, and both the sweep-local and
  machine-wide concurrency it ran against) — deliberately a *separate* file, since
  `stats.json`'s byte-stability is what verifies a change didn't alter results, and
  timestamps in it would break that. Read it back with `analysis/run_timing.py`.
  A run that *fails to launch* (fork() under memory pressure) is retried up to
  `--max-spawn-attempts` times; a run that launched and exited non-zero never is,
  since `--seed` makes it deterministic and re-running only reproduces it.
  Failures and retries are printed to **stderr the moment they happen** (one line
  each, pointing at that run's `log.txt`) so a watcher can cancel a broken sweep
  early instead of learning at the end; progress stays on stdout.
- `analysis/` scripts are **post-run analysis only** — they read `--output-dir`
  output back in and produce plots/tables/reports; they never run a simulation
  or drive a `Method` themselves. That's `hitl_pmp/cli.py`'s job (`python -m
  hitl_pmp.cli --env ... --method ... --output-dir ...`). If an `analysis/`
  script is calling `Problem`/`Method`/`Environment` directly instead of
  invoking the CLI and reading its output, that's a sign that the CLI-side
  wiring it depends on shipped in a later PR than it should have.

## How to write a PR description, and what to report

Every PR body — tooling and docs as much as experiments — is a **TL;DR** followed by
seven `##` sections, in this order:

`Question / goal` · `Background` · `Hypothesis` · `Guidance given` · `Methods` ·
`Results` · `Recommendation`

- **Background is not optional and goes before Hypothesis.** PRs are reviewed long after
  the conversation that produced them, and a reader six months out has none of it. Say
  what the code did before, name the PR/experiment/defect this follows from, and define
  the mechanism the change turns on. `Question / goal` is what was attempted; `Background`
  is what someone needs in order to understand that goal at all.
- **`Hypothesis: None — implementation task, not an experiment`** is a correct answer for
  a bugfix, refactor, rebase or docs change. Never invent one to fill the slot.
- **Do not add a TDD section.** Write the failing test first and watch it fail — that is
  the working discipline — but keep it out of the write-up. No red-green narration, no
  quoted pytest output. The test in the diff and the pass counts are the evidence.

**Report counts as `x/y`, never a bare percentage.** Everywhere: prose, tables, PR bodies,
experiment logs, axis labels, analysis output. A percentage hides the denominator, and the
denominators here are small and uneven — Tossing Room's fixed test set is 14 TRASH /
14 RECYCLING / **2** EMPTY, so "EMPTY: 100%" is really `2/2`, which is almost no evidence,
while "TRASH: 100%" is `14/14`. A percentage may accompany a count (`27/30 (90%)`), never
replace it. Write "null result" in full; reserve a bare `null`, in backticks, for the code
value.

**When a later change makes a published number provisional or wrong, the staleness note goes
in the committed `docs/experiment-logs/` entry as well as the PR body.** The PR body is not
where a reader six months out looks; the log is. Put it where someone landing mid-page will
see it, not only at the bottom. **Never edit, restate or recompute a published number** — add
a clearly-marked note beside it, so both what was originally reported and why it is now
provisional stay visible. State each reason at the strength it is warranted, and mark an
unverified part as unverified.

**Any quantitative result needs a figure, not just a table.** A table makes the reader
reconstruct the shape one number at a time, and the shape — a gap closing, a curve
flattening, two arms diverging — is the thing worth seeing. Plot per-seed spread rather
than only a mean: with ten seeds a bar chart of two means hides one seed driving the whole
effect. Keep the table too; the figure shows the shape, the table carries the numbers.

**Where a figure or video lives depends on the kind of PR:**

- **Experiment-log PRs commit them**, alongside the `docs/experiment-logs/` entry, and
  reference them from the body by `raw.githubusercontent.com` URL **pinned to the commit
  SHA, not the branch**. `curl` each and confirm `200` with the right content-type.
- **Every other PR leaves a drag-drop `TODO` block** where the image belongs, and serves
  the file on the scratch web server (`127.0.0.1:8765`) for Josh to drop in. Dragging a
  file into the GitHub editor uploads it to `user-attachments`, which is permanent and
  lives in neither repo — this matters most on **upstream** PRs, where a maintainer should
  not have to carry our illustration. A `raw.githubusercontent` link into a topic branch
  breaks when the branch is deleted, and one into a force-pushed-away commit renders today
  and breaks silently later.
- A PR that *produces* an artifact — a renderer, a demo, a plot script — should show one
  even when it has no "results".

## Architecture

`src/hitl_pmp/` is reusable library code; `tests/` mirrors it 1:1. `analysis/` (scripts/
notebooks producing results/figures) will import from `hitl_pmp`, never the reverse.

### The `core/` interfaces: instances where there's real state, static containers where there isn't

`core/` holds six **fixed abstract interfaces**: `Problem`, `Method`, `Renderer`
(top-level), plus `Environment`, `HumanOracle`, `Tasks` (nested *under*
`core/problem/`, not siblings of it — see below for why).

The dividing line is **does this class carry real per-run state**:
- `Environment`, `Problem`, `Tasks`, `Method` genuinely do (`current_state`,
  `env`/`tasks`/`human`, RNG streams, etc.), so they're real pydantic
  (`BaseModel, abc.ABC`) instances, constructed with that state as keyword
  constructor arguments (e.g. `LightSwitchEnvironment(grid_size=10)`,
  `LightSwitchProblem(env=env, tasks=tasks)`) — every method is a normal instance
  method (`self`, not `@staticmethod`), and a concrete domain's own genuine
  structural *constants* (e.g. `LightSwitchEnvironment.robot`/`.light_type`/
  `.action_space`) still stay `ClassVar`, since those really are the same for every
  instance that will ever exist. `Metrics` (`core/metrics/metrics.py`) got the same
  treatment even though it isn't one of the six interfaces (see below).
- `HumanOracle` and `Renderer` have no state of their own to hold between calls, so
  they stay static-method containers — but both now take the one `Environment`
  *instance* they need to read/mutate as an explicit per-call argument
  (`HumanOracle.execute_human_command(..., env: Environment)`,
  `Renderer.render_frame(*, state, env, label=None)`) rather than ever reaching for
  a global. The same rule extends to any concrete business logic underneath these
  interfaces: genuinely stateless helpers (e.g.
  `environments/lightswitch/action_oracle_policy.py`'s `ActionOraclePolicy.get_action`)
  stay static-method containers too — never a bare module-level function, except a
  short lambda where an interface demands a positional callable (`Predicate.holds`,
  `Policy`) — but anything that needs a specific `Environment` instance's config now
  takes that instance as an explicit parameter instead of reading a class attribute.

`Metrics` (`core/metrics/metrics.py`) sits alongside these but isn't abstract: every
method there is already a genuine, reusable default (nothing here needs different
behavior than "one task type, no real human-intervention tracking" yet), so there is no
forced-must-override method the way `Problem` still has `run_task_episode`. Callers
construct `Metrics()` directly, no per-domain subclass; a future `Method`/environment
that needs different behavior overrides just the specific method that differs. One fresh
instance per run, so there is no shared state to clear.

```
core/
├── problem/
│   ├── problem.py            Problem — composition root / facade
│   ├── environment/
│   │   ├── environment.py     Environment — the one real-world/ground-truth instance
│   │   └── types.py            State, Object, Type, Action
│   ├── human/
│   │   ├── human.py            HumanOracle — the human-cost model
│   │   └── types.py            Cost, CommandStartStateDescription, CommandGoalDescription
│   └── tasks/
│       ├── tasks.py            Tasks — task/goal generation
│       └── types.py             Task, Goal, Predicate, GroundAtom
├── method/
│   ├── method.py               Method — the agent side
│   └── types.py                 LabeledAction, Policy, Rollout, SetupCommand, SetupCommandTarget, Skill, GroundSkill, Variable, LiftedAtom
├── metrics/
│   └── metrics.py               Metrics — the evaluation protocol
└── renderer/
    └── renderer.py              Renderer, VideoWriter
```

`Problem.run_task_episode` takes an optional `renderer: type[Renderer] | None = None`
and returns `(succeeded, frames)` — every episode is optionally recordable through
this one call (no separate rendering-only codepath, which would duplicate the loop).

`src/hitl_pmp/cli.py` is the global CLI entrypoint (`python -m hitl_pmp.cli --env
<name> --method <name> ...`, e.g. `--env lightswitch --method skill-oracle`); both
`--env` and `--method` are required. `--env` registers a domain's own config flags
(via `environments/<name>/cli.py`'s `add_arguments`) but that domain is never run
directly — `--method` is what actually drives a registered `core.Method` (via its
own `methods/<name>/cli.py`, e.g. `methods/oracle/cli.py`'s `SkillOracleCli` for the
privileged oracle baselines that predate any real learning `Method`) through
`src/hitl_pmp/practice_loop.py`'s `PracticeLoop`, the one
execution harness every `Method` runs through regardless of whether it learns (see
the `core/` section above for why `Metrics`, what it records evaluations into, is
fully concrete). A *learning* `Method` distinguishes its two phases through
`Method.get_practice_policy` (explores and records training data during an
interaction period) versus `get_task_policy` (exploits only, on held-out evaluation
tasks — learning from it would be training on the test set), plus `end_cycle()` for
per-cycle retraining; both are concrete defaults, so non-learning baselines need no
boilerplate. All flags are named, no positional arguments. `--output-dir DIR`
(global) writes `stats.json` (the run's serialized `Metrics` — raw fields only, so
readers reconstruct a `Metrics` and call its own computation methods) and, if the
environment has a `renderer.py`, a demo `episode.mp4`. It also writes
`config_snapshot.json` (`config_snapshot.py`'s `ConfigSnapshot`) — the conditions the
run happened under: the *resolved* argparse namespace (so defaulted flags are recorded,
not just passed ones), this repo's commit + dirty flag, the same pair for Fast Downward
and both KINDER upstreams, and the Python/torch/numpy/platform stack. It locates
checkouts through the import system rather than a hardcoded path, and never raises. A
*separate* file for the same reason `timing.json` is: `stats.json`'s byte-stability is
what verifies a change didn't alter results, and a commit SHA in it would break that on
every commit. `--num-render-checkpoints N` (global) instead records N evaluation sweeps
spread evenly from before any practice through the end of training, as
`episode_<transitions>.mp4`. `--record-full-loop PATH` (global) records the *entire outer
loop* — practice periods included, which nothing else renders — to one seekable annotated
`.mp4`. Off by default and a pure observer: a recorded run takes the same actions and
writes a byte-identical `stats.json`. See `recording/README.md`.

**Why `Environment`/`HumanOracle`/`Tasks` nest under `problem/`**: the design doc defines
only `Problem` and `Method` (plus `Metrics`), bundling task generation, the human command
and the standard MDP functions onto one class. Splitting them out buys one dynamics
implementation reusable across different `HumanOracle`/task-distribution pairings, and
Gym-compatibility for RL baselines — but they still belong to `Problem`, so they nest
under it. `Problem` stays a **facade**: `get_current_state`, `take_action`,
`get_valid_actions`, `hard_reset`, `sample_train_task`, `sample_test_task`,
`calculate_cost_for_human_command` and `execute_human_command` are concrete one-line
passthroughs to `Problem.env`/`.tasks`/`.human`. The **only** abstract method on `Problem`
is `run_task_episode`. Full rationale and a mermaid dependency graph:
`src/hitl_pmp/core/README.md`.

**Why this breaks from Gym's `reset()`-is-free assumption**: a robot can take
**irreversible** actions, so ending an episode doesn't imply a free reset — a human
must sometimes intervene, at a cost. `Environment.take_action`/`get_valid_actions`
operate implicitly on that instance's own tracked `current_state: State | None`
(no explicit `state` param — this is not a reusable dynamics function for
hypothetical "what-if" planning; a `Method` that needs to plan carries its own model
for that). `set_state` is a privileged external override used by `HumanOracle` via
`Problem.execute_human_command`, distinct from `take_action`'s normal forward dynamics.
`hard_reset()` is harness-only (before a run starts), never called by the agent.
`HumanOracle.execute_human_command` takes `env: Environment` (the instance) directly
and is responsible for mutating it (e.g. `env.set_state(...)`) to reflect whatever
actually happened — it returns nothing; querying cost beforehand is
`calculate_cost_for_human_command`'s separate, side-effect-free job.

### Naming: "robot"/"actor" is ours, "agent"/"agentic" means an LLM

**If we implement it, it is a robot or an actor. If it is LLM-driven, it is an agent.**

- **robot / actor** — a `Method`, a policy, EES, random-skills, the oracle: anything acting
  in the environment under our own code.
- **agent / agentic** — Claude, Claude Code, `prpl-agent-utils`, the pure-agent baseline,
  and the subagents that do this project's work.

Not a style preference — the collision is real and has already happened. The
human-in-the-loop ladder shipped an arm named `agent-signal` meaning *the robot raises
`InteractionComplete`*, nothing to do with an LLM, while a sibling task was building a
genuine LLM **pure agent** baseline. From the word alone a reader cannot tell which is
which, and both appear in the same PR stack.

Applies to arm names, CLI flags, class names, PR titles and prose alike. `methods/
pure_agent/` keeps its name: that one really is an LLM agent, and "the pure agent" is Tom
Silver's own term for Step 2.5 of the recipe it implements.

### Conventions (enforced by lint, not just documented)

- **Pydantic only** — `dataclasses`/`attrs` are banned via ruff `TID251`.
- **Keyword-only everywhere** — ruff `PLR0917` with `max-positional-args = 0`. Only
  `self`/`cls`, unavoidable dunders (`__getitem__`, silenced with `# noqa: PLR0917`),
  and third-party calls are exempt.
- **Lowercase filenames only** — ruff `N999`.
- **No `if TYPE_CHECKING:` guards** — banned via `TID251`. Where two subpackages each
  need a type the other owns, import the target's `types.py` directly (never its ABC
  file) — neither `types.py` imports the sibling ABC back, so there's no real cycle.
- **No `__init__.py` re-exports** — every name has exactly one import path (e.g.
  `from hitl_pmp.core.problem.environment.types import State`).
- **Imports absolute across subpackages, relative within one** — ruff `TID252` bans
  `..`-parent-relative imports.
- **Data lives in the `types.py` of the module it supports** as pydantic `BaseModel`s
  — never a shared "bucket" file. `Task`/`Goal` are deliberately *not*
  frozen/hashable (they wrap a mutable numpy-backed `State`); `Object`/`Type`/
  `GroundAtom`/`Predicate` *are* frozen (they sit in dict keys / a `frozenset`).
- **Files/classes organized top-down** — most composite/important first, e.g. in
  `tasks/types.py`: `Task` → `Goal` → `Predicate` → `GroundAtom`. No
  `model_rebuild()` calls needed: pydantic resolves forward references lazily on
  first real use.
- **`Type` declares a feature schema** (`feature_names: tuple[str, ...]`, `dim`
  property), not just a name — lets `State`'s validator reject a feature vector whose
  length doesn't match `obj.type.dim`, and lets `State.get(obj=, feature_name=)` look
  up by name instead of raw index. No `parent`/inheritance on `Type` — deliberately
  deferred until a domain actually needs a type hierarchy.

### Sibling folders (concrete implementations; each has its own README)

- `environments/` — concrete `Environment` + `Tasks` + `Problem` per domain, one
  subfolder each (e.g. `environments/lightswitch/`). A domain subfolder holds
  `environment.py`, `tasks.py`, `problem.py`, and optionally `predicates.py` (only if
  a planning-based `Method` needs symbolic `GroundAtom`s for that domain) and
  `skills.py` (only if a `Method` selects lifted `Skill`s rather than acting
  directly in raw action space — declares `Skill` `ClassVar`s plus
  `sample_params`/`compute_action` static methods; see `core/README.md`'s
  `Skill`/`GroundSkill` section and `environments/lightswitch/skills.py`).
  `environments/tossing3d/` is the one domain that does **not** implement its own
  dynamics: it wraps KINDER's `Tossing3D` simulator and its parameterized controllers
  (see the `reference/` section above), so it additionally holds `kinder_backend.py` —
  the single module allowed to import KINDER, and only lazily. Three consequences worth
  knowing before touching it: one `take_action` is a whole *skill* (hundreds of MuJoCo
  ticks), `set_state` can only restore an **episode-initial** state and raises otherwise
  (a flat `State` cannot carry MuJoCo's `qpos`/`qvel`), and it defaults to *our*
  `scripts/task_configs/Tossing3D-o1-coincident.json` rather than upstream's stock `o1`,
  under which a cube landing **in** the bin is a scored failure. Its simulator-backed
  tests gate on `importlib.util.find_spec("kinder")` — the *import* package name; the
  distribution is `kindergarden` — so they skip cleanly on CI.
- `humans/` — will hold concrete `HumanOracle` implementations, the v0 (unconditional) →
  v3 (natural-language, capability-aware) axis from the design doc. Domain-agnostic:
  a `HumanOracle` knows nothing about any specific `Environment`'s dynamics. **None
  exist yet** — the folder is a README and an `__init__.py`, which is why
  `Metrics.num_human_interventions()` reports nothing: intervention is not
  *representable* yet, not merely unobserved.
- `methods/` — concrete `Method`/baseline implementations. `oracle/` holds the
  privileged `skill-oracle`. `practice_makes_perfect/` is the *original* PMP/EES paper
  reproduction on Light Switch — a faithfulness repro, not this project's own research
  contribution — and so far contains **`ees_method.py` and `random_skills_method.py`
  only**, plus `competence_models.py` and `wrapped_sampler.py`. The paper's other
  baselines (Fail Focus, Competence Gradient, Skill Diversity, Task-Relevant, Task
  Repeat, MAPLE-Q) are **not implemented**. Neither are this project's own planned
  baselines (trivial fixed-skill planner, `planning_to_practice.py`, `pure_vla.py`,
  `in_context_vla.py` — see the design doc's baseline progression and expected failure
  modes), which wait on that reproduction.
- `adapters/` — bidirectional `core.Environment` ↔ Gym/Gymnasium bridge:
  `from_gym.py` wraps a third-party `gym.Env` to satisfy `core.Environment`;
  `to_gym.py` wraps this project's `core.Environment` to expose the Gym interface for
  RL libraries like SB3/RLlib. Not mirror images — different directions, different
  jobs. Neither exists yet.
- `recording/` — `--record-full-loop`'s implementation: `LoopRecorder` (the hooks
  `practice_loop.py` calls) plus `StatusBarOverlay`, which composes the annotation
  *around* frames a `core.Renderer` already produced rather than teaching any domain
  renderer about loop state. Sits between `practice_loop.py` and `core/` in the
  import layering. See its own README for the four reset kinds and why they are
  labelled apart.
- `planning/` — bridges `Predicate`/`GroundAtom`/`Skill` to real Fast Downward (PDDL
  planning), for planning-based `Method`s only — needed for `methods/
  practice_makes_perfect/`, since EES's competence-cost-aware task planning has no
  built-in-planner substitute (predicators' own non-FD `astar` planner doesn't
  support per-operator costs at all). Pure deep-RL baselines (MAPLE-Q) never import
  it. Not vendored/bundled — see the Setup section above and `planning/README.md`
  for install steps. `pddl.py`'s `PddlWriter` renders the symbolic layer as PDDL;
  `fast_downward.py`'s `FastDownwardPlanner` runs predicators' own three-stage
  protocol (translate → patch per-ground-skill costs into the SAS file → search with
  `seq-opt-lmcut`). Its tests genuinely shell out to a real FD rather than being
  skipped, so a missing/broken install fails loudly instead of silently.

### The paper's own codebase: `reference/predicators`

The original "Practice Makes Perfect" paper was built on the `predicators` TAMP codebase.
This project deliberately does **not** extend it (it's entangled and hard to extend), but
it is the reference implementation for porting any paper environment/behavior faithfully
— e.g. the paper's "Light Switch" environment is `GridRowEnv` in
`predicators/envs/grid_row.py`, with its skills/NSRTs in
`predicators/ground_truth_models/grid_row/`. When the paper's prose is imprecise or
silent on an exact number (tolerances, sampling ranges, defaults), treat that code as
ground truth over the paper text, and check `predicators/settings.py` for the actual
default config values used.

Read it at `reference/predicators` (see the `reference/` section above). Earlier versions
of this file pointed at a sibling fork, `../hitl-practice`, which is **not present on this
machine** — if that fork exists somewhere and matters, restore the pointer.
