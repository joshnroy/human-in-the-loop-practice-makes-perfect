# CLAUDE.md

Guidance for Claude Code working in this repository. **The code is truth.** This file
carries only what the code cannot say: setup, traps that cost hours, and process rules.

Design doc / paper notes live in Notion (ask Josh for access).

## Setup

**Prefix every command with `scripts/with_env.sh`.** It activates the `hitl-pmp` conda
env (Python 3.10), sets `FD_EXEC_PATH`, sets `PYTHONPATH` to its *own* checkout's `src/`,
and pins the EGL rendering backend. Run it bare to print what it resolved.

An agent sandbox cannot execute `source`, `export` or `VAR=x cmd` forms at all, and an
`export` in one call is gone by the next — so for agents the wrapper is not a
convenience, it is the only way to get a correct environment. See the `hitl-env` skill.

```bash
conda activate hitl-pmp
pip install -e ".[dev]"
```

**Fast Downward** is required by any planning-based `Method` (`--method ees`, and
`planning/`'s own tests). Deliberately not vendored: build it once and it is found
automatically if it sits beside this repo, or point `FD_EXEC_PATH` at any other checkout.

```bash
cd .. && git clone https://github.com/aibasel/downward.git && cd downward && ./build.py
```

### `reference/`: third-party checkouts, pinned as git submodules

| path | url |
| --- | --- |
| `reference/kinder-baselines` | `joshnroy/kinder-baselines` |
| `reference/kindergarden` | `joshnroy/kindergarden` |
| `reference/predicators` | `Learning-and-Intelligent-Systems/predicators` |

Two of the three point at forks on purpose: a submodule *hard-fails* when its pinned SHA
is force-pushed away, and a fork Josh controls cannot be force-pushed under us.
`predicators` is read-only reference, pinned on its default branch (`master`, not `main`).

A fresh clone gets empty directories until you populate them:

```bash
git submodule update --init                  # or: git clone --recurse-submodules
scripts/update_reference_repos.sh            # initialise anything missing
scripts/update_reference_repos.sh --check    # report only, clone nothing
```

That script **reports** — never resets — a submodule that is dirty or off the pin, so it
cannot detach someone's mid-investigation checkout. Exit `0` current/initialised, `1` real
failure, `2` skipped or drifted.

**`git worktree add` does not populate submodules.** That is deliberate leverage:
kindergarden's 912 MB is opt-in per worktree (1.1 GB for all three). Nothing breaks
without them — the KINDER-backed tests gate on `importlib.util.find_spec("kinder")` and
**skip**. But CI installs KINDER, so an empty worktree is now *weaker* than CI, not
equivalent. A worktree that will run the Tossing3D tests must populate `reference/`.

**Bumping a pin is a deliberate act, never a side effect of syncing:**
`git -C reference/<name> fetch && git -C reference/<name> checkout <sha>`, then commit the
changed gitlink here. **The two KINDER pins bump together** — each requires the other.

### KINDER (the `Tossing3D` benchmark) — a required dependency

It installs **into `hitl-pmp` itself**, as the `tossing3d` extra. Populate the submodules
first or both `-e` paths are empty directories and pip fails:

```bash
git submodule update --init reference/kindergarden reference/kinder-baselines
pip install -e ".[dev,tossing3d]" \
    -e reference/kindergarden -e reference/kinder-baselines/kinder-models
```

Install from `reference/` specifically, so the tree that is *read* and the tree that is
*run* are the same one. Verify: `kinder.__file__` and `kinder_models.__file__` must both
resolve under `reference/`.

Three transitive ceilings come with it, none of them ours to lift — installing the extra
**downgrades numpy, scipy and pillow** in whatever environment it touches:

| ceiling | imposed by | resolves to |
| --- | --- | --- |
| `numpy<2.0,>=1.23.5` | `pybullet_helpers 0.1.1` | 1.26.4 |
| `scipy==1.14.0` (**exact** pin) | `pybullet_helpers 0.1.1` | 1.14.0 |
| `pillow<12.0` | `moviepy` | 11.3.0 |

**Only if you need IKFast**, install system BLAS/LAPACK once. `pybullet_helpers` compiles
IKFast from C++ the first time a controller asks for inverse kinematics — on Tossing3D
that means `pick_shelf`, not the toss sequence. Without these the build fails; do not
substitute wheel-internal libraries.

```bash
sudo apt install libblas-dev liblapack-dev libgfortran5   # IKFast / pick_shelf only
```

### Four KINDER traps, each of which costs an hour

- **The distribution is `kindergarden`; the import package is `kinder`.**
  `import kindergarden` raises `ModuleNotFoundError`. `kinder_models` lives in the
  `kinder-baselines` monorepo, subdirectory `kinder-models`.
- **`MUJOCO_GL=egl` and `PYOPENGL_PLATFORM=egl` are required, and ordering matters.**
  `register_all_environments()` forces `osmesa` when `DISPLAY` is unset; under `osmesa`
  `import mujoco` raises, and `_check_deps` swallows *every* exception — so all
  `Dynamic3D` envs are skipped **in silence** and `kinder.make("kinder/Tossing3D-o1-v0")`
  fails much later with `NameNotFound`. Import a dynamic3d **module** (e.g.
  `kinder.envs.dynamic3d.envs`, *not* the `kinder.envs.dynamic3d` package, which does not
  pull in `mujoco`) before that call, and set both variables back to `egl` after it.
  `scripts/with_env.sh` exports both; the import ordering is still yours to get right.
  **CI diverges and renders through OSMesa**: an `ubuntu-latest` runner has no EGL
  driver, so `ci.yml`'s `test` job installs `libosmesa6-dev` and sets `MUJOCO_GL=osmesa`
  with `PYOPENGL_PLATFORM` empty — kinder-baselines' own recipe — while the workstation
  stays on `egl`.
  **The backend is inheritable, not hardcoded**: `configure_headless_rendering` snapshots
  `MUJOCO_GL`/`PYOPENGL_PLATFORM` at module import and re-asserts *that*, defaulting to
  `egl`. A value set before KINDER is imported is honoured; the `osmesa`
  `register_all_environments()` writes afterwards is still undone.
- **Memory.** The unbounded PyBullet leak is fixed upstream (`PyBulletSim` disconnects
  from a `weakref.finalize`), but a planner grounds a fresh sim per sampling attempt, so
  holding many alive at once is still expensive — measured **~2.23 GiB per Tossing3D
  run**, roughly 5x what the tooling assumes. Never run an unbounded loop; wrap anything
  iterative in a memory-capped scope (see below). Do **not** add a `_release`-style
  explicit `close()` on top of the finalizer — that double-disconnects. `close()` itself
  is safe and idempotent.
- **First `reset()` downloads ~1 GB** of MimicLabs scene assets into the checkout.
  Automatic and idempotent, but it needs network and a few minutes. Suppress with
  `DISABLE_AUTO_DYNAMIC3D_SCENES_DOWNLOAD`, which is what kindergarden's own CI sets.

**The Tossing3D scene moves with the `reference/kindergarden` pin.**
`environments/tossing3d/` selects no scene of its own, so a pin bump can change the
geometry every measured number was taken under.
`test_the_shipped_scene_still_puts_the_bin_on_the_box_that_scores` reads the installed
KINDER's own task JSON and fails loudly if the bin comes off the scoring box, so the
coupling is observable rather than silent.

### Contributing upstream to `kindergarden` / `kinder-baselines`

Neither repo has a `CONTRIBUTING.md`, and both `.gitignore` `CLAUDE.md`, so their
conventions live here. Line length 88 (black is the authority; pylint says 89); isort
profile black, `multi_line_output = 2`; mypy `strict_equality` / `disallow_untyped_calls`
/ `warn_unreachable`. Titles are `<package>: <lowercase imperative>`; bodies are
`## Summary` / `## Test plan` / `## Followup`. Four traps:

- **`run_autoformat.sh` reformats but never asserts a clean diff**, and CI runs that same
  script — so unformatted code does not fail CI. Repo-wide it rewrites ~28 unrelated files
  from tool-version drift; format your own and revert the rest.
- **The pylint plugin path is per-package.** Point `PYTHONPATH` at the package directory.
  With the wrong one pylint prints `E0013` and **silently runs without the plugin, still
  reporting 10.00/10**.
- **`np.random` is banned** by that plugin; `default_rng` and `Generator` are allow-listed.
- Two runtime facts before debugging a planner: `TrajectorySamplingFailure` is **not an
  `Exception` subclass**, so `except Exception` misses every sampling failure; and
  refinement requires `final_abstract_state == ns`, **exact set equality**.

`kindergarden` specifics: Dynamic3D env registration is **filesystem-derived** — dropping
a JSON in `tasks/` registers an environment with no code change, and a malformed one
breaks *every* environment. `scripts/docs/generate_env_docs.py` matches on the env class's
module file, so JSON/asset/scene edits go stale **silently** — regenerate with
`--env <Name>` or `--force`.

`kinder-baselines` specifics: the env-model dispatcher string must equal the module
filename stem **including case** (`tidybot3d_sweep3D`). `invalid-name` and
`duplicate-code` are disabled, so CapWords locals and near-copy env models are idiomatic.
Selective CI falls back to all ten packages when a change touches `.github/`, `scripts/`,
a root `run_*.sh`, or anything outside a package. **black and isort actively disagree** on
`from X import (Y as Z,)`, live on `main` today.

## Commands

```bash
pytest                        # run all tests
ruff check .                  # lint  (--fix to autofix)
ruff format .                 # format
mypy src                      # typecheck (src only)
lint-imports                  # enforce the core/environments/methods dependency direction
scripts/check_doc_links.sh    # no self-URLs or dangling links in committed logs
```

CI runs these as three jobs — `lint` (`ruff check`, `ruff format --check`, `lint-imports`,
`check_doc_links.sh`), `typecheck` (`mypy src`), `test` (`pytest -q`). `lint-imports` must
print `Contracts: 1 kept, 0 broken.`; anything else is a real architectural error, not a
lint nit. `main` allows squash-merge only.

**Running the gate from a git worktree** needs two environment variables, or ~29 tests
fail for unrelated reasons: `PYTHONPATH=<worktree>/src`, because the editable install
resolves `hitl_pmp` to the *main* checkout, and `FD_EXEC_PATH=<path-to>/downward`, because
Fast Downward is found by a sibling-directory convention that does not resolve from inside
`.claude/worktrees/`. `scripts/with_env.sh` sets both.

**CI's `test` job runs `pytest` twice** — `tests/environments/tossing3d` alone, then
`--ignore` of it — because MuJoCo's OSMesa and torch's bundled triton each load their own
LLVM and segfault when they share one. The workstation renders through EGL, which links no
LLVM, so **the local gate stays a single `pytest`**. kinder-baselines avoids the same
collision the same way.

**Run the gate locally; do not block on GitHub CI.** Local is ~1 minute against CI's ~10,
and it is the same commands. Open the PR, report whatever state CI is in, and finish. If
CI later fails on something local passed, *that divergence* is the bug worth reporting.

**A branch that is `BEHIND` gets rebased before it is surfaced** — it has never been
tested against what it would merge into, and `MERGEABLE` only means "no textual conflict".
Prefer resuming the agent that owns the branch over rebasing in the main checkout.

**One exception: an agent far into an experiment finishes first, then rebases.** If the
incoming changes cannot affect the result, take the results and rebase afterwards — but
"cannot affect the result" is a claim to check. If `main` moved the dynamics, the sampler
or the analysis module, the rebase means a **re-run**, not a replay. State which it was.

Two things that go stale under any rebase: `raw.githubusercontent.com` figure URLs in a PR
body pinned to a full SHA (re-pin as the last step after the final force-push), and
`file:line` citations in prose — cite a **symbol**, with the line as a convenience.

## Running experiments

- Drive runs through `scripts/run_sweep.py`, never a hand-rolled shell loop. Fixed seeds,
  never randomly drawn; a single `--seed` fully determines a run. It writes the
  `<results-root>/<method>/<seed>/` layout `analysis/` globs for, plus a `timing.json`
  beside each `stats.json` (read it back with `analysis/run_timing.py` — do not
  re-measure). `timing.json` is separate on purpose: `stats.json`'s byte-stability is what
  verifies a change did not alter results.
- **`analysis/` is post-run only** — it reads `--output-dir` output back in and never
  drives a `Method`. That is `hitl_pmp/cli.py`'s job.
- **Always cap memory.** Both systemd managers here run `DefaultOOMPolicy=stop`, so an
  uncapped OOM tears down the **entire session** — this happened on 2026-08-03 (~48 GB,
  two mid-flight agents lost). Swap is 100% consumed and all of it is tmpfs, so a spike
  goes straight to the OOM killer.

  ```bash
  systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue <your command>
  ```

  A `--scope` dies with the shell that launched it; long runs need a **service**, not a
  scope. Use `16G` for a full sweep, `2-8G` for probes so a leak hits a wall in seconds.
- **Budget concurrency globally.** `--max-workers` defaults to all 24 cores *per sweep*,
  so two agents taking the default both run ~1.4x slower for no extra throughput. Check
  `pgrep -af '[h]itl_pmp\.cli'` and `/proc/loadavg` first; aim for ~22 concurrent runs
  across *all* agents. Concurrency has been measured not to affect results, so yielding
  cores costs wall-clock only.
- **A sweep must not read any dependency out of the shared checkout.** A worktree isolates
  `src/`, but the editable installs resolve by absolute path, so `import kinder` still
  loads the *main* checkout's `reference/`. Populate the worktree's own `reference/` before
  launching and verify both `__file__`s land under it. On 2026-08-13 a 14,000-cell sweep
  ran for over an hour against the shared tree and blocked an unrelated checkout the whole
  time.
- **Paired tests when arms share seeds** — an unpaired test throws away that structure.
  **Never assert an effect without a p-value.** If you cannot compute a test, report the
  raw numbers and say explicitly that no inference is supported.

## Ask before changing the state of anything public

**Never create or change the state of a publicly visible repository object without
explicit permission for that specific action.** Lab repos
(`Princeton-Robot-Planning-and-Learning/*`) count as public, as does this one.

Covered: opening a PR **including a draft**, merging, closing, opening or commenting on an
issue, posting a review, requesting reviewers, adding labels, editing a PR or issue body,
pushing a branch to a public remote. Not covered: reading, fetching, local branches and
commits, work inside a fork you were told to use, and CI that runs on its own.

**"It's only a draft" is not a reason to skip asking** — maintainers are notified, and
anything posted under Josh's account is a statement he is accountable for.

**An agent's PR *stays* a draft — always, on every repo.** `gh pr create --draft`; never
`gh pr ready`, including flipping one back as part of a rebase or a fix. Josh promotes them
himself. Since a draft notifies maintainers anyway, this is not about privacy: it is about
who is asserting "this is finished and worth your time".

**Permission for one action is not permission for the next.** Approval to open a PR is not
approval to comment on it, edit its body, or open a follow-up issue. Ask again, naming the
exact object, the exact repo, and what it would say. Put this in every subagent brief that
can reach a public repo.

**`gh pr edit` fails** on all three repos with a Projects-classic GraphQL error. Use
`gh api -X PATCH repos/:owner/:repo/pulls/<n> -F body=@file` and read `.body` back, or pass
`--body-file` up front. **`gh pr checks` has no `--json` flag** on gh 2.46.0, so a wait
loop filtering on it exits immediately and *looks like a pass* while CI is pending — poll
`gh api repos/:owner/:repo/commits/<sha>/check-runs` instead.

## Stop and ask when the premise changes

**If something is ambiguous, or a fact the task rests on turns out to be false, stop and
find a human.** Do not pick the likeliest reading and carry on, do not work around the
blocker, and do not quietly widen or narrow scope. Report the partial work and what you
would need to know to continue. Five cases:

- The brief conflicts with the code, the docs, or the live system.
- A fact the task depends on is false.
- Two readings would produce materially different work, and nothing distinguishes them.
- Something changed underneath you — an upstream merge, another agent's push, a moved
  constant — so the thing you are doing is no longer the thing that was asked for.
- You are about to do something the brief does not clearly imply, to make progress.

**The asymmetry is the whole argument.** Stopping costs one message. Continuing on a wrong
premise costs the entire run and can produce a confident, well-formatted result that is
silently wrong. Agent wall-clock is roughly the number of tool calls times ~15s.

**A discrepancy is a finding, not a thing to reconcile.** If the code does not match the
brief, that mismatch is often the most valuable output — say so instead of quietly making
one side match the other. Put this rule in every subagent brief.

## Task organization: tiers of tasks, each a stack of PRs

A **task** is the unit of planning; a **PR is a subtask**. Tiers, and within a tier by
dependency:

1. **Correctness** — the environment, its skills, or the measurement pipeline are *wrong*.
   First regardless of size, because every later result depends on them.
2. **Methods and experiments** — the long-running scientific work.
3. **Infra** — tooling, docs, ergonomics. **Exception:** infra that makes tier-2 work
   significantly faster is promoted into tier 2.

**When several tasks are ready, start the longest-running agent work first.** **Surface
open decisions before delegating**, not after.

## Workflow: one independent feature per PR, stacked in dependency order

Decompose multi-piece work into genuinely independent features *before* any branch is
created. Two features are independent if neither imports/calls the other.

- Write the full dependency-ordered list up front, most-foundational first.
- Build and open **one PR at a time**, stacked on the previous one's branch. If a PR's
  diff spans more than one entry, split it before opening.
- If a later PR reveals an earlier one's scope was wrong, fix the ordering going forward
  rather than quietly re-bundling.

## How to write a PR description, and what to report

Every PR body — tooling and docs as much as experiments — is a **TL;DR** followed by seven
`##` sections, in this order:

`Question / goal` · `Background` · `Hypothesis` · `Guidance given` · `Methods` ·
`Results` · `Recommendation`

- **Background is not optional and goes before Hypothesis.** A reader six months out has
  none of the conversation. Say what the code did before, name the PR/experiment/defect
  this follows from, and define the mechanism the change turns on.
- **`Hypothesis: None — implementation task, not an experiment`** is a correct answer for a
  bugfix, refactor, rebase or docs change. Never invent one.
- **Do not add a TDD section.** Write the failing test first and watch it fail — that is
  the discipline — but keep it out of the write-up. The test in the diff is the evidence.

**Report counts as `x/y`, never a bare percentage.** Everywhere: prose, tables, PR bodies,
logs, axis labels, analysis output. Tossing Room's fixed test set is 14 TRASH /
14 RECYCLING / **2** EMPTY, so "EMPTY: 100%" is really `2/2`. A percentage may accompany a
count (`27/30 (90%)`), never replace it. Write "null result" in full.

**Never edit, restate or recompute a published number.** When a later change makes one
provisional or wrong, add a clearly-marked note beside it — in the committed
`docs/experiment-logs/` entry as well as the PR body, because the log is where a reader six
months out looks. State each reason at the strength it is warranted.

**Any quantitative result needs a figure, not just a table.** The shape — a gap closing, a
curve flattening, two arms diverging — is the thing worth seeing. Plot per-seed spread, not
only a mean. Keep the table too.

### Training-curve style, fixed project-wide

- **Colour carries the arm's role, never anything else.** Blue (`#0072B2`) is the arm that
  *has* an assistance mechanism available (`scheduled`, `on-stuck`, `at-random`,
  `on-no-applicable-skill`) whether or not it ever fires; orange (`#D55E00`) is the arm
  nothing helps (`never`, `no-human`, any control with no assistance mechanism at all).
  This is deliberately about whether the mechanism *exists*: an arm whose trigger never
  activates is still blue, because the finding is that the mechanism existed and did
  nothing. Grey dotted is reserved for reference/ceiling arms. No fourth hue — encode a
  second axis with linestyle.
- **Linestyle carries the subgroup within a colour.** Solid is the main population, dashed
  `(0, (4, 2))` the secondary (e.g. a stuck/stranded split). An arm with no such split gets
  one solid bold line and *says so* in its legend entry ("no stranding here").
- **Reference/ceiling arms are flat horizontal lines, never a curve** — plotting a
  non-learner as a wandering line invites a reader to hunt for a trend that isn't there.
- **Faint per-seed traces (`alpha≈0.16`, `linewidth≈0.8`) drawn first, underneath the bold
  subgroup means (`linewidth≈2.3`), on every training curve.** They are the point: a bold
  mean over a bimodal population describes none of its seeds.
- **No `(x/N)` suffix in the axis label.** The axis label is the bare quantity
  (`solved per seed`); the denominator goes in the panel title (`TRASH tasks (of 14)`).
- **Legend entries carry the exact count**, e.g. `env resets — mean, n=10`.

Any subagent building a training-curve figure follows this section without being
re-briefed.

### Where a figure or video lives

- **Experiment-log PRs commit them**, beside the `docs/experiment-logs/` entry, referenced
  **by repo-relative link** — `![alt](2026-08-07-foo.png)`. **Never a URL in a committed
  file**; CI's `lint` job enforces this via `scripts/check_doc_links.sh`. `main` allows
  squash-merge only, so merging mints a new commit and **orphans every SHA that branch
  pinned** — and the failure is silent, because GitHub keeps orphaned commits reachable, so
  a dead pin returns `200` with byte-correct content for months. Videos use link syntax
  (`[seed 3, stranded](foo.mp4)`): a relative link resolves to the blob page, which GitHub
  renders with a player, whereas a raw `.mp4` URL merely downloads.
- **The PR body is the exception**, since it cannot resolve repo-relative paths. Bodies use
  `raw.githubusercontent.com` **pinned to a full SHA, not a branch**, re-pinned as the very
  last step after the final force-push and verified three ways: `curl` for `200` with the
  right content-type; `sha256` of the *fetched bytes* against the **working-tree** file
  (never `git cat-file`); and `git merge-base --is-ancestor <sha> <branch>`, since `200`
  and `sha256` both pass on an already-orphaned commit. **Do all three against the raw URL,
  never the `blob/` page** — a blob URL returns `200` and `text/html` for *every* path, so
  hashing it gets you the sha of a web page and the check passes while proving nothing. At
  a raw URL, `application/octet-stream` on an `.mp4` is the **correct** content type, not a
  failure.
- **Every other PR leaves a drag-drop `TODO` block** where the image belongs, and serves
  the file on the scratch web server (`http://agni:8765/<file>`) for Josh to drop in. This
  matters most on **upstream** PRs, where a maintainer should not have to carry our
  illustration.
- A PR that *produces* an artifact — a renderer, a demo, a plot script — should show one
  even when it has no "results".

## Architecture

`src/hitl_pmp/` is reusable library code; `tests/` mirrors it 1:1. `analysis/` imports from
`hitl_pmp`, never the reverse.

`core/` holds six fixed abstract interfaces: `Problem`, `Method`, `Renderer`, plus
`Environment`, `HumanOracle` and `Tasks` nested *under* `core/problem/`. **The dividing
line is whether a class carries real per-run state**: `Environment`, `Problem`, `Tasks`,
`Method` and `Metrics` do, so they are real pydantic (`BaseModel, abc.ABC`) instances with
normal instance methods; `HumanOracle` and `Renderer` do not, so they stay static-method
containers that take the one `Environment` instance they need as an explicit per-call
argument rather than reaching for a global. A concrete domain's genuine structural
*constants* stay `ClassVar`.

`Problem` is a **facade**: everything but `run_task_episode` is a concrete one-line
passthrough to `Problem.env`/`.tasks`/`.human`.

**Why this breaks from Gym's `reset()`-is-free assumption**: a robot can take
**irreversible** actions, so ending an episode does not imply a free reset — a human must
sometimes intervene, at a cost. `Environment.take_action`/`get_valid_actions` operate on
that instance's own tracked `current_state` (no explicit `state` param — a `Method` that
needs to plan carries its own model). `set_state` is a privileged external override used by
`HumanOracle`; `hard_reset()` is harness-only.

`src/hitl_pmp/cli.py` is the global entrypoint (`python -m hitl_pmp.cli --env <name>
--method <name> ...`); both flags are required, all flags are named. `--output-dir` writes
`stats.json` (the serialized `Metrics`, raw fields only), `config_snapshot.json` (the
resolved argparse namespace plus every checkout's commit — separate from `stats.json` for
the same byte-stability reason as `timing.json`), and a demo `episode.mp4`.

### Naming: "robot"/"actor" is ours, "agent"/"agentic" means an LLM

**If we implement it, it is a robot or an actor. If it is LLM-driven, it is an agent.** Not
a style preference — the collision has already happened: an arm named `agent-signal`
meaning *the robot raises `InteractionComplete`* shipped alongside a genuine LLM pure-agent
baseline, and from the word alone a reader cannot tell which is which. Applies to arm
names, CLI flags, class names, PR titles and prose. `methods/pure_agent/` keeps its name;
that one really is an LLM agent.

### Conventions (enforced by lint, not just documented)

- **Pydantic only** — `dataclasses`/`attrs` banned via ruff `TID251`.
- **Keyword-only everywhere** — ruff `PLR0917`, `max-positional-args = 0`. Only
  `self`/`cls`, unavoidable dunders (`# noqa: PLR0917`), and third-party calls are exempt.
- **Lowercase filenames only** — ruff `N999`.
- **No `if TYPE_CHECKING:` guards** — banned via `TID251`. Import the target's `types.py`
  directly, never its ABC file.
- **No `__init__.py` re-exports** — every name has exactly one import path.
- **Imports absolute across subpackages, relative within one** — ruff `TID252`.
- **Data lives in the `types.py` of the module it supports**, never a shared bucket file.
  `Task`/`Goal` are deliberately *not* frozen (they wrap a mutable numpy-backed `State`);
  `Object`/`Type`/`GroundAtom`/`Predicate` *are* (they sit in dict keys / a `frozenset`).
- **Files/classes organized top-down**, most composite first. No `model_rebuild()` needed.
- **No bare module-level functions** for concrete business logic — static-method
  containers, except a short lambda where an interface demands a positional callable.
- **No external links in committed files** (Notion is private), and comments only where
  they explain a *why*.

### Sibling folders

`environments/` (concrete `Environment`+`Tasks`+`Problem` per domain), `humans/` (concrete
`HumanOracle`s — **none exist yet**, which is why `Metrics.num_human_interventions()`
reports nothing: intervention is not *representable* yet), `methods/`, `adapters/` (a Gym
bridge, not yet written), `recording/` (`--record-full-loop`), `results_writer/`,
`planning/` (PDDL + Fast Downward, for planning-based `Method`s only).

`environments/tossing3d/` is the one domain that does not implement its own dynamics: it
wraps KINDER, so it holds `kinder_backend.py`, the single module allowed to import KINDER
and only lazily. Three consequences: one `take_action` is a whole *skill* (hundreds of
MuJoCo ticks), `set_state` can only restore an **episode-initial** state and raises
otherwise (a flat `State` cannot carry MuJoCo's `qpos`/`qvel`), and the geometry moves with
the pin.

**Paper baselines are out of scope.** Fail Focus, Competence Gradient, Skill Diversity,
Task-Relevant, Task Repeat and MAPLE-Q are deliberately not being built. Do not propose
them.

### The paper's own codebase: `reference/predicators`

The original "Practice Makes Perfect" paper was built on the `predicators` TAMP codebase.
This project deliberately does **not** extend it, but it is the reference implementation
for porting any paper environment faithfully — the paper's "Light Switch" is `GridRowEnv`
in `predicators/envs/grid_row.py`. **When the paper's prose is imprecise or silent on an
exact number, that code is ground truth over the paper text**; check
`predicators/settings.py` for the actual defaults.
