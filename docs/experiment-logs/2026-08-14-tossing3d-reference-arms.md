# Reset-free EES is not worse than random skills — the baseline it was read against was

**Answer: no, and the comparison that suggested otherwise was against a number produced
under different conditions.** Measured at the same 100 cycles, the same 10 fixed seeds, the
same separate evaluation `Problem` and the same two KINDER pins as the EES sweep,
`random-skills` scores **`6.2/100`** reset-free — not the **`24/100`** line from #133 that
`docs/experiment-logs/2026-08-13-tossing3d-reset-policy-new-pin.md` observed reset-free EES
sitting below. Against the matched baseline, reset-free EES (`12.2/100`, quoted from that
page, not recomputed) is **higher on `7/10` seeds, lower on `1/10`, tied on `2/10`**, mean
`+0.60` tasks per seed, exact paired sign-flip **`p = 0.046875`**.

**So "the learner is worse than picking skills at random" is not supported.** That reading
was an artefact of the mismatched baseline, which is exactly why #247 flagged it as
arithmetic rather than a test and why this sweep exists.

**The `+0.60` itself is marginal and should not be over-read.** `p = 0.046875` sits just
under `0.05`, the observed effect is slightly *below* this design's minimum detectable
effect of `0.68`, and `2/10` seeds tied. The claim this page supports is the **negative**
one — reset-free EES is not below matched random skills — not a confident claim that it is
meaningfully above.

**And the stranding is confirmed to be a property of the domain, not of EES.** Reset-free
`random-skills` — which has no planner, no competence model and no sampler to corrupt —
strands on **`10/10`** seeds from cycle **`1`**, takes **`37`** transitions across all ten
seeds, and idles **`990/1000`** practice cycles. EES's own figures at these pins are `10/10`,
cycle `1`, `35` transitions, `990/1000`. Two methods with nothing in common but the
operators collapse the same way.

## Question / goal

**Is reset-free EES actually worse than random skills, measured under identical
conditions?** #247 reported reset-free EES at `12.2/100` and noted it sat below the
`24/100` `random-skills` reference, on `10/10` seeds. It was explicit that this was "a
description of two numbers, not a test". This page runs the test.

## Background

`analysis/practice_makes_perfect/tossing3d_reset_free_arms.py` carries two non-learning
reference levels as module constants — `RANDOM_SKILLS_PER_SEED = 2.4` and
`ORACLE_PER_SEED = 10.0`, both labelled `(#133)` — and draws them as horizontal lines on
every reset-policy figure. They are the only reference points any Tossing3D reset-policy
page has had.

**Both predate the conditions they are now read against.** #133 ran **20 cycles**, not 100,
and predates **#160**, which gave Tossing3D a separate evaluation `Problem`. Without that
separate `Problem` every evaluation episode's `reset_to_task` writes into the practice
environment, so a reset-free arm is reset `--num-test-tasks` times per sweep anyway and
`never` is a label rather than a condition — `PracticeLoop.run` now refuses the combination
outright for that reason. Since then the pins have also moved twice (#246 most recently),
and the throw moved with them: the canonical oracle rollout's resting *x* shifted `+41.6 mm`.

So the `24/100` line describes a different number of cycles, a different evaluation
protocol and different dynamics from the arm it was being compared against. Nothing was
wrong with quoting it as history; it simply cannot carry a comparison.

`--practice-reset-policy never` turns off the per-period reset. On Tossing3D that is
inseparably also "no scene variety": the only way to obtain a new initial state is
`env.reset(seed=...)`, so handing the robot a new scene and resetting it are the same
physical act.

## Hypothesis

Registered in writing before any `stats.json` from this sweep existed. At the time the only
numbers read were #247's own EES arms, reproduced from their committed runs.

> **H1 (primary).** Under matched conditions the ordering **reverses**: random-skills will
> score *below* reset-free EES, not above. The two arms fail in different places —
> EES-`never` plans the skill sequence correctly and only lacks a trained sampler, while
> random-skills draws the *sequence itself* uniformly on a domain where `Toss`
> unconditionally deletes `Reachable`/`Holding` and dead-ends the episode. Getting the
> sequence right is a precondition for scoring at all.
>
> The competing mechanism, stated so it could not be retrofitted: a sampler fit on ~3.5
> transitions of degenerate stranded data could concentrate mass in a worse-than-uniform
> region, which would put EES below random-skills. That is what #247's flagged comparison
> would have implied, and this sweep is the test of it.
>
> **H2.** `random-skills` `scheduled` vs `never` will be a **null result** on the
> evaluation axis. `RandomSkillsMethod` carries no learned state across cycles and
> evaluation runs on a separate `evaluation_problem` reset per episode, so the only channel
> from practice to evaluation is the method's own `_rng`, shared by both phases through
> `choose_ground_skill`. Evaluation collapse is therefore **not available to this method by
> construction**, and arm 2 is not a test of "is stranding a domain property" on that axis.
> Where it *is* a test is the practice axis, and there I predict random-skills strands like
> EES did.
>
> **H3.** `skill-oracle` near `100/100`, and flat by construction.

**H1 confirmed. H2 confirmed. H3 confirmed exactly.** The part H1 got wrong by omission is
how *small* the reversal is: predicting the direction was right, but the margin is one
seed's worth of tasks and only marginally significant.

## Guidance given

- Three arms, `--env tossing3d`, 10 fixed seeds (`0-9`), 100 cycles, matching the EES
  sweep's conditions exactly, driven by `scripts/run_sweep.py`.
- **Reconstruct the conditions from the committed `config_snapshot.json`, not from prose**,
  and verify with a key-for-key comparison ignoring only `num_cycles`, `output_dir`,
  `seed`, `method` and `practice_reset_policy`. Report the match as `x/y`.
- Populate the worktree's **own** `reference/` submodules at the gitlinks `main` records
  and point `PYTHONPATH` at them, because the shared checkout's submodules are on topic
  branches; verify from a probe's own snapshot before running the full sweep.
- Launch as a detached systemd **service**, not a `--scope`; verify the memory cap in the
  kernel cgroup; budget concurrency machine-wide at ~22.
- Paired test across the ten seeds, MDE beside any null result, counts as `x/y`.
- Register the hypothesis **before** looking at the numbers.
- Never edit, restate or recompute a published number; add a marked note beside it.

## Methods

Three arms plus the two EES arms read back from #247's sweep for comparison:

| arm | method | reset policy | cycles | seeds |
| --- | --- | --- | --- | --- |
| 1 | `random-skills` | `scheduled` | 100 | 0-9 |
| 2 | `random-skills` | `never` | 100 | 0-9 |
| 3 | `skill-oracle` | `scheduled` | — see below | 0-9 |
| (#247) | `ees` | `scheduled` | 100 | 0-9 |
| (#247) | `ees` | `never` | 100 | 0-9 |

**`skill-oracle` cannot run 100 cycles, and that is by construction rather than an
oversight.** `SkillOracleCli.run` hardcodes `num_cycles=0` and registers no `--num-cycles`
flag at all, so the arm is exactly one evaluation sweep of 10 tasks per seed. Its `100/100`
therefore pools `10/10` on each of ten seeds from a **single** sweep, where every other
arm's `x/100` pools a ten-sweep mean per seed. The denominators match; the averaging behind
them does not, and the arm is drawn as a flat line for the same reason.

### The decision rule, unchanged from #178/#179 and #247

Reused rather than reinvented, and **delegated in code** to
`Tossing3DResetFree.late_scores`/`pooled`/`stranding_onset` so the two experiments cannot
drift apart on the definitions the comparison rests on.

- A seed's score is its **mean solved count over the last 10 sweeps** (`LATE`).
- **Paired** across the ten seeds, because every arm ran the same fixed seed set.
- Exact paired sign-flip on per-seed differences, **with the MDE beside it**.
- Stranding onset is the first cycle of the *terminal* run of zero-transition cycles.

The pipeline was validated by pointing it at #247's committed runs before this sweep
finished: it independently reproduces `73.4/100`, `12.2/100`, `3069` vs `35` transitions,
`0/10` vs `10/10` stranded, `990/1000` idle cycles and that page's whole per-seed
transition table.

### Provenance, read from the runs rather than from the brief

All **`30/30`** runs completed with exit status `0`.

| field | value | agreement |
| --- | --- | --- |
| `kindergarden_commit` | `c9f00e82f94c807f5a92c76a29f55cc572cdd2a2` | `30/30`, clean |
| `kinder_models_commit` | `9e881264d6868a391fff8e3090b9ea44bea1d231` | `30/30`, clean |
| `fast_downward_commit` | `6230635ccff53e1df38ead53b057a2a0e9160275` | `30/30` |

All three equal the EES sweep's own recorded values, so the arms being compared ran against
the same simulator and the same planner.

**Resolved argument namespaces, key for key against the EES snapshot**, ignoring the five
keys named in the guidance: `random-skills` matches **`17/17`** comparable keys on all
`20/20` of its runs; `skill-oracle` matches **`16/16`** on all `10/10` of its runs. **`0`**
mismatches anywhere. The keys absent from the comparison are the 12 EES-only method flags
(`exploration_epsilon`, `sampler_max_train_iters`, the three `reproduce_predicators_*`, the
three help-seeking flags, and so on) that `RandomSkillsCli` and `SkillOracleCli`
deliberately do not register — `skill-oracle` additionally lacks `num_cycles`, which is the
16-vs-17 difference.

**`git_commit` is not uniform across the 30, and the reason is mine rather than the
sweep's.** `ConfigSnapshot` computes `git_dirty` from `git status --porcelain`, which counts
**untracked** files, and this analysis module was written in the same worktree while the
sweep ran. The distribution is `26/30` at `a5aea86` clean, `1/30` at `e516455` clean and
`3/30` at `e516455` dirty. Two things make it inert:

- **Every one of the 20 `random-skills` runs — the arms every claim on this page rests on —
  recorded the identical `(a5aea86, clean)` state.** Only the 10 `skill-oracle` runs, which
  finished early and score `10/10` on every seed regardless, span the three states.
- `e516455` and `a5aea86` have **byte-identical `src/` and `reference/` trees** (`git
  rev-parse <ref>:src` gives `b8db0bd` at both; `:reference` gives `880af93` at both). The
  only difference between them is the two new files under `analysis/` and `tests/`, neither
  of which `hitl_pmp.cli` imports.

Recorded rather than smoothed over, because a reader checking the snapshots will see three
states and deserves to know which of them could have mattered.

### Isolation, which was necessary rather than precautionary

`git worktree add` does not populate submodules, and the `kinder`/`kinder_models` editable
installs resolve by absolute path to the **main** checkout. Verified directly: without an
overriding `PYTHONPATH`, `kinder.__file__` in this worktree resolves into the shared
checkout, whose `reference/kindergarden` was sitting at `1355404` on the topic branch
`josh/feature/mujoco-substep-control-schedule` — another agent's mid-flight work, and not
the pin. The sweep ran with `PYTHONPATH` seeded with this worktree's own
`reference/kindergarden/src` and `reference/kinder-baselines/kinder-models/src`, confirmed
by printing both `__file__`s and then re-confirmed from a 2-cycle probe's own
`config_snapshot.json` before the full sweep launched.

### Cost

Both sweeps ran as detached `systemd --user` services with `MemoryMax=16G` and
`OOMPolicy=continue`, verified in the kernel (`memory.max = 17179869184` on both cgroups),
at ~22 concurrent runs machine-wide.

| arm | median per run | n |
| --- | --- | --- |
| `random-skills` / `scheduled` | `5040.1 s` | 10 |
| `random-skills` / `never` | `4527.6 s` | 10 |
| `skill-oracle` / `scheduled` | `77.3 s` | 10 |
| `ees` / `scheduled` (#247) | `6795.8 s` | 10 |
| `ees` / `never` (#247) | `5235.3 s` | 10 |

`random-skills` is cheaper than `ees` in both policies, as expected — it never calls Fast
Downward — but the saving is ~26% rather than an order of magnitude, because both arms pay
for the same 101 evaluation sweeps. Sweep wall-clock was `5124.1 s` (20 runs, 11 workers)
and `4622.9 s` (10 runs, 10 workers). `analysis/run_timing.py` globs `*/*/timing.json`, so
it must be pointed at each policy directory rather than the sweep root.

## Results

### The matched baseline is a quarter of the line it replaces

| arm | `LATE` window | sweeps averaged |
| --- | --- | --- |
| `skill-oracle` / `scheduled` | **`100.0/100`** | 1 (see Methods) |
| `ees` / `scheduled` (#247, quoted) | **`73.4/100`** | 10 |
| `ees` / `never` (#247, quoted) | **`12.2/100`** | 10 |
| `random-skills` / `never` | **`6.2/100`** | 10 |
| `random-skills` / `scheduled` | **`5.0/100`** | 10 |
| — the `#133` line these replace | `24/100` | 20-cycle, pre-#160 |

`skill-oracle` returning exactly `100/100` — `10/10` on every one of ten seeds — is worth
more than its place in the ordering suggests: it is an independent check that the pin bump
did not break the throw, so nothing else on this page can be explained away by a broken
domain.

![Two learning curves against practice cycle with three flat reference lines. The blue scheduled-EES curve climbs from about 1 to about 7 out of 10 over the first sixty cycles; the orange reset-free EES curve sits flat between 1 and 1.5 for all hundred cycles. A grey dotted line at 10 marks the skill-oracle ceiling, and two grey dashed lines at 0.50 and 0.62 mark random-skills under env resets and never reset respectively — both clearly below the orange reset-free EES curve. Ten faint per-seed traces underlie each bold mean.](2026-08-14-tossing3d-reference-arms-curves.png)

**Figure 1. The reset-free arm is flat, and the matched random-skills lines sit below it.**
Bold pooled mean over faint per-seed traces. The two `random-skills` arms and `skill-oracle`
are **reference lines rather than curves**: none of them learns, so a curve would invite a
reader to look for a trend in a constant. The two grey dashed lines are nearly coincident
(`0.50` and `0.62` per seed), which is the null result of the control drawn rather than
asserted.

### The comparison the sweep was run to make

| comparison | direction | mean per-seed | exact sign-flip | MDE |
| --- | --- | --- | --- | --- |
| reset-free: `ees` − `random-skills` | higher `7/10`, lower `1/10`, tied `2/10` | `+0.60` | **`p = 0.046875`** | `0.68` |
| `scheduled`: `ees` − `random-skills` | higher `10/10`, lower `0/10`, tied `0/10` | `+6.84` | **`p = 0.00195312`** | `0.82` |
| control: `random-skills` `never` − `scheduled` | higher `4/10`, lower `2/10`, tied `4/10` | `+0.12` | `p = 0.34375` — **null result** | `0.28` |

**Reset-free EES is above matched random-skills, not below.** The direction is the opposite
of what the `24/100` line implied. But the margin is small: `+0.60` tasks per seed out of
10, `p` only just under `0.05`, an observed effect slightly under the design's MDE, and two
tied seeds. **The supported claim is "not worse", not "clearly better".**

**Under `scheduled` the same comparison is unambiguous** — `+6.84` on `10/10` seeds,
`p = 0.00195312`, the floor an exact sign-flip can return at ten paired seeds. EES learns a
great deal when it is allowed to practise; the reset-free arm's problem is that it barely
practises at all.

**The control is a null result** — `p = 0.34375`, MDE `0.28`, and `55` seeds would be needed
for 80% power at the observed `+0.12`. This is the *expected* outcome and not evidence that
the domain is benign: as registered in H2, `RandomSkillsMethod` carries no learned state
across cycles and evaluation runs on an isolated `Problem`, so an evaluation collapse is not
available to it by construction. Reading this null as "reset-free practice is fine" would
invert its meaning.

![Two panels of per-seed slope lines. Left, labelled as the question the experiment was run to answer, joins each seed's reset-free random-skills score to its reset-free EES score; seven lines rise, one falls, two are flat, annotated p = 0.0469. Right, labelled as the control, joins random-skills under env resets to random-skills under never reset; four rise, two fall, four are flat, annotated as a null result with p = 0.344.](2026-08-14-tossing3d-reference-arms-paired.png)

**Figure 2. Per seed, not two bars.** With ten seeds a bar chart of two means hides one seed
driving the whole movement. Left panel: the rise is real but modest, and seed 3 moves the
other way. Right panel: the control scatters both directions around zero, which is what a
null result looks like.

### Stranding is a property of the domain, confirmed on a second method

| measure | `ees` / `sched` | `ees` / `never` | `rand` / `sched` | `rand` / `never` |
| --- | --- | --- | --- | --- |
| total practice transitions (10 seeds × 100 cycles) | `3069` | **`35`** | `5452` | **`37`** |
| seeds ever stranded | `0/10` | **`10/10`** | `0/10` | **`10/10`** |
| stranding onset | — | cycle `1` | — | cycle `1` |
| practice cycles taking zero steps | `0/1000` | **`990/1000`** | `0/1000` | **`990/1000`** |

**This is the strongest result on the page.** `random-skills` has no planner, no competence
model and no sampler — there is nothing in it to be corrupted by degenerate data — and it
strands on exactly the same schedule as EES, to within two transitions across ten seeds.
Whatever closes the world after a couple of actions is in the domain's operators, not in the
learner.

![Two panels sharing a log-scaled y axis, cumulative practice transitions against practice cycle. In each panel ten blue env-reset traces rise steadily to a few hundred transitions by cycle 100, while ten orange never-reset traces jump to between 1 and 9 in the first cycle and are then perfectly flat for the remaining ninety-nine. The left panel is EES with 3069 versus 35 transitions, the right is random-skills with 5452 versus 37, and both legends report 0 of 10 versus 10 of 10 seeds stranded.](2026-08-14-tossing3d-reference-arms-practice.png)

**Figure 3. Two methods, one collapse.** A robot that keeps practising is a line that keeps
rising; a stranded one goes flat and stays flat. The y axis is symlog so that the flat
orange traces at 1-9 transitions stay legible beside blue traces three orders of magnitude
above them.

`random-skills` under `scheduled` collects **more** transitions than EES (`5452` against
`3069`) while scoring far less (`5.0/100` against `73.4/100`) — it is the paper's own point
about this baseline, visible directly: it gathers experience without improving.

### Per seed

| seed | `rand`/`sched` | `rand`/`never` | `oracle` | `ees`/`sched` | `ees`/`never` | `ees − rand` (never) |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.7 | 0.9 | 10.0 | 6.1 | 0.9 | `0.0` |
| 1 | 0.6 | 0.4 | 10.0 | 7.3 | 1.7 | `+1.3` |
| 2 | 0.3 | 0.9 | 10.0 | 7.4 | 0.9 | `0.0` |
| 3 | 0.4 | 1.1 | 10.0 | 8.5 | 0.4 | `-0.7` |
| 4 | 0.1 | 0.3 | 10.0 | 7.7 | 1.5 | `+1.2` |
| 5 | 0.6 | 0.6 | 10.0 | 7.1 | 0.9 | `+0.3` |
| 6 | 0.7 | 0.4 | 10.0 | 7.8 | 0.7 | `+0.3` |
| 7 | 0.6 | 0.6 | 10.0 | 5.8 | 1.8 | `+1.2` |
| 8 | 0.5 | 0.5 | 10.0 | 7.7 | 1.1 | `+0.6` |
| 9 | 0.5 | 0.5 | 10.0 | 8.0 | 2.3 | `+1.8` |
| **sum** | **5.0** | **6.2** | **100.0** | **73.4** | **12.2** | **`+6.0`** |

The `ees`/`sched`, `ees`/`never` and `oracle` columns are quoted from #247 and #133-era
work respectively where they were already published; the `oracle` column is re-measured here
and happens to agree exactly.

### What this does *not* establish

- **It does not show that reset-free EES is meaningfully better than random skills.** The
  `+0.60` is marginal on every axis a reader should check. It rules out "worse"; it does not
  establish "better" with any confidence.
- **It does not rescue the reset-free arm.** Both reset-free arms are near the floor —
  `12.2/100` and `6.2/100` against a `100/100` ceiling and a `73.4/100` scheduled arm. The
  interesting gap is still `73.4` versus `12.2`.
- **It does not test the degenerate-sampler mechanism directly.** H1's competing hypothesis —
  that a sampler fit on ~3.5 transitions is worse than uniform — is *not supported* by this
  data, since EES ends up above uniform rather than below. But "not supported" here comes
  from an outcome measurement, not from inspecting what the sampler actually learned.
- **It does not decompose no-reset from no-scene-variety**, which are inseparable on this
  domain by construction.
- **The control's null result carries no information about the domain**, for the
  construction reason given above. It confirms the instrument, not the environment.

## Recommendation

**Stop citing `24/100` as the `random-skills` reference on Tossing3D, and stop reading
#247's `12.2 < 24` observation as evidence of anything.** The matched number is `6.2/100`
reset-free and `5.0/100` scheduled. #247 was right to flag its own comparison as untested;
this page is the retraction of the reading, not of that page.

Three follow-ups, in the order the evidence argues for them:

1. **Re-point `tossing3d_reset_free_arms.py`'s reference constants once #247 merges.**
   `RANDOM_SKILLS_PER_SEED = 2.4` and its `(#133)` label are now known to be
   non-comparable, and they are drawn on every reset-policy figure. This PR deliberately
   does **not** edit them: #247 is still open against the same file, and silently changing a
   published reference level would both conflict with that branch and violate the
   never-restate-a-published-number rule. It wants a marked note beside the old value, which
   is a one-line change best made after the merge.
2. **Add the staleness note to #247's own committed log.** Its "`never` now sits below the
   `24/100` `random-skills` reference on `10/10` seeds" sentence is the one a reader six
   months out will land on. It cannot be edited from this branch — that page is not on
   `main` yet — so it needs doing on #247's branch before merge, or immediately after.
3. **Point the `HumanOracle` ladder at this domain**, unchanged as the next experiment. This
   page strengthens the case rather than weakening it: the gap a human would close is
   `73.4` versus `12.2`, and the stranding it would rescue is now confirmed to be a property
   of the domain rather than an artefact of one learner.

## Artifacts

- Runs, all three arms, 10 seeds each:
  [`2026-08-14-tossing3d-reference-arms-runs/`](2026-08-14-tossing3d-reference-arms-runs)
  (`stats.json`, `config_snapshot.json` and `timing.json` per run, in
  `<policy>/<method>/<seed>/`. `log.txt`, `progress.jsonl` and the evaluation `.mp4`s are
  deliberately not committed.)
- Analysis: `analysis/practice_makes_perfect/tossing3d_reference_arms.py`, covered by
  `tests/analysis/practice_makes_perfect/test_tossing3d_reference_arms.py`.
- Figures: [curves](2026-08-14-tossing3d-reference-arms-curves.png),
  [paired](2026-08-14-tossing3d-reference-arms-paired.png),
  [practice](2026-08-14-tossing3d-reference-arms-practice.png).
- The EES arms compared against are #247's, read from
  `/home/josh/hitl-sweeps/2026-08-13-tossing3d-reset-policy-new-pin/` and committed on that
  PR's own branch.
