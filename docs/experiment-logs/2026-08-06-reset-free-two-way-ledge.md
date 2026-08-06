# Reset-free practice with the ledge made two-way: the positive control

**Status: pre-registered, not yet run.** This file's first commit contains no results.
The numbers arrive in a later commit on the same branch.

## Question / goal

Reset-free practice is worse than scheduled-reset practice on Tossing Room (split
throws). Is that because practice is reset-free, or because *this domain has an
irreversible action*?

The two are not the same claim, and only the second is a statement about the method. If
removing the irreversibility restores reset-free performance, the finding is "EES cannot
recover from an irreversible mistake, and reset-free practice is what exposes that" — a
claim about irreversibility. If reset-free practice still fails in a world with no
irreversible action, then irreversibility was not the mechanism and the negative result
means something else.

## Background

`environments/tossingroomsplit` is a 7-room hallway. The robot and the item pile start
in room 3; the recycling bin and its button are in room 1, the trash bin and its button
in room 6. A one-way ledge blocks the **rightward** step out of room 2. That single
blocked edge is the only edge from rooms {0,1,2} into {3…6}, and the pile is the only
source of items, so rooms 0–2 are absorbing at any horizon: a robot that walks left once
can never pick up, throw, or press the trash button again.

**PR #115** measured `--practice-reset-policy scheduled` against `never`, 10 paired
seeds, EES, 10 cycles × 150 steps, 30 test tasks. Its published result: **151/300 vs
85/300** at the final sweep, −22.0pp against a 10.9pp MDE, exact two-sided sign-flip
p = 2/256 = 0.0078, `never` worse in 8/10 seeds and better in 0/10. Those numbers are
cited here, not re-derived.

#115's own write-up attributes that to **two entangled mechanisms**, and this experiment
only removes one of them:

1. **Stranding.** The ledge severs rooms 0–2 from the pile, so a practice period that
   ends past it stays past it. A follow-up investigation established this from primary
   run data: 9/10 `never` seeds strand permanently, and across 74 post-onset periods and
   roughly 11,000 skill executions the executed skills are exclusively `MoveRoom` (8,827)
   and `PressRecycling` (2,199) — zero pickups, zero throws, zero `PressTrash`, in 0/9
   recoveries. There is no recovery machinery: all three `Method.reset_environment`
   implementations return `False` without writing, and `TossingRoomSplitProblem` never
   sets `human`, so `execute_human_command` cannot fire.
2. **The training distribution.** `reset_to_task` is the only thing that installs a
   Task's `initial_state`, so under `never` the bins' `throw_distance` and the items'
   `weight` — the throw samplers' own input row — stay frozen at their canonical values.
   #115 measured 194 greedy throws at **1** distinct required-force target under `never`,
   against 86 targets over 440 throws under `scheduled`.

**PR A of this stack** (`--two-way-ledge`) makes that one blocked edge traversable
rightward. It removes mechanism 1 exactly and leaves mechanism 2 completely untouched.
That asymmetry is the reason the prediction below is "partial recovery" rather than
"recovery".

### What `--two-way-ledge` also changes, and why the design has four arms

The two-way world is **easier**, in three ways that are properties of the domain and not
of any method:

* EMPTY stops being an ordering task. Recycling-first costs 9 against trash-first's 10,
  so the longest shortest solve is 9 and the evaluation horizon drops **12 → 11**.
* RECYCLING stops being one-attempt-per-period. The walk back from room 1 to the pile
  exists, so a practice period buys as many recycling attempts as trash ones.
* Rooms 0–2 stop being absorbing — the intended effect.

So a two-way success count is **not comparable to a one-way success count**. What *is*
comparable across the two worlds is the **within-world gap** between the two reset
policies, because both arms of a given world face the same domain difficulty. That is
why this experiment runs a full 2×2 rather than one new arm: without a
`two-way / scheduled` arm there is no control for the world getting easier, and any
improvement in `two-way / never` would be unattributable.

The one-way arms are re-run here at this branch's commit rather than taken from #115's
committed aggregate. `--two-way-ledge` defaults off and a default run was verified
byte-identical to 291da9a (`cmp` on `stats.json`, EES, seed 0, 2 cycles × 30 steps,
6 test tasks, `OMP_NUM_THREADS=1` on both), so the one-way arms are expected to
reproduce #115 exactly. If they do not, that divergence is itself the finding.

## Hypothesis

**Direction.** I expect the reset-free penalty — `scheduled` minus `never`, within a
world — to be substantially **smaller** in the two-way world than in the one-way world.

**Magnitude: partial, not full.** Mechanism 2 above survives the flag untouched, so I do
not expect the penalty to reach zero. My prior is that it shrinks by more than half but
that a real penalty remains.

**Confidence: moderate, and one observation already points the other way.** #115's seed 1
is the single `never` seed that never strands. It practiced *more* successfully than its
scheduled pair (97/174 successful throws against 73/157) and still scored **5/14** TRASH
against that pair's **14/14**. That is mechanism 2 acting alone, with stranding absent —
which is precisely the situation the two-way world creates for every seed. If seed 1 is
representative, the two-way `never` arm will still collapse and this experiment will
return a null result on its primary question.

**A second observation, also disclosed.** A single full-length timing probe
(seed 0, `--two-way-ledge --practice-reset-policy never`) was run before this
pre-registration in order to size the sweep. Its printed practice-side summary recorded
**95 `ThrowRecycling`** and **94 `ThrowTrash`** attempts across the run — i.e. no
stranding, which is the mechanism working as intended. Its evaluation outcomes were not
inspected, and it is a single seed. It is reported here because it was seen, not because
it is evidence.

**Per-family prediction.** TRASH carried nearly all of #115's damage (113/140 → 39/140),
so TRASH is where recovery should show if it shows anywhere. RECYCLING was a null result
in #115 and additionally changes character under the flag (it becomes repeatable), so I
expect it to be uninformative about the mechanism either way. EMPTY has 2 test tasks per
seed — 20 per arm — and was 20/20 in both of #115's arms; I expect it to support no
inference again.

**Null result matters more than success here.** If the two-way `never` arm still fails,
that says irreversibility was not the mechanism, which is a more interesting and more
disruptive finding than confirmation. It will be reported plainly and not softened.

## Guidance given

Josh, verbatim: *"this is sort of intentional — the point is that practice makes perfect
gets stuck when it gets stuck — that's the whole point — showing that it doesn't work in
the reset free setting… and also run a followup (with a flag) that disables the one way,
making it two way. that should work."*

Also given: build the flag as its own PR with the experiment stacked on top; report `x/y`
per goal family with the MDE derived from its own two denominators; report the stranding
rate as a **measured outcome, not an assumption**; commit the raw data per seed, because
an earlier experiment log (#103) committed none and is now unreproducible; and if the
result is null, say so plainly.

## Analysis plan, fixed in advance

**Four arms, seeds 0–9 fixed (never drawn).** All arms: `--env tossingroomsplit --method
ees --num-test-tasks 30 --num-cycles 10 --max-steps-per-interaction 150`.

| arm | `--practice-reset-policy` | `--two-way-ledge` |
| --- | --- | --- |
| `one-way/scheduled` | `scheduled` | no |
| `one-way/never` | `never` | no |
| `two-way/scheduled` | `scheduled` | yes |
| `two-way/never` | `never` | yes |

**Primary quantity: the reset-free penalty**, per seed, per world — that seed's
`scheduled` final-sweep solved count minus its `never` final-sweep solved count, out of
30. The primary question is whether that penalty is smaller in the two-way world.

**Two paired tests, both exact, both on the 10 fixed seeds:**

1. *Within the two-way world.* `scheduled` vs `never`, exact two-sided sign-flip
   permutation test on the mean paired difference — `PairedTests.sign_flip`, the same
   test #115 used — with the exact two-sided Wilcoxon signed-rank test as the
   distribution-free companion.
2. *The interaction, which is the positive control's actual claim.* Per-seed
   `penalty(one-way) − penalty(two-way)`, same exact sign-flip test. A positive,
   significant interaction is the evidence that irreversibility is the mechanism.

**Reported as counts.** `x/y` everywhere — per arm, per family, per seed — never a bare
percentage. A percentage may accompany a count, never replace it.

**MDE, derived per comparison from its own two denominators**, using this repo's
constant `2.801585` (two-sided α = 0.05, 80% power):

```text
MDE = 2.801585 * sqrt( p1(1 - p1)/n1 + p2(1 - p2)/n2 )
```

Where the variance is exactly zero the MDE is reported as **degenerate** and the
comparison is stated to support **no inference** — it is not a null result. EMPTY, at
20 per arm and 20/20 in both of #115's arms, is the likely case.

**Decision rule, fixed now.** As in #115, **anything under 10pp is treated as "no
meaningful difference" regardless of what a test says.** On top of that:

* **The positive control succeeds** if (i) the two-way `scheduled` − `never` difference
  is under 10pp in magnitude, *or* is not significant at exact sign-flip p < 0.05; **and**
  (ii) the measured stranding rate in `two-way/never` is at or near zero.
* **Partial recovery** if the penalty shrinks by more than 10pp but a penalty of 10pp or
  more remains. This is the outcome I expect.
* **Null result on the primary question** if the two-way penalty is within 10pp of the
  one-way penalty — irreversibility was not the mechanism. Written as "null result", in
  full.

**The stranding rate is measured, not assumed.** "The robot no longer gets stuck" is the
mechanism this experiment turns on, so it is reported as an outcome. #111's
`Metrics.practice_outcomes_per_cycle` gives per-cycle, per-lifted-skill attempts and
successes, so a stranded cycle is directly visible as **zero attempts of every
`Pickup*` and `Throw*` skill in that cycle**. Reported as `x/y` stranded cycles over the
10 seeds × 10 cycles = 100 cycles per arm, and as the number of seeds that ever strand.

**Manipulation checks, both required to pass before any outcome is read.**
`num_practice_resets` must be 10 in every `scheduled` run and 0 in every `never` run —
the flag must not have changed what that field counts. Every run must end at exactly
1500 online transitions, and every arm's realised test-set composition must be
14 TRASH / 14 RECYCLING / 2 EMPTY per seed.

**Raw data is committed** — `stats.json`, `config_snapshot.json` and `timing.json` for
all 40 runs — so this experiment stays re-analysable without a re-run.

**One known hazard, stated rather than controlled.** `torch.manual_seed` pins initial
weights but not reduction order, so the ambient intra-op thread count is silently a
second input to every sampler result (draft PR #112, not merged). `scripts/run_sweep.py`
pins `OMP_NUM_THREADS=1` on its children, so all four arms are run through it and no arm
is ever compared against a bare CLI run.

## Methods

*(to be filled in with the run.)*

## Results

*(to be filled in with the run.)*

## Recommendation

*(to be filled in with the run.)*
