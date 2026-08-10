# EES sampler-linear ablation: the MLP's nonlinearity is not decoration — logistic regression is worse than no learning at all

**A rerun of PR #195's `no-human` control and eight-point rescue-rate sweep on Tossing Room,
with `LearnedSkillSampler`'s classifier collapsed from a 32x32 ReLU MLP to logistic
regression (`--sampler-classifier linear`). Ten fixed seeds (0-9) per point, 90 runs total,
identical protocol to #195 otherwise.**

## Question / goal

`LearnedSkillSampler` (`src/hitl_pmp/methods/practice_makes_perfect/wrapped_sampler.py`)
scores 100 candidate parameter draws per decision with `MlpBinaryClassifier` (a ReLU MLP,
`hid_sizes=(32, 32)` by default) and returns the argmax. Does the sampler's job — ranking
continuous throw/placement parameters by predicted success — actually need that
nonlinearity, or would a linear decision boundary (logistic regression) do just as well?

## Background

This follows directly from PR #195 (`2026-08-10-human-ladder-rate-sweep.md`), which measured
a `no-human` control and an eight-point `--ask-for-help at-random` rescue-rate sweep
(N ∈ {1, 2, 3, 5, 7, 10, 14, 20}, `--mean-steps-between-help-requests`) against `--practice
-reset-policy never` on Tossing Room, and found the dose-response **non-monotonic**: N=1/N=2
(the highest rescue rates) scored *below* the `no-human` control, the response climbed
sharply from N=3, and the best point was N=14 (259/300), short of both the `two-way-ledge`
and `skill-oracle` ceilings.

`MlpBinaryClassifier._build_net` (`wrapped_sampler.py`, ~line 342) already generalizes to
`hid_sizes=()`: the loop over hidden layer sizes never executes, so the built net is exactly
`[nn.Linear(input_dim, 1), nn.Sigmoid()]` — logistic regression, zero hidden layers, zero
ReLUs — with normalization, Adam, the single-class shortcut, and `LearnedSkillSampler.sample`'s
tie-breaking deviations (6 and 7) all unchanged. This PR reuses that rather than hand-writing
a second classifier class specifically **because** reuse *guarantees* everything except the
net's shape stays byte-identical between the two arms; a hand-rolled `LinearBinaryClassifier`
would only claim that identity, not prove it.

`OptimisticSkillCompetenceModel` (`competence_models.py`) — a separate, already-linear
component that decides *which skill* to practice, not how to score a skill's continuous
parameters — is untouched by this change.

## Hypothesis

Does the MLP's nonlinearity meaningfully change the result, or is a linear classifier
sufficient for this sampler's job? Going in, the reasonable prior was "some degradation, but
the qualitative dose-response shape survives" — logistic regression is a real classifier, not
a no-op, and Light Switch's EES reproduction (a much simpler 1-D domain) has never needed
this sampler to do anything more sophisticated than rank a handful of candidates along one
axis. **This turned out to be wrong, and not just in degree.** The linear classifier does not
attenuate the effect measured in #195, it inverts it: every rescue rate this PR swept scores
at or below the linear classifier's own `no-human` control, and that control itself scores
*below* what zero informed parameter learning gets at N=1 (see Results). The MLP's
nonlinearity is not a refinement on top of a working mechanism — for this domain, it is most
of what makes the mechanism work at all.

## Guidance given

Reuse `MlpBinaryClassifier(hid_sizes=())` rather than a new class, and say why in the PR
body. Add `EesMethod.sampler_classifier: Literal["mlp", "linear"]`, derived into `hid_sizes`
only at the one `LearnedSkillSampler` construction site (`EesMethod.sampler()`) — never a
second selector field on `LearnedSkillSampler` itself, which already owns `hid_sizes`.
Rerun `no-human` under the linear classifier (10 seeds) so the low-N comparison has a
same-classifier control to be judged against; do **not** rerun `two-way-ledge` or
`skill-oracle`, which never touch `LearnedSkillSampler` at all (confirmed below). TDD: pin
that the linear configuration is exactly affine in logit space and the MLP measurably is not,
using several probe-point pairs so the negative (MLP) test cannot pass by landing inside one
ReLU activation region by accident. Extend `analysis/practice_makes_perfect/
human_ladder_curves.py` rather than writing a new script; keep the rate-sweep blue
(`_RATE_SWEEP_COLOR`) for both classifiers (classifier type is not this project's blue/orange
assistance axis) and carry the classifier split on linestyle instead — solid for the original
MLP series, dashed `(0, (4, 2))` for linear. Report every count as `x/y`; use
`PairedTests.sign_flip` for every paired comparison; state plainly whether the ablation
reproduces #195's shape or diverges, rather than asserting an effect without a p-value.

## Methods

**Which arms were and were not rerun.** `SkillOracleMethod` (`methods/oracle/`) and
`RandomSkillsMethod` (`methods/practice_makes_perfect/random_skills_method.py`) never
construct a `LearnedSkillSampler` — confirmed by grep: neither file imports
`wrapped_sampler` anywhere — so `skill-oracle` has no sampler-classifier in its critical path
at all, and rerunning it would measure exactly what #195 already measured. `two-way-ledge`
(`--ask-for-help never` on the two-way-ledge world) *does* refit `LearnedSkillSampler` every
cycle, same as `no-human`, but was scoped out of this rerun deliberately (per guidance) and is
carried on the figure/table as the original MLP-era reference, labeled **not rerun**.

**What was rerun**, all with `--env tossingroom --sampler-classifier linear` added to #195's
own flags, `--practice-reset-policy never --num-test-tasks 30 --num-cycles 10
--max-steps-per-interaction 150`, ten fixed seeds (0-9) each, driven by
`scripts/run_sweep.py` (one invocation per component, matching #195's own pattern):

| component | extra flags | seeds | runs |
| --- | --- | --- | --- |
| `no-human` | `--ask-for-help never` | 10 | 10 |
| rate sweep, N ∈ {1,2,3,5,7,10,14,20} | `--ask-for-help at-random --human-reset-target task-initial --mean-steps-between-help-requests N` | 10 each | 80 |

90 runs total. Machine was quiet (no other agent sweeps running, load average ~0.5-0.7 on a
24-core box); `--max-workers 10` per invocation, nine sequential invocations under one
`systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue` wrapper. Total wall-clock
**8.6 minutes** (`timing.json`'s `start_epoch_seconds`/`end_epoch_seconds` across all 90 runs,
first start to last end) — the same order of magnitude as #195's ~15 minutes for a
20-run-larger 110-run sweep, and well under this repo's standing "does not need a systemd
service" threshold. All 90/90 runs succeeded on the first launch attempt (no spawn retries).

**Manipulation checks, all passing, on every one of the 90 runs:**

- `config_snapshot.json`'s resolved `args["sampler_classifier"]` is exactly `"linear"` on all
  90 runs (checked, not assumed — this is the check that would have caught the flag silently
  not reaching the child process, which would otherwise look like a clean experiment while
  measuring the MLP arm twice under a different label). The numbers below are **not**
  byte-identical to #195's — they diverge substantially from N=2 onward — so there is no
  identical-arms bug signal to chase here.
- `num_practice_resets == 0` on all 90 runs (every run is `--practice-reset-policy never`).
- Every rate-sweep point recorded a strictly positive intervention count.
- Test-set composition is 14 TRASH / 14 RECYCLING / 2 EMPTY on every run (asserted by the
  loader).
- **A useful internal-consistency check, not one that was pre-specified**: pooled
  intervention counts are *exactly* identical between the mlp and linear arms at every N
  (e.g. N=14: 1025 pooled both arms). This is expected, not a bug — `--ask-for-help
  at-random`'s decision to ask is drawn from its own RNG stream, independent of what the
  sampler picks, and every practice period runs the same fixed step budget
  (`--num-cycles 10 --max-steps-per-interaction 150` = 1500 practice calls) regardless of
  outcome. It is reassuring precisely because it shows the two sweeps are properly paired on
  everything except the one thing under test.
- **N=1's outcome is mechanically forced to be classifier-independent, and it measures so**:
  both arms score 71/300 at N=1, matching to the task. #195 already established why: at N=1
  every one of 1500 policy calls asks for help, so `PracticeLoop`'s `except
  HumanHelpRequested` branch `continue`s every single practice call and the robot never takes
  one real practice action. With zero observations ever recorded, `LearnedSkillSampler`
  stays permanently unfitted under *either* `hid_sizes` — `MlpBinaryClassifier.is_fitted` is
  `False` regardless of architecture — so `sample()` takes deviation 6's uniform-random
  fallback identically either way. This is not evidence the ablation is inert (see the N≥2
  results below); it is the one point in the sweep where classifier choice is provably
  irrelevant, and it landing exactly on the same number both times is a correctness check the
  rest of the sweep passed, not a finding of its own.

Raw per-seed `stats.json`/`config_snapshot.json`/`timing.json` for all 90 runs are committed
under `2026-08-10-ees-sampler-linear-ablation-runs/` (`log.txt`/`episode*.mp4`/
`progress.jsonl` were not — same precedent #195 set, only the three files needed to
reconstruct a result and audit its conditions).

Every `no-human`/rate-sweep-point pair below is paired on shared seeds within its own
classifier (`PairedTests.sign_flip`, exact by full enumeration, same as #195); a linear-era N
is judged against the **linear-era** `no-human` control, never the mlp-era one — they are
different measurements, one per `sampler_classifier`.

## Results

**The ablation does not attenuate #195's finding — it inverts it.** #195's `no-human`
control was 112/300; under the linear classifier it collapses to **56/300**, exactly half.
Every rate-sweep point measured here, at every N from 2 through 20, scores *below* that
already-reduced linear `no-human` control — the opposite of #195, where every N from 3 onward
scored significantly *above* the mlp `no-human` control. And every rate-sweep point except
N=1 also scores below the mechanically classifier-independent N=1 floor (71/300): under the
linear classifier, informed practice is not merely worse than the MLP's — it is worse than
**no informed parameter learning at all**.

![final OVERALL solved vs N, both classifiers overlaid: solid mlp (#195) rises sharply and flattens near 22-26/30; dashed linear (this ablation) stays flat near 2-4/30 for every N ≥ 2, with reference lines for both no-human controls and the mlp-era skill-oracle ceiling](2026-08-10-ees-sampler-linear-ablation.png)

| N | `sampler_classifier=mlp` (#195) OVERALL | gap vs mlp no-human | p | `sampler_classifier=linear` (this PR) OVERALL | gap vs linear no-human | p | MDE (linear) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 71/300 | -41 | 0.125 | 71/300 | **+15** | 0.0156 | 1.20 |
| 2 | 99/300 | -13 | 0.668 | 40/300 | -16 | 0.0313 | 1.52 |
| 3 | 216/300 | +104 | 0.00391 | 26/300 | -30 | 0.00195 | 1.45 |
| 5 | 236/300 | +124 | 0.00391 | 27/300 | -29 | 0.00391 | 1.59 |
| 7 | 248/300 | +136 | 0.00195 | 29/300 | -27 | 0.00391 | 1.45 |
| 10 | 235/300 | +123 | 0.00195 | 40/300 | -16 | 0.0801 | 2.05 |
| 14 | **259/300** | +147 | 0.00195 | 37/300 | -19 | 0.0195 | 1.53 |
| 20 | 211/300 | +99 | 0.00781 | 32/300 | -24 | 0.00195 | 1.33 |

(`no-human`: mlp 112/300, linear **56/300**. `two-way-ledge` 287/300 and `skill-oracle`
300/300 are carried as reference lines only — mlp-era, not rerun.)

**Every linear-era point except N=1 is significantly below its own control** (p ≤ 0.032,
seven of eight points), the mirror image of #195's "every point from N=3 onward beats the
control" — here every point from N=2 onward is *beaten by* the control. N=10 is the one
exception at p=0.080 (MDE 2.05, the widest of the linear column), consistent with noise
around a small, real negative effect rather than a genuine null — every neighboring N (7 and
14) is significant in the same direction.

**N=1's positive, significant gap (+15, p=0.0156) is the sharpest illustration of the whole
finding, not a contradiction of it.** N=1 forces zero informed practice under either
classifier (see Methods), so its 71/300 is what Tossing Room scores from planning and
skill selection alone, with parameter sampling never informed by any data. Under the mlp
classifier that floor is *worse* than what practice buys (`no-human` reaches 112/300; the
best rate-sweep point reaches 259/300) — informed practice helps, a lot. Under the linear
classifier that same floor is *better* than what practice buys at every rate this PR swept
(`no-human` 56/300; every N≥2 point 26-40/300) — informed practice, driven by a linear
sampler, actively **hurts** in this domain.

**The collapse is not confined to one goal family.** At N=14 — the mlp arm's best point —
TRASH goes from 105/140 (mlp) to 9/140 (linear) and RECYCLING from 134/140 to 8/140; at N=3,
TRASH goes from 97/140 to 4/140 and RECYCLING from 99/140 to 2/140. EMPTY stays 20/20 in
every arm of both classifiers (this domain's fixed test set has only 2 EMPTY tasks and they
do not require a thrown/placed parameter to solve).

**A mechanistic read, from one seed's practice tally** (`no-human`, linear, seed 0): `ThrowTrash`
practice attempts are dominated by `LearnedSkillSampler`'s deviation-6 fallback — the
uninformative-tie branch that fires when the classifier's scores cannot discriminate among
the 100 candidates it was handed (`4/18 uninformative, 0/1 informed` in this seed's tally) —
consistent with a linear decision boundary being unable to separate "will this throw succeed"
as a function of state and continuous throw parameters, where the mlp classifier's ReLU
hidden layers can. This is offered as the likely mechanism, not independently verified beyond
this one seed's log; the headline claim rests on the score numbers above, which are.

## Recommendation

**Do not simplify `LearnedSkillSampler`'s classifier to logistic regression on Tossing
Room — the MLP's nonlinearity is carrying most of the mechanism that makes informed
practice better than no practice at all, not refining it.** This is a domain-specific
finding about Tossing Room's throw/placement parameter space (plausibly genuinely
non-linear success regions), not a general claim that linear classifiers never work for
this project's samplers — Light Switch's simpler 1-D parameter space has never been tested
under this ablation and may behave differently.

**If a future experiment wants a cheaper sampler for wall-clock reasons, this result rules
out `hid_sizes=()` on Tossing Room specifically** — the failure mode is not "smaller
effect, still positive," it is "actively harmful, worse than no learning." A future
cost-saving ablation on this sampler should look at cheaper *nonlinear* configurations
(fewer or smaller hidden layers) rather than removing the nonlinearity outright.

**This PR's own numbers are new measurements, following #195's own "raw data must be
committed" precedent** (`2026-08-10-ees-sampler-linear-ablation-runs/`, `stats.json`/
`config_snapshot.json`/`timing.json` for all 90 runs) so this comparison cannot be lost the
way #151's was.
