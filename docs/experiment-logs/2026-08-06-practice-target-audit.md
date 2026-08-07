# Which skills does EES actually practise on Light Switch and Ball-Ring?

**Audit, 2026-08-06.** Tossing3D turned out to have a parameterised skill that EES never
practised and that nothing in `stats.json` could show. This asks whether the same defect
class is present on the two domains the PMP reproduction actually runs on, and whether
any published conclusion rests on it.

**Answer: no published Light Switch or Ball-Ring conclusion is affected.** The instrument
does surface something new on Ball-Ring — three of its five parameterised skills are never
practised — but no committed claim depends on those three being learned, and the one skill
every Ball-Ring conclusion *does* rest on is practised in 10/10 seeds.

> **These numbers depend on an unmerged PR.** They were produced on a branch stacked on
> **#119** (`SamplerConsultation`), which is open and not merged. If #119 changes, re-run
> before citing.

## What was missing, and what was added

PR #111 gave `stats.json` a per-skill record of practice **executions**; #119 split its
fallback pool so "no sampler exists" reads apart from "a sampler was consulted and could
not discriminate". Neither can show a skill EES declines to practise, because **an
execution is not a decision**. EES plans to a chosen candidate's preconditions and executes
the whole prefix on the way, so a skill can accumulate hundreds of executions without ever
having been the thing EES wanted to practise. That is exactly how Tossing3D's
`MoveToThrowPose` recorded 175/175 executions while being dropped from every candidate
list.

So this adds `PracticeTargetTally`, a second per-window record keyed by lifted skill,
counting **decisions**: `scored` (entered the ranked candidate list), `declined_perfect`
(scored `-inf`, which under `skip_perfect` means and only means a measured success rate of
exactly 1.0), `selected` (the candidate the explorer committed to), `unreachable`
(outranked the winner but no plan reached its preconditions). A skill absent from the
record entirely was never a candidate at all — the third state, and the one a bare zero
cannot distinguish from the other two.

**EES's behaviour is untouched.** Every call site reads a value the surrounding code had
already computed; no branch, no RNG draw and no planner call depends on it. Pinned by
`test_recording_practice_targets_does_not_change_what_ees_does`, which asserts a recording
and a non-recording method produce the identical ranked list *and* leave their RNGs in the
identical state.

## Protocol

Both sweeps at the published protocols, seeds 0-9 fixed:

```bash
python -m scripts.run_sweep --env lightswitch --methods ees --num-seeds 10 \
  --results-root <dir> --shared-args "--grid-size 25 --num-test-tasks 10" \
  --method-args "ees=--num-cycles 10 --max-steps-per-interaction 150"

python -m scripts.run_sweep --env ballring --methods ees --num-seeds 10 \
  --results-root <dir> --shared-args "--num-test-tasks 10" \
  --method-args "ees=--num-cycles 25 --max-steps-per-interaction 100 \
    --competence-window-size 2 --competence-recency-size 2 \
    --exploration-epsilon 0.5 --sampler-max-train-iters 10000"

python -m analysis.practice_makes_perfect.practice_diagnostics --results-root <dir> \
  --target-output <png> --target-skills <names>
```

Raw `stats.json` and `config_snapshot.json` for all 20 runs are committed under
`2026-08-06-practice-target-audit/`.

## Light Switch: nothing to see, and that is the right answer

Pooled over 10 seeds. `declined perfect`, `scored` and `unreachable` count *ground-skill
scoring decisions*, so they are large — every grounding is rescored on every call.
`selected` is the interpretable one.

| skill | `param_dim` | selected | scored | declined perfect | seeds selecting it |
|---|---|---|---|---|---|
| TurnOnLight | 1 | 7131 | 8846 | 242 | 10/10 |
| JumpToLight | 1 | 1928 | 9088 | 0 | 10/10 |
| TurnOffLight | 1 | 29 | 7967 | 869 | 2/10 |
| MoveRobot | **0** | 0 | 0 | 227164 | 0/10 |

**`MoveRobot` is never practised, and that is correct, not a defect.** It declares
`param_dim = 0` — there is no sampler, none is possible, and nothing to learn — and it
succeeds 2873/2873. A skill with no parameters *should* be dropped once it is reliable.
This is the case the Tossing3D failure superficially resembles and is not: there the
declined skill was `param_dim = 1` and carried the domain's only meaningful learnable
parameter.

**`TurnOnLight` — the skill every Light Switch claim rests on — is practised heavily and
its sampler is genuinely informed**: 7131 selections, and 2222/2729 informed draws out of
6708 executions. The reproduction log's claim that "the `TurnOnLight` sampler having been
specialized away from its uniform prior" is supported by the selection record, not
undermined by it.

`TurnOffLight` is the only mildly interesting row: selected in 2/10 seeds and declined as
already-perfect in 9/10. It is not silently starved — it still draws 553/619 informed,
because with `reproduce_predicators_explore_target_only` off (the default) *every* skill
executed during practice explores and records training rows, target or not. It is simply
outranked; `unreachable` is 0/10 seeds, so it never lost on reachability.

![Light Switch practice-target decisions](https://raw.githubusercontent.com/joshnroy/human-in-the-loop-practice-makes-perfect/2e26c7345292ce9a5c10083346475a180655bc37/docs/experiment-logs/2026-08-06-lightswitch-practice-targets.png)

## Ball-Ring: three of five parameterised skills are never practised

Ten of Ball-Ring's fifteen skills are `param_dim = 0`; all ten are declined once reliable,
which is correct for the same reason `MoveRobot` is. The five that carry parameters:

| skill | selected | scored | declined perfect | seeds selecting it | executions (succ/att) | informed |
|---|---|---|---|---|---|---|
| PlaceBallOnTable | 3275 | 11894 | 0 | 10/10 | 0/3280 | 0/3280 |
| PlaceCupWithoutBallOnTable | 2803 | 6016 | 62 | 10/10 | 2002/3190 | 902/1067 |
| PlaceBallOnFloor | **0** | 0 | 6008 | **0/10** | 231/231 | 0/231 |
| PlaceCupWithBallOnFloor | **0** | 0 | 5341 | **0/10** | 150/150 | 0/150 |
| PlaceCupWithoutBallOnFloor | — | — | — | **0/10** | never executed | — |

**`PlaceBallOnFloor` and `PlaceCupWithBallOnFloor` are the Tossing3D signature exactly.**
Each succeeds on every attempt in every seed (per-seed 31/31, 26/26, 21/21, 31/31, 12/12,
18/18, 20/20, 25/25, 24/24, 23/23 for the first), so its measured success rate is exactly
1.0, so `score_ground_skill` returns `-inf`, so `choose_practice_target` drops every
grounding — in 10/10 seeds, from the first window onward. Their `param_dim = 2` samplers
are consulted on all 231/231 and 150/150 executions and never once discriminate.

The rate that fires `skip_perfect` is read off the competence history, which *excludes*
epsilon-random attempts, while the execution tally includes them — so the two are not
generally the same number. Here they are: both skills record 0 epsilon-random draws, and
every attempt succeeded, so the competence subset is all-successes too. The conclusion does
not rest on conflating them.

**`PlaceCupWithoutBallOnFloor` is the third state**: never executed in any seed, so never a
candidate. Absent, not declined — a different fact with a different fix.

**The decisive skill is fine.** Every Ball-Ring conclusion rests on the cup-placement
sampler ("the entire learning problem is specializing the cup-placement sampler"). That is
`PlaceCupWithoutBallOnTable`: selected in 10/10 seeds, 902/1067 informed draws. It is
practised and its sampler is learned.

![Ball-Ring practice-target decisions](https://raw.githubusercontent.com/joshnroy/human-in-the-loop-practice-makes-perfect/2e26c7345292ce9a5c10083346475a180655bc37/docs/experiment-logs/2026-08-06-ballring-practice-targets.png)

### Is the floor-place result a defect?

**Probably not, and this log does not claim it is.** `PlaceBallOnFloor`'s add-effects are
`BallOnFloor` and `BallNotInCup`. Drop a ball on the floor and it is on the floor and not
in a cup — for essentially any placement the sampler could draw. The success predicate is
insensitive to the parameter *by construction*, and the parameter was added (2026-08-02
fidelity work) to model where the ball lands, not to be learned. A skill that genuinely
cannot fail has nothing for practice to improve, and declining it is `skip_perfect` working.

What separates this from Tossing3D is which skill it happened to. There, the
non-discriminating predicate sat on the domain's *only* meaningful learnable parameter, so
"EES does not learn here" was measuring nothing. Here it sits on three peripheral
placements while the decisive one is practised throughout.

## What this does not answer

- **`PlaceBallOnTable` never succeeds: 0/3280 executions, 0 in every one of 10 seeds**,
  while being selected as a practice target in 10/10. That is a live anomaly — it is the
  mirror image of the pre-fix `PlaceBallOnFloor` defect (2026-08-02) — but it is a
  different defect class from the one audited here (EES is practising it, loudly), and it
  was **not investigated**. It deserves its own look.
- **This Ball-Ring arm scored 91/100 where the published `iters10k` arm at identical
  settings scored 99/100 — unexplained.** Both arms share the fixed seeds 0-9, so they
  are paired:

  | seed | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
  |---|---|---|---|---|---|---|---|---|---|---|
  | published | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 | 9/10 | 10/10 | 10/10 |
  | this audit | 10/10 | 10/10 | 9/10 | 10/10 | 8/10 | 9/10 | 10/10 | 8/10 | 7/10 | 10/10 |

  5/10 seeds are tied and **5/5** of the non-tied ones move the same way, mean −0.80
  solved tasks per seed. An exact paired permutation test over all 1024 sign flips gives
  **p = 0.0625** — but that is the *floor* for 5 non-tied pairs (2 × 2⁻⁵), so **no
  two-sided test on this data can reach 0.05**. The right summary is "a consistent
  one-directional shift that this design cannot resolve", not "no difference" and not
  "a significant regression".

  **Both numbers came through `scripts/run_sweep.py`** — mine directly, the published one
  per the command in `2026-08-03-ballring-iters.md`'s "Reproducing the runs". That rules
  out the obvious candidate: #112 pinned the sampler's torch reductions, but `run_sweep`
  already pins `OMP_NUM_THREADS=1`, so #112 is a **no-op for any swept run** and cannot
  explain this. An earlier draft of this log offered #112 as the explanation; it is wrong
  and has been withdrawn. The provenance is documentary rather than verified from
  artifacts — that arm's raw directories did not survive the move between machines, so
  there is no `config_snapshot.json` to confirm it against, which is itself the reason
  this audit commits all 20 of its own.

  **This does not bear on the conclusions above**, which are structural and hold under
  either realisation: `PlaceBallOnFloor` succeeds on every attempt in every seed, and
  `PlaceCupWithoutBallOnTable` is selected in every seed. Light Switch reproduced
  exactly at 100/100.
- **No `skip_perfect` counterfactual was run.** Whether practising the two declined
  floor-place skills would change anything is untested; this audit deliberately did not
  alter EES.
