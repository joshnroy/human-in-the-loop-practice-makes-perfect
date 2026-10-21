# Human-cost sweep — seed 0

> **STALENESS NOTE (added 2026-10-21, when this log was committed).** Every Tossing3D
> number on this page was measured before kindergarden `0dac89e` (2026-10-19), which
> fixes the bin to the floor once it is placed; before it, a free 0.1 kg bin could be
> tipped by the cube it caught, dragged by picks and pushed by the base. They were also
> measured before the later pick-controller changes. None of them is evidence about the
> current scene. Nothing on this page is edited or recomputed.
>
> **Committed 2026-10-21 from the main checkout's uncommitted files.** Links to
> external sites and private Notion pages were removed (their text is kept), and links
> to paths outside this repository became plain paths. Most `artifacts/` evidence named
> here is not committed; it lives in the main checkout's untracked `artifacts/`
> directory, and the `/tmp` paths no longer exist.

Date: 2026-10-08. Twenty-one fresh runs: seven full-agentic on the workstation, seven EES and seven expectimax on Della.

Both human reset skills have price c in {1, 3, 5, 8, 10, 15, 20}. Each human invocation still counts as one practice step. Robot-step price = 1; human weight = 1; lambda = 3e-6.

All runs: 85,000 counted-step ceiling; frozen evaluation every 1,700 steps; ten held-out tasks with 500 robot steps each; stop after three consecutive 10/10 evaluations. Seed 0, canonical practice world 125, fresh initialization.

Agentic: same executed expanded hint plus future-cost explanation prompt as seed 5; runtime a13e5ef, prompt fdac5a9; Claude Opus 5.5, high effort, $20 cap each ($140 total ceiling). The newer Notion review wording is not used. No previous learned policy or practice data supplied.

Expectimax: depth 6, grid 25x16, local-trend competence, observation penalty 0, 1024 cost-filter particles, competence-aware human forecast. EES retains previous poster-style settings with reset gate disabled.

| Human price | Local agentic service | EES job | Expectimax job |
|---:|---|---:|---:|
| 1 | hitl-human-cost-01-agentic-seed0-20261008 | 15228102 | 15228109 |
| 3 | hitl-human-cost-03-agentic-seed0-20261008 | 15228103 | 15228110 |
| 5 | hitl-human-cost-05-agentic-seed0-20261008 | 15228104 | 15228111 |
| 8 | hitl-human-cost-08-agentic-seed0-20261008 | 15228105 | 15228112 |
| 10 | hitl-human-cost-10-agentic-seed0-20261008 | 15228106 | 15228113 |
| 15 | hitl-human-cost-15-agentic-seed0-20261008 | 15228107 | 15228114 |
| 20 | hitl-human-cost-20-agentic-seed0-20261008 | 15228108 | 15228115 |

Della monitor job: 15228116. Persistent 900-second read-only monitors; Della monitoring continues independently of workstation.

Validation: four launch checks passed after their initial failing run. All seven settings verified through actual accounting and prompt rendering: one robot step plus two human calls has experience 3 and cost 1 + 2c. Live agentic protocols and prompt prefixes match the prepared settings. Della startup records verify the configuration paths and observed human charges.

Records:

```text
artifacts/human-cost-sweep-seed0-20261008/planned-runs.json
artifacts/human-cost-sweep-seed0-20261008/local-launch-receipt.json
artifacts/human-cost-sweep-seed0-20261008/della-launch-receipt.json
artifacts/human-cost-sweep-seed0-20261008/validation.json
```

Potential prompt changes remain separate ablations: complete continuation accounting; comparison over repeated tosses; measured recovery/transfer diagnostics; explicit recovery-versus-reset choices; concise decision records. A prescribed near-side curriculum would be a capability control rather than spontaneous strategy selection.

## High-cost extension

Launched 2026-10-08 around 13:33 EDT: nine additional fresh seed-0 runs at human prices 100, 200, and 1000. Same frozen source, executed prompts, initial policy, practice budget, evaluation schedule, early stopping, and learning settings as above. Only both human-skill prices change. Each human duration remains 1; robot price and human weight remain 1; lambda remains 3e-6. Agentic uses Opus 5.5 high effort, $20 each ($60 additional maximum). The accounting clarification proposed after inspecting earlier explanations is not added to this cost-only extension.

| Human price | Local agentic service | EES job | Expectimax job |
|---:|---|---:|---:|
| 100 | hitl-human-cost-100-agentic-seed0-20261008 | 15231406 | 15231409 |
| 200 | hitl-human-cost-200-agentic-seed0-20261008 | 15231407 | 15231410 |
| 1000 | hitl-human-cost-1000-agentic-seed0-20261008 | 15231408 | 15231411 |

Della monitor job 15231412; individual persistent local monitors; 900-second intervals. Launch records and validation are in `artifacts/human-cost-high-sweep-seed0-20261008/`. Four launch tests passed following their initial failing run; actual accounting and rendered prompts validated at all three prices. Submission is not an outcome; consult run status and evaluations for results.

## Continuation-cost prompt variant

Launched 2026-10-08: four fresh seed-0 agentic runs at human costs 1, 20, 100, and 1000. Matched existing-prompt runs at the same prices are the comparison group. This is a diagnostic subset of the sweep, with no extra EES or expectimax launches.

The only source change is this approved paragraph, inserted before the deployment-interface description:

> When choosing a practice action, compare its immediate cost plus the expected accumulated robot and human cost of the subsequent practice needed to achieve comparable deployment performance. An action with a higher immediate cost may be preferable if it enables cheaper repeated practice and reduces total future cost. Compare alternatives through equivalent end states, including preparation, pickup, recovery, and any later validation. Use rollout measurements to support cost estimates; state uncertainty where alternatives remain untested.

The expanded hint, earlier future-cost explanation request, controller bootstrap, runtime, and other parameters are unchanged. Opus 5.5 high effort; $20 cap per run ($80 maximum for the variant sweep); 85,000 counted steps; measurements every 1,700 steps; ten held-out tasks with 500 robot steps; stop after three consecutive 10/10 evaluations. Human duration 1, robot price 1, human weight 1, lambda 3e-6.

| Human price | Local service |
|---:|---|
| 1 | hitl-continuation-cost-01-agentic-seed0-20261008 |
| 20 | hitl-continuation-cost-20-agentic-seed0-20261008 |
| 100 | hitl-continuation-cost-100-agentic-seed0-20261008 |
| 1000 | hitl-continuation-cost-1000-agentic-seed0-20261008 |

Independent persistent 900-second read-only monitors; model never receives held-out results. Three tests passed following initial failure, actual accounting and rendered prompts validated, and full copied source compared: only the task prompt differs. Manifest, exact prompts, validation, and submission receipts are in `artifacts/agentic-continuation-cost-seed0-20261008/`. These are launch records, not completed outcomes.

## Serialization correction and fresh replacements

2026-10-08: PR #416 fixes frozen evaluation serialization of NumPy action vectors and schedules; stack is #413 → #416 → #414 → #415. Fix commit 6f70655a5; restacked head 0e3cc6907a9efcd1dc87fd71d9e741248d283a2a. Existing numeric values, shapes, bounds, objective, and prompts are unchanged.

The original-prompt cost-10 run ended with this error; original costs 200 and 1000 and continuation-prompt costs 1 and 20 were stopped after the same failure. All five are superseded, with their artifacts and zero-step error evaluations preserved. Each replacement starts fresh with the same seed, prompt, prices, model/effort, $20 budget, experience budget, measurement interval, and stopping rule. Other successful or unaffected runs were left in place; no Della jobs were stopped.

| Prompt | Human price | Replacement service |
|---|---:|---|
| original | 10 | hitl-jsonfix-original-10-agentic-seed0-20261008 |
| original | 200 | hitl-jsonfix-original-200-agentic-seed0-20261008 |
| original | 1000 | hitl-jsonfix-original-1000-agentic-seed0-20261008 |
| continuation | 20 | hitl-jsonfix-continuation-20-agentic-seed0-20261008 |
| continuation | 1 | hitl-jsonfix-continuation-01-agentic-seed0-20261008 |

Validation: regression tests failed before the fix; 59 tests passed at the prerequisite and 72 at the restacked head (one skip each). A saved formerly failing checkpoint executed 243 robot steps with no error through the real evaluator; it did not solve that episode. Three launch tests passed; old and replacement prompts and costs match. Fresh learning runs are used because failed evaluations prevented correct early stopping. Provenance, audit snapshots, launch receipts, and smoke-test results are in `artifacts/serialization-relaunch-20261008/`.
