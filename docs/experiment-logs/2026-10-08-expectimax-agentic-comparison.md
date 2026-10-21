# Expectimax vs full-agentic: human-reset strategy audit

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

Date: 2026-10-08. **Analysis of existing runs; no new experiment launched.**

Comparison document (private Notion page) (private draft) · Results and graphs (private Notion page) · Explanation-prompt PR #415.

Seed 5 explicitly declined autonomous retrieval despite acknowledging the hint. Its final response said:

> Because throws are defined relative to the base, I practised directly on far-side bins. A far reset cost only 1, versus 30+ robot steps to recover the cube myself, and the hint's robot-side practice wasn't needed.

This is a retrospective explanation, not a recorded comparison before each action. The requested decision summaries were not emitted. Saved thinking blocks contain no readable text.

At 6,800 counted steps, means over original seeds 0–4:

| Method | Near resets | Far resets | All requests | Held-out score / 10 |
| --- | ---: | ---: | ---: | ---: |
| Expectimax depth 6/grid | 7.4 | 1.0 | 8.4 | 2.2 |
| Full-agentic expanded hint | 9.4 | 42.8 | 52.2 | 10.0 |

Seed 5 is a separate explanatory run, excluded from these means. It finished at 11,746 counted steps with 62 far resets and no near resets, final 10/10, and three consecutive perfect evaluations including the final one. Its far requests were six placement samples, twelve floor-calibration attempts, and forty-four controller tests. Ten calibration throws actually landed on the robot-side floor; zero near-bin resets does not mean zero near-side practice.

The broader claim that agentic always ignores the strategy is contradicted by seed 1: it executed six near-side floor toss–retrieve cycles using 570 robot steps and zero human requests. The interface audit found no host block on near resets or autonomous reuse.

For a real similar-state comparison, agentic seed 4 switched from a successful near-bin test to far tests to try the “real task.” Expectimax seed 0 decision 57 chose a near reset after a cube landed in its near bin: near Q = 0.1606330734, far Q = 0.1606202607. Both requested human help; the physical states, controllers, and histories were not identical. Earlier expectimax made 26 near toss dispatches without intervening human resets, mostly misses onto the floor.

Current costs are one per robot step and one per human request, with human weight 1 and λ = 3e-6. Therefore cost equals counted experience; both methods cost 6,800 at the matched checkpoint. Seed 5's reset-price explanation omits subsequent robot pickup and preparation. Its final thirteen far tests cost 2,376 total. No measured seed-5 near-recovery counterfactual establishes which full continuation was cheaper. Expectimax also receives shared competence structure and abstract predicates; this is not an isolated prompting comparison.

Proposed prompt changes, **not implemented or launched by this audit**:

1. Compare complete continuation costs, including robot pickup after human resets.
2. Compare several tosses, accounting for repeated resets and expected deployment improvement.
3. Measure uncertain recovery and transfer costs when the information is worth its cost.
4. Make resets explicit decisions instead of unconditional test-driver setup.
5. Use a prescribed near-side curriculum as a separate capability control.

Recommended first ablation: 1 + 4, keeping the objective fixed. Record concise structured decisions and observed costs. Enforcing records in the host would be a separate instrumentation change.

Evidence bundle:

```text
artifacts/agentic-expectimax-comparison-20261008/notion-comparison.md
artifacts/agentic-expectimax-comparison-20261008/comparison-evidence.json
artifacts/agentic-expectimax-comparison-20261008/publication.json
artifacts/five-seed-status-20261008/data.json
artifacts/decision-comparison-20261008/comparison.json
artifacts/decision-comparison-20261008/expectimax-first100.json
artifacts/full-agent-normalized-2xhint-seed5-20261008/audit.json
artifacts/full-agent-normalized-2xhint-seed5-20261008/run/coding/stream.jsonl:216
```

Log identifier: `exp-20261007-expectimax-agentic-comparison-audit`. The prefix follows the existing publisher convention; the record date is 2026-10-08.
