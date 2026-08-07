# analysis

Post-run analysis only — scripts here read `--output-dir` output already produced
by `python -m hitl_pmp.cli --env <name> --method <name> ... --output-dir DIR`
(`DIR/stats.json`, the run's `core.Metrics`, and `DIR/episode.mp4`) and turn it into
plots/tables/reports. **Never** run a simulation or drive a `Problem`/`Method`
directly — that's `hitl_pmp.cli`'s job, via `src/hitl_pmp/practice_loop.py`'s
`PracticeLoop`. If a script here is calling `Problem`/`Method`/`Environment`
directly instead of invoking the CLI and reading its output, that's a sign the
CLI-side wiring it depends on shipped in a later PR than it should have — see
`../CLAUDE.md`'s workflow section.

`stats.json` is a raw `Metrics.model_dump_json()` (just `evaluations`/`task_name`,
no derived fields) — a reader reconstructs the instance via
`Metrics.model_validate_json(...)` and calls its own computation methods
(`task_training_curve()`, `percentage_success_overall_test()`, etc.), so there's
exactly one place those are computed, not a second copy living in `analysis/`.

- `run_timing.py` — per-run wall-clock as a function of the concurrency a run
  actually experienced, read back from the `timing.json` files
  `scripts/run_sweep.py` writes beside each `stats.json` (see
  [`../scripts/README.md`](../scripts/README.md)). Takes `--results-root DIR` laid
  out as `DIR/<method>/<seed>/timing.json`, found by **filename glob, never by file
  mtime**. Prints elapsed seconds bucketed by observed concurrency plus a per-sweep
  summary; `--per-run` adds one row per run. Concurrency here means the
  *machine-wide* `hitl_pmp.cli` count, not the sweep's own in-flight count — several
  agents' sweeps share this box, and using the sweep-local number would credit
  another sweep's load to this one's `--max-workers`. It is the one script here that
  imports from `scripts/` (`RunTiming`, the record's schema): the reader validates
  against the writer's own definition rather than keeping a second copy that can
  drift.
- `practice_makes_perfect/random_skills.py` — aggregates `RandomSkillsMethod`
  (and, for comparison, any other `--method`'s) test success rate across seeds and
  `--grid-size` values, given a `--results-root DIR` laid out as
  `DIR/<method>/<grid_size>/<seed>/stats.json`. Prints a table; `--output PATH.png`
  additionally plots success rate vs. `grid_size`, one line per method.
- `practice_makes_perfect/ees.py` — the paper's own Figure 4 view for the EES
  reproduction: fraction of evaluation tasks solved vs. number of online
  transitions, one line per approach, solid = mean across seeds and shading =
  standard error. Takes `--results-root DIR` laid out as
  `DIR/<method>/<seed>/stats.json` (grid size fixed, seeds varying — the axis that
  matters here is *training progress*, not environment size). A method with a
  single checkpoint (`--method skill-oracle`, which never practices) is drawn as a
  flat dashed reference level rather than a lone point.
- `practice_makes_perfect/practice_diagnostics.py` — the *why* behind that curve,
  from the same `stats.json` files: per lifted skill and per window, how often it
  was practiced, how often that worked, and how much of it the sampler's classifier
  actually chose (`practice_outcomes_per_cycle`), plus how often the planner was
  asked and came back empty (`planning_failures_per_cycle` /
  `planning_attempts_per_cycle`, which nothing plotted before). Same
  `DIR/<method>/<seed>/stats.json` layout, `--output PATH.png` for the figure. Reads
  a run's own record and keys on nothing domain-specific, so it serves every `--env`
  — which is what distinguished it from the Tossing Room skill-trace script, whose
  overlapping tallies were Tossing-Room-only by construction (retired with the three
  superseded Tossing Room domains).
- `practice_makes_perfect/tossing3d_practice_diagnosis.py` — applies the decision rule
  pre-registered in `docs/experiment-logs/2026-08-06-tossing3d-practice-diagnosis.md`
  to a Tossing3D sweep and renders that log's figure. Domain-specific where the script
  above is not, because the rule names this domain's three skills: the standoff
  (`MoveToThrowPose`), the skill carrying the real success signal (`Toss`), and `Pick`
  as the **positive control** — a run in which the control also shows zero informed
  draws is reporting an instrument fault, and the module says so rather than
  concluding. The verdict deliberately reads no episode counts; see its docstring for
  why that is load-bearing on this domain today.
