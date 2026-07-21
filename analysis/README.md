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

- `practice_makes_perfect/random_skills.py` — aggregates `RandomSkillsMethod`
  (and, for comparison, any other `--method`'s) test success rate across seeds and
  `--grid-size` values, given a `--results-root DIR` laid out as
  `DIR/<method>/<grid_size>/<seed>/stats.json`. Prints a table; `--output PATH.png`
  additionally plots success rate vs. `grid_size`, one line per method.
