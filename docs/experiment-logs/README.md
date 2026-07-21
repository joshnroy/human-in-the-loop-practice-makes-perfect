# experiment-logs

Dated, one-off writeups of experiments run in the course of development — results,
plots, and the reasoning behind them — kept as a permanent record alongside the PR
that produced them. One markdown file per experiment (`YYYY-MM-DD-<topic>.md`),
with any images it references living alongside it under the same date/topic prefix.

This is **not** the same thing as [`../../analysis/`](../../analysis/README.md):
`analysis/` holds *reusable* scripts that read `--output-dir` output and produce
plots/tables/reports on demand, run again by anyone at any time; a file here is a
point-in-time record of one specific run of those scripts (or of ad hoc
exploration), written up once and not re-generated. A PR whose description
references numbers/plots from a real experiment should link the corresponding file
here rather than pasting the full writeup inline, so the PR description stays
short and the actual data has a permanent, linkable home.
