# EXP-22c / EXP-23c analysis

Analysis only. Experiment commit76732331; no experiment code changes.
Source pipeline: disk backup scratchpad/x22b from2026-09-28.

Run every command using this worktree scripts/with_env.sh inside a systemd service
with MemoryMax=6G, MemorySwapMax=0 and OOMPolicy=continue. Keep8–12GiB available.

1. Check all63/63 Della array tasks are COMPLETED with ExitCode0:0.
2. Run pull.py once in a capped service. It uses della.py, skips unnecessary
   decision logs, checksum-verifies local copies and does not delete remote logs.
3. Run run.py in a capped service. It collects only completed runs, streams skill
   logs and decisions, checks practice/eval count agreement, and publishes the site. EES additionally uses
   the prior practice_events parity and final-capped-action omission checks.
4. Rerun run.py when workstation57/57 finishes; previous scan outputs are reused.

Outputs: data/ holds per-run rows, validated skill scans, compact decision events,
EXP23 categories, paired tests and checksum evidence. Generated website lives in
sibling hitl-pmp-site/tossing3d-baseline/exp22c. Parent owns HANDOFF/home.py updates.

Statistics: original comparison family, within-machine pairing by seed. Exact
sign-flip p-values enumerate all2^n paired sign assignments. Supplementary paired
t-tests are undefined for nonzero constant differences; never substitute p=0.
No arm pooling to inflate sample size; no cross-machine replay assumption.

EXP23: optional means PickCube appears among logged root valid actions at a
reset decision; needed means it does not. This is a model-availability label,
not a physical impossibility claim. All cost1/5/20 comparisons use Della; cost5
is Model B lambda3e-4 GP-off from EXP22c. Reset destinations use objects[-1].

Poster: Josh explicitly requested all recorded practice episodes. The main linked
timeline now covers all50/50 cycles of workstation B lambda3e-6 GP-off seed0;
site_header owns the full timeline generators in the checkpoint worktree and
backup scratchpad/exp22c-full-timeline. Only cycle1 has a rendered clip; its action
boxes link to exact video offsets. Remaining49 cycles do not imply video exists.
The former first10-cycle draft is historical only. Demo selects the earliest
robot-side reset followed by tossing (cycle1); it is recorded-state replay.
