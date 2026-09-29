# Persistent Sweep launch gate

The minute timer runs independently of any chat or agent. Scientific integration writes
`/home/josh/Documents/repos/research/hitl-pmp-results/sweep-launch/readiness.json`
atomically only after checks pass. Missing readiness waits; a non-READY status blocks.

Required READY fields: `revision` (full SHA), `source_dir` (absolute, clean tracked checkout),
`manifest: {path, sha256}`, nonempty `evidence: [{path, sha256}]`, and `submit_argv`
containing a literal `{token}` argument. The wrapper submits exactly one array named
`sweep-<token>` and prints final JSON `{"job_ids": ["12345"]}`. Evidence records the
scientific checks; this generic launcher validates provenance, not scientific judgment.
The wrapper must independently verify remote pinned code/dependency and manifest hashes.

State and transitions live alongside readiness. States are waiting, blocked, ready,
submitting, submitted. A durable submission intent is fsynced before executing the
wrapper. On a crash/timeout the next tick reconciles squeue and sacct; even empty
accounting never triggers another submission. This deliberately trades automatic retry
for no duplicates after uncertain delivery. To resolve an unmatched intent, inspect
both remote scheduler and wrapper logs before an operator authorizes a separate attempt.
Never delete state merely to make a blocked launch retry. Submitted failures stay
submitted: the launcher never restarts an experiment.

Run `scripts/with_env.sh python scripts/install_sweep_launch_watch.py` under a capped
scope to install. The service has MemoryMax=6G and MemorySwapMax=0. The existing
research-monitor tmux and 30-minute health timer remain independent. The minute timer
can launch ready code but cannot finish code, resolve scientific blockers, or wake an AI.
`transitions.jsonl` is a local event log, not proof that a human notification was delivered.

Stop automatic launching with `systemctl --user stop sweep-ready-launch.timer`.
