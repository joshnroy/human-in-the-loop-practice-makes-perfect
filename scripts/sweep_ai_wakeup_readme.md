# Actual Sweep AI continuation

`sweep-ai-wakeup.timer` runs on each half-hour (`:00` and `:30`), independently of the desktop chat. It invokes authenticated `codex exec`, not merely a job monitor. Installation uses `scripts/install_sweep_ai_wakeup.py --config <explicit-json> --state-dir <durable-dir>` under a capped scope. The installed service is capped at2GiB with MemorySwapMax=0 and OOMPolicy=continue. The independent readiness/launch gate retains exclusive control over scientific submission.

Current installation configuration and ledger:
`/home/josh/Documents/repos/research/hitl-pmp-results/sweep-launch/ai-wakeup/`.
The working source is isolated in `.codex-worktrees/sweep-ai-wakeup`.

Each configured task specifies its exact isolated worktree, owner-status path, continuation prompt, and authorized additional writable directories. Owner status records `owner`, `worktree`, `last_progress_utc`, optional `lease_expires_utc`, `status`, `current_units`, `progress_paths`, and durable next steps. Without an explicit expiration, the lease lasts45minutes after progress. Active agents must refresh their lease at checkpoints; do not retain one while idle. Mark unresolved scientific authorization `blocked-pending-...`. The supervisor never resumes such a task simply because time passed.

The supervisor uses an exclusive lock. A fresh owner lease prevents takeover. Active validation with recent progress prevents takeover; stalled validation after2hours gets a read-only diagnosis and is never automatically killed/restarted. Only an expired lease without active validation permits mutable continuation. The worker receives the existing scientific boundaries and must persist the next checkpoint before returning. Default turn limit20minutes. JSONL transcripts and the final message are retained even with ephemeral CLI sessions. A crash/timeout is recorded, not converted into an experiment retry.

Installed CLI detail: `--approve-for-me` itself selects workspace-write plus automatic review and cannot be combined with `--sandbox`. Mutable continuation uses that flag alone. Stalled-job diagnosis uses `--sandbox read-only`. No bypass flags, ignored rules, or disabled approval hooks are used.

Validation:8/8 supervisor tests cover leases, pending approval, active validation, read-only stalled diagnosis, wrong worktree, failed CLI, and duplicate prevention. Real authenticated filesystem smoke test1/1: Codex invoked a memory-capped tool, wrote and verified the expected marker, and returned exit0. This demonstrates tool-using continuation, not only model connectivity.

To stop the AI continuation timer: `systemctl --user stop sweep-ai-wakeup.timer` under a capped scope. This does not stop existing scientific jobs or the independent readiness timer. Logs prove local worker execution; they do not prove a user notification was delivered or wake the original desktop conversation.
