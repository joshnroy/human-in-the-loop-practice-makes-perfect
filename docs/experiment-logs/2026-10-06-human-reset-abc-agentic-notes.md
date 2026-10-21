# Agentic human-reset A/B/C, seed 0

Three jobs, optional delegation, identical enriched r10 initial library. A retains probability-one human search; B infers competence and learning rate; C pins human learning rate zero while retaining competence inference. All successful self-loops are unavailable during practice and evaluation. Failed attempts may self-loop. Explicit human failure edges model unavailable/declined intervention leaving source unchanged. No physical failures injected, human execution unchanged.

48 agentic method and 38 runtime tests pass. Docker preflight passed. Zero-rate semantic preflight confirms competence falls on failure and learning rate stays exactly zero after refit. Della probe found no Docker; local queue uses persistent tmux hitl-human-reset-abc-20261006, maximum two runs concurrently, existing shared host slice6CPUs15GiB and two Docker slots3CPUs7.5GiB each. C starts when a slot frees.

Monitor uses existing tested per-run supervisor, with aggregate periodic chat monitor owned by PDDL agent. Local implementation/source snapshots retained. No publication.

## Supervisor repair 2026-10-06 01:27 UTC

Queue accidentally imported stale canonical monitor lacking the earlier directory-only guard, causing NotADirectoryError on coding-session.json. Added guard and passed 21 monitor tests, including persistent-session-file regression. Replaced only queue supervision with explicit attach-to-existing-services mode; original A/B services stayed active with original 21:12:14 EDT start times. No experiment restart or state loss. Both monitor snapshots fresh, no actual-bug trigger. C remains queued. Artifacts are on persistent repo disk, not tmpfs.

Full supervisor parity follow-up: restored complete previously running r9 monitor from /tmp, including synthetic provider-error handling, native recovery, evaluator tool checks, and Tenacity429 telemetry. Full monitor validation122tests+12subtests passed. Supervision reattached only, original experimentservices unchanged.
