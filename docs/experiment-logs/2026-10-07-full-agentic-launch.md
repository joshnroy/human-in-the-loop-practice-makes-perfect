# Full-agentic workstation launch — 2026-10-07

The user authorized the workstation run after reboot. The corrected service
started at 12:24:51 EDT and was verified active with successful Claude broker
responses and separate coding/evaluation containers. This is launch evidence,
not a completed experimental result.

## Configuration and provenance

The method runs through the frozen RoboCode transport with Claude Code OAuth,
Claude Opus 5.5 at high effort, and a $20 model-cost cap. It starts with the
original pre-pilot bootstrap, seed 0, 85,000 counted practice steps, measurement
snapshots every 1,700 steps, one counted step per human invocation, and ten
held-out tasks with a 500-controller-step limit. Learning remains agent-selected.
No additional wall-time limit was imposed. The run uses the agreed frozen
simulator dependencies; their bytes were verified against the Della manifest.

Source revision: `5a073c449aef909e3747270b3f3a97e882d61f30`.
Service: `hitl-full-agent-step-20261007-02.service`.
Output: `artifacts/full-agent-step-20261007/run-02`.
The long output path is durable; short-lived sockets use
`/tmp/hitl-agentic-scratch`. The service is limited to 15 GiB and six CPU equivalents;
Docker execution retains its existing per-container limits and shared resource pool.

[Source hashes](2026-10-07-full-agentic-launch/source-manifest.json),
[sandbox configuration](2026-10-07-full-agentic-launch/sandbox.json),
[exact successful launch command](2026-10-07-full-agentic-launch/launch-command-02.json),
[launch receipt](2026-10-07-full-agentic-launch/launch-receipt-02.json), and
[entrypoint](2026-10-07-full-agentic-launch/launch_runner.py.txt) preserve the launch.
Authentication stays on the host; no credentials are included in these records.

## Validation and initial startup correction

The agentic PR's lint failure was reproduced using CI's Ruff 0.16.10, then fixed
by annotating the planning MCP context manager as a generator. The fix was
propagated through the draft PR stack without rewriting the published experiment
commits. The exact-version Ruff checks and formatting pass; the runtime,
full-agentic and measurement suite passed 58 tests with one skipped. GitHub lint
was confirmed successful after the fix; the other CI groups were still running
at this capture.

A [no-model preflight](2026-10-07-full-agentic-launch/preflight.py.txt)
verified frozen imports, real controller and human-step accounting, receipt
idempotence, immutable measurement snapshots, Docker socket communication, and
an independent held-out evaluation with a short smoke-test horizon.
The [preflight result](2026-10-07-full-agentic-launch/preflight-result.json)
is test evidence and is excluded from the experiment's practice and model budgets.

The first service start at 12:23:48 EDT failed before any model call or practice
step because the launch's temporary directory exceeded the Unix socket pathname
limit. Both coding and evaluation startup encountered this infrastructure issue.
The failed `run/` directory was retained. The corrected start used a short
`TMPDIR` and a fresh `run-02/` directory with the full, unspent $20 cap.
The [first launch receipt](2026-10-07-full-agentic-launch/launch-receipt.json)
and [command](2026-10-07-full-agentic-launch/launch-command.json) remain archived.

## Monitoring and interruption

Read `run-02/status.json`, `run-02/evaluations/*/results.json`, and the service
journal for progress. Completed evaluation records are separate from pending
snapshots. Write `run-02/STOP` only when a stop is requested; the runner then
records its actual endpoint and drains pending evaluations. A workstation reboot
does not preserve the live simulator, so it is not an exact experiment-resume path.

Della connectivity timed out after reboot. The user deferred restoring it and
asked to proceed with this local run. No Della jobs were stopped or resubmitted.
The [Della restart record](2026-10-07-reset-competence.md) retains the job IDs;
no fresh Della results or completion claims are made here.
