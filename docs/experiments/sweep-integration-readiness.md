# Sweep integration checkpoint (not pilot-ready)

## Current checkpoint, September28 23:50 EDT

Josh explicitly approved the issue/fix table, resolving the earlier authorization block. The drawer, pick-candidate, reachable-cube Sweep, declared-region observation, shared explicit human reset, and deployment-skill corrections are implemented. Focused Sweep/model tests31/31 pass; baseline EES tests70/70 pass. The strengthened physical smoke held the wiper, executed one human reset, and verified the full shared start target:1/1 passed, with1initial reset,1controller action,1human intervention, no resampling. A missing recorder API dependency from the76732331 import was diagnosed and the exact matching sampler_draws.py imported; six affected source files now pass mypy. No learning pilot has launched.

Exact depth20 practice search remains the primary unresolved gate. A delegated optimizer is testing exact caches and sound pruning; initial60-second probes remain incomplete. Wiper physical recovery and native calibration/frozen launch manifest also remain prerequisites. The older checkpoint below is retained as history, including its now-resolved approval block.

## Historical18:23 checkpoint

The implementation is isolated on `codex/sweep-learning-integration`. It contains the original Sweep physical controllers plus the approved EXP22c method/core changes from76732331, preserving Sweep KINDER8f600231 and kinder-baselines427ad6cd. Local dependency copies have independent Git metadata and clean pinned content.

Implemented: continuous-practice environment with an explicit no-automatic-reset invariant; stock OpenDrawer/PickWiper/Sweep parameter forwarding;28 individual robot operators (including explicit OpenGripper); fixed valid-start candidate filter and shared seed manifest; separate evaluation environment with all5-cube goal and per-cube diagnostic; unique state/replay files for every episode; per-action physical effect/tick records; CLI registration and canonical grid ModelB adapter; exact deployment expectation and caching; actual resolved method configuration and sampler-draw recording.

Verification:27/28 targeted tests pass;1/28 strict expected failure documents future ParkWiper→cube-pickability lookahead. Adapter9/9 source files passed mypy and Ruff after recording/metadata additions. Seed0 initialization1/1 passed. Stock smoke executed3/3 controllers, one initial reset, zero human resets and zero learning cycles. Effects: OpenDrawer0/1, PickWiper1/1, Sweep0/1. The smoke deliberately followed the old scripted sequence despite failed drawer opening, and logged Sweep's false preconditions; these are not learned-policy scores.

## Frozen candidate validity

64/64 initializations examined before any controller action. Practice0..31:31/32 valid; only28 excluded for wiper region+yaw. Evaluation10000..10031:32/32 valid. Ascending first3/first10 selection gives practice0,1,2 and evaluation10000..10009. No seed resampling or outcome selection. Exact results and candidate SHA256: `scratchpad/sweep-integration/start-validation.json`. The settled wiper origin needs the existing1cm support-height tolerance for start/reset validity; evaluation goal checks retain exact upstream containment.

## Blocking decisions / unfinished work

- Human reset execution is intentionally not implemented until shared-start semantics are reconciled with core reset_movables' robot-untouched contract. Current symbolic human effect includes RobotHome; it must not be asserted unless execution actually achieves it.
- Future Pickable facts after parking a held wiper cannot be truthfully inferred from HandEmpty alone. No optimistic fake effect has been inserted.
- Current symbolic Sweep may predict all5 success when some cubes are unreachable (e.g. on the floor). Requiring all5 InPile would also forbid useful partial completion when other cubes are already in the drawer. Conditional effects alone are unsafe because EES labels skill success from unconditional add-effects only.
- Actual EES evaluation currently plans with all robot skills; the new ModelB deployment descriptor assumes stock-only fixed policy. The two must be aligned by an approved choice.
- Partial drawer positions can fall between Open and Closed predicates, requiring a sound action model.
- Automatic approval review rejected the attempted partial-drawer/deployment-skill correction as a potential design change without exact authorization. It did not execute; related delegated edits were also stopped before mutation.
- Exact complete-recovery expectimax remains computationally unverified: canonical grid25×16,1024costparticles,weight0 depth20 did not finish30s; no horizon shortening, solver switch or branch approximation was made.
- Wiper recovery fix remains a separate draft dependency. No integration PR should precede it.
- Freeze calibrated random-exploration competence, native cost levels, and complete horizon coverage before the8-arm pilot. Candidate numerical proposals are not pilot results.

No Sweep learning experiment has been launched. Do not infer readiness from successful imports or passing isolated tests.
