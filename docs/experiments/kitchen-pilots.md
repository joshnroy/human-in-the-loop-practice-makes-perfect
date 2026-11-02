# Kitchen pouring pilots and overnight runs

This is a separate `pouring3d_kitchen` experiment. It preserves the original
`pouring3d` configuration, source assets and results. Practice uses cold dumplings
and a dry pan, with explicit costed human bulk reset; deployment has visual water
and requires jug placement, simulated cooking activation and retreat, with no human.
See [the environment contract](../environments/pouring3d-kitchen.md).

## Frozen comparison

| Stage | Methods | Seeds | Practice ticks | Evaluation |
| --- | --- | --- | ---: | --- |
| pilot | EES, Curtis, expectimax | 0, 2 | 6,400 | Initial/final; 3 tasks, one per count |
| overnight | EES, Curtis, expectimax | 0, 2 | 25,600 | Initial/every 6,400 ticks; 9 tasks, three per count |

The overnight stage restarts from the paired seeds rather than resuming pilot
checkpoints. Both stages use separate Pour1/2/3 samplers, the ordinary shared
learners, tilt 87–95 degrees, lateral amplitude 0–25 mm and duration 0–30 seconds.
Feasibility witnesses are never supplied to learned samplers. Support failures and
unsuccessful pours remain in the measured outcomes.

Settings otherwise follow the [original pouring pilots](pouring3d-pilots.md):
7,500 evaluation ticks, 2,500 ticks per skill, human interruption cost 5, robot tick
cost 0.01, one counted tick per human request, no scheduled/free resets, no
perfect-score early termination. The kitchen cost file prices the actual
`human_bulk_pour_back` action. Costs are recorded once by the ordinary accounting
interface; the environment's interruption ledger is an audit, not a second charge.

Curtis and expectimax use the existing `SkillPlansPracticeDomain` and unmodified
inference/search algorithms. As in the original pouring study, their planning
utility estimates the learned pick–pour–place subtask, averaged over requested
counts. Actual deployment evaluation additionally executes activation and retreat
and measures `MealStarted`. These deployment-only support actions are not trained
or assigned invented practice observations. Thus the model's predicted utility is
a proxy, not the measured full-task success rate. Report both pour and full-task
outcomes; do not treat low productivity or early STOP as an infrastructure failure.

## Launch and provenance

Use `scripts/with_env.sh python -m scripts.kitchen_launch check`, then `manifest`
with `--stage pilot` or `--stage overnight`, `--out FILE --results-root DIR`.
`run --manifest FILE --go --max-workers 6` verifies clean source/dependency commits,
resolved CLI flags, costs, private dependency imports and native initial state.
Existing output directories are never overwritten. Freeze local commits before
creating manifests; never change code used by an active worker.

The launch gate checks method construction, a native witness for each count,
a continuous failed-pour/recovery/retry cycle, and completion/readability of all
six pilots. Pilot failure/STOP rates are scientific outcomes. Exceptions, incorrect
accounting, forbidden human deployment actions, hidden free resets, or corrupt
recordings are blocking correctness problems. Retain failed diagnostics and label
any replacement revision; never silently pool them with the corrected runs.

Run local jobs in memory-capped services within `agents.slice`, budgeting workers
against other workstation users. No physical robot, water or heater is operated.
Simulation success does not validate hardware feasibility or thermal physics.
