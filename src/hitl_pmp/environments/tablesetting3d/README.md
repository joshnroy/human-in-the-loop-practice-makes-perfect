# TableSetting3D

Four heavy, unbreakable plates start at their individual kitchen-counter locations.
TidyBot must place each at its assigned dining-table setting. It can always set the
table and reset the task independently, carrying one plate per trip. A human can
return all outstanding plates in a stack, but may spill that stack.

This first version is a **skill-level simulator with a MuJoCo visualization**.
Robot motion is scripted and human handling is abstract. Neither grasp feasibility
nor human stack stability is inferred from contact dynamics. Plates are rigid
cylinders, with no breakage. The payload limits (robot 1.5 kg, plates 1.2 kg each,
human 6 kg) are benchmark assumptions, not hardware specifications.

## State, actions, and resets

The native environment is `kinder/TableSetting3D-o4-v0`, implemented in
`reference/kindergarden/src/kinder/envs/dynamic3d/tablesetting3d.py`.
Its observation is the robot x/y followed by four plate x/y/z positions. True
human competence is excluded from observations. The project `Environment` adapter
is `TableSetting3DEnvironment`, with the same actions encoded as one-element arrays.

| Action | Effect |
| --- | --- |
| 0–3 | Transport the selected plate to its assigned table setting. |
| 4–7 | Return the selected plate to its assigned kitchen location. |
| 8 | Request one human stack reset of all plates not already home. |
| 9 | Wait one simulated second without changing physical state. |

Robot actions transport exactly one plate. Navigation follows an open aisle;
spills land at separate, reachable floor positions. Human resets preserve robot
position and dishes already home. Reset completion means every dish is home;
it does not require restoring the robot's initial position. The next task may
start from any reachable robot pose. A spill never triggers an automatic repair
or a free reset. `reset()`/`hard_reset()` are privileged harness operations only.

Goal checking uses each dish's assigned position, including height. A plate on the
floor below the right setting does not satisfy the goal. Task completion returns
Gym `terminated=True`; persistent practice callers may then issue costed reset
actions on the same instance.

## Competence and costs

`human_competence` is the probability that a requested stack reset succeeds.
The same probability applies to any nonempty stack in this first version.
Success returns the plates home; failure leaves them at fixed floor landing pads.
Only the Bernoulli outcome is random, seeded by Gym reset. Spill severity, human
speed, and the cost of requesting help remain fixed across competence conditions.

Costs are simulated seconds, using robot speed 0.3 m/s, human speed 1 m/s,
3-second surface pickups, 14-second floor pickups, and 3-second placements.
The longer floor pickup represents a slower recovery maneuver; it is a chosen
benchmark parameter, not measured robot timing. Human attempts include a fixed
2-second request, 5-second handling charge, and collection/return travel.
No separate spill penalty is imposed: the consequence is robot recovery work.
Rewards are negative action costs; `info` also reports cost, human interventions,
observed human success, and reset completion.

After the default four-plate table-setting sequence:

- Robot reset: about 110.8 seconds.
- Human reset attempt: about 13.72 seconds.
- Robot recovery after a spill: about 131.07 seconds.

The reference comparison is `human_attempt + (1 - estimated_p) * spill_recovery`
versus robot reset. Its crossover is about 0.26 in this particular state; the
costs depend on the current robot and plate positions. It compares one human
attempt followed by robot recovery with a fixed-order autonomous reset, not all
possible policies or optimal collection routes.

## Demo and video

```bash
scripts/with_env.sh python -m scripts.table_setting3d_demo \
  --output-dir artifacts/table-setting3d/reliable --human-competence 0.9 --seed 0
scripts/with_env.sh python -m scripts.table_setting3d_demo \
  --output-dir artifacts/table-setting3d/unreliable --human-competence 0.01 --seed 0
scripts/with_env.sh python -m scripts.table_setting3d_comparison \
  --first artifacts/table-setting3d/reliable \
  --second artifacts/table-setting3d/unreliable \
  --output artifacts/table-setting3d/comparison.mp4
```

Use the repository's memory-limited execution wrapper for simulator runs as
described in `CLAUDE.md`. Each command produces `episode.mp4`, `initial.png`, and
`results.json`. Video motion is accelerated with `--speedup` (default 12); costs
remain simulated seconds. `--no-video` runs without initializing MuJoCo.

Every action record includes its action ID, physical state before/after, realised
cost, and exact half-open video frame range. Results record source and scene
fingerprints, video frame count, and video checksum. The comparison command
checks these records before selecting the last adaptive cycle from each run and
the first observed spill in the second run, if one occurred. It writes a clip
manifest beside the comparison video and refuses stale or mismatched inputs.
The comparison is a demonstration of two seeded runs, not a statistical result.

The demo schedules three calibration requests, observes success/failure, and
updates a Beta(1,1) competence belief. From the fourth cycle, a small reference
selector compares expected reset costs using that belief. It never reads the
true competence to choose an action. Calibration failures are recovered and
charged. With seed 0, the examples above yield three successful or three failed
calibration resets, respectively, then human or autonomous reset choices.

This reference selector is **not yet integrated with EES or the belief-space
practice planner**, and the environment is not registered in the project's
global `--env` dispatcher. The native Gym environment, project Environment
adapter, and dedicated demo are the supported interfaces for this prototype.
The tests cover recovery, exact cost accounting, seeded outcomes, partial resets,
competence isolation, learned reset choice, native registration, and rendering.
Recovery/cost checks enumerate all 81 combinations of home, table, and floor
locations for the four plates.

## Decision needed before practice-planner integration

Robot skills in this prototype are fixed and reliable. Calibration learns the
human's reliability; repeating robot transport does not improve a robot skill.
Connecting the environment to robot-skill practice therefore requires deciding
whether to retain a human-reliability-only experiment or add a learnable robot
skill alongside the reliable fallback. No new robot failure model or learning
curve has been selected.

The reference reset selector makes one human attempt and then recovers with the
robot on failure. The environment itself permits additional human requests. A
general planner may choose retries and reach a different competence threshold;
the reference selector's threshold should not be presented as a globally optimal
policy boundary.
