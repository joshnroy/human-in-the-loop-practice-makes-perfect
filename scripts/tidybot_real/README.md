# Real TidyBot launch and webcam recorder

`launch.sh` starts a real-robot (or `fake`) Tossing3D run of `hitl_pmp.cli`. It asks one
y before driving the robot, tells the webcam recorder to record the run, and writes its
cost file and run folder under `HERE`. Every setting is an environment variable whose
default is the 2026-11-08 trial 3 value; the script's header lists them.

## The 2026-11-08 launches

That day ran from three near-copies of one script, kept in the results folder. Each is
now a setting of this one:

| file on 2026-11-08 | what it was for | equivalent |
| --- | --- | --- |
| `launch.sh` | trial 3 restart: seed 2, F=50, a 20-task evaluation after every 2nd cycle, no step-0 evaluation (the step-0 point is the morning run's 9/20, same robot code) | `BUDGET=100000 launch.sh real` |
| `launch-trial3-eval10.sh` | trial 3 again from a later launch worktree (`launch-20261108f`); the settings and run folder are trial 3's | `WORKTREE=.../agent-wt/launch-20261108f BUDGET=100000 launch.sh real` |
| `launch-seed3.sh` | trial 4: seed 3, F=50, a step-0 evaluation, then 10-task evaluations after cycles 3 and 6 (about 60 and 120 practice actions) | `WORKTREE=.../agent-wt/launch-20261108f SEED=3 NUM_TEST_TASKS=10 MEASURE_EVERY_CYCLES=3 INITIAL_MEASUREMENT=yes RUN_TAG=F50-6cyc-20261108e BUDGET=100000 launch.sh real` |

Despite its name, `launch-trial3-eval10.sh` differed from `launch.sh` only in the
worktree: it still asked for 20 test tasks.

Earlier trials: seed 0 (the morning run), seed 1 (`run-real-curtis-w10-20261108b`,
stopped in cycle 1). Trial 3's code was launch branch `launch/real-20261108d` (main
`eb857ef12` plus #768, #770, #771, #773 and #774; #773 makes a timed human reset cost
`10 * (RESET_COST + 0.1 * seconds)`). The robot code is recorded in each run folder's
`RUN-NOTES.md`.

The robot's RPC authkey is read from `ROBOT_AUTHKEY_FILE`; it is never in this repo.

## `webcam/`

`server.py` records a USB webcam continuously as timestamped H.264 segments with a
per-frame CSV sidecar, serves a live view on `--port` (8766), and gives `launch.sh` the
`/record/start` and `/record/stop` endpoints that cut a run's own segments into
`<run dir>/webcam/`. It also publishes frames over `multiprocessing.connection` on
`--frame-port` (6004) when given `--frame-authkey-file`; `client.py`'s
`WebcamImageClient` reads them with the same key file. The key is the one prpl-tidybot's
ceiling cameras use (`CONN_AUTHKEY` in its `third_party/constants.py`); put it in a file
outside this repo.

Both files are copies of the service that runs on agni, kept close to it: they follow
this repo's lint except `PLR0917` (keyword-only parameters), waived per file.
