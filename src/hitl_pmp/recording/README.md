# `recording/` — the whole outer loop as one annotated video

`--record-full-loop PATH` (a global CLI flag) records an entire `PracticeLoop` run to a
single seekable `.mp4`: every practice period, every evaluation episode of every sweep,
and every environment reset, in the order they happened.

```bash
python -m hitl_pmp.cli --env tossingroomsplitpickupweight --method ees --seed 0 \
    --num-cycles 3 --max-steps-per-interaction 30 --num-test-tasks 3 \
    --record-full-loop /tmp/full_loop.mp4
```

Keep the run small. Every practice step and every evaluation step becomes a rendered
frame, so this is a tool for *seeing the structure*, not for recording a full-scale
experiment; a few cycles is plenty.

## What it is not

`--num-render-checkpoints` records **evaluation episodes only**, one clip per
checkpoint, of test task 0 only. Practice periods have never been rendered at all —
which is where the behaviour worth watching is (what the method chose to practice, how
often the harness rescued it, what a reset cost it). This records the loop itself.

## Why the reset kinds are labelled separately

The environment's state jumps for four structurally different reasons, and a viewer
cannot tell them apart from each other, or from a skill's effect, without being told:

| kind | when | in `practice_loop.py` |
| --- | --- | --- |
| `HARD` | once, before anything else | `problem.hard_reset()` |
| `PERIOD` | top of every practice cycle | `reset_to_task(task)` before the step loop |
| `INTERVAL` | every k steps *within* a period | `--practice-reset-interval` |
| `EVALUATION_TASK` | once per test task | inside `Problem.run_task_episode` |

Each gets its own colour, its own caption over the frame, and a `RESET` field in the
bar; the marker is held for several frames, because a one-frame marker on an
instantaneous state jump is easy to scrub straight past. `PERIOD`'s necessity is
actively under review (see `PracticeLoop`'s own docstring, which argues both sides),
which is what this recording exists to make inspectable.

## Structure

- `types.py` — `LoopStatus` (one frame's annotation, frozen), `LoopPhase`, `ResetKind`.
- `overlay.py` — `StatusBarOverlay`, which composes the bar and markers *around* a frame
  a `core.Renderer` already produced. Deliberately not part of any domain renderer: a
  domain's job is to draw its own state, and teaching each one about cycles and sweeps
  would spread harness knowledge across every environment.
- `loop_recorder.py` — `LoopRecorder`, the hooks `PracticeLoop` calls.

## The one invariant

**Recording never changes what a run does.** `LoopRecorder` draws from no RNG, takes no
action and decides nothing; it is handed state the loop already had. A recorded run
takes exactly the same actions as an unrecorded one — it just renders far more of them.
`tests/test_method_runner.py` pins the resulting `stats.json` byte-identical, and the
same comparison was run at full scale on Tossing Room EES (seed 0, 3 cycles, identical
SHA-256).

Frames are streamed to `core.renderer.VideoStream` one at a time and dropped, never
accumulated — same rule, and same reason, as `PracticeLoop`'s checkpoint streaming.
