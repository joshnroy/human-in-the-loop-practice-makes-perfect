# Tossing3D POMDP checkpoint and resume

Scheduled-reset POMDP runs with `--output-dir` now atomically save
`checkpoint.pkl` after the initial evaluation and after each completed
practice/refit/evaluation cycle. A crash during a cycle loses that cycle only;
resume starts immediately after the last durable boundary. A crash before the
initial evaluation completes has no checkpoint to resume.

For a run without recording/rendering, set `--checkpoint /path/checkpoint.pkl`
explicitly. A checkpoint contains the learner's fitted classifiers, normalization,
training datasets, competence histories, POMDP belief, sampler and method RNGs,
task RNG streams, Python/NumPy/torch RNG states, fixed test tasks, loop position,
cumulative metrics, and the runner's counter baselines. Live policies, simulator
handles and disposable translation caches are not serialized.

```bash
scripts/with_env.sh python -m hitl_pmp.cli \
  --env tossing3d --method pomdp --seed 1 --num-cycles 20 \
  --output-dir /tmp/tossing-seed1

# After interruption: repeat all original experiment flags, change the output directory.
scripts/with_env.sh python -m hitl_pmp.cli \
  --env tossing3d --method pomdp --seed 1 --num-cycles 20 \
  --resume /tmp/tossing-seed1/checkpoint.pkl \
  --output-dir /tmp/tossing-seed1-resumed
```

`--num-cycles` remains the total target, not the number of additional cycles.
Resume checks the experiment configuration, source-code fingerprint, Python and
dependency versions, schema, and payload checksum before unpickling. Keep the same
checkout and pinned KINDER dependencies; this is a recovery format, not a promise
of cross-version model portability. Recording flags and output paths may change.

The resume output directory must be new or empty. Existing recordings are never
truncated. The new segment's `stats.json` contains the complete cumulative run;
its videos, state logs and JSONL sidecars cover only the resumed segment. A partial
cycle may be visible in the old segment but is discarded as training experience.
Progress timestamps/elapsed times refer to the segment, not combined wall time.
The checkpoint destination defaults to the new output directory; alternatively
give `--checkpoint` explicitly. Checkpoints are written through a temporary file,
flushed to disk, atomically replaced, and directory-synced.

## Limits and trust

- **Only load your own trusted local checkpoints.** Pickle deserialization can
  execute arbitrary code. A checksum detects corruption, not malicious content.
- **Scheduled practice resets only.** The next practice task and every evaluation
  task rebuild the simulator from their saved/continued scene-seed streams, so no
  mid-episode simulator state is required. Existing `never` runs are unchanged and
  do not auto-checkpoint; explicit checkpoint/resume requests reject that policy.
  KINDER's float32 object-centric replay snapshots are not exact MuJoCo/controller
  checkpoints, so using them to resume a reset-free experiment would perturb it.
- The guarantees apply to the current CPU learner and matching dependencies.
  Checkpoints do not migrate hardware, resume a partial optimizer step, extend the
  original cycle budget, or merge W&B/video/JSONL recordings across segments.
- A checkpoint is not an experiment result: final cumulative results still live
  in `stats.json`, written after the requested total cycles finish.
