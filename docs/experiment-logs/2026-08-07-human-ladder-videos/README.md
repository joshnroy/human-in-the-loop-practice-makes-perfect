# One training video per arm of the human-in-the-loop ladder

Eight `--record-full-loop` recordings, one per arm of
[the ladder](../2026-08-07-human-ladder.md). `--record-full-loop` is the only thing that
records **practice** periods — `--num-render-checkpoints` records evaluation episodes only —
which is what makes these watchable at all, since the whole experiment is about what happens
*during* practice.

Read the status bar: it names the phase, the cycle, the step, and the reset kind. `HUMAN` is
the reset kind to watch for. It is labelled apart from `PERIOD` and `INTERVAL` because it is
the only state jump that is **charged**, and on a reset-free arm it is the only jump that
happens during practice at all — so counting the `HUMAN` markers in a clip reads that arm's
intervention budget straight off the video. See `src/hitl_pmp/recording/README.md`.

## These are illustrations at reduced scale, not the measured runs

**No number in any experiment log is read off these videos.** They run
`--num-cycles 2 --max-steps-per-interaction 80 --num-test-tasks 3`, where the measured arms
run `10 × 150 × 30`. That is `recording/README.md`'s own instruction — *"Keep the run small …
a tool for seeing the structure, not for recording a full-scale experiment"* — and it is also
a size question: these are about 6 MB for eight, against roughly 30 MB per arm at full scale.

Everything else is the arm's real configuration: same `--env tossingroom`, same
`--practice-reset-policy never`, same `--ask-for-help` and `--human-reset-target`, same
`--seed`.

## Which seed each video shows, and why

**The representative seed is the one whose final score is closest to that arm's median over
its ten seeds**, ties broken by the lower seed number. A stated rule, so "representative" is
not a judgement call and no arm is silently shown at its best.

Each arm name links to its clip. A repo-relative link resolves to the file's blob page,
which GitHub renders with a video player; a raw `.mp4` URL would serve
`application/octet-stream` and merely download. One click is the unavoidable cost — no
syntax embeds an inline player from a committed relative path.

| arm | `--method` | seed | that seed scored | arm median | rescues in the clip |
|---|---|---|---|---|---|
| [`no-human`](2026-08-07-human-ladder-no-human.mp4) | `ees` | 4 | 7/30 | 7.0 | 0 |
| [`stuck-initial`](2026-08-07-human-ladder-stuck-initial.mp4) | `ees` | 5 | 20/30 | 20.0 | 4 |
| [`stuck-random`](2026-08-07-human-ladder-stuck-random.mp4) | `ees` | 3 | 20/30 | 19.5 | 4 |
| [`at-random-initial`](2026-08-07-human-ladder-at-random-initial.mp4) | `ees` | 1 | 17/30 | 18.0 | 1 |
| [`at-random-random`](2026-08-07-human-ladder-at-random-random.mp4) | `ees` | 1 | 17/30 | 15.5 | 1 |
| [`two-way-ledge`](2026-08-07-human-ladder-two-way-ledge.mp4) | `ees` | 0 | 30/30 | 30.0 | 0 |
| [`skill-oracle`](2026-08-07-human-ladder-skill-oracle.mp4) | `skill-oracle` | 0 | 30/30 | 30.0 | 0 |
| [`random-skills`](2026-08-07-human-ladder-random-skills.mp4) | `random-skills` | 2 | 0/30 | 0.0 | 0 |

The "rescues in the clip" column is the reduced-scale run's own count, not the measured
arm's. It is nonetheless the clearest thing in the set: the `on-stuck` clips show 4 `HUMAN`
markers each against the `at-random` clips' 1, which is the ~3.4× rate difference the log's
cost table quantifies at full scale, visible directly.

`skill-oracle`'s clip is much the shortest (148 KB against 740–920 KB) because that arm never
practises at all: there is no practice period to record, only the evaluation sweep.
