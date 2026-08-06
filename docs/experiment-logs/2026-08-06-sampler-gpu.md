# The sampler classifier should not move to the GPU — but it should stop using 24 CPU threads

**TL;DR.** `fit()` really is where a run's time goes: **154.4/188.2 seconds** of one
25-cycle `tossingroomsplit` EES run, over 47 sampler refits. But the GPU is not the fix.
Against a CPU pinned to one torch thread, the RTX 5090 is **slower at every training-set
size measured, 8 through 2048 rows** — there is no crossover. A crossover *does* appear
against the CPU path as shipped, bracketed between **n = 96 and n = 128**, and it is not
a GPU win: it is the point where torch starts parallelising a 12→32→32→1 net across all
24 cores and the thread-barrier overhead makes the same fit **53x slower** (0.348 s at one
thread against 18.510 s as shipped). CUDA context init, which I expected to be the
decisive per-process tax, is a fifth of a second and turned out not to matter.
**Recommendation: do not add a device option. Pin torch to one intra-op thread instead.**

![fit time against dataset size, both devices](./2026-08-06-sampler-gpu-crossover.png)

![where a real run's wall clock goes](./2026-08-06-sampler-gpu-fit-share.png)

## Question / goal

`MlpBinaryClassifier` is pure CPU — no `.cuda()`, no `.to(device)`, no device handling
anywhere in the sampler stack. The machine has an idle RTX 5090 and a CUDA-enabled torch.
**Is moving the sampler's classifier to the GPU worth doing?** Two sub-questions, in the
order that decides the answer:

1. What fraction of a real run is actually spent inside `fit()`? If it is 2%, nothing
   else matters.
2. If it is large, does the GPU make `fit()` faster, and above what dataset size?

## Background

`methods/practice_makes_perfect/wrapped_sampler.py`'s `MlpBinaryClassifier` is the ported
`MLPBinaryClassifier` stack from predicators: min/max input normalization, a single-class
shortcut, and a 32×32 ReLU net — 1,505 parameters at input dim 12 — trained **full batch** with Adam
on binary cross-entropy. `LearnedSkillSampler.fit` refits it *from scratch* on every
observation ever made, once per skill per learning cycle, which is what
`_ClassifierWrappedSamplerLearner._learn_nsrt_sampler` does. `EesMethod._refit_samplers`
passes `max_train_iters=EesMethod.sampler_max_train_iters`, whose default is predicators'
own `10000`.

Two facts from earlier work bound the design space here:

- **Iteration count is results-affecting**, so it cannot be traded for speed.
  `docs/experiment-logs/2026-08-03-ballring-iters.md` measured that raising it from 10,000
  to 100,000 drops held-out argmax success from 0.988 to 0.930 (paired, t = 5.67, 10/10
  seeds) while train BCE falls 5.9e-3 → 2.8e-5. It is therefore held **fixed** inside
  every comparison below, and only the device varies.
- **`stats.json` byte-stability is how this project verifies a change did not alter
  results** (`tests/scripts/test_reproducibility.py`). Any speed change has to survive
  that, and cuDNN kernels are not bit-identical to CPU ones — which is a strong prior
  against a GPU path before any timing is done.

Nothing in the repo calls `torch.set_num_threads`, so every run takes torch's default:
one intra-op thread per core, 24 on this box. `scripts/run_sweep.py` then launches one
process per seed, each with its own 24-thread pool.

## Hypothesis

Recorded before measuring, and reproduced here unedited:

> 1. The GPU will be SLOWER than CPU at every n in {8..2048} at input dim 11-12 … I
>    predict NO crossover below n = 2048. If a crossover exists at all I expect it above
>    n ~ 10^4-10^5 rows, which no real run reaches. Additional per-process cost: CUDA
>    context init ~0.5-2 s, paid once per process.
> 2. I expect fit() to be a MINORITY but non-trivial share of a tossingroomsplit EES run
>    — my guess is 5-25% of wall clock.
> 3. I predict the answer is "do not implement a device option".

**Two of the three were wrong.** (2) badly — fit is the *majority* of a run, not a
minority. (1) partly — a crossover does exist, and far lower than predicted, though not
for the reason a crossover would normally mean. CUDA init was overestimated by ~10x. Only
(3), the conclusion, survived, and it survived for a different reason than the one that
motivated it.

## Guidance given

Josh's brief, condensed: measure the end-to-end share **first**, because a micro-benchmark
showing a 10x fit speedup that moves total runtime by 1% is academic. Report the crossover
n, not the ratio at one size, since the crossover is the transferable answer. Treat CUDA
context init as its own line item because it is paid per process and `run_sweep` is one
process per seed. Hold iteration count fixed. Warm up, repeat, report medians,
`torch.cuda.synchronize()` before stopping any GPU timer. Implement a device option
**only** if (1) and (2) justify it; a null result is a good outcome and saves carrying an
abstraction for nothing. Cap memory, budget concurrency — three other agents were running.

## Methods

Everything ran on the shared 24-core box with an RTX 5090, torch 2.13.0+cu130, inside
`systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0 -p OOMPolicy=continue`.

**End-to-end share.** `scripts/profile_sampler_fit_share.py` wraps
`MlpBinaryClassifier.fit` and `.predict_proba` in timers and then runs the ordinary CLI —
so the run measured is the run the CLI would otherwise have done. One seed:
`--env tossingroomsplit --method ees --seed 0 --num-cycles 25
--max-steps-per-interaction 100`.

**Micro-benchmark.** `scripts/bench_sampler_device.py`, medians of 3 timed repetitions
after 2 warmup repetitions, input dim 12 (measured, not assumed: every classifier row in
the profiled run was 12 wide), 1000 iterations, n from 8 to 2048. The CPU arm calls the
shipped class unmodified; the GPU arm is `CudaMlpBinaryClassifier`, which overrides
`_train` and `predict_proba` only, and which `tests/scripts/test_bench_sampler_device.py`
pins to agree with the CPU arm on the fitted probabilities to 1e-4. Early stopping is
disabled in both arms (`n_iter_no_change = 10**9`) so both execute exactly 1001
iterations — a device-dependent early stop would confound iteration count with device.
`torch.cuda.synchronize()` brackets every GPU timing.

The grid was run at 1000 iterations while real runs use 10,000. The two are consistent:
the shipped path at n = 16 costs 0.321 s per 1000 iterations here, and the profiled run's
median refit at a comparable n was 3.33 s at 10,000 — a factor of 10.4, i.e. the loop is
linear in iteration count and the grid transfers. 1000 was chosen so that the pathological
CPU arm, which reaches 101 s per 1000 iterations at n = 2048, stayed affordable on a
shared box.

**Threads.** A third arm re-runs the identical CPU grid with `torch.set_num_threads(1)`.
This is a *separate arm*, never pooled with the default CPU arm — the driver's own
docstring records that *calling* `set_num_threads` is not a no-op even when passed the
value already in effect. A separate sweep varies only the thread count at fixed n = 16.

**CUDA context init.** Measured in a **fresh process per sample**, five samples, since it
is a per-process lazy initialisation that reads zero on a second measurement in the same
process.

**Byte-reproducibility.** Two separate checks. Seed 0 was re-run through a wrapper that
calls `torch.set_num_threads(1)` before `Cli.main`, with no change to `src/`, and its
`stats.json` compared byte-for-byte against the baseline run's. Separately,
`--mode bitwise` fits the same data with the same seed on both devices and compares the
resulting score vectors for exact equality.

## Results

### 1. `fit()` is 154.4/188.2 seconds of a real run

47 refits in one 25-cycle run, of which **46/47 actually trained** (1/47 took the
single-class shortcut, which returns without building a net). Median trained refit: 3.33 s
(min 3.15, max 4.45). Training sets were small and 12 columns wide: **2 to 41 rows**,
median 15.5. `predict_proba` — 235 calls — cost **0.040 s in total**, four thousandths of
the fit cost, so essentially all of it is training and none of it scoring.

Two consequences. A `fit()` speedup does move total runtime, so the question was worth
asking. And the real training sets a 25-cycle run reaches (≤41 rows) sit *below* the
threshold where anything interesting happens.

### 2. There is no crossover against a one-thread CPU

Median seconds per `fit()`, input dim 12, 1000 iterations, 3 repetitions:

| n rows | CPU as shipped | CPU, 1 thread | CUDA (RTX 5090) |
| ---: | ---: | ---: | ---: |
| 8 | 0.308 | 0.332 | 0.704 |
| 16 | 0.321 | 0.334 | 0.613 |
| 32 | 0.323 | 0.338 | 0.647 |
| 48 | 0.312 | 0.338 | 0.606 |
| 64 | 0.324 | 0.340 | 0.696 |
| 96 | 0.324 | 0.342 | 0.609 |
| 128 | **18.510** | 0.348 | 0.629 |
| 192 | **21.534** | 0.364 | 0.642 |
| 256 | **21.171** | 0.359 | 0.836 |
| 512 | **34.688** | 0.382 | 0.833 |
| 2048 | **101.259** | 0.532 | 0.849 |

**Against the one-thread CPU arm there is no crossover at any measured n.** CPU grows
smoothly from 0.332 s to 0.532 s across a 256x increase in rows — the work really is
negligible next to the per-iteration Python and autograd overhead — and stays below the
GPU's 0.606–0.849 s throughout. The GPU line is essentially flat in n, which is the
signature of a launch-bound loop: ~30 tiny kernel launches per iteration, plus a forced
device synchronization every iteration from the `loss.item()` the best-loss checkpoint
needs.

**Against the CPU as shipped there is a crossover, bracketed between n = 96 and
n = 128** — and it is a CPU pathology, not a GPU win. At n = 128 the shipped path is
**53x slower than the same fit at one thread** (18.510 s against 0.348 s), and by
n = 2048 it is **190x slower** (101.259 s against 0.532 s). Below the threshold the two
CPU arms are within 8.6% of each other at every n (largest gap at n = 48, 0.3115 s against
0.3383 s) and I would not claim a difference between them.

### 3. The mechanism is threads, and the penalty scales with how busy the box is

Same fit throughout — n = 16, dim 12, 1000 iterations, medians of 3 — varying only torch's
intra-op thread count. Each row records the 1-minute load average it was taken at, because
that turns out to matter more than anything else here:

| torch threads | median `fit()` | 1-min load at measurement |
| :--- | ---: | ---: |
| 1 | 0.338 | 43.6 |
| 2 | 0.752 | 42.4 |
| 4 | 6.031 | 43.9 |
| 8 | 30.086 | 50.0 |
| 16 | 72.107 | 58.6 |
| 24 | 120.426 | 67.7 |
| shipped default (no `set_num_threads` call) | 0.873 | 66.2 |

![fit time against torch thread count](./2026-08-06-sampler-gpu-threads.png)

Two things follow, and they are the load-bearing part of this write-up.

**One thread is the only setting that is stable.** It cost 0.32–0.34 s in every
measurement I took, at 1-minute loads from 11 to 44. Every setting above one is
contention-sensitive and can degrade without bound: an earlier sweep of the same grid,
taken at load ~15, put 1, 2, 4, 8 and 16 threads *all* at 0.32 s with only 24 threads
degraded (to 12.0 s). The table above, taken while another agent was running 19 concurrent
`hitl_pmp.cli` processes, shows the same sweep an order of magnitude worse from 4 threads
up. **The absolute numbers in this section are therefore not reproducible; the ordering
is.** A sweep is exactly the condition that produces the load, since every one of those 19
processes spawns its own 24-thread pool — 456 threads on 24 cores.

**"24 threads" and "the shipped default" are not the same trigger**, which is why the
table lists them separately. At n = 16, on the same box at the same moment, the default
costs 0.873 s while an explicit `set_num_threads(24)` costs 120.426 s — 138x apart at an
identical thread count on paper: torch does not fan out a matrix this small unless the call
has eagerly created the pool. At n ≥ 128 the default is slow *without* any explicit call,
because the tensors have crossed torch's own parallelisation threshold. Two distinct
mechanisms, one consequence. This experiment establishes that thread count is what drives
the cost; it does **not** isolate which of the two triggers fires in any given real run,
and the results-table figures at n ≥ 128 are the default-path trigger.

**Caveat carried forward:** the grid in result 2 was measured at 1-minute loads of 18–26
(recorded by hand from `/proc/loadavg` during the run, not by the benchmark, which did not
yet record it). So the shipped-CPU arm's 18–101 s figures are contended-machine numbers
and would be smaller on an idle box. The one-thread arm would not change.

### 4. CUDA context init is a fifth of a second per process, not the tax I predicted

Five fresh processes, one measurement each: context creation 0.135–0.230 s, first cuBLAS
matmul 0.065–0.068 s, so about 0.20–0.30 s per process in total. An earlier set of five, taken at much lower load, gave 0.077–0.088 s
and 0.060–0.063 s. Either way it is real but an order of magnitude below my 0.5–2 s prior,
and it does not change any conclusion here. Reported because I said I would measure it
separately, and because the prediction was wrong.

### 5. Byte-reproducibility: the thread fix survives it, a GPU path cannot

**The one-thread CPU path is byte-identical.** Seed 0 re-run with
`torch.set_num_threads(1)` produced a `stats.json` with the same md5 as the baseline run's
(`906d3930187d3cb8d3cecb5e9ac57ac1`), so `tests/scripts/test_reproducibility.py`'s
guarantee is untouched by the change recommended below.

**A GPU path could not be.** Fitting the same data with the same seed on both devices and
comparing the resulting scores: **0/10 seeds bit-identical**, worst absolute score
difference 3.577e-05. That is far coarser than byte-identity, and since those scores feed
an argmax over candidates, some of those differences will flip a choice and change the
run. `stats.json` byte-stability is how this project verifies a change did not alter
results, so this is disqualifying on its own, independently of the timings above.

## Recommendation

1. **Do not add a device option to `wrapped_sampler.py`.** The GPU loses to a single CPU
   thread at every dataset size measured, the real datasets are 2–41 rows, and a GPU path
   would have to thread the device through `predict_proba` as well as `_train` — the
   benchmark subclass could not get away with overriding one method, which is the cheapest
   possible evidence of how invasive the real change would be. Set against that, it breaks
   `stats.json` byte-stability outright (0/10 seeds bit-identical) for no speed.
2. **Pin torch to one intra-op thread**, as its own PR. This is the finding worth acting
   on. It is a no-op for a 25-cycle run (≤41 rows, already below the threshold), but a
   250-cycle run reaches several hundred rows per sampler and crosses it, at which point
   every refit costs tens of seconds instead of a third of a second. That PR should carry
   the byte-identical `stats.json` check as its evidence, and should decide deliberately
   between `torch.set_num_threads(1)` in the process entrypoint and an `OMP_NUM_THREADS`
   setting in `run_sweep`; this experiment does not settle which.
3. **Do not chase the GPU again without re-reading result 2.** The one number that looks
   like it argues for a GPU — 18.5 s against 0.63 s at n = 128 — is a threading artifact
   on the CPU side. Fix that and the GPU is behind everywhere.

## Reproducing

```bash
# 1. End-to-end share of one real run.
scripts/with_env.sh python scripts/profile_sampler_fit_share.py \
  --profile-out fitshare.json \
  --env tossingroomsplit --method ees --seed 0 \
  --num-cycles 25 --max-steps-per-interaction 100 --output-dir run/

# 2. The device x n grid, two arms.
scripts/with_env.sh python scripts/bench_sampler_device.py \
  --out bench-default.json --reps 3 --dims 12 --iters 1000 \
  --ns 8 16 32 48 64 96 128 192 256 512 2048
scripts/with_env.sh python scripts/bench_sampler_device.py \
  --out bench-threads1.json --reps 3 --dims 12 --iters 1000 --devices cpu --threads 1 \
  --ns 8 16 32 48 64 96 128 192 256 512 2048

# 3. Thread sweep at fixed n. One process per setting, since set_num_threads is
#    process-global and sticky; omit --threads for the shipped default.
for T in 1 2 4 8 16 24; do
  scripts/with_env.sh python scripts/bench_sampler_device.py \
    --out threads-$T.json --reps 3 --dims 12 --iters 1000 --ns 16 --devices cpu --threads $T
done
scripts/with_env.sh python scripts/bench_sampler_device.py \
  --out threads-default.json --reps 3 --dims 12 --iters 1000 --ns 16 --devices cpu

# 4. CUDA context init. One FRESH process per sample -- a second call in the same
#    process reads zero.
for i in 1 2 3 4 5; do
  scripts/with_env.sh python scripts/bench_sampler_device.py --mode cuda-init --out init-$i.json
done

# 5. Bit-identity of CPU- vs GPU-fitted scores.
scripts/with_env.sh python scripts/bench_sampler_device.py --mode bitwise --out bitwise.json

# 6. Byte-reproducibility of the one-thread CPU path: run seed 0 twice, once through a
#    wrapper that calls torch.set_num_threads(1) before hitl_pmp.cli.Cli.main, then
#    md5sum both stats.json.

# 7. The three figures.
scripts/with_env.sh python -m analysis.practice_makes_perfect.sampler_device_bench \
  --bench-json bench-default.json bench-threads1.json \
  --out docs/experiment-logs/2026-08-06-sampler-gpu-crossover.png
scripts/with_env.sh python -m analysis.practice_makes_perfect.sampler_fit_share \
  --profile-json fitshare.json \
  --out docs/experiment-logs/2026-08-06-sampler-gpu-fit-share.png
scripts/with_env.sh python -m analysis.practice_makes_perfect.sampler_thread_sweep \
  --threads-json docs/experiment-logs/2026-08-06-sampler-gpu-threads.json \
  --out docs/experiment-logs/2026-08-06-sampler-gpu-threads.png
```

Every number in this page has its recording committed alongside it:
`2026-08-06-sampler-gpu-bench-default.json`, `-bench-threads1.json`, `-threads.json`,
`-cuda-init.json`, `-bitwise.json` and `-fitshare.json`. All three figures regenerate
from exactly those files.

**Provenance, stated plainly.** The device grid (result 2) and the fit-share profile
(result 1) were collected with this same benchmark and profiler logic *before* it was
tidied into `scripts/bench_sampler_device.py` and `scripts/profile_sampler_fit_share.py`,
so two things about those two files differ from what the committed scripts would write
today: their top-level metadata keys were renamed on the way in, and the fit-share file
predates the `trained_fit_count` field (that count, 46/47, is recomputed from its `fits`
list). The timing code and the per-row schema are unchanged. The committed grid script was
re-run afterwards as a check: n = 16 gave 0.308 s CPU / 0.605 s CUDA against the recorded
0.321 / 0.613, and n = 96 gave 0.323 / 0.620 against 0.324 / 0.609. Results 3, 4 and 5
were collected with the committed script and need no such caveat.
