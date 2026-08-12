# KINDER Tossing3D: what actually works at upstream `main`

**Written 2026-08-05.** This is a validation record, not an integration. Its job was to
answer, with evidence, whether the `Tossing3D` environment behaves the way seven closed
pull requests assumed it did — before any of that work was reconsidered. **Nothing on
`main` imported KINDER when this was written; `--env tossing3d` has since landed (#77) and
does, lazily.** Its numbers were checked against this file.

Every claim below names the command that produced it. Where a previously-recorded
claim did **not** reproduce, this file says so in bold rather than repeating the old
number.

**Audited 2026-08-06** against `main` at `db2589f`. §1, §2, §3 and §5 reproduce unchanged;
**§4 no longer holds at upstream `main`** — see the note there.

## What was validated

| repo | remote | branch | commit validated |
| --- | --- | --- | --- |
| `reference/kindergarden` | `Princeton-Robot-Planning-and-Learning/kindergarden` | `main` | `cdf1b8ba0ed0d4fbf0390e336bea748e83d517d5` |
| `reference/kinder-baselines` | `Princeton-Robot-Planning-and-Learning/kinder-baselines` | `main` | `4c731dc81d68ee6888ef3a989034991cd0694630` |

> **Note, 2026-08-07.** The `remote` column above is a record of where these checkouts
> pointed when the validation ran, not where they point now. Both are **git submodules**
> pinned to fork branches today — `joshnroy/kinder-baselines` @ `11eace5` and
> `joshnroy/kindergarden` @ `4113237`. The measurements below are left exactly as
> published; only the pointer is out of date. `scripts/update_reference_repos.sh --check`
> prints the current pins.
>
> **Note, 2026-08-12.** The `kinder-baselines` pin has since moved `11eace5` → `3524010`
> (a rebase of the same branch onto upstream `main` @ `4760956`). That is not only a
> pointer change: `4760956` is upstream PR #103, which **turns base-motion
> collision-checking on** — it was hardcoded off for every measurement recorded on this
> page. The measurements below are still left exactly as published, but any of them that
> depended on base motion planning should be treated as **provisional** until re-measured
> at the new pin. Nothing on this page has been recomputed.
>
> **Second note, 2026-08-12.** The `kinder-baselines` pin has moved again, `3524010` →
> `1b564a1` (`joshnroy/kinder-baselines` PR #8), a clean +3-commit fast-forward that makes
> the toss's release speed a parameter. **This one changes no default behaviour**: a caller
> that passes no release speed gets exactly the `(140, 300, 200)` deg/s profile the old
> inline literals produced, asserted against the pinned checkout by
> `tests/environments/tossing3d/test_kinder_pin.py`. So it does not add to the provisional
> status above. What it *does* mean is that every measurement on this page was taken when
> the release speed could not be selected at all — each is evidence about **140 deg/s and
> no other speed**. Nothing on this page has been recomputed.

Both SHAs were read back from the checkouts rather than assumed, and both were
confirmed to be the current tip of their remote's default branch **at the time of
writing**:

```
$ git ls-remote https://github.com/Princeton-Robot-Planning-and-Learning/kindergarden.git refs/heads/main
cdf1b8ba0ed0d4fbf0390e336bea748e83d517d5	refs/heads/main
$ git ls-remote https://github.com/Princeton-Robot-Planning-and-Learning/kinder-baselines.git refs/heads/main
4c731dc81d68ee6888ef3a989034991cd0694630	refs/heads/main
```

**`kinder-baselines` has since moved.** Its `main` is `9512b9edbdd17ecfd7d9a6350a9408ade94d4bad`
as of 2026-08-06 — PR #87, the PyBullet leak fix, which is why §4 no longer reproduces.
`kindergarden` `main` is still `cdf1b8b`, so §1, §2, §3 and §5 are unaffected.

**`cdf1b8b` is one commit ahead of the `39eb7e08` that the earlier Tossing3D work
pinned** (`docs/tossing3d-integration-status.md` §4). The intervening commit changes
cluttered-retrieval initial-state sampling, which does not touch Tossing3D — but every
number in this file is measured at `cdf1b8b`, not at the pin, and should not be
compared against a `39eb7e08` measurement without checking that first.

Runs were executed from the `reference/` install (`kinder.__file__` and
`kinder_models.__file__` both resolve under `reference/`), inside
`systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0 -p OOMPolicy=continue`.
The cap was verified real before trusting any run:

```
$ systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0 -p OOMPolicy=continue -- cgcheck.sh
memory.max      = 8589934592
memory.swap.max = 0
```

This matters: `systemctl --user show -p DefaultOOMPolicy` reports `DefaultOOMPolicy=stop`
on this machine, so an unbounded run that trips the kernel OOM killer tears down the
whole user session, not just the run.

---

## 1. Does `kinder.make("kinder/Tossing3D-o1-v0", ...)` construct and `reset()` cleanly?

**Yes — but only if a `kinder.envs.dynamic3d` *module* is imported before
`register_all_environments()`. Importing the package is not enough, and the failure is
silent.**

This is a correction to the guidance this task started from, which said to import
`kinder.envs.dynamic3d`. That specific import does **not** work.

`register_all_environments()` sets `MUJOCO_GL=osmesa` when `DISPLAY` is unset
(`src/kinder/__init__.py:67-74`), *before* it probes its backends. Under `osmesa`,
`import mujoco` raises — this machine has no OSMesa libraries:

```
$ python osmesa_check.py osmesa
  File ".../mujoco/gl_context.py", line 38, in <module>
    from mujoco.osmesa import GLContext as _GLContext
AttributeError: 'NoneType' object has no attribute 'glGetError'
MUJOCO_GL=osmesa: import mujoco FAILED -> AttributeError: 'NoneType' object has no attribute 'glGetError'

$ python osmesa_check.py egl
MUJOCO_GL=egl: import mujoco OK
```

`_check_deps` (`src/kinder/__init__.py:34-46`) catches **every** exception and returns
`False`, so the whole `Dynamic3D` category is skipped without a word, and the failure
surfaces much later as a registry miss:

```
gymnasium.error.NameNotFound: Environment `Tossing3D-o1` doesn't exist in namespace kinder.
```

The fix is to get `mujoco` into `sys.modules` *before* the call, while `MUJOCO_GL` is
still `egl`. A/B over four import orderings, all with `MUJOCO_GL=egl` set at the top of
the process and `DISPLAY` unset:

| pre-import | `mujoco` in `sys.modules` before register | `Tossing3D-o1` registered |
| --- | --- | --- |
| none | False | **False** |
| `import mujoco` | True | True |
| `import kinder.envs.dynamic3d` (the package) | False | **False** |
| `import kinder.envs.dynamic3d.envs` (a module) | True | True |

`MUJOCO_GL` reads `osmesa` immediately after `register_all_environments()` in all four
cases, which is why it must be set back to `egl` afterwards as well.

With the corrected ordering, construction and reset are clean — `make_ok: true`,
`reset_ok: true` on every seed tried (10/10 across all runs below), and
`_check_goals()` is `false` at reset, as it should be.

**One-time cost not previously recorded:** the first `reset()` against a fresh
checkout auto-downloads ~1 GB of MimicLabs scene assets from Google Drive into
`src/kinder/envs/dynamic3d/models/assets/mimiclabs_scenes`. It is automatic and
idempotent, but it takes a few minutes and needs network.

---

## 2. Does the oracle skill sequence run end to end, and does `_check_goals()` return true for a successful throw?

**Yes to both — and the two answers come apart in a way that is the whole story of this
domain.**

The sequence used is upstream's own, from
`kinder-models/tests/dynamic3d/tossing/test_tossing_parameterized_skills.py::test_pick_ground_toss`:
`pick_shelf` → `move_to_target(bin_0, standoff)` → `move_arm_to_conf(pre-toss)` → `toss`.
Deliberately upstream's parameters, not this repo's oracle constants, so the result does
not depend on any closed branch.

**Every skill terminated on every seed: 10/10.** No hangs, no controller that ran out
of steps.

But the *scoring* splits cleanly on one parameter — `move_to_target`'s standoff
distance, which is how far the base stops from the bin:

| standoff | seeds | cube comes to rest at x | rest z | `_check_goals()` true |
| --- | --- | --- | --- | --- |
| **1.35** (upstream's own test value) | 0, 1, 2, 3, 4, 125 | 2.216, 2.216, 2.216, 2.181 (+2 off-nominal) | ~0.0444 | **0/6** |
| **1.55** | 0, 2, 4, 125 | 2.021, 1.997, 2.021, 1.999 | ~0.0249 | **4/4** |

Two of the six at standoff 1.35 were off-nominal rather than bin landings: seed 1 went
sideways (`y = -0.495`) and seed 3 never released (`z = 0.395`). The other four landed
at x ≈ 2.18–2.22 with z ≈ 0.0444 — that z is the bin's floor (0.02 m bottom panel plus
the cube's 0.025 m half-extent), i.e. **the cube is sitting in the bin**. All four
scored a failure.

At standoff 1.55 the cube rests at z ≈ 0.0249 — its own half-extent, i.e. **bare
floor** — short of the bin, and all four score a success.

So: `_check_goals()` does return true for a successful throw, and it is not broken. It
is simply satisfied *only* by throws that miss the bin. See question 3.

---

## 3. Is the bin/goal-region mismatch still present at `main`?

**Yes, the mismatch is still there. But the specific framing that the goal box and the
bin "do not overlap" is WRONG, and was wrong in the brief this task started from — they
do overlap, by 6.9 cm on x.** `docs/tossing3d-integration-status.md` §5.3 already had
this right; the overlap is real but nothing can come to rest in the usable part of it.

Both figures read live, not re-derived: the goal box from `Region.bbox` on
`blocks_goal_region`, the bin from the compiled MuJoCo geoms (world-frame AABB over all
5 geoms whose body/geom name contains `bin_0`, via `sim.model.mj_model` /
`sim.data.mj_data`).

| | x | y | z |
| --- | --- | --- | --- |
| `blocks_goal_region` live `Region.bbox` | **[1.8500, 2.1500]** | [-0.1500, 0.1500] | [0.0000, 0.1500] |
| bin footprint, seed 0 | **[2.0808, 2.3808]** | [-0.1496, 0.1504] | [-0.0001, 0.1999] |
| bin footprint, seed 125 | **[2.0801, 2.3801]** | — | — |

`blocks_goal_region` is a single box (`goal_region_num_boxes: 1`), and its bbox is the
task JSON's `[1.90, 2.10]` inflated by `ground_placement_threshold = 0.05` on every side
(`MujocoGround`'s `ground_placement_threshold` and `_create_regions`,
`objects/base.py:840` and `:874-881`) — unchanged at `main`.

**Against the earlier record:** the goal box `[1.8500, 2.1500]` matches exactly. The bin
footprint was previously recorded as `[2.0807, 2.3807]`; measured here it is
`[2.0808, 2.3808]` on seed 0 and `[2.0801, 2.3801]` on seed 125. The bin's settled
position varies slightly with the initial-state RNG, so the old single figure sits
inside the range rather than disagreeing with it. **Report this quantity per-seed; it is
not a constant.**

The consequence, restated correctly:

- The two boxes **overlap on x over [2.0808, 2.1500] — 6.9 cm.** They are not disjoint.
- The bin's near wall occupies roughly x ∈ [2.081, 2.101], so once the cube's 0.025 m
  half-extent is counted, the sliver a cube could actually *rest* in and still score is
  about x ∈ [2.126, 2.150] — under 2.5 cm.
- Nothing lands there. Measured directly: 4/4 in-bin landings at standoff 1.35 rest at
  x ≈ 2.18–2.22, past 2.15, and score **0/4**.
- Meanwhile 4/4 landings on open floor at x ≈ 2.00 score **4/4**.

**A cube that lands in the bin is a scored failure, and a cube that misses the bin and
lands on bare floor is a scored success.** That is reproduced at `main`, from a clean
control pair, and it is the single most misreadable thing about this domain. It is the
claim `CLAUDE.md` cites this file for, and it was still true at `cdf1b8b` on 2026-08-06.

**It has an expiry date.** kindergarden PR #126, "Move the Tossing3D-o1 bin back inside
`blocks_goal_region`" (https://github.com/Princeton-Robot-Planning-and-Learning/kindergarden/pull/126),
is open and would make stock `o1` behave the way this repo's coincident config already
does. When it merges, this section and `CLAUDE.md`'s citation of it both need re-measuring.

---

## 4. Does the PyBullet leak still reproduce at `main`?

**It did at `4c731dc8`, and it does not any more.** `kinder-baselines` PR #87,
squash-merged as `9512b9e` on 2026-08-06, gives `PyBulletSim` a `weakref.finalize` that
disconnects its PyBullet client when the sim is garbage collected, so a sequential run
releases as it goes; `PyBulletSim.close()` (`utils.py:596`) now calls that finalizer rather
than `p.disconnect` directly, and no longer has zero callers. The upstream issue this
section motivated is `kinder-baselines` #88, now closed.

What the fix does **not** remove is the cost of holding many sims alive at once, so the
"run it under a memory cap" conclusion below still stands. The rest of this section is the
measurement as taken at `4c731dc8`, kept because every §5 number in
`docs/tossing3d-integration-status.md` predates the fix.

Run from
`scratch/kinder-pybullet-leak-evidence/repro.py`, 20 iterations, inside the 8 GB scope.

`--mode leak` (upstream's own per-execution `lifted.ground(objects)` pattern):

```
iter 0 : rss=610.8 MB   live_pybullet_clients=1
iter 9 : rss=1932.7 MB  live_pybullet_clients=10
iter 19: rss=3332.3 MB  live_pybullet_clients=20
---- CHECK RESULT ----
iterations run: 20/20
first RSS: 610.8 MB   final RSS: 3332.3 MB
final RSS growth: 2721.5 MB
peak live pybullet clients: 20
FAIL: RSS UNBOUNDED: growth 2721.5 MB over 20 iterations exceeds bound 64.0 MB
FAIL: CLIENTS UNBOUNDED: peak live PyBullet clients 20 exceeds bound 3
```

- **Live PyBullet clients: exactly one more per skill execution, 20/20.** Never released.
- **RSS: 2721.5 MB over 20 iterations ≈ 136 MB per skill execution**, linear, no plateau.
  Consistent with the ~150 MB/execution previously recorded.

`--mode control-close` (same loop, `PyBulletSim.close()` after each iteration) over the
same 20 iterations and the same 310 env steps:

```
final RSS growth: -24.3 MB
peak live pybullet clients: 0
PASS: bounded across all iterations
```

So the client was genuinely leaked and genuinely releasable — the leak was not inherent to
the workload, which is the evidence PR #87 acted on. **Any iterative use of these
controllers must still run under a memory cap**, for the many-sims-at-once cost noted
above.

---

## 5. Does `Tossing3D-o2.json` still ship the bin at x = 2.0 with a `blocks_goal_region` byte-identical to `o1`'s?

**Yes to both.**

```
$ sed -n '23,28p' Tossing3D-o1.json | sha256sum
75424c1fbc20271375b3df12ac7ce984503a968fa5dcdc748bf6e3dd45af5dbe  -
$ sed -n '23,28p' Tossing3D-o2.json | sha256sum
75424c1fbc20271375b3df12ac7ce984503a968fa5dcdc748bf6e3dd45af5dbe  -
```

Same hash, and the full-file diff confirms `blocks_goal_region` is not among the
differing lines. The four differences are the variant description, the bin position, the
extra `cube_1`, and the extra goal atom:

```
$ diff Tossing3D-o1.json Tossing3D-o2.json
4c4
<     "variant_specific_description": "Get one cube into the bin.",
>     "variant_specific_description": "Get two cubes into the bin.",
14c14
<             "ranges": [[2.23, -0.0005, 2.231, 0.0005]],
>             "ranges": [[2.0, -0.0005, 2.001, 0.0005]],
...
```

`o1` puts `bin_init_region` at x = 2.23; **`o2` puts it at x = 2.0**, against a goal
region that is character-for-character the same. The pairing "bin at 2.0, goal region at
1.90–2.10" is therefore something upstream itself still ships, not an invention — which
was the basis of the earlier argument, and it holds at `main`.

---

## Caveats on this machine

**IKFast is a stock build.** `pick_shelf` needs IKFast, which `pybullet_helpers`
compiles from C++ on first use against the three paths `compile.py` defaults to:
`/usr/lib/x86_64-linux-gnu/{libblas.a, lapack/liblapack.a}` and
`libgfortran.so.5.0.0`. All three are now present, from Ubuntu's own packages:

```
$ dpkg -l libblas-dev liblapack-dev libgfortran5 | grep ^ii
ii  libblas-dev:amd64   3.12.1-7ubuntu1       amd64  Basic Linear Algebra Subroutines 3, static library
ii  libgfortran5:amd64  16-20260322-1ubuntu1  amd64  Runtime library for GNU Fortran applications
ii  liblapack-dev:amd64 3.12.1-7ubuntu1       amd64  Library of linear algebra routines 3 - static version
```

The compiled module and its `build/` were deleted and rebuilt from scratch with **no
environment overrides at all** — no `BLAS_DIR`/`LAPACK_DIR`/`LIBGFORTRAN_DIR`, and no
`CC`/`CXX`/`LDSHARED` — just `python setup.py` in the venv, ending `ikfast module
ikfast_kortex imported successful`. Done twice, the two builds are byte-identical
(`sha256 1a1868b258eddd1f…`), so this is reproducible rather than a one-off.

What it links is entirely system libraries:

```
$ readelf -d ikfast_kortex.cpython-310-x86_64-linux-gnu.so | grep NEEDED
 (NEEDED)  Shared library: [libgfortran.so.5]
 (NEEDED)  Shared library: [libstdc++.so.6]
 (NEEDED)  Shared library: [libm.so.6]
 (NEEDED)  Shared library: [libgcc_s.so.1]
 (NEEDED)  Shared library: [libc.so.6]
```

No OpenBLAS, and no `RUNPATH`/`RPATH` at all — the module imports with
`LD_LIBRARY_PATH` unset. BLAS and LAPACK come in *statically* from the `.a` files:
74 Fortran-style symbols are defined in the module's own text (`dgemm_`, `dgetrf_`,
`dgeev_`, …). The linker pulls in only what IKFast references, so `dgesv_` and
`dgesvd_` are **absent** — expect a partial set, not all four.

**The earlier build here was not stock, and §2 re-measured on the stock one is
unchanged.** Before those packages were installed, the same module was pointed —
through `compile.py`'s `BLAS_DIR`/`LAPACK_DIR`/`LIBGFORTRAN_DIR` hooks — at OpenCV's
bundled OpenBLAS 0.3.3 inside the venv's own wheels. `readelf -d` on that build listed
`libopenblas-r0-f650aae0.3.3.so` with no `RUNPATH`, i.e. it depended on the loader
finding a library inside another wheel's private directory. §2 was measured against
it, and §4's leak run executes the same `pick_shelf`. The oracle sequence was then
re-run end to end on the stock build, seed 125: all four skills terminate in **71 / 23 / 16 / 18**
steps, standoff 1.35 leaves the cube at `x=2.2197 y=0.0103 z=0.0444` with
`_check_goals() = False`, and standoff 1.55 at `x=2.0268 y=0.0105 z=0.0249` with
`_check_goals() = True` — §2's bin-floor-versus-bare-floor split, unchanged. That
reproduction, not the link itself, is the evidence nothing here depended on the
substituted libraries. (Resting *x* is sensitive to the `pick_shelf` parameter draw
— a different draw moves it to 2.1888 with the same `z` and the same verdict — so
compare §2's per-seed figures on `z` and the verdict, not on `x` to four places.)

**A previous session's shim is still on disk and must not be reused.** An earlier
attempt at the same problem symlinked `libblas.a`/`liblapack.a` at an OpenBLAS inside a
now-deleted scratch venv (dangling), and supplied a hand-compiled `libgfortran.so.5.0.0`
built from a 1-byte `empty.c` that exports **0 symbols** — verified with `nm -D`. That is
a stub, not a library. It was never used for any number here, and the system packages
above supersede it entirely: there is no longer any reason to point a build at it.

**Run wiring.** The venv lives at `/home/josh/Documents/repos/research/kinder-venv` and
installs editable from `reference/`. It was seeded from conda's `python3.10` (this
machine has no system 3.10), so `sysconfig`'s `CC`/`LDSHARED` still name conda's
`compiler_compat` linker — but with the system BLAS/LAPACK/gfortran packages present
the IKFast build succeeds straight through that, unmodified. Overriding
`CC`/`CXX`/`LDSHARED` to plain `gcc`/`g++`, and putting `opencv_python.libs` and
`numpy.libs` on `LD_LIBRARY_PATH` at import time, were both needed for the earlier
substituted build; neither is needed now.

## What this record does not establish

- **Nothing about whether EES or any method should use this domain.** No learning run
  was made here.
- **Nothing about `Tossing3D-o2`.** Question 5 is a static read of its task JSON; `o2`
  was never constructed or stepped.
- **Nothing about the closed Tossing3D branches.** They were deliberately not read,
  built, or run. Where this file cites `docs/tossing3d-integration-status.md`, it is
  citing the durable record on `main`, and it re-measured rather than trusted it.
