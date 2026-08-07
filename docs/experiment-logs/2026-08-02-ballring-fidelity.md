# Ball-Ring: three environment-fidelity fixes, and a null result

An audit of our Ball-Ring port against `../hitl-practice` (predicators) found three
places where the port diverged from the reference. Two are real defects that corrupted
skill competence signals; one is latent but wrong, in shared `planning/` code. All three
are fixed here.

**The headline metric did not move.** Over 10 seeds the final evaluation success rate
went from 98% to 99% — a paired *p* of 0.59. This is a correctness/fidelity change, not
a performance one, and the section at the end says plainly what that does *not* explain.

---

## Fix 1: floor-placement jitter

predicators' `place_on_floor_sampler`
(`predicators/ground_truth_models/ball_and_cup_sticky_table/nsrts.py:287-303`) scatters
the placed object in a disk about the room centre:

```python
obj_to_place = objs[-1]
size = state.get(obj_to_place, "radius") * 2
dist = rng.uniform(0, size)
theta = rng.uniform(0, 2 * np.pi)
x = x_c + dist * np.cos(theta)
```

Our port returned the room centre exactly. So floor-placing the cup and then the ball put
**both objects at the identical point** (0.5, 0.5).

`BallInCup` is `Circle.contains_circle`, i.e. `dist(centres) + ball_radius <= cup_radius`.
At the paper config (`num_tables=5`) the numbers are:

| quantity | value |
|---|---|
| `size` (`table_radius * objs_scale`) | 0.024491 |
| `cup_radius` (`size * 1.05`) | 0.025716 |
| `ball_radius` (`size * 0.95`) | 0.023266 |
| **tolerance** (`cup_radius - ball_radius`) | **0.002449** |
| cup jitter disk radius (`2 * cup_radius`) | 0.051431 |
| ball jitter disk radius (`2 * ball_radius`) | 0.046533 |

Two coincident centres are trivially within 0.002449 of each other, so `BallInCup` held
with probability **1**. `PlaceBallOnFloor`'s add-effects are `BallNotInCup` and
`BallOnFloor`, so it recorded a **failure on essentially every attempt** — a competence
curve for that skill driven entirely by a geometry bug.

The three floor-place skills (`PlaceBallOnFloor`, `PlaceCupWithoutBallOnFloor`,
`PlaceCupWithBallOnFloor`) go `param_dim=0 -> 2`.

> **Note added 2026-08-06 — those three parameters are never learned.** Nothing above is
> retracted; this log's numbers stand and its headline is an explicit null result. But a
> reader can reasonably take "goes `param_dim=0 -> 2`" to mean EES now practises these
> skills, and measurement says it does not.
> [`2026-08-06-practice-target-audit.md`](2026-08-06-practice-target-audit.md) finds that
> over 10 seeds `PlaceBallOnFloor` (231/231) and `PlaceCupWithBallOnFloor` (150/150)
> succeed on every attempt, so `skip_perfect` drops every grounding of them from the
> practice-target list in **10/10 seeds**, and their samplers return an informed draw
> **0/231** and **0/150** times; `PlaceCupWithoutBallOnFloor` is never executed at all.
> That is most likely correct rather than a defect — the add-effects are insensitive to
> where on the floor the object lands, so there is nothing for practice to improve — and
> no conclusion in this log or any other depends on those three being learned.

### Measured, through the real path

Driven end-to-end through `BallRingSkills.compute_action -> BallRingEnvironment._simulate`
(never a reimplementation of the sampler), floor-placing the cup and then the ball:

| arm | n | `BallInCup` rate |
|---|---|---|
| before (pinned to the room centre) | 2,000 | **100.000%** (2000/2000) |
| after (jitter restored) | 120,000 | **0.573% ± 0.022pp** (688 hits) |

**Why the real-path number can be trusted as the geometric one.** In every one of those
trials the placed object's final `(x, y)` was *bit-exactly* the point the action
commanded, and `env.ball_in_cup` agreed with `separation <= tolerance` on all 30,000
trials of a checked subsample (155 hits, 155 agreements). Nothing in `_simulate` —
not `place_smooth_fall_prob`, not `place_ball_fall_prob`, not
`sample_floor_point_around_table` — perturbs a floor placement. So the real path *is* the
geometry, and the geometry can be solved directly rather than sampled.

### The geometric prediction, derived independently

Both points are drawn radially uniform (`dist ~ U(0, R)`, `theta ~ U(0, 2*pi)`) — **not**
area-uniform, so the density goes as 1/r and concentrates near the centre. This is the
step most likely to be got wrong by eye. Conditioning on the two radii, the angle between
them is uniform, so

```
P(separation <= tol | r_b, r_c) = arccos(C)/pi,   C = (r_b^2 + r_c^2 - tol^2) / (2 r_b r_c)
```

clipped at the degenerate `C <= -1` (always) and `C >= 1` (never) cases, then integrated
over `r_b ~ U(0, 2*ball_radius)`, `r_c ~ U(0, 2*cup_radius)`.

| method | result |
|---|---|
| midpoint quadrature, n = 5,000 / 10,000 / 20,000 | **0.6053%** (converged to 4 d.p.) |
| Monte Carlo, n = 4×10⁶ | 0.604% ± 0.004pp |
| real path, n = 120,000 | 0.573% ± 0.022pp |

The real-path measurement sits 1.5σ from the converged analytic value. There is no
discrepancy to explain.

### Correcting two circulated figures

- **"0.545%"** — the earlier measurement. It was a 3,000-trial estimate, i.e.
  0.545 ± 0.14pp. It is a perfectly ordinary draw from a 0.605% process; it was never
  in tension with anything. The same goes for the "geometry-only prediction of 0.618%",
  which was a 400,000-draw Monte Carlo (±0.012pp), 1.1σ from the converged value.
  **Report these with their error bars, not as bare decimals.** The converged number is
  0.605%.
- **"~5%" is wrong**, and its provenance is exact: it is the *single*-jitter rate, where
  the cup jitters and the ball stays pinned at the centre. Then `BallInCup` needs
  `d_cup <= tol` with `d_cup ~ U(0, 2*cup_radius)`, giving
  `tol / (2*cup_radius) = 0.1/2.1 = 1/21 = 4.762%`. predicators jitters **both** objects,
  which is what drops it by nearly an order of magnitude.

Cup and ball radii derive deterministically from `num_tables`, not from the layout draw,
so measuring against one sampled initial state is not a single-sample artifact.

---

## Fix 2: the navigation annulus

`navigate_action` probed 24 angles at a **single** distance,
`radius + 0.5 * (reachable_thresh - radius)`, and on total failure returned an
**unchecked, colliding** pose. `_simulate` rejects a colliding pose, so `NavigateTo*`
became a silent no-op that recorded a failure predicators cannot produce — its
`navigate_to_obj_sampler` (`nsrts.py:459-489`) is a `while True` rejection loop over
`dist ~ U(size, reachable_thresh)` that never gives up.

The single distance genuinely fails whenever the target sits on a table: every angle at
the annulus midpoint is still inside the table's own circle, so all 24 candidates
collide.

Measured over 300 sampled initial states × 7 targets (5 tables + ball + cup), with the
old scan reconstructed in-process so both arms run against bit-identical states, and a
no-op defined **behaviourally** (`take_action` leaves the robot's `(x, y)` unchanged):

| scan | silent no-ops | which targets |
|---|---|---|
| old, single distance | **153 / 2100** | all 153 the ball, on its start table |
| new, annulus fractions `(0.5, 0.7, 0.85, 0.95, 0.99)` | **0 / 2100** | — |

0.5 is tried first, so **1947 of 2100 target poses return a bit-identical action** —
exactly the 2100 − 153 that already worked. The fix is strictly additive; no previously
working navigation moved.

(The outermost fraction stops at 0.99, never 1.0: a pose at exactly `reachable_thresh`
can round to just over it and read as unreachable.)

---

## Fix 3: repeated-object groundings

`SkillGrounder.abstract_state` carried `if len(set(combo)) != len(combo): continue`, but
predicators' `abstract()` documents *"Duplicate arguments in predicates are allowed."*
(`predicators/utils.py:2697`) and its `get_object_combinations` applies no distinctness
filter. Our own `SkillGrounder` class docstring already claimed it "deliberately does NOT
force distinct objects" — the class contradicted itself.

**It was not dead code, and the first draft of this fix's docstring got the reason
wrong.** Same-type-twice predicates *do* exist here — Light Switch's
`Adjacent(cell, cell)` and Tossing Room's `Adjacent(room, room)` / `CanMoveRoom(room,
room)`. The filter was *inert*, not unreachable: none of those relations ever holds
reflexively, so the combinations it skipped are exactly the ones `predicate.holds` would
have rejected anyway. Verified by enumeration rather than asserted — over 60 sampled
initial states per domain, dropping the filter changes the abstraction of **0 states in
all three domains**, and produces no extra atoms anywhere. Ball-Ring has no
same-type-twice predicate at all.

So nothing observable changes today. It is still worth fixing: this is shared `planning/`
code, and the first reflexive relation anyone adds (a `SameRoom`, an `Equal`) would
silently lose its diagonal.

Note that `applicable_ground_skills` was already filter-free and is untouched — which
matters, because Tossing Room's `MoveRoom(robot, from_room, to_room)` and `Press` both
take two same-type parameter slots.

---

## The fixes broke a test — for an interesting reason

`tests/environments/test_operator_dynamics_fidelity.py` is the cross-domain walk that
checks a symbolic operator model never permits more than the dynamics allow. It has a
**coverage floor**: every declared skill must be enumerated somewhere, so the property
test cannot pass vacuously.

After the jitter fix, that floor failed on Ball-Ring: the walk never reached a state
where `PlaceBallInCupOnFloor` was applicable. **The coverage floor had been passing
because of the bug.**

The mechanism. `PlaceBallInCupOnFloor` needs `HoldingBall` and `IsReachableCup` believed
at once. Every ball-acquiring skill requires `IsReachableBall`/`IsReachableSurface`, and
every `NavigateTo*` wipes all three reachability atoms via `ignore_effects`, so the only
believed route is a `NavigateToCup` executed *while already holding the ball*. With both
floor placements pinned to the room centre, a robot standing next to the floor cup was
also standing next to the ball — `IsReachableCup` came **free** at pick time and the walk
reached the state 12 times without ever making that navigation. Restoring the jitter
correctly removed the coincidence:

| | before fix | after fix |
|---|---|---|
| believed `PlaceBallInCupOnFloor` applicability | 12 | **0** |
| `PickCupWithBallFromFloor` executions | 12 | 3 |
| `PickBallFromFloor` executions | 11 | 3 |

This is structural, not stochastic: 8→16 walks, 40→80 steps, and oracle horizon 12→24 all
left the count at **0**. Per-walk execution-count resets made it *worse* (two skills lost
instead of one).

The skill itself is unaffected — a direct 20-trial probe (floor-place the cup, hold the
ball, `NavigateToCup`, `PlaceBallInCupOnFloor`) found it applicable and achieving
`BallInCup` **20/20**. And the navigation involved is *not* covered by
`_BALL_RING_NAVIGATION_TO_AN_OBJECT_OFF_THE_FLOOR`, which is scoped to the cup being off
the floor; here the cup is on the floor and the navigation genuinely works. So this was a
**reachability gap in the walk's search heuristic**, never a licensed no-op.

**The fix**: the walk's advance rule was "take the least-executed applicable skill". It is
now "advance toward the least-executed skill reachable within one more step" — one-step
lookahead over the symbolic model only, no environment execution, no extra trials. A
binary "does this unlock a never-executed skill?" flag was tried first and is too greedy:
it chases each new skill the instant it appears and abandons the rest, losing coverage of
`PickBallFromFloor` and `PickCupWithBallFromFloor`.

The new rule is also far less knife-edge than the old one. Across 6 perturbations of the
walk parameters (6/8/12 walks, 30/40/60 steps, oracle horizon 8/12, oracle length 8/12/24)
all three domains report `missing=[]` and `violations=0` every time.

**The detector was re-verified, not just observed to be quiet.** `missing=[]` and
`violations=0` are *negative* results: they say the test does not fire, not that it still
can — and the change makes the walk deliberately less undirected, which is precisely what
the module docstring credits with finding bug #1. So both bugs this file exists to catch
were reconstructed in-process (no checkout) and re-run under the **new** rule:

| reconstructed model | violations |
|---|---|
| pre-#27 Ball-Ring: `ignore_effects` dropped from every `NavigateTo*` | **656** |
| pre-#28 Tossing Room: `PileInRoom` dropped from `Pickup` | **388** |
| both current models (control) | 0 |

The new advance rule still catches both. (This is what the Tossing Room log meant by "a
green assertion that can never fire is worse than no assertion" — after changing a
detector, confirm it still detects.)

---

## Result: a clean null

10 seeds, Ball-Ring, 10,000 sampler iterations, fixed test set, arms run sequentially on
one base. Final evaluation sweep at 2,500 transitions:

| arm | final mean % | sd | worst seed |
|---|---|---|---|
| `ignore_effects` only (= current `main`) | 98 | 4.2 | 90 |
| **+ these fidelity fixes** | **99** | **3.2** | **90** |
| *predicators (reference)* | *91* | *12.0* | *70* |

Per-seed differences: `[0, 0, 0, +10, 0, +10, 0, -10, 0, 0]` — three seeds moved, each by
a single task out of ten, two up and one down.

The design is **paired** (the same 10 seeds in both arms), so the paired test is the right
one:

- **paired t-test: t = 0.557, df = 9, mean difference +1.0 points, p = 0.59**
- unpaired pooled two-sample t-test: t = 0.600, df = 18, p = 0.56

The previously circulated *p* = 0.55 was the unpaired figure. Either way this is a null:
**+1 point, and we cannot distinguish it from zero.**

Raw data: `fix-results/{ignore_effects,envfix}/ees/*/stats.json` and
`fix-results/predicators-ballring-25cyc.json` (not in this repo).

---

## What this does NOT explain

We score **99%** where predicators' own reference runs score **91%**. The prediction going
in was that these fidelity fixes would pull us *down* toward the reference — that our
Ball-Ring was simply easier than predicators'. **That prediction was wrong.** The fixes
moved the number a point in the *other* direction and did not close the gap at all.

So *"our Ball-Ring is easier than predicators'"* is **not** the explanation for us
exceeding the reference. Two of the three fixes made the domain strictly harder (a free
`BallInCup` removed; a silent navigation no-op that had been costing real failures), and
the success rate still did not fall.

The remaining hypothesis — that predicators' own 91% is *depressed* by wall-clock planning
timeouts in its parallel reference runs, rather than our 99% being inflated — is being
tested separately. Nothing here bears on it either way.

---

## Reproducing

The durable artifacts are committed tests, not scripts:

- `tests/environments/ballring/test_skills.py::test_floor_placing_the_cup_then_the_ball_rarely_produces_ball_in_cup`
  — measures the `BallInCup` rate through the real path and checks it against an
  independently computed geometric prediction.
- `tests/environments/ballring/test_skills.py::test_navigate_never_falls_back_to_an_unchecked_colliding_pose`
  and `::test_navigate_still_prefers_the_midpoint_of_the_annulus` — the annulus scan
  always finds a pose, and previously-working navigations are unchanged.
- `tests/planning/test_grounding.py` — a repeated-object grounding is produced.
- `tests/environments/test_operator_dynamics_fidelity.py` — the cross-domain walk,
  including the coverage floor discussed above.

The 300-state navigation sweep, the 120,000-trial `BallInCup` run, the quadrature, and the
predicate-arity enumeration were one-off scratch measurements, reported here rather than
committed as scripts.
