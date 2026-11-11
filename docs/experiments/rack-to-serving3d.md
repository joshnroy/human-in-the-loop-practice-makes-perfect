# Rack-to-serving 3D: implementation investigation

## Status and task story

**2026-11-01 update:** see [physics repair and task-validity audit](rack-physics-validation-20261101.md)
for the missing rolling-contact settings, inherited tray contact-priority issue,
scraper release/follow-through repair, shared settled-success semantics, and
native five-food transfer witness. Earlier failures below remain historical
evidence. The completed eight-trial matrix finds that one constant **0°, 0.06 m/s**
solves all four tested layouts; **0°, 0.03 m/s** already solves the fixed deployment
initialization. Overnight learning remains paused: nontrivial parameter-learning
motivation and full robot-reset coverage are not established. Geometry and the
deployment distribution have not been silently made harder.

This is an isolated implementation investigation, not a validated kitchen workflow
or an overnight-readiness claim. No learning run was relaunched as part of this
work. The existing shelf environment remains separate and unchanged.

The robot transfers a batch of cooked dumplings from a counter-height cooling rack
onto an adjacent flat slate serving tray, using a physical broad bench scraper.
The rack is movable, rather than a shelf fixed to a cupboard. The story concerns
already-cooked food becoming ready for transfer; it is not a cooking task.

- **Training:** a busy human can be interrupted for the supported batch-return or
  tool-recovery operations. Each attempt has a finite, strictly positive cost
  (`human_interruption_cost=5` by default), charged before scene changes, including
  unsuccessful attempts. Help is neither free nor prohibited by an infinite cost.
- **Deployment:** human recovery is unavailable. The robot can wait for readiness
  and must act without human help. Waiting is an available action, not evidence
  that a recovery or transfer will succeed.
- **Readiness:** the configurable elapsed-time ledger defaults to
  `cooling_seconds=300`, `initial_elapsed_seconds=0`, and `wait_seconds=30`.
  Returning a batch to the rack does not restart cooling. This is a task-level
  timing abstraction, not a temperature measurement, thermal simulation, or
  food-safety determination.

## Hardware provenance and distinct variants

The purchase reference is the **Spring Chef “10 x 15” wire cooling rack**, with
**37.5 x 25.5 cm dimensions reported by the user**. The product label and reported
dimensions are recorded as supplied; they have not been independently researched,
converted into a replacement specification, or verified against delivered hardware.
The model assigns 0.375 m to width (world y) and 0.255 m to depth (world x).

**Simulation update (2026-11-11):** the default is now
`surface_kind="wire_solid"`: it renders exactly the same wires and feet, but the
wires are visual-only and one invisible solid slab supplies contact across the
rack footprint. The slab top matches the old wire-top height, its thickness is
the wire diameter, and it uses the same friction and contact dimension. The
feet and counter weld remain. Both MuJoCo and the motion planner use the slab;
no controller or collision allowance changes. Historical wire runs below retain
the original wire collision model and must be replayed with `surface_kind="wire"`.

`surface_kind="wire"` uses individual round wire colliders with open gaps and four
feet. Wire radius, spacing, foot geometry, friction, **rack height (0.025 m)**, and
**rack mass (0.5 kg)** are configurable assumptions. In particular, height and mass
are **UNVERIFIED**, not manufacturer specifications. Measure the purchased item
before interpreting simulation behavior as hardware evidence.

`surface_kind="concept"` is a separate, invented perforated-platform concept with
flat rectangular webs and actual openings. It is not another name for the purchased
wire rack and must not inherit claims about that rack. Neither historical variant contains a
hidden solid supporting plate or a raised front lip.

The scraper defaults to a 0.25 m-wide, 0.12 m-high blade, 0.001 m blade thickness,
and 0.275 kg total mass. Its blade and full-width top grip span local x. These are
modeling assumptions inherited from the prototype geometry, not a measured tool
specification. The food is represented by five rigid ellipsoids, not deformable
dough or filling.

## Layout and physical interpretation

All dimensions are metres. Values below are nominal defaults, not hardware
measurements:

| Quantity | Default |
| --- | --- |
| Counter top | 0.900 |
| Counter footprint | x = 0.98–1.70; y = −0.50–0.50; supported by legs |
| Rack center | x = 1.500; y = 0 |
| Rack front / exit edge | x = 1.3725; exit is toward decreasing x |
| Rack top | 0.925 |
| Slate tray center | x = 1.2125; y = 0 |
| Tray footprint | 0.300 depth (x) × 0.375 width (y) |
| Tray thickness / top | 0.012 / 0.912 |
| Rack-to-tray edge gap | 0.010 |

The default rack top is above the tray top, and the front exit has no modeled
blocking lip. The small gap and relative heights must be measured on the actual
setup, including under load. Runtime transfer checks must use live poses and
measured edges, rather than assuming the nominal 10 mm gap remains unchanged.

The rack has a free joint and rests on the counter through frictional contact.
Scraper contact can therefore translate or rotate it. The slate tray also has a
free joint: it starts on the counter but is not fixed, infinitely heavy, or assumed
ungraspable. Its graspability and whether a robot can safely move it remain
unverified. Repositioning either object changes the transfer geometry.

For compatibility with KINDER SweepOffShelf3D and ShelfSkills, `cupboard_1` supplies
counter-and-leg colliders plus non-colliding shelf metadata sites. There is no
duplicate fixed rack-shelf collider. The rack is the separate native object
`rack_0`; the scraper is `wiper_0`, the tray is `box_0`, and the dumplings are
`cube_0` through `cube_4`. Backend instance overrides measure the live rack/tray
instead of applying the original shelf or enclosing-box interpretation.

## Model limits and required hardware checks

Native rigid contact does not establish realistic food adhesion, deformation,
tearing, or snag behavior. Round wires can make contact with rigid proxies, but
that does not validate whether cooked dumpling skin catches on real wires or
releases cleanly. Friction and mass choices require calibration; visually
plausible motion is not sufficient evidence.

Before any hardware-readiness claim, run supervised tests with the intended
**cooled, cooked dumplings**, actual rack, scraper, tray, and counter:

1. **Release:** measure sticking and release from the rack and scraper across the
   intended food preparations and surface conditions; inspect food damage.
2. **Wire catching:** check whether the blade or dumpling skin catches, tears, or
   becomes trapped at wires, crossings, and the exit edge. Treat the perforated
   concept as a separate test article if it is ever built.
3. **Scraper-induced rack movement:** measure rack translation, rotation, and
   lifting under contact, together with changes in gap and relative support
   height. Also check tray motion and loss of support during transfer.
4. **Tool and tray handling:** validate actual grasp/release, reach, clearance,
   and stability. Do not infer tray ungraspability from the chosen task policy.
5. **Food hygiene:** establish an appropriate food-contact cleaning and handling
   protocol for the real materials and robot tooling, including contamination and
   spill handling. Simulation recovery of a counter/floor proxy is not permission
   to return real spilled food to service.

The configured cooling timer provides no thermal or hygiene validation and no
claim that a particular wait duration makes food safe to handle or serve.

## Validation evidence and limits

The verified [native contract report](../../artifacts/rack-contracts-final.json)
records **21/21 native checks passed**, with `complete=true`, `passed=true`, and
`uses_privileged_test_setup=true`. It covers both wire and concept scenes: measured
gap and height ordering, initial food support, deployment rejection of human
interventions, elapsed-time waiting, food-location classification, rack movement
under an applied test force, and the tray's free joint. A training check confirms
that two explicit interventions incur cost without restarting cooling.

**These are contract checks using privileged setup, not successful robot pickup or
sweep evidence.** Food is explicitly placed for classification tests, and rack
movement is induced by an external test force rather than a robot scraper stroke.
The tray check establishes that it is not fixed; it does not establish a feasible
grasp. The wait check uses a shortened 0.2-second threshold, not a validation of the
default cooling interval or any thermal behavior. Pickup and sweep are not claimed
to be working on the strength of this report.

From the assigned worktree, run the combined regression suite:

```bash
scripts/with_env.sh pytest -q \
  tests/environments/racktoserving3d \
  tests/environments/sweepoffshelf3d/test_symbolic.py \
  tests/environments/sweepoffshelf3d/test_cli.py \
  tests/test_cli.py \
  tests/test_step_protocol.py \
  tests/test_step_protocol_generic.py
```

These tests include configuration, geometry, task semantics, and CLI/routing
contracts. Record the actual result and skip reasons for the revision tested;
passing tests alone do not demonstrate successful physical robot manipulation.

The final [test output](../../artifacts/test-summary-final.txt) records **325 passed,
4 skipped**. The four skips are existing shelf symbolic combinations in which a
fallen wiper cannot also be held. Package mypy, Ruff check, formatting check, and
`git diff --check` pass. The original shelf environment and its tests have no diff.

- [Wire-rack two-view video](../../artifacts/rack-serving-final/two-views.mp4) and
  [matching report](../../artifacts/rack-serving-final/report.json) show an actual
  robot attempt, not a successful transfer montage. Readiness waiting and scraper
  pickup succeed. Sweep preparation is rejected because the arm would intersect
  the counter/tray. Return reaches the home area but fails the seating controller's
  5.1 mm tracking check; it is not scored as a successful return. Explicit training
  human recovery follows, costing 5 units per requested intervention.
- [Robot reset probe](../../artifacts/rack-reset-final/two-views.mp4) and
  [report](../../artifacts/rack-reset-final/report.json) start with **privileged
  test setup** placing one proxy on the tray, clearly labeled in the video.
  The robot's grasp approach fails IK; there is no successful robot return to claim.
  The subsequent human batch return is a separate, charged operation.
- [Concept-platform attempt](../../artifacts/rack-individual-v4/two-views.mp4) and
  [report](../../artifacts/rack-individual-v4/report.json) retain a failed individual
  pickup from the distinct perforated geometry. That recording precedes the
  final reduction of food pregrasp height from 12 cm to 4 cm; do not treat it as
  final-controller validation or evidence about the purchased wire rack.
- **No hardware evidence is available.** The cooled-cooked-food checks above remain
  required; there is no hardware-validation result implied by this document.

## Launch-readiness follow-up, 2026-10-31

**Not launch-ready: physical delivery remains a blocker.** The evidence above is
retained history; the following wire-rack runs use the revised controller. No scene
geometry, masses, friction coefficients, planner search algorithm, or old benchmark
implementation was changed in this follow-up. Cooling physics is not required:
these transfer checks use `cooling_seconds=0`, not a thermal model.

The controller now circulates around the counter and pushes from behind the rack.
It retains the 45-degree scraper tilt, collision-checks the complete blade and
handle, and remeasures the tool-to-hand transform after transport and during the
stroke. This corrects the stale-grasp transform caused by physical twisting in
the pads. Tiny physical rack rotations no longer invalidate legitimate shallow
upward support; deep penetration and side collisions remain rejected. Scraper
return reorients outside the counter, then verifies actual support before release.

| Check | Observed result | Evidence |
| --- | --- | --- |
| 0-degree, 0.12 m/s sweep | All five briefly contact the tray, then four spill; sweep stops at a clearance guard. Return seats the scraper successfully. **Not a successful delivery.** | [Video](../../artifacts/rack-push-feedback-v2/two-views.mp4), [report](../../artifacts/rack-push-feedback-v2/report.json) |
| 0-degree, 0.03 m/s sweep | Stroke and scraper return finish without controller error, but all five spill onto the floor. **Command completion is not task success.** | [Video](../../artifacts/rack-slow-v1/two-views.mp4), [report](../../artifacts/rack-slow-v1/report.json) |
| Slow-sweep velocity trace | During the stroke, measured scraper translation is about 0.027 m/s; food can move substantially faster after reaching the tray. This is not simply a speed parameter being ignored. | [Trace](../../artifacts/rack-slow-trace/physics-trace.jsonl), [report](../../artifacts/rack-slow-trace/report.json) |
| Passive tray diagnostic | Three privileged drop/tilt cases remain on the tray after five seconds. Contacts use dimension 3: configured torsional/rolling coefficients are not active. This does not by itself establish the cause of sweep spills. | [Report](../../artifacts/rack-passive-contact.json) |
| Individual robot reset | A fresh helper picks one tray food in 455 control steps and returns it to the rack in 361. The initial tray placement is privileged test setup, not a successful sweep. This is one feasible reset, not clutter-wide recovery coverage. | [Report](../../artifacts/rack-reset-integrated-food-final-p/report.json) |

The same integrated reset is reproduced with rendering in the
[two-view video](../../artifacts/rack-reset-integrated-food-video-q/two-views.mp4)
and [report](../../artifacts/rack-reset-integrated-food-video-q/report.json).

The two tested sweep settings do **not** prove that all parameters fail. A
near-vertical pushing pose is a proposed controller experiment, not an approved
change to the retained 45-degree pose or a demonstrated fix.

The broad scraper pickup targets the actual handle rather than the thin blade and
requires bilateral handle contact. Individual food pickup uses a calibrated angled
grasp and partial closure rather than crushing the small proxy with full closure.
These are simulator calibrations, not hardware grasp guarantees. Robot reset
failures must remain visible; training may use the explicit costed human fallback.

The CLI now has an exact-control-step adapter with separate training/deployment
environments, actual-state goal checking, and interruption adoption without an
extra physics step. The bounded method preflight plan is in
[rack-cli-preflight.md](rack-cli-preflight.md); a parser accepting solver flags is
not evidence that either solver ran successfully. No learning run was launched.
Both Curtis/determinized A* and expectimax now have rack-specific domain wiring;
their search implementations are unchanged. The deployment-value approximation
and its limitations are documented in that preflight plan.

Validation after these changes: **373 targeted regression tests passed**, four
skipped and one deselected ([output](../../artifacts/rack-followup-tests.txt));
**24 rack-domain tests passed** ([output](../../artifacts/rack-domain-tests.txt));
**14 coverage-manifest tests passed**. Mypy passes for the 21 environment modules
and the two domain/CLI modules; targeted Ruff checks pass. The refreshed native
contract report passes **21/21 checks** for timing, classification, human costs,
deployment rejection, and both surface geometries
([report](../../artifacts/rack-contracts-followup.json)). The separate native
three-control-step interruption test also passes. None of these checks substitutes
for a stable physical sweep delivery.

The initial five-food layout is fixed. Distinct random seeds alone are not distinct
initial layouts. Coverage must evaluate the seeded human batch-return layouts and
ask whether **each layout has at least one successful parameter setting**, not
whether one shared setting solves all layouts. That coverage is still unverified.
`scripts/validate_rack_coverage.py` currently writes a bounded, explicitly untested
manifest only; it does not execute or aggregate a sweep grid. Its
[draft manifest](../../artifacts/rack-coverage-draft/manifest.json) starts no workers
and makes no coverage claim.
Do not launch method comparisons before a stable physical delivery witness and
the bounded learner integration checks pass. Record wire and concept results
separately; neither rigid-proxy success nor failure establishes real cooked-food
release, adhesion, deformation, or snagging behavior.

## Full-background rendering, 2026-11-01

The demo script now accepts `--scene-bg`, using the existing KINDER/MimicLabs
background rather than inventing a kitchen reconstruction. The wide view includes
the room; the second view retains the close oblique view of rack and tray. Reports
record `scene_bg`, and the video includes an explicit `air/other` count for food
that has no supported-location classification. This flag does not change the
scraper controller, friction, or food geometry.

The first background render exposed a scoring bug: the ground plane belongs to
the background's `scene` body, not `world`, so floor spills were unclassified.
The backend now recognizes contacted horizontal planes as floor candidates;
the existing food-height check still applies. Walls, boxes, and separated contacts
do not acquire floor semantics. A regression test fails before the fix and passes
afterward. The earlier `rack-background-demo-20261101` recording is retained but
must not be used for its floor counts.

To reproduce a full two-view cycle, use a fresh output directory and the normal
bounded `run-heavy` scope:

```bash
scripts/with_env.sh python -m scripts.rack_serving_preflight \
  --output artifacts/NEW-background-demo --scene-bg \
  --cooling-seconds 0 --speed .03 --recovery
```

The recovery portion includes explicit training human interventions; it is not
an autonomous deployment success or an overnight experiment.
