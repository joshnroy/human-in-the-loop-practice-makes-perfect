# adapters/kinder

A **generic** bridge from any KINDER environment to this project's `core`
vocabulary. Nothing in this package names a particular KINDER environment;
`environments/tossing3d/` is its first consumer, not its subject.

## Why one bridge works for every KINDER environment

Every KINDER environment ships the same two entry points:

```text
create_lifted_controllers(action_space, ...) -> dict[str, LiftedParameterizedController]
<Env>StateAbstractor(sim).state_abstractor(state)  -> RelationalAbstractState
```

So the symbolic layer and the executable layer both have a fixed shape, and the
only per-environment knowledge left is *which* predicates and controllers a
domain's operators are written over. That is genuinely domain knowledge, and it
stays in the domain.

## The four modules

| module | what it does |
| --- | --- |
| `bootstrap.py` | the EGL/`DISPLAY` dance that has to happen before `import kinder`, and the module-not-package import that makes registration work |
| `state_translation.py` | `ObjectCentricState` ↔ `core.State`, losslessly, both directions |
| `abstraction.py` | the one-shot state abstractor as per-predicate `core.Predicate`s, cached |
| `controllers.py` | lifted controllers as `core.Variable`s, parameter draws, and executions |

`types.py` holds `ControllerRun`, the one piece of plain data this package
produces.

## Delegation, not reimplementation — and what that fixed

The rule is that no number upstream owns is written down again here.

The concrete case: hitl declared `TOSS_RELEASE_MS_BOUNDS = (300, 1400)` while
the controller's own measured band was `(700, 840)` — a sampling window about
nine times too wide, so the large majority of draws could not score. Both
numbers were internally consistent and nothing detected the gap, because there
was no mechanism by which upstream narrowing its band could narrow hitl's.
`KinderControllers.sample_params` now calls the controller's own
`sample_parameters`, so there is no second number.

The same rule applies to the symbolic layer. A predicate here is a membership
test in the atom set upstream computed, not a reimplemented classifier, so
"six classifiers kept in agreement with upstream's by test" stops being a thing
that can drift.

## Three things the shapes force

**The abstractor computes everything at once, and it is expensive.** It does a
`PyBulletSim.set_state(state)` — a full forward-kinematics pass — then
classifies. `core.Predicate.holds` asks one predicate at a time and
`Goal.is_satisfied` asks for all of them in a row, so the naive wrapper pays for
that pass once per predicate. Hence a cache.

**The abstractor is not pure, so the state is not a sufficient cache key.**
`Tossing3DStateAbstractor.state_abstractor` says so itself: *"poses come from
`state`, the goal region from the live simulator"*. One `core.State` therefore
maps to different atom sets either side of a scene rebuild. The key is
`(generation, state contents)`, and the *simulator's owner* bumps the generation
through `invalidate()` — nothing about a `core.State` can reveal that
`env.reset(seed=...)` happened, but the code that called it knows exactly.

**The whole feature schema crosses, not the subset a domain reads.** KINDER's
classifiers are not arithmetic over poses; `_check_holding` runs forward
kinematics off the arm joints. A `core.State` carrying four of the robot's
thirty-eight features cannot rebuild a state the abstractor will accept.

## Two translation traps

**`?robot` must not reach PDDL.** KINDER names variables in PDDL style;
`core.Variable.name` deliberately carries no sigil because
`PddlWriter._variable_str` adds one at write time. A name crossing unchanged
renders `??robot`, which Fast Downward's translator splits into two tokens —
and the failure is silent, because `EesMethod._next_plan` catches
`PlanningFailure` and degrades to a no-op. `KinderControllers.variables` strips
it.

**`params_space` is `None`.** `LiftedParameterizedController` declares the
attribute but every Tossing3D controller leaves it unset, so a skill's
`param_dim` cannot be read off the controller. A domain declares it, and a
simulator-gated fidelity test compares the declaration against what the
controller's own sampler actually returns.

## What is deliberately asymmetric

Translating *to* KINDER, an object this package does not know is **dropped**; an
object KINDER expects and the `core.State` omits is a **`KeyError`**. Both are
the only safe behaviour in their direction: a domain may carry pseudo-objects
KINDER has no place for (Tossing3D's `scene`, holding the seed a rebuild needs),
while an object KINDER expects cannot be defaulted, because a zero vector is a
real pose as far as the classifiers are concerned.

Likewise, an atom over a predicate the domain did not declare **raises**.
Dropping it would leave the operator model quietly incomplete, which is the
exact class of drift this package exists to remove.

## What stays outside

Building the gym environment, choosing a camera, collecting frames, and deciding
when the scene was rebuilt are the domain's business. `KinderControllers.run`
takes the simulator step as an injected callable rather than owning an
environment, which is what keeps this package generic instead of growing a
Tossing3D-shaped hole.

## Testing

`tests/adapters/kinder/` uses hand-built states and stand-in controllers, and
runs offline — `relational_structs` is pure Python and imports without MuJoCo.
The properties under test are about *delegation and caching*, which a real
simulator would bury under seconds of motion planning. The real abstractor and
the real controllers are exercised through these same entry points in
`tests/environments/tossing3d/`.
