"""One live Tossing3D scene, shared across this package's tests.

## Why there is a shared scene at all

This domain used to be testable offline. `observations.py` built hand-made
`KinderObservation`s -- a plain-floats boundary type -- so the state translation, the six
predicate classifiers, the skills and the oracle could all be exercised with no MuJoCo.
That layer is gone, and not by accident: the predicates are now upstream's
(`kinder_models.dynamic3d.tossing.state_abstractions`), `Holding` runs forward kinematics
off the arm joints, and `MovableInGoalRegion` reads the goal region off the live
simulator through the same `check_in_region` call `_check_goals()` makes. A predicate that
consults a `PyBulletSim` cannot be evaluated against a dict of floats.

So every test in this package that needs a `State` needs a simulator. That is the
accepted cost of not re-implementing upstream's symbolic layer -- see
`environments/tossing3d/predicates.py` for the trade, and CLAUDE.md for why KINDER became
a required dependency and CI installs it.

## Why session-scoped, and what that buys

Measured on this machine: constructing `Tossing3DEnvironment` is free (it is lazy), the
**first** `reset_to_seed` costs **3.4 s** because it builds the gym env, the MuJoCo model
and the `PyBulletSim`, and every subsequent reset costs **0.3 s**. Building one scene per
test would therefore add roughly 3.1 s x (number of tests) of pure setup for no additional
coverage.

`live_env` resets the one shared scene to `CANONICAL_SEED` before handing it over, so each
test still starts from the same known state -- the isolation that matters -- while paying
the build once per session.

## When *not* to use these fixtures

Build your own environment, and close it, whenever the test:

* closes the environment, or asserts on `close()`;
* needs a different `variant` (e.g. `o2`, which `backend()` refuses);
* needs `record_substeps` on, or any other backend flag that changes the wrapper stack;
* asserts about construction itself (that constructing imports no simulator).

Sharing a scene across those would leak state into unrelated tests. Everything else --
reading features, evaluating predicates, grounding skills, sampling parameters -- is a
read of the scene and belongs on the shared one.
"""

import importlib.util

import pytest

from hitl_pmp.core.problem.environment.types import State
from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment

# Upstream's own `test_pick_ground_toss` seed, and the one most numbers in this package's
# docs were measured at.
CANONICAL_SEED = 125

# The import package is `kinder`; the distribution is `kindergarden`. Both halves are
# checked because `kinder_models` lives in a different repo and a half-populated
# `reference/` is a real state a worktree can be in.
requires_kinder = pytest.mark.skipif(
    importlib.util.find_spec("kinder") is None or importlib.util.find_spec("kinder_models") is None,
    reason="KINDER is an optional extra (`kindergarden` + `kinder_models`)",
)


@pytest.fixture(scope="session")
def _shared_scene():
    """The one live scene for the whole session. Never depend on this directly."""
    env = Tossing3DEnvironment()
    env.reset_to_seed(seed=CANONICAL_SEED)
    try:
        yield env
    finally:
        env.close()


@pytest.fixture
def live_env(*, _shared_scene: Tossing3DEnvironment) -> Tossing3DEnvironment:
    """The shared scene, rewound to `CANONICAL_SEED` before this test runs."""
    _shared_scene.reset_to_seed(seed=CANONICAL_SEED)
    return _shared_scene


@pytest.fixture
def live_state(*, live_env: Tossing3DEnvironment) -> State:
    """The translated `core.State` of the canonical scene's initial state."""
    return live_env.get_current_state()
