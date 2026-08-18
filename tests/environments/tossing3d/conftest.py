"""Shared fixtures for the Tossing3D tests.

## Why `no_kinder_import` exists

Four tests in this package assert that some operation stays **lazy** -- constructing a
`Tossing3DEnvironment`, registering the domain's CLI, toggling substep recording,
resolving a playback rate for a run that writes no video. Laziness is load-bearing:
`hitl_pmp.cli` imports every registered environment's CLI, so if this domain's import
chain reached MuJoCo then `--env lightswitch` would stop working on a machine without the
optional extra.

They used to say so with `assert "mujoco" not in sys.modules`. That is a **session-global
proxy for a per-call property**, and it holds only while nothing earlier in the entire run
has built a scene. It was safe when this package's tests were mostly offline. It is not
safe now: the simulator-backed tests share the same session, so whether these four pass
depends on pytest's file collection order rather than on the code they are testing. On a
sibling branch, where more of the suite had become simulator-backed, exactly this broke --
`test_toggling_substep_recording_imports_no_simulator` failed while the laziness it names
was completely intact.

This fixture asserts the property directly instead. Every KINDER import in this package
goes through one door -- `KinderBackend.api()`, which is the only place `import kinder`
appears -- so making that door raise turns "did this operation import KINDER?" into a
question about *this* call rather than about the whole process. Order-independent, and it
names the offender when it trips.

**It is narrower than what it replaces, and neither form dominates.** This is a
*behavioural* guard: it catches code that reaches the import door at runtime. A
module-scope `import kinder` added to, say, `cli.py` would sail straight past it, and the
`sys.modules` check would have caught that. The trade is deliberate -- a guard that fails
for reasons unrelated to the code under test gets weakened or deleted, and this one
cannot -- but the gap is real and worth knowing rather than discovering.

**It is hardening rather than a bug fix on this branch**: the four tests pass here either
way, because the collection order happens to be favourable. The value is that they will
keep passing for the right reason as more of this package becomes simulator-backed.
"""

import pytest


@pytest.fixture
def no_kinder_import(*, monkeypatch: pytest.MonkeyPatch):
    """Fail loudly, and by name, if the code under test imports KINDER at all."""
    from hitl_pmp.environments.tossing3d.kinder_backend import KinderBackend

    def refuse(self) -> None:  # noqa: PLR0917 - stands in for a bound method
        raise AssertionError(
            "this operation imported KINDER; it is supposed to stay lazy so that the rest "
            "of the CLI still works on a machine without the optional extra"
        )

    monkeypatch.setattr(KinderBackend, "api", refuse)
    return refuse
