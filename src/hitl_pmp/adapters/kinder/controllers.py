"""KINDER's lifted parameterized controllers, in `core`'s vocabulary.

Every KINDER environment exposes `create_lifted_controllers(action_space, ...)`, returning
`{name: LiftedParameterizedController}`. Each of those declares typed `variables`, grounds
to concrete objects, and carries its own `sample_parameters(state, rng)`. This module
turns that into the three things `core` asks for -- a `Skill`'s `parameters`, a draw of
its continuous parameters, and an execution -- **by delegation, never reimplementation**.

## Why the sampler is delegated rather than described

This is the part that was actually wrong before, not merely duplicated. hitl declared its
own `TOSS_RELEASE_MS_BOUNDS = (300, 1400)` beside the controller's own measured band of
`(700, 840)`, so it drew from a window about nine times too wide and the large majority of
its draws could not score. Nothing detected it, because both numbers were internally
consistent -- there was simply no mechanism by which upstream narrowing its band could
narrow hitl's. `sample_params` below calls `sample_parameters` on the controller itself,
so there is no second number to keep in step.

The state is passed through, not withheld. `SkillProvider.sample_params` is documented as
a *state-independent* draw, and most KINDER samplers are (`MoveToTossLocationAndToss`
opens with `del x`), but `PickCubeController`'s reads the target's pose and
rejection-tests the resulting base pose against other cubes. Handing it a blank state
would silently change what upstream samples, which is the failure this module exists to
prevent -- so the live state goes in and the deviation is stated here rather than hidden.

## The `?` that has broken this repo before

KINDER follows PDDL convention and names its variables `?robot`. `core.Variable.name`
deliberately does **not** carry the sigil -- `PddlWriter._variable_str` adds one at write
time -- so a name crossing unchanged renders `??robot`, which Fast Downward's translator
splits into two tokens. Every plan call in the domain then fails, and silently, because
`EesMethod._next_plan` catches `PlanningFailure` and degrades to a no-op. `variables()`
strips it.

## What stays outside this module

Building the gym environment, choosing a camera, collecting frames, and deciding when the
scene is rebuilt are all the *domain's* business, so `run` takes the simulator step as an
injected callable rather than owning an environment. That is what keeps this package
generic across every KINDER environment instead of growing a Tossing3D-shaped hole.
"""

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict

from hitl_pmp.adapters.kinder.state_translation import KinderStateTranslator
from hitl_pmp.adapters.kinder.types import ControllerRun
from hitl_pmp.core.method.types import Variable

# KINDER names its variables in PDDL style; `core` does not. See the module docstring.
KINDER_VARIABLE_PREFIX = "?"


class KinderControllers(BaseModel):
    """One environment's lifted controllers, plus the translation around them."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    translator: KinderStateTranslator
    # `create_lifted_controllers(...)`'s own return value, held opaquely so this class is
    # generic over every KINDER environment.
    lifted_controllers: dict[str, Any]

    def variables(self, *, key: str) -> tuple[Variable, ...]:
        """One controller's typed parameters, as `core.Variable`s.

        What a domain checks its own hand-written `Skill.parameters` against: the operator
        *semantics* are the domain's to state, but the signature is upstream's, and a
        fidelity test comparing the two makes a renamed or reordered parameter loud.
        """
        controller = self._controller(key=key)
        return tuple(
            Variable(
                name=kinder_variable.name.removeprefix(KINDER_VARIABLE_PREFIX),
                type=self.translator.core_types[kinder_variable.type.name],
            )
            for kinder_variable in controller.variables
        )

    def sample_params(
        self,
        *,
        key: str,
        object_names: Sequence[str],
        state: Any,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Draw this controller's continuous parameters, from its own sampler.

        `state` is a `core.State`; it is translated back to a KINDER state on the way in,
        because that is what the sampler reads.
        """
        ground = self._ground(key=key, object_names=object_names)
        kinder_state = self.translator.to_kinder_state(state=state)
        return np.asarray(ground.sample_parameters(kinder_state, rng), dtype=float)

    def run(
        self,
        *,
        key: str,
        object_names: Sequence[str],
        params: np.ndarray | None,
        state: Any,
        step: Callable[[np.ndarray], Any],
        limit: int,
        reset_kwargs: Mapping[str, Any] | None = None,
    ) -> ControllerRun:
        """Drive one controller to termination, and report what happened.

        The loop body is upstream's own, from its `test_*_parameterized_skills` tests:
        `controller.step()` into the simulator, then `controller.observe(next_state)`,
        breaking on `terminated()`. `step` is injected and must return the KINDER state
        the simulator reached.

        Neither a controller that exceeds `limit` nor one that raises propagates: both are
        ordinary outcomes of continuous parameters that did not work out, and the caller
        has to be able to keep going. See `types.ControllerRun`.

        `reset_kwargs` is forwarded verbatim and only when supplied, because these
        keywords are per-controller (`disable_collision_objects` on one, `release_speed`
        on another) and passing one to a controller that does not declare it is a
        `TypeError`.
        """
        ground = self._ground(key=key, object_names=object_names)
        kinder_state = self.translator.to_kinder_state(state=state)
        try:
            ground.reset(kinder_state, params, **dict(reset_kwargs or {}))
        except Exception as exc:  # noqa: BLE001  (any planner failure is a failed skill)
            return ControllerRun(steps=0, terminated=False, error=f"{type(exc).__name__}: {exc}")

        for index in range(limit):
            try:
                next_state = step(ground.step())
            except Exception as exc:  # noqa: BLE001  (same reasoning as above)
                return ControllerRun(
                    steps=index, terminated=False, error=f"{type(exc).__name__}: {exc}"
                )
            ground.observe(next_state)
            if ground.terminated():
                return ControllerRun(steps=index + 1, terminated=True)
        return ControllerRun(steps=limit, terminated=False)

    def _controller(self, *, key: str) -> Any:
        if key not in self.lifted_controllers:
            raise KeyError(
                f"no controller named {key!r}; this environment offers "
                f"{sorted(self.lifted_controllers)}"
            )
        return self.lifted_controllers[key]

    def _ground(self, *, key: str, object_names: Sequence[str]) -> Any:
        controller = self._controller(key=key)
        objects = tuple(self.translator.kinder_objects[name] for name in object_names)
        return controller.ground(objects)
