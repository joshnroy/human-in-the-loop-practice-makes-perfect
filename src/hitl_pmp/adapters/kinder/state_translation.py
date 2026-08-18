"""Lossless two-way translation between KINDER's `ObjectCentricState` and `core.State`.

The two type systems describe the same thing in different vocabularies. KINDER uses
`relational_structs` -- `ObjectCentricState`, `Object`, `Type`, with the feature schema
held *beside* the type in the state's own `type_features` map. `core/problem/` uses its
own `State`/`Object`/`Type`, with the schema declared *on* the type
(`Type.feature_names`). Neither knows about the other, which is exactly why this sits in
`adapters/`: the import-linter contract puts this layer directly above `core` and below
`environments`, so a domain can reach it and `core` can stay ignorant of KINDER.

## Why the whole schema, and not the subset a domain reads

The obvious economy -- keep only the features a domain's own predicates touch -- is the
thing that makes this translation useless. KINDER's classifiers are not arithmetic over
poses: `Tossing3DStateAbstractor._check_holding` pushes the state into a live PyBullet
sim and runs forward kinematics off the arm joints. So a `core.State` carrying four of
the robot's thirty-eight features cannot rebuild a state the abstractor will accept, and
"the state is a lossy projection" stops being a documented caveat and becomes a wrong
answer.

Copying the schema wholesale also removes a class of drift. A hand-written subset has to
be checked against upstream by a test; a copied schema is right by construction, and a
feature upstream renames surfaces as a missing key rather than as a silently shifted
index.

## What is deliberately asymmetric

Going *to* KINDER, an object the translator does not know is **dropped**; an object it
knows and the `core.State` omits is a **`KeyError`**. That is not an inconsistency, it is
the only pair of behaviours that is safe in each direction:

- A domain may carry pseudo-objects KINDER has no place for -- Tossing3D's `scene`, which
  holds the seed a scene rebuild needs and the step count `set_state` refuses on. Passing
  one to KINDER is meaningless, so it is dropped.
- An object KINDER *expects* cannot be defaulted. A zero vector is a real pose as far as
  the abstractor is concerned, so a gap has to be loud.
"""

from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict

from hitl_pmp.core.problem.environment.types import Object, State, Type


class KinderStateTranslator(BaseModel):
    """A fixed cast of objects and types, and the translation between the two worlds.

    Built once per scene from a KINDER state, then reused: the cast of a KINDER task is
    fixed by its task JSON, so nothing here has to be re-derived per step. Frozen,
    because a translator that changed under a cached abstract state would silently
    invalidate it.

    `kinder_type_features` is carried verbatim rather than rebuilt, because
    `ObjectCentricState`'s constructor wants exactly that mapping and reconstructing an
    equivalent one is a chance to get the ordering wrong.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    # Keyed by KINDER's own type / object *names*, which are what a domain names in its
    # operator declarations. Both sides of each mapping are needed: the core half to hand
    # to `core` code, the KINDER half to rebuild a state KINDER accepts.
    core_types: dict[str, Type]
    core_objects: dict[str, Object]
    kinder_types: dict[str, Any]
    kinder_objects: dict[str, Any]
    kinder_type_features: Any

    @staticmethod
    def from_kinder_state(*, kinder_state: Any) -> "KinderStateTranslator":
        """Derive the whole translation from one KINDER state.

        Every type the state *knows about* gets a `core.Type`, not only those with an
        object present -- the mapping then covers any state of this scene rather than
        just this one, and an unused entry costs nothing.
        """
        core_types: dict[str, Type] = {}
        kinder_types: dict[str, Any] = {}
        for kinder_type, feature_names in kinder_state.type_features.items():
            core_types[kinder_type.name] = Type(
                name=kinder_type.name, feature_names=tuple(feature_names)
            )
            kinder_types[kinder_type.name] = kinder_type

        core_objects: dict[str, Object] = {}
        kinder_objects: dict[str, Any] = {}
        for kinder_object in kinder_state:
            core_objects[kinder_object.name] = Object(
                name=kinder_object.name, type=core_types[kinder_object.type.name]
            )
            kinder_objects[kinder_object.name] = kinder_object

        return KinderStateTranslator(
            core_types=core_types,
            core_objects=core_objects,
            kinder_types=kinder_types,
            kinder_objects=kinder_objects,
            kinder_type_features=kinder_state.type_features,
        )

    def to_core_state(self, *, kinder_state: Any) -> State:
        """One KINDER state as a `core.State`, every feature carried across.

        `float64` throughout. KINDER's own observation space is float32 and MuJoCo
        integrates in float64; taking whatever the state holds and widening it means this
        translation never introduces a rounding step of its own.
        """
        return State(
            data={
                self.core_objects[kinder_object.name]: np.asarray(
                    kinder_state[kinder_object], dtype=np.float64
                ).copy()
                for kinder_object in kinder_state
            }
        )

    def to_kinder_state(self, *, state: State) -> Any:
        """A `core.State` back as a KINDER `ObjectCentricState`.

        Imported here rather than at module scope so this module stays importable without
        the optional extra -- the same lazy-import discipline the rest of this package
        follows.
        """
        from relational_structs import ObjectCentricState

        by_name = {obj.name: vector for obj, vector in state.data.items()}
        data: dict[Any, np.ndarray] = {}
        for name, kinder_object in self.kinder_objects.items():
            if name not in by_name:
                raise KeyError(
                    f"this core.State has no object named {name!r}, which the KINDER "
                    f"scene requires: it carries {sorted(by_name)}. A missing object "
                    "cannot be defaulted -- a zero vector is a real pose to KINDER's "
                    "own classifiers."
                )
            data[kinder_object] = np.asarray(by_name[name], dtype=np.float64).copy()
        return ObjectCentricState(data, self.kinder_type_features)
