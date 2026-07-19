from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator


class State(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    data: dict[Object, np.ndarray]

    @model_validator(mode="after")
    def _check_feature_dims(self) -> State:
        for obj, features in self.data.items():
            if len(features) != obj.type.dim:
                raise ValueError(
                    f"{obj.name} has {len(features)} features, but its type "
                    f"{obj.type.name!r} declares {obj.type.dim}: {obj.type.feature_names}"
                )
        return self

    def __getitem__(self, obj: Object) -> np.ndarray:  # noqa: PLR0917 (dunder: no `*` possible)
        return self.data[obj]

    def get(self, *, obj: Object, feature_name: str) -> float:
        """Look up a named feature (e.g. "x") rather than a raw vector index."""
        idx = obj.type.feature_names.index(feature_name)
        return self.data[obj][idx]

    def set(self, *, obj: Object, feature_name: str, feature_val: float) -> None:
        """Set a named feature (e.g. "x") rather than a raw vector index."""
        idx = obj.type.feature_names.index(feature_name)
        self.data[obj][idx] = feature_val


class Object(BaseModel):
    """A ground instance of a Type: a specific, named entity (e.g. "block1" of Type
    "block"), as opposed to Type itself, which is just the category. This is what
    makes GroundAtom "ground" — a Predicate applied to actual Objects, not free
    variables ranging over a Type."""

    model_config = ConfigDict(frozen=True)

    name: str
    type: Type


class Type(BaseModel):
    """Declares a schema of named features (e.g. ["x", "y", "z"]), not just a name —
    this is what lets State enforce that every Object's raw feature vector actually
    matches its type, and what lets State.get/set look up a feature by name instead
    of a raw index. No parent/inheritance: nothing here consumes it yet (no
    is_instance()/ancestor-walking exists), and no current domain needs a type
    hierarchy — see core/README.md for when it'd earn its way back in."""

    model_config = ConfigDict(frozen=True)

    name: str
    feature_names: tuple[str, ...]

    @property
    def dim(self) -> int:
        return len(self.feature_names)


Action = np.ndarray
