from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict


class State(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    data: dict[Object, np.ndarray]

    def __getitem__(self, obj: Object) -> np.ndarray:  # noqa: PLR0917 (dunder: no `*` possible)
        return self.data[obj]


class Object(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    type: Type


class Type(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    parent: Type | None = None


Action = np.ndarray
Cost = float

for _model in (State, Object, Type):
    _model.model_rebuild()
