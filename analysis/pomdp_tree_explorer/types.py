"""Small cached metadata; full search traces remain on disk."""

from typing import Any

from pydantic import BaseModel


class DecisionEntry(BaseModel):
    offset: int
    length: int
    metadata: dict[str, Any]


class DecisionIndex(BaseModel):
    signature: tuple[int, int]
    entries: list[DecisionEntry]
