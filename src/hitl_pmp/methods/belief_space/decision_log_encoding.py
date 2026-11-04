"""Render belief-space diagnostics so two identically seeded runs log the same bytes.

Two things made the decision log differ between processes. Skills, ground skills and
their effects hold frozensets, whose iteration order follows string hashing and so
`PYTHONHASHSEED`; pydantic's JSON dump and its repr both list a frozenset in that order.
Predicates hold lambdas, whose default repr carries a memory address. Here a set is
listed sorted and an address is dropped; nothing else about the rendering changes, so a
consumer reading `action["skill"]["name"]` or matching a predicate name in an atom's text
reads what it always did.

This is applied where a record is rendered for the log, never to anything the planner
computes or looks up.
"""

import json
import re
from collections.abc import Iterable, Mapping
from typing import Any, ClassVar

from pydantic import BaseModel
from pydantic_core import to_jsonable_python

from hitl_pmp.core.method.types import GroundSkill
from hitl_pmp.core.problem.tasks.types import GroundAtom

_ADDRESS = re.compile(r" at 0x[0-9a-fA-F]+")


class DecisionLogEncoding:
    """Hash-order- and address-free JSON values and texts for the decision log."""

    MAX_CACHED_VALUES: ClassVar[int] = 4096
    # Keyed by identity and holding the model, as `GroundSkillText` does: search logs the
    # same ground skill objects thousands of times, and an id is never reused while cached.
    _values: ClassVar[dict[int, tuple[BaseModel, Any]]] = {}

    @staticmethod
    def value(*, value: Any) -> Any:
        """`model_dump(mode="json", fallback=str)`, with sets sorted and addresses dropped.

        A frozen model's rendering never changes, so it is computed once per object. The
        returned structure may be shared between records and must not be mutated."""
        if isinstance(value, BaseModel) and value.model_config.get("frozen"):
            cached = DecisionLogEncoding._values.get(id(value))
            if cached is not None and cached[0] is value:
                return cached[1]
            rendered = DecisionLogEncoding._render(value=value)
            if len(DecisionLogEncoding._values) >= DecisionLogEncoding.MAX_CACHED_VALUES:
                DecisionLogEncoding._values.clear()
            DecisionLogEncoding._values[id(value)] = (value, rendered)
            return rendered
        return DecisionLogEncoding._render(value=value)

    @staticmethod
    def text(*, value: Any) -> str:
        """`str(value)`, with sets sorted and addresses dropped."""
        if isinstance(value, BaseModel):
            return " ".join(
                DecisionLogEncoding._field_text(name=name, value=item)
                for name, item in value.__repr_args__()
            )
        return DecisionLogEncoding._repr(value=value)

    @staticmethod
    def atoms(*, atoms: Iterable[GroundAtom]) -> list[str]:
        """The sorted texts of a set of atoms, as `sorted(str(atom) ...)` gave them."""
        return sorted(DecisionLogEncoding.text(value=atom) for atom in atoms)

    @staticmethod
    def ground_skill_keys(
        *, costs: Mapping[str, Any], ground_skills: Iterable[GroundSkill]
    ) -> dict[str, Any]:
        """Re-render keys written as `str(ground_skill)`, possibly suffixed.

        The forecast dictionaries are keyed by `str(ground_skill)`, the lookup key search
        uses, which is consistent within one process but not across two. The keys are
        rewritten only as they are logged; a key naming no listed ground skill is kept.
        Insertion order is kept, so the log lists the forecasts as before."""
        texts = {
            str(ground_skill): DecisionLogEncoding.text(value=ground_skill)
            for ground_skill in ground_skills
        }
        # Longest first, so a prefix that is itself a ground skill's text never wins.
        prefixes = sorted(texts, key=len, reverse=True)

        def rekey(*, key: str) -> str:
            if key in texts:
                return texts[key]
            prefix = next((p for p in prefixes if key.startswith(p)), None)
            return key if prefix is None else texts[prefix] + key[len(prefix) :]

        return {rekey(key=key): item for key, item in costs.items()}

    @staticmethod
    def _render(*, value: Any) -> Any:
        if isinstance(value, BaseModel):
            return {
                name: DecisionLogEncoding.value(value=getattr(value, name))
                for name, field in type(value).model_fields.items()
                if not field.exclude
            }
        if isinstance(value, (set, frozenset)):
            items = [DecisionLogEncoding.value(value=item) for item in value]
            return sorted(items, key=lambda item: json.dumps(item, sort_keys=True))
        if isinstance(value, (list, tuple)):
            return [DecisionLogEncoding.value(value=item) for item in value]
        if isinstance(value, dict):
            return {
                DecisionLogEncoding._key(key=key): DecisionLogEncoding.value(value=item)
                for key, item in value.items()
            }
        return to_jsonable_python(value, fallback=DecisionLogEncoding._fallback)

    @staticmethod
    def _key(*, key: Any) -> Any:
        rendered = to_jsonable_python(key, fallback=DecisionLogEncoding._fallback)
        return rendered if isinstance(rendered, str) else json.dumps(rendered)

    @staticmethod
    def _fallback(value: Any) -> str:  # noqa: PLR0917 -- pydantic passes it positionally
        return _ADDRESS.sub("", str(value))

    @staticmethod
    def _field_text(*, name: str | None, value: Any) -> str:
        rendered = DecisionLogEncoding._repr(value=value)
        return rendered if name is None else f"{name}={rendered}"

    @staticmethod
    def _repr(*, value: Any) -> str:
        if isinstance(value, BaseModel):
            fields = ", ".join(
                DecisionLogEncoding._field_text(name=name, value=item)
                for name, item in value.__repr_args__()
            )
            return f"{type(value).__name__}({fields})"
        if isinstance(value, (set, frozenset)):
            if not value:
                return f"{type(value).__name__}()"
            items = ", ".join(sorted(DecisionLogEncoding._repr(value=item) for item in value))
            return f"{{{items}}}" if type(value) is set else f"{type(value).__name__}({{{items}}})"
        if isinstance(value, tuple):
            parts = [DecisionLogEncoding._repr(value=item) for item in value]
            return f"({parts[0]},)" if len(parts) == 1 else f"({', '.join(parts)})"
        if isinstance(value, list):
            return f"[{', '.join(DecisionLogEncoding._repr(value=item) for item in value)}]"
        if isinstance(value, dict):
            items = ", ".join(
                f"{DecisionLogEncoding._repr(value=key)}: {DecisionLogEncoding._repr(value=item)}"
                for key, item in value.items()
            )
            return f"{{{items}}}"
        return _ADDRESS.sub("", repr(value))
